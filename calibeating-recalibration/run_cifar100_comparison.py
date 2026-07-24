import fire
import os
import torch
import torchvision as tv
from torch.utils.data import DataLoader

from models import DenseNet
from calibeating import TopLabelCalibeating
from recalibration import TopLabelRecalibrator
from stream_metrics import StreamMetrics


def online_comparison(data, save, depth=40, growth_rate=12,
                      n_passes=100, eps=0.05, n_conf_bins=15,
                      log_every=200000, stream_batch_size=128,
                      shuffle_passes=True, seed=0):
    model_path = os.path.join(save, 'model.pth')
    if not os.path.exists(model_path):
        raise RuntimeError('Cannot find %s' % model_path)
    state_dict = torch.load(model_path, map_location='cpu')
    if (depth - 4) % 3:
        raise Exception('Invalid depth')
    block_config = [(depth - 4) // 6 for _ in range(3)]
    model = DenseNet(growth_rate=growth_rate, block_config=block_config,
                     num_classes=100)
    model.load_state_dict(state_dict)
    model = model.cuda().eval()
    for p in model.parameters():
        p.requires_grad_(False)

    mean = [0.5071, 0.4867, 0.4408]
    stdv = [0.2675, 0.2565, 0.2761]
    test_transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=mean, std=stdv),
    ])
    test_set = tv.datasets.CIFAR100(data, train=False, transform=test_transforms,
                                    download=True)

    total_T = len(test_set) * n_passes
    print('CIFAR-100, online stream of %d samples (%d test x %d passes)'
          % (total_T, len(test_set), n_passes))

    calib = TopLabelCalibeating(num_classes=100, n_conf_bins=n_conf_bins)
    recal = TopLabelRecalibrator(num_classes=100, eps=eps,
                                 T=total_T, seed=seed)
    calib_cascade = TopLabelCalibeating(num_classes=100, n_conf_bins=n_conf_bins)
    recal_cascade = TopLabelRecalibrator(num_classes=100, eps=eps,
                                         T=total_T, seed=seed + 7)

    print('  F0  raw softmax')
    print('  F1  simple calibeating (top-label, %d bins)' % n_conf_bins)
    print('  F2  recalibration (Blackwell, eps=%.3f, m=%d)' % (eps, recal.m))
    print('  F3  cascade: recalibration on calibeating output')

    metrics = {k: StreamMetrics(num_classes=100) for k in ('F0', 'F1', 'F2', 'F3')}
    g = torch.Generator()
    g.manual_seed(seed)
    global_step = 0

    with torch.no_grad():
        for pass_idx in range(n_passes):
            stream_loader = DataLoader(
                test_set, batch_size=stream_batch_size, pin_memory=True,
                shuffle=shuffle_passes,
                generator=g if shuffle_passes else None,
                num_workers=2,
            )
            for batch_x, batch_y in stream_loader:
                batch_x = batch_x.cuda(non_blocking=True)
                logits = model(batch_x)
                p_raw_batch = torch.softmax(logits, dim=1).cpu()

                for i in range(batch_x.size(0)):
                    global_step += 1
                    p_raw = p_raw_batch[i]
                    y_int = int(batch_y[i].item())

                    c_calib = calib.predict_and_update(p_raw, y_int)
                    c_recal = recal.predict_and_update(p_raw, y_int)
                    c_stage1 = calib_cascade.predict_and_update(p_raw, y_int)
                    c_cascade = recal_cascade.predict_and_update(c_stage1, y_int)

                    for fk, fv in (('F0', p_raw), ('F1', c_calib),
                                   ('F2', c_recal), ('F3', c_cascade)):
                        metrics[fk].update(fv, y_int)

                    if global_step % log_every == 0 or global_step == total_T:
                        print('[%7d]  ' % global_step + '  '.join(
                            '%s B=%.4f ECE=%.4f' % (
                                k, metrics[k].report()['brier'],
                                metrics[k].report()['ece'])
                            for k in ('F0', 'F1', 'F2', 'F3')))

    print('\n=== Cumulative final (n=%d, eps=%.3f, m=%d) ===' %
          (metrics['F0'].n, eps, recal.m))
    cols = ('forecaster', 'Brier', 'NLL', 'ECE_15bin',
            'ECE_L1_exact', 'K_L2_exact', 'Acc')
    print('  '.join('%-15s' % c for c in cols))
    print('-' * 105)
    labels = {'F0': 'F0 raw', 'F1': 'F1 calibeating',
              'F2': 'F2 recalibration', 'F3': 'F3 cascade'}
    for k in ('F0', 'F1', 'F2', 'F3'):
        r = metrics[k].report(k_resolution=2)
        print('  '.join([
            '%-15s' % labels[k],
            '%-15.5f' % r['brier'],
            '%-15.5f' % r['nll'],
            '%-15.5f' % r['ece'],
            '%-15.5f' % r['ece_l1_exact'],
            '%-15.6f' % r['k_l2_exact'],
            '%-15.4f' % r['acc'],
        ]))

    out_path = os.path.join(save, 'cifar100_comparison_metrics.pt')
    torch.save({'eps': eps, 'n_conf_bins': n_conf_bins, 'n_passes': n_passes,
                'final': {k: metrics[k].report(k_resolution=2)
                          for k in ('F0', 'F1', 'F2', 'F3')}},
               out_path)
    print('\nSaved: %s' % out_path)


if __name__ == '__main__':
    fire.Fire(online_comparison)
