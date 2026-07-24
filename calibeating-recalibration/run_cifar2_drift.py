import fire
import os
import numpy as np
import torch
import torchvision as tv
from torch.utils.data import DataLoader

from models import DenseNet
from train_cifar2 import BinaryCifar
from scalar_calibeating import ScalarCalibeating, ScalarStreamMetrics
from recalibration import OnlineRecalibrator


def cifar2_drift(data, save, depth=40, growth_rate=12,
                 eps=0.05, n_conf_bins=15,
                 n_passes=100, flip_at_pass=50, flip_prob=0.2,
                 log_every=20000, stream_batch_size=128,
                 shuffle_passes=True, seed=0, cascade_seed=7,
                 flip_seed=1234):
    if (depth - 4) % 3:
        raise Exception('Invalid depth')
    block_config = [(depth - 4) // 6 for _ in range(3)]
    model = DenseNet(growth_rate=growth_rate, block_config=block_config,
                     num_classes=1)
    model.load_state_dict(torch.load(os.path.join(save, 'model.pth'),
                                     map_location='cpu'))
    model = model.cuda().eval()
    for p in model.parameters():
        p.requires_grad_(False)

    mean = [0.4914, 0.4822, 0.4465]
    stdv = [0.2470, 0.2435, 0.2616]
    test_transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=mean, std=stdv),
    ])
    base_test = tv.datasets.CIFAR10(data, train=False, transform=test_transforms,
                                    download=True)
    test_set = BinaryCifar(base_test)
    total_T = len(test_set) * n_passes
    flip_at_step = len(test_set) * flip_at_pass
    flip_rng = np.random.default_rng(flip_seed)

    print('CIFAR-2 cats-vs-dogs drift stream')
    print('  T=%d, flip starts at step %d (pass %d), flip_prob=%.2f'
          % (total_T, flip_at_step, flip_at_pass, flip_prob))

    calib = ScalarCalibeating(n_conf_bins=n_conf_bins)
    recal = OnlineRecalibrator(eps=eps, T=total_T, L=2.0, seed=seed)
    calib_stage1 = ScalarCalibeating(n_conf_bins=n_conf_bins)
    recal_stage2 = OnlineRecalibrator(eps=eps, T=total_T, L=2.0, seed=cascade_seed)

    metrics_full = {k: ScalarStreamMetrics() for k in ('F0', 'F1', 'F2', 'F3')}
    metrics_post = {k: ScalarStreamMetrics() for k in ('F0', 'F1', 'F2', 'F3')}

    print('  F0 raw  F1 calibeating(%d bins)  F2 recal(eps=%.3f, m=%d)  F3 cascade'
          % (n_conf_bins, eps, recal.m))

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
                logits = model(batch_x).squeeze(-1)
                p_raw_batch = torch.sigmoid(logits).cpu().numpy()

                for i in range(batch_x.size(0)):
                    global_step += 1
                    p_raw = float(p_raw_batch[i])
                    y_orig = int(batch_y[i].item())
                    if (global_step > flip_at_step
                            and flip_rng.uniform() < flip_prob):
                        y_int = 1 - y_orig
                    else:
                        y_int = y_orig

                    c1 = calib.predict_and_update(p_raw, y_int)
                    c2, _ = recal.predict_sample_update(p_raw, y_int)
                    c3a = calib_stage1.predict_and_update(p_raw, y_int)
                    c3, _ = recal_stage2.predict_sample_update(c3a, y_int)

                    for fk, fv in (('F0', p_raw), ('F1', c1),
                                   ('F2', c2), ('F3', c3)):
                        metrics_full[fk].update(fv, y_int)
                        if global_step > flip_at_step:
                            metrics_post[fk].update(fv, y_int)

                    if global_step % log_every == 0 or global_step == total_T:
                        msg = '[%7d] phase=%s  ' % (
                            global_step,
                            'POST-FLIP' if global_step > flip_at_step else 'PRE-FLIP')
                        for fk in ('F0', 'F1', 'F2', 'F3'):
                            r = metrics_full[fk].report()
                            msg += '%s B=%.4f ECE=%.4f  ' % (fk, r['brier'], r['ece'])
                        print(msg)

    def print_table(metrics, name):
        print('\n=== %s (n=%d) ===' % (name, metrics['F0'].n))
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

    print_table(metrics_full, 'Cumulative full stream (eps=%.3f, m=%d)'
                % (eps, recal.m))
    print_table(metrics_post, 'POST-FLIP only (the drift-stress phase)')


if __name__ == '__main__':
    fire.Fire(cifar2_drift)
