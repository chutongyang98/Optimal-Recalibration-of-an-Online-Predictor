import fire
import os
import time
import torch
import torchvision as tv
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler
from models import DenseNet


CAT_CLASS = 3
DOG_CLASS = 5


class BinaryCifar(Dataset):

    def __init__(self, base):
        self.base = base
        self.indices = []
        for i in range(len(base)):
            _, label = base[i]
            if label == CAT_CLASS:
                self.indices.append((i, 0))
            elif label == DOG_CLASS:
                self.indices.append((i, 1))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        orig_idx, binary_label = self.indices[i]
        x, _ = self.base[orig_idx]
        return x, binary_label


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    sum_loss = 0.0
    bce = nn.BCEWithLogitsLoss(reduction='sum')
    with torch.no_grad():
        for x, y in loader:
            x = x.cuda(non_blocking=True)
            y = y.float().cuda(non_blocking=True)
            logits = model(x).squeeze(-1)
            sum_loss += bce(logits, y).item()
            preds = (logits > 0).long()
            correct += (preds == y.long()).sum().item()
            total += y.size(0)
    return sum_loss / total, correct / total


def train(data='/home/ubuntu/data/cifar10', save='runs/densenet_cifar2',
          valid_size=1000, seed=0,
          depth=40, growth_rate=12, n_epochs=100, batch_size=64,
          lr=0.1, wd=1e-4, momentum=0.9):
    torch.manual_seed(seed)
    if not os.path.exists(save):
        os.makedirs(save)

    if (depth - 4) % 3:
        raise Exception('Invalid depth')
    block_config = [(depth - 4) // 6 for _ in range(3)]

    mean = [0.4914, 0.4822, 0.4465]
    stdv = [0.2470, 0.2435, 0.2616]
    train_transforms = tv.transforms.Compose([
        tv.transforms.RandomCrop(32, padding=4),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=mean, std=stdv),
    ])
    test_transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=mean, std=stdv),
    ])

    print('Loading CIFAR-10...')
    base_train_aug = tv.datasets.CIFAR10(data, train=True, transform=train_transforms,
                                         download=True)
    base_train_eval = tv.datasets.CIFAR10(data, train=True, transform=test_transforms,
                                          download=False)
    base_test = tv.datasets.CIFAR10(data, train=False, transform=test_transforms,
                                    download=False)

    train_set = BinaryCifar(base_train_aug)
    valid_set = BinaryCifar(base_train_eval)
    test_set = BinaryCifar(base_test)
    print('CIFAR-2 (cats vs dogs): train=%d, test=%d' %
          (len(train_set), len(test_set)))

    indices = torch.randperm(len(train_set))
    train_indices = indices[:len(indices) - valid_size]
    valid_indices = indices[len(indices) - valid_size:]

    train_loader = DataLoader(train_set, pin_memory=True, batch_size=batch_size,
                              sampler=SubsetRandomSampler(train_indices))
    valid_loader = DataLoader(valid_set, pin_memory=True, batch_size=256,
                              sampler=SubsetRandomSampler(valid_indices))
    test_loader = DataLoader(test_set, pin_memory=True, batch_size=256, shuffle=False)

    model = DenseNet(growth_rate=growth_rate, block_config=block_config,
                     num_classes=1)
    if torch.cuda.device_count() > 1:
        model_wrapper = nn.DataParallel(model).cuda()
    else:
        model_wrapper = model.cuda()
    print(model_wrapper)

    bce = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model_wrapper.parameters(), lr=lr,
                          momentum=momentum, weight_decay=wd, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(0.5 * n_epochs), int(0.75 * n_epochs)], gamma=0.1)

    best_valid_acc = 0.0
    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        model_wrapper.train()
        running_loss = 0.0
        n_seen = 0
        n_correct = 0
        for x, y in train_loader:
            x = x.cuda(non_blocking=True)
            y = y.float().cuda(non_blocking=True)
            logits = model_wrapper(x).squeeze(-1)
            loss = bce(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            n_correct += ((logits > 0).long() == y.long()).sum().item()
            n_seen += x.size(0)
        train_loss = running_loss / n_seen
        train_acc = n_correct / n_seen

        valid_loss, valid_acc = evaluate(model_wrapper, valid_loader)
        scheduler.step()

        print('Epoch %3d  train_loss=%.4f train_acc=%.4f  '
              'valid_loss=%.4f valid_acc=%.4f  (%.1fs)' %
              (epoch, train_loss, train_acc, valid_loss, valid_acc,
               time.time() - t0))

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), os.path.join(save, 'model.pth'))
            torch.save(valid_indices, os.path.join(save, 'valid_indices.pth'))
            print('  saved (best valid acc: %.4f)' % best_valid_acc)

    test_loss, test_acc = evaluate(model_wrapper, test_loader)
    print('\nFinal best-checkpoint test_loss=%.4f test_acc=%.4f' %
          (test_loss, test_acc))


if __name__ == '__main__':
    fire.Fire(train)
