from collections import defaultdict
import torch


class StreamMetrics:

    def __init__(self, num_classes, ece_bins=15):
        self.num_classes = num_classes
        self.ece_bins = ece_bins
        self.n = 0
        self.sum_brier = 0.0
        self.sum_nll = 0.0
        self.n_correct = 0
        self.confs = []
        self.preds = []
        self.trues = []

    def update(self, p, y):
        max_conf, pred = torch.max(p, dim=0)
        max_conf = float(max_conf)
        pred = int(pred)
        self.n += 1
        sq = float(p.pow(2).sum()) - 2.0 * float(p[y]) + 1.0
        self.sum_brier += sq
        self.sum_nll += float(-torch.log(p[y].clamp(min=1e-12)))
        if pred == y:
            self.n_correct += 1
        self.confs.append(max_conf)
        self.preds.append(pred)
        self.trues.append(int(y))

    def ece(self):
        if self.n == 0:
            return 0.0
        confs = torch.tensor(self.confs)
        preds = torch.tensor(self.preds)
        trues = torch.tensor(self.trues)
        accs = preds.eq(trues).float()
        edges = torch.linspace(0, 1, self.ece_bins + 1)
        ece = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            in_bin = (confs > lo) & (confs <= hi)
            n_in = int(in_bin.sum())
            if n_in == 0:
                continue
            acc_bin = float(accs[in_bin].mean())
            conf_bin = float(confs[in_bin].mean())
            ece += abs(conf_bin - acc_bin) * (n_in / self.n)
        return ece

    def k_l2_at_resolution(self, decimals):
        if self.n == 0:
            return 0.0
        n_v = defaultdict(int)
        pos_v = defaultdict(int)
        for c, p, y in zip(self.confs, self.preds, self.trues):
            v = round(c, decimals)
            n_v[v] += 1
            if p == y:
                pos_v[v] += 1
        n_total = sum(n_v.values())
        return sum((n / n_total) * (v - pos_v.get(v, 0) / n) ** 2
                   for v, n in n_v.items())

    def ece_l1_at_resolution(self, decimals):
        if self.n == 0:
            return 0.0
        n_v = defaultdict(int)
        pos_v = defaultdict(int)
        for c, p, y in zip(self.confs, self.preds, self.trues):
            v = round(c, decimals)
            n_v[v] += 1
            if p == y:
                pos_v[v] += 1
        n_total = sum(n_v.values())
        return sum((n / n_total) * abs(v - pos_v.get(v, 0) / n)
                   for v, n in n_v.items())

    def report(self, k_resolution=None):
        if self.n == 0:
            return {'n': 0}
        out = {
            'n': self.n,
            'brier': self.sum_brier / self.n,
            'nll': self.sum_nll / self.n,
            'acc': self.n_correct / self.n,
            'ece': self.ece(),
        }
        if k_resolution is not None:
            out['k_l2_exact'] = self.k_l2_at_resolution(k_resolution)
            out['ece_l1_exact'] = self.ece_l1_at_resolution(k_resolution)
        return out
