import math
from collections import defaultdict


class ScalarCalibeating:

    def __init__(self, n_conf_bins=15, fallback='p'):
        if fallback not in ('p', 'half'):
            raise ValueError("fallback must be 'p' or 'half'")
        self.n_conf_bins = n_conf_bins
        self.fallback = fallback
        self.counts = [0] * n_conf_bins
        self.sums = [0.0] * n_conf_bins

    def _bin_id(self, p):
        idx = int(p * self.n_conf_bins)
        if idx >= self.n_conf_bins:
            idx = self.n_conf_bins - 1
        if idx < 0:
            idx = 0
        return idx

    def predict(self, p):
        bid = self._bin_id(float(p))
        n = self.counts[bid]
        if n == 0:
            return float(p) if self.fallback == 'p' else 0.5
        return self.sums[bid] / n

    def update_state(self, p, y):
        bid = self._bin_id(float(p))
        self.counts[bid] += 1
        self.sums[bid] += float(y)

    def predict_and_update(self, p, y):
        c = self.predict(p)
        self.update_state(p, y)
        return c

    def num_bins_seen(self):
        return sum(1 for c in self.counts if c > 0)


class ScalarStreamMetrics:

    def __init__(self, ece_bins=15, eps_clip=1e-12):
        self.ece_bins = ece_bins
        self.eps_clip = eps_clip
        self.n = 0
        self.sum_brier = 0.0
        self.sum_nll = 0.0
        self.n_correct = 0
        self.preds = []
        self.ys = []

    def update(self, p, y):
        p = float(p)
        y = int(y)
        self.n += 1
        self.sum_brier += (p - y) ** 2
        if y == 1:
            self.sum_nll += -math.log(max(p, self.eps_clip))
        else:
            self.sum_nll += -math.log(max(1.0 - p, self.eps_clip))
        if (p > 0.5) == bool(y):
            self.n_correct += 1
        self.preds.append(p)
        self.ys.append(y)

    def ece_l1_binned(self):
        if self.n == 0:
            return 0.0
        edges = [(i / self.ece_bins, (i + 1) / self.ece_bins)
                 for i in range(self.ece_bins)]
        ece = 0.0
        for lo, hi in edges:
            in_bin = [(p, y) for p, y in zip(self.preds, self.ys)
                      if (lo < p <= hi) or (lo == 0.0 and p == 0.0)]
            n_in = len(in_bin)
            if n_in == 0:
                continue
            mean_p = sum(p for p, _ in in_bin) / n_in
            mean_y = sum(y for _, y in in_bin) / n_in
            ece += abs(mean_p - mean_y) * (n_in / self.n)
        return ece

    def k_l2_at_resolution(self, decimals=2):
        if self.n == 0:
            return 0.0
        n_v = defaultdict(int)
        sum_v = defaultdict(float)
        for p, y in zip(self.preds, self.ys):
            v = round(p, decimals)
            n_v[v] += 1
            sum_v[v] += y
        k = 0.0
        for v, n in n_v.items():
            mean_y = sum_v[v] / n
            k += (n / self.n) * (v - mean_y) ** 2
        return k

    def ece_l1_at_resolution(self, decimals=2):
        if self.n == 0:
            return 0.0
        n_v = defaultdict(int)
        sum_v = defaultdict(float)
        for p, y in zip(self.preds, self.ys):
            v = round(p, decimals)
            n_v[v] += 1
            sum_v[v] += y
        e = 0.0
        for v, n in n_v.items():
            mean_y = sum_v[v] / n
            e += (n / self.n) * abs(v - mean_y)
        return e

    def report(self, k_resolution=2):
        if self.n == 0:
            return {'n': 0}
        return {
            'n': self.n,
            'brier': self.sum_brier / self.n,
            'nll': self.sum_nll / self.n,
            'acc': self.n_correct / self.n,
            'ece': self.ece_l1_binned(),
            'ece_l1_exact': self.ece_l1_at_resolution(k_resolution),
            'k_l2_exact': self.k_l2_at_resolution(k_resolution),
        }
