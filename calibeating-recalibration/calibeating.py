import torch


class SimpleCalibeating:

    def __init__(self, num_classes, binning='conf', n_conf_bins=15,
                 fallback='b', device=None):
        if binning not in ('conf', 'topclass', 'topclass-conf'):
            raise ValueError("binning must be 'conf', 'topclass', or 'topclass-conf'")
        if fallback not in ('b', 'uniform'):
            raise ValueError("fallback must be 'b' or 'uniform'")
        self.num_classes = num_classes
        self.binning = binning
        self.n_conf_bins = n_conf_bins
        self.fallback = fallback
        self.device = torch.device(device) if device is not None else torch.device('cpu')
        self.counts = {}
        self.sums = {}

    def _bin_id(self, p):
        max_p, argmax = torch.max(p, dim=0)
        conf_idx = int(max_p.item() * self.n_conf_bins)
        if conf_idx >= self.n_conf_bins:
            conf_idx = self.n_conf_bins - 1
        if self.binning == 'conf':
            return conf_idx
        if self.binning == 'topclass':
            return int(argmax.item())
        return int(argmax.item()) * self.n_conf_bins + conf_idx

    def predict(self, p):
        bid = self._bin_id(p)
        if bid not in self.counts:
            if self.fallback == 'b':
                return p.detach().clone()
            return torch.full_like(p, 1.0 / self.num_classes)
        return (self.sums[bid] / self.counts[bid]).to(p.device)

    def update(self, p, y):
        bid = self._bin_id(p)
        if bid not in self.counts:
            self.counts[bid] = 0
            self.sums[bid] = torch.zeros(self.num_classes, device=self.device)
        self.counts[bid] += 1
        self.sums[bid][int(y)] += 1.0

    def predict_and_update(self, p, y):
        c = self.predict(p)
        self.update(p, y)
        return c

    def num_bins_seen(self):
        return len(self.counts)


class TopLabelCalibeating:

    def __init__(self, num_classes, n_conf_bins=15, decimals=None):
        self.num_classes = num_classes
        self.n_conf_bins = n_conf_bins
        self.decimals = decimals
        self.counts = {}
        self.positives = {}

    def _bin_id(self, max_p):
        if self.decimals is not None:
            return int(round(max_p * (10 ** self.decimals)))
        idx = int(max_p * self.n_conf_bins)
        if idx >= self.n_conf_bins:
            idx = self.n_conf_bins - 1
        return idx

    def predict(self, p):
        max_p, argmax = torch.max(p, dim=0)
        max_p_f = float(max_p)
        argmax_i = int(argmax)
        bid = self._bin_id(max_p_f)
        n = self.counts.get(bid, 0)
        if n == 0:
            q = max_p_f
        else:
            q = self.positives[bid] / n

        rest_mass = 1.0 - max_p_f
        out = torch.empty_like(p)
        if rest_mass > 0:
            scale = (1.0 - q) / rest_mass
            out.copy_(p * scale)
        else:
            out.fill_((1.0 - q) / max(self.num_classes - 1, 1))
        out[argmax_i] = q
        return out

    def update(self, p, y):
        max_p, argmax = torch.max(p, dim=0)
        bid = self._bin_id(float(max_p))
        self.counts[bid] = self.counts.get(bid, 0) + 1
        if int(argmax) == int(y):
            self.positives[bid] = self.positives.get(bid, 0) + 1
        else:
            self.positives.setdefault(bid, 0)

    def predict_and_update(self, p, y):
        c = self.predict(p)
        self.update(p, y)
        return c

    def num_bins_seen(self):
        return len(self.counts)
