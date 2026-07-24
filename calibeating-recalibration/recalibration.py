import math
import numpy as np
import torch


def squared_loss(p, y):
    return (p - y) ** 2


def project_simplex_2d(v):
    t = 0.5 * (v[0] - v[1] + 1.0)
    t = max(0.0, min(1.0, t))
    return np.array([t, 1.0 - t])


class MLOOSquared:

    def __init__(self, m):
        self.m = m
        self.n = m + 1
        self.net = np.arange(self.n) / m

    def __call__(self, w_1, w_2, u1, q):
        p = self.net
        G = np.empty((self.n, 2))
        G[:, 0] = w_1 * u1 * (p - 0.0) + w_2 * (squared_loss(p, 0.0) - squared_loss(q, 0.0))
        G[:, 1] = w_1 * u1 * (p - 1.0) + w_2 * (squared_loss(p, 1.0) - squared_loss(q, 1.0))

        third_q = (G[:, 0] <= 0) & (G[:, 1] <= 0)
        if third_q.any():
            i = int(np.argmax(third_q))
            return np.array([i], dtype=np.int64), np.array([1.0])

        positive = G[:, 0] > 0
        if not positive.any():
            return np.array([self.m], dtype=np.int64), np.array([1.0])
        j_plus_1 = int(np.argmax(positive))
        j = j_plus_1 - 1
        if j < 0:
            return np.array([0], dtype=np.int64), np.array([1.0])

        h_j = G[j, 0] - G[j, 1]
        h_jp1 = G[j_plus_1, 0] - G[j_plus_1, 1]
        denom = h_jp1 - h_j
        if abs(denom) < 1e-12:
            t = 0.5
        else:
            t = (-h_j) / denom
            t = max(0.0, min(1.0, t))
        return np.array([j, j_plus_1], dtype=np.int64), np.array([1.0 - t, t])


class OnlineRecalibrator:

    def __init__(self, eps, T, L=2.0, eta_outer=None, eta_inner=None, seed=0):
        self.eps = eps
        self.L = L
        self.T = T
        self.m = max(1, math.ceil(8 * max(1.0, math.sqrt(L)) / eps))
        self.n = self.m + 1

        self.net = np.arange(self.n) / self.m
        self.eps1 = 1.0 / self.m
        self.eps2 = 4.0 * L / (self.m ** 2)
        self.mloo = MLOOSquared(self.m)

        self.u1 = np.zeros(self.n)
        self.w = np.array([0.0, 1.0])

        self.eta_inner = math.sqrt(self.n / max(T, 1)) if eta_inner is None else eta_inner
        self.eta_outer = (math.sqrt(2.0) / (2.0 * L * math.sqrt(max(T, 1)))
                          if eta_outer is None else eta_outer)

        self.rng = np.random.default_rng(seed)

    def _round_to_net(self, q):
        idx = int(round(float(q) * self.m))
        idx = max(0, min(self.m, idx))
        return idx, self.net[idx]

    def predict(self, q_t):
        _, q_round = self._round_to_net(q_t)
        idx, probs = self.mloo(self.w[0], self.w[1], self.u1, q_round)
        return idx, probs, q_round

    def sample(self, idx, probs):
        return int(self.rng.choice(idx, p=probs))

    def update(self, q_round, idx, probs, y_t):
        a = np.zeros(self.n)
        a[idx] = probs
        v1 = a * (self.net - y_t)
        v2 = float((a * (squared_loss(self.net, y_t) - squared_loss(q_round, y_t))).sum())
        g_outer = np.array([
            float(self.u1 @ v1) - self.eps1,
            v2 - self.eps2,
        ])
        self.w = project_simplex_2d(self.w + self.eta_outer * g_outer)
        self.u1 = np.clip(self.u1 + self.eta_inner * v1, -1.0, 1.0)

    def predict_sample_update(self, q_t, y_t):
        idx, probs, q_round = self.predict(q_t)
        i_chosen = self.sample(idx, probs)
        p_sampled = float(self.net[i_chosen])
        self.update(q_round, idx, probs, y_t)
        return p_sampled, q_round


class TopLabelRecalibrator:

    def __init__(self, num_classes, eps, T, seed=0):
        self.num_classes = num_classes
        self.recal = OnlineRecalibrator(eps=eps, T=T, L=2.0, seed=seed)

    @property
    def m(self):
        return self.recal.m

    def predict_and_update(self, p, y_true):
        max_p, argmax = torch.max(p, dim=0)
        max_p_f = float(max_p)
        argmax_i = int(argmax)
        y_binary = int(argmax_i == int(y_true))

        p_sampled, _ = self.recal.predict_sample_update(max_p_f, y_binary)

        out = torch.empty_like(p)
        rest_mass = 1.0 - max_p_f
        if rest_mass > 0:
            scale = (1.0 - p_sampled) / rest_mass
            out.copy_(p * scale)
        else:
            out.fill_((1.0 - p_sampled) / max(self.num_classes - 1, 1))
        out[argmax_i] = p_sampled
        return out
