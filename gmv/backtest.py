"""Backtest machinery: look-ahead-safe data access, drifting weights, the past-data rule, and inference over days."""
import numpy as np


class Past:
    """Rows before day t, at most `limit` of them; every read is checked."""
    def __init__(self, r, t, limit):
        assert 0 < limit <= t <= len(r), (limit, t, len(r))
        self.r, self.t, self.limit = r, t, limit

    def last(self, L):
        assert 0 < L <= self.limit, f"reads {L} rows, allowed {self.limit}"
        out = self.r[self.t - L:self.t]
        assert len(out) == L
        return out


def hold(W, Rp):
    """Buy W (K x p) and hold over Rp (days x p): daily returns (K x days) and the drifted weights at the end."""
    G = np.cumprod(1 + Rp, axis=0)
    V = W @ G.T
    return V / np.c_[np.ones(len(W)), V[:, :-1]] - 1, W * G[-1] / V[:, -1:]


def turnover(Ws, Dend):
    """Σ|w_new - drifted old| at each rebalance (K x rebalances), NaN at the first."""
    out = np.full(Ws.shape[:2], np.nan)
    out[1:] = np.abs(Ws[1:] - Dend[:-1]).sum(axis=2)
    return out.T


def rule(rets, Ws, Dend, cands, reb, look):
    """At each rebalance, the candidate with the lowest variance of its out-of-sample returns over the last
    `look` days. Returns the rule's daily returns, its turnover and the index chosen (-1 before the look-back)."""
    n = len(Ws)
    ch = np.full(n, -1)
    out = np.full(rets.shape[1], np.nan)
    turn = np.full(n, np.nan)
    for k in range(n):
        s = k * reb
        if s < look:
            continue
        c = cands[int(np.argmin(rets[cands, s - look:s].var(axis=1)))]
        ch[k] = c
        out[s:s + reb] = rets[c, s:s + reb]
        if ch[k - 1] >= 0:
            turn[k] = np.abs(Ws[k, c] - Dend[k - 1, ch[k - 1]]).sum()
    return out, turn, ch


def net(G, U, reb_days, cost):
    """Daily returns net of `cost` basis points times turnover on each rebalance day."""
    X = G.copy()
    X[:, :, reb_days] -= cost * 1e-4 * np.nan_to_num(U)
    return X


def nw_tstat(d):
    """t-statistic of the mean of a daily series, Newey-West standard error."""
    T = len(d)
    L = int(4 * (T / 100) ** (2 / 9))
    e = d - d.mean()
    s = e @ e / T
    for k in range(1, L + 1):
        s += 2 * (1 - k / (L + 1)) * (e[k:] @ e[:-k]) / T
    return d.mean() / np.sqrt(s / T)


def block_counts(T, rng, block, boot=1000):
    """boot x T: how often each day enters a moving-block bootstrap resample."""
    nb = -(-T // block)
    starts = rng.integers(0, T - block + 1, size=(boot, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(boot, -1)[:, :T]
    C = np.zeros((boot, T))
    np.add.at(C, (np.repeat(np.arange(boot), T), idx.ravel()), 1)
    return C


def boot_var(X, C):
    """X: portfolios x series x days; C: resample weights (rows sum to 1). Variance per series and resample,
    averaged over portfolios."""
    flat = X.reshape(-1, X.shape[2]).astype(float)
    m1, m2 = flat @ C.T, (flat ** 2) @ C.T
    return (m2 - m1 ** 2).reshape(X.shape[0], X.shape[1], -1).mean(axis=0)


def compare(X, bv, i, j):
    """Variance ratio of series i to j with its 95% bootstrap interval, and the Newey-West t of the daily
    difference in squared deviations (negative: i is less risky)."""
    v = X.var(axis=2, ddof=1).mean(axis=0)
    r = bv[i] / bv[j]
    dev = (X[:, [i, j]] - X[:, [i, j]].mean(axis=2, keepdims=True)) ** 2
    return {"ratio": v[i] / v[j], "95% low": np.quantile(r, 0.025), "95% high": np.quantile(r, 0.975),
            "t-stat": nw_tstat((dev[:, 0] - dev[:, 1]).mean(axis=0))}
