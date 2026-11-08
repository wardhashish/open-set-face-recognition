"""gate_eval.py -- the scoreboard used for every model (given; do not edit).

All thresholds are chosen on val. The metric functions you implemented in Part 2 are passed in, so this file
contains no metric logic of its own.
"""
VERSION = "A4-2026.1"

from types import SimpleNamespace

import numpy as np

from gate_train import l2n, pair_scores

NEEDED = ("f1_sweep", "roc_metrics", "open_set_decide", "open_set_f1", "best_open_set_threshold", "dir_at_far")


def f1_at(s, y, t):
    pred = s >= t
    return float(2 * (pred & (y == 1)).sum() / max(pred.sum() + (y == 1).sum(), 1))


def neg_weights(y_val, pi_target):
    pv = y_val.mean()
    return np.where(y_val == 1, 1.0, (pv / (1 - pv)) * ((1 - pi_target) / pi_target))


class Scoreboard:
    def __init__(self, D, **fns):
        missing = [k for k in NEEDED if k not in fns]
        if missing:
            raise ValueError(f"missing functions: {missing}")
        self.D, self.f = D, SimpleNamespace(**fns)
        self.results, self.emb = {}, {}

    def open_set_scan(self, E, split):
        o = self.D[split].os
        pred, dmin, _ = self.f.open_set_decide(E[o.gal_idx], E[o.probe_idx], np.inf)
        return o.gal_y[pred], dmin, o.probe_y

    def evaluate(self, name, E_val, E_test, normalize=True, tau_v=None, tau_o=None, score_fn=None, store=True):
        f, V, T = self.f, self.D.val, self.D.test
        if normalize:
            E_val, E_test = l2n(E_val), l2n(E_test)
        sc = score_fn or (lambda E, i, j: pair_scores(E, i, j))
        sv, st = sc(E_val, V.pairs.i, V.pairs.j), sc(E_test, T.pairs.i, T.pairs.j)
        yv, yt = V.pairs.y, T.pairs.y
        tau_naive = f.f1_sweep(sv, yv)["best_thr"]
        if tau_v is None:
            tau_v = f.f1_sweep(sv, yv, neg_weights(yv, yt.mean()))["best_thr"]
        r = f.roc_metrics(st, yt)
        out = dict(V_F1=f1_at(st, yt, tau_v), V_F1_naive=f1_at(st, yt, tau_naive), AUC=r["auc"], EER=r["eer"],
                   TAR1e2=r["tar"][1e-2], TAR1e3=r["tar"][1e-3])
        for tier, nm in ((1, "V_F1_hard"), (0, "V_F1_rand")):
            m = T.pairs.tier == tier
            out[nm] = f1_at(st[m], yt[m], tau_v)
        pid_v, dm_v, tr_v = self.open_set_scan(E_val, "val")
        pid_t, dm_t, tr_t = self.open_set_scan(E_test, "test")
        if tau_o is None:
            tau_o = f.best_open_set_threshold(dm_v, pid_v, tr_v)[0]
        o = f.open_set_f1(pid_t, dm_t <= tau_o, tr_t)
        out.update(O_F1=o["f1"], O_prec=o["prec"], O_FAR=o["far"], DIR_far1pct=f.dir_at_far(dm_t, pid_t, tr_t, 0.01)[0])
        out["S"] = 0.5 * (out["V_F1"] + out["O_F1"])
        out.update(tau_v=float(tau_v), tau_o=float(tau_o))
        if store:
            self.results[name] = out
            self.emb[name] = (E_val, E_test)
        return out

    def show(self, names=None):
        cols = ["V_F1", "V_F1_naive", "V_F1_hard", "V_F1_rand", "AUC", "EER", "TAR1e2", "O_F1", "DIR_far1pct", "S"]
        print(f"{'model':<26}" + "".join(f"{c:>12}" for c in cols))
        for n in (names or self.results):
            print(f"{n:<26}" + "".join(f"{self.results[n][c]:>12.4f}" for c in cols))
