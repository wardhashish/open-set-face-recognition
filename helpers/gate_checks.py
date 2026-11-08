"""gate_checks.py -- check cells for the functions you implement (given; do not edit).

Each check receives the function(s) under test, so the notebook stays the only place where your code lives.
run_check(name, fn) prints a tick or the reason for failure and records the result in CHECKS.
"""
VERSION = "A4-2026.1"

import numpy as np
import torch
import torch.nn.functional as F

CHECKS = {}


def run_check(name, fn):
    try:
        fn(); CHECKS[name] = True; print(f"✔ {name}")
    except NotImplementedError:
        CHECKS[name] = False; print(f"✘ {name}: not implemented yet")
    except AssertionError as e:
        CHECKS[name] = False; print(f"✘ {name}: {e}")
    except Exception as e:
        CHECKS[name] = False; print(f"✘ {name}: {type(e).__name__}: {e}")


def check_f1_sweep(f1_sweep):
    rng = np.random.default_rng(0)
    for _ in range(25):
        n = 150; s = rng.integers(0, 12, n).astype(float); y = rng.random(n) < 0.3; w = rng.choice([1, 2, 3], n).astype(float)
        r = f1_sweep(s, y, w); best = -1
        assert len(r["thr"]) == len(np.unique(s)), "one entry per DISTINCT score expected (ties are a block)"
        assert np.all(np.diff(r["thr"]) < 0), "thr must be in descending order"
        for t in np.unique(s):
            pred = s >= t; tp = (w * (pred & y)).sum(); fp = (w * (pred & ~y)).sum(); fn = (w * (~pred & y)).sum()
            f = 2 * tp / (2 * tp + fp + fn); best = max(best, f)
            i = int(np.where(r["thr"] == t)[0][0]); assert abs(r["f1"][i] - f) < 1e-9, f"F1 wrong at threshold {t}"
        assert abs(r["best_f1"] - best) < 1e-9, "best_f1 wrong"
        rep = np.repeat(np.arange(n), w.astype(int)); assert abs(f1_sweep(s[rep], y[rep])["best_f1"] - r["best_f1"]) < 1e-9, "weights must act as multiplicities"
    s = np.r_[np.ones(5), np.zeros(5)]; y = np.r_[np.ones(5), np.zeros(5)]
    assert f1_sweep(s, y)["best_thr"] == 1.0 and f1_sweep(s, y)["best_f1"] == 1.0, "perfectly separable case"
    s = np.array([3., 3, 2, 2, 1]); y = np.array([1, 0, 1, 1, 0]); r = f1_sweep(s, y)
    assert r["best_thr"] == 2.0 and abs(r["best_f1"] - 6 / 7) < 1e-12, "hand example: expect thr=2, F1=6/7"


def check_roc(roc_metrics):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    s = np.r_[rng.normal(2, 1, 60000), rng.normal(0, 1, 60000)]; y = np.r_[np.ones(60000), np.zeros(60000)]
    m = roc_metrics(s, y)
    assert abs(m["auc"] - roc_auc_score(y, s)) < 1e-9, "AUC differs from the reference"
    assert abs(m["eer"] - 0.1587) < 0.006, f"EER of two unit Gaussians 2 sigma apart should be ~0.159, got {m['eer']:.4f}"
    assert m["fpr"][0] == 0 and m["tpr"][0] == 0, "ROC must start at (0,0)"
    si = rng.integers(0, 25, 1500).astype(float); yi = rng.random(1500) < 0.4
    mi = roc_metrics(si, yi, far_targets=(0.01, 0.1, 0.3)); assert abs(mi["auc"] - roc_auc_score(yi, si)) < 1e-9, "AUC with ties"
    for f, tv in mi["tar"].items():
        best = 0.0
        for t in np.unique(si):
            p = si >= t
            if (p & ~yi).sum() / (~yi).sum() <= f: best = max(best, (p & yi).sum() / yi.sum())
        assert abs(best - tv) < 1e-9, f"TAR@FAR={f}: expected {best:.4f}, got {tv:.4f}"


def check_open_set(open_set_decide, open_set_f1, best_open_set_threshold, dir_at_far):
    G = np.array([[0, 0], [10, 0.]]); gal_y = np.array([7, 9]); P = np.array([[.1, 0], [9.5, 0], [.2, 0], [5, 5], [20, 0.]]); true = np.array([7, 9, 9, -1, -1])
    pred, dmin, acc = open_set_decide(G, P, 1.0)
    assert np.allclose(dmin, [.01, .25, .04, 50, 100]) and list(pred[:3]) == [0, 1, 0] and list(acc) == [True, True, True, False, False], "open_set_decide on the hand example"
    r = open_set_f1(gal_y[pred], acc, true)
    assert abs(r["f1"] - 2 / 3) < 1e-12 and abs(r["prec"] - 2 / 3) < 1e-12 and r["far"] == 0, f"hand example at tau=1: expected F1=2/3, got {r}"
    pred, dmin, acc = open_set_decide(G, P, 60.0); r = open_set_f1(gal_y[pred], acc, true)
    assert abs(r["f1"] - 4 / 7) < 1e-12 and r["far"] == 0.5, "hand example at tau=60: F1=4/7, FAR=1/2"
    tau, f1 = best_open_set_threshold(dmin, gal_y[pred], true); assert abs(tau - 0.25) < 1e-9 and abs(f1 - 2 / 3) < 1e-12, f"best threshold: expected (0.25, 2/3), got ({tau}, {f1})"
    rng = np.random.default_rng(3)
    for _ in range(15):
        n = 250; dm = np.round(rng.random(n) * 5, 1); tr = np.where(rng.random(n) < .4, rng.integers(0, 5, n), -1); pr = np.where(rng.random(n) < .6, tr, rng.integers(0, 5, n))
        tau, f1 = best_open_set_threshold(dm, pr, tr)
        assert abs(f1 - max(open_set_f1(pr, dm <= t, tr)["f1"] for t in np.unique(dm))) < 1e-9, "best_open_set_threshold is not optimal"
        assert abs(open_set_f1(pr, dm <= tau, tr)["f1"] - f1) < 1e-9, "returned tau does not achieve the returned F1"
        for far in (0.0, 0.05, 0.2, 0.5):
            d_, t_ = dir_at_far(dm, pr, tr, far); unk = tr < 0
            assert (dm[unk] <= t_).mean() <= far + 1e-12, "dir_at_far: FAR constraint violated"
            assert not [t for t in np.unique(dm[unk]) if t > t_ and (dm[unk] <= t).mean() <= far + 1e-12], "dir_at_far: tau is not the largest feasible"
            assert abs(d_ - ((dm <= t_) & (pr == tr) & (tr >= 0)).sum() / (tr >= 0).sum()) < 1e-12, "DIR value"


def check_calib(platt_fit, shift_logit):
    rng = np.random.default_rng(5); n = 40000
    s = np.r_[rng.normal(1, 1, n), rng.normal(-1, 1, n)]; y = np.r_[np.ones(n), np.zeros(n)]
    A, Bb = platt_fit(s, y)                                   # equal-variance Gaussians => true log-odds is exactly 2*s
    assert abs(A - 2) < 0.08 and abs(Bb) < 0.08, f"expected (A,B)~(2,0), got ({A:.3f},{Bb:.3f})"
    A, Bb = platt_fit(s, y, np.where(y == 1, 1.0, 4.0))       # negatives x4  <=> prior 0.2
    assert abs(Bb - np.log(0.25)) < 0.1, f"with weights emulating prior 0.2, B should be ~log(1/4)={np.log(.25):.3f}, got {Bb:.3f}"
    assert abs(shift_logit(0.0, 0.5, 0.2) - np.log(0.25)) < 1e-12 and abs(shift_logit(1.3, 0.3, 0.3) - 1.3) < 1e-12


def check_pdist(pairwise_sq_dist):
    torch.manual_seed(0); E = torch.randn(7, 5); D_ = pairwise_sq_dist(E)
    assert D_.shape == (7, 7) and torch.allclose(D_, torch.cdist(E, E) ** 2, atol=1e-4), "values differ from ||a-b||^2"
    assert (D_ >= 0).all(), "negative squared distances (numerical error not clamped)"
    assert D_.diagonal().abs().max() < 1e-6, "diagonal must be (numerically) zero"
    E2 = torch.randn(4, 3); E2 = torch.cat([E2, E2[:1], torch.zeros(1, 3)]).requires_grad_()
    pairwise_sq_dist(E2).sum().backward(); assert torch.isfinite(E2.grad).all(), "non-finite gradient with duplicated / zero rows"


def check_triplet(triplet_loss):
    tl = triplet_loss
    y = torch.tensor([0, 0, 1, 1]); E = torch.tensor([[0, 0], [1, 0], [0, 3], [0, 4.]])
    for mn, exp in [("hard", 1.25), ("semihard", 1.25), ("all", 1.5)]:
        l, d = tl(E, y, 10.0, mn); assert abs(l.item() - exp) < 1e-5, f"{mn}: hand example 1 expects {exp}, got {l.item():.5f}"
    y = torch.tensor([0, 0, 1, 2]); E = torch.tensor([[0, 0], [2, 0], [1, 0], [0, 3.]])
    for mn, exp, fa in [("hard", 4.0, 1.0), ("semihard", 0.0, 0.0), ("all", 4.0, 0.5)]:
        l, d = tl(E, y, 1.0, mn); assert abs(l.item() - exp) < 1e-5, f"{mn}: hand example 2 expects {exp} (singleton identities have no positives!), got {l.item():.5f}"
        assert abs(float(d["frac_active"]) - fa) < 1e-5, f"{mn}: frac_active expected {fa}, got {float(d['frac_active']):.4f}"
    torch.manual_seed(0); ls = [tl(E, y, 1.0, "random")[0].item() for _ in range(800)]
    assert abs(np.mean(ls) - 2.0) < 0.25, f"random mining: expected mean loss ~2.0 on hand example 2, got {np.mean(ls):.3f}"
    yy = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, -1, -2, -3])
    for mn in ["hard", "semihard", "all", "random"]:
        Ee = torch.randn(12, 6, requires_grad=True); l, _ = tl(F.normalize(Ee, dim=1), yy, 0.3, mn); l.backward()
        assert torch.isfinite(l) and torch.isfinite(Ee.grad).all(), f"{mn}: NaN/inf in loss or gradient (distractors? empty masks?)"
    Ee = F.normalize(torch.randn(12, 6), dim=1); p = torch.randperm(12)
    for mn in ["hard", "semihard", "all"]:
        assert abs(tl(Ee, yy, 0.3, mn)[0].item() - tl(Ee[p], yy[p], 0.3, mn)[0].item()) < 1e-5, f"{mn}: loss must not depend on the order of the batch"
    yc = torch.tensor([0, 0, 1, 1, 2, 2]); Ec = F.normalize(torch.eye(6)[yc] + 0.01 * torch.randn(6, 6), dim=1)
    for mn in ["hard", "semihard", "all", "random"]: assert tl(Ec, yc, 0.3, mn)[0].item() == 0.0, f"{mn}: well separated clusters must give loss 0"


def check_grads(analytic_grads):
    torch.manual_seed(0)
    for _ in range(5):
        a, p, n = (torch.randn(6, dtype=torch.float64, requires_grad=True) for _ in range(3))
        L = ((a - p) ** 2).sum() - ((a - n) ** 2).sum() + 0.3; L.backward()
        ga, gp, gn = analytic_grads(a.detach(), p.detach(), n.detach())
        for g, t, nm in ((ga, a, "a"), (gp, p, "p"), (gn, n, "n")):
            assert torch.allclose(torch.as_tensor(g, dtype=torch.float64), t.grad, atol=1e-9), f"gradient wrt {nm} is wrong"


def check_pair(PairHead, pair_bce_loss):
    head = PairHead(4); y = torch.tensor([0, 0, 1, 1, -1]); E = torch.randn(5, 4)
    assert head(E[:2], E[2:4]).shape == (2,), "head must return one logit per pair"
    assert torch.allclose(head(E[:2], E[2:4]), head(E[2:4], E[:2])), "head must be symmetric in its two inputs"
    assert head(E[:3, None], E[None, :4]).shape == (3, 4), "broadcasting (n,1,D) vs (1,K,D) must give (n,K)"
    with torch.no_grad(): head.fc.weight.zero_(); head.fc.bias.fill_(1.0)
    y4 = torch.tensor([0, 0, 1, 1]); sp = lambda x: float(np.log1p(np.exp(x)))
    l, d = pair_bce_loss(E[:4], y4, head, "natural"); assert abs(l.item() - (2 * sp(-1) + 4 * sp(1)) / 6) < 1e-5, "natural BCE value (2 positive / 4 negative pairs)"
    l, d = pair_bce_loss(E[:4], y4, head, "balanced"); assert abs(l.item() - (sp(-1) + sp(1)) / 2) < 1e-5, "balanced BCE value"
    assert abs(float(d["pos_frac"]) - 1 / 3) < 1e-6, "pos_frac"
    Ew = E.clone().requires_grad_(); l, _ = pair_bce_loss(Ew, y, head, "balanced"); l.backward(); assert torch.isfinite(Ew.grad).all()


def check_scaling(tau_at_far, scaling_experiment, E):
    s = np.arange(100.0)                                   # 100 negatives with scores 0..99
    t = tau_at_far(s, 0.05); assert (s >= t).sum() == 5, f"FAR 5% of 100 negatives = 5 accepted, got {(s >= t).sum()}"
    t = tau_at_far(s, 0.0); assert (s >= t).sum() == 0, "FAR 0 => nothing accepted"
    far, per = scaling_experiment(E, 0.0, [5], n_rep=2, seed=0)        # tau_d = 0 => (almost) nothing accepted
    assert far[5][0] < 0.01 and per[5][1].sum() > 0, "scaling_experiment sanity"
    far, per = scaling_experiment(E, 10.0, [5], n_rep=2, seed=0)       # tau_d huge => everything accepted
    assert far[5][0] == 1.0, "with a huge threshold every stranger must be accepted"


def check_scn(scenario_rates, expected_daily_cost):
    dm = np.array([.1, .2, .3, 5, 6.]); pr = np.array([1, 2, 3, 1, 2]); tr = np.array([1, 2, 9, -1, -1])
    r = scenario_rates(dm, pr, tr, 1.0)
    assert abs(r["p_ok"] - 2 / 3) < 1e-12 and abs(r["p_wrong"] - 1 / 3) < 1e-12 and r["p_rej"] == 0 and r["p_stranger"] == 0, f"rates on the hand example: {r}"
    r2 = scenario_rates(dm, pr, tr, 5.5); assert abs(r2["p_stranger"] - 0.5) < 1e-12, "one of the two strangers is admitted at tau=5.5"
    sc = dict(N=1000, u=.2, c_fa=500, c_fr=3); assert abs(expected_daily_cost(r, sc) - 1000 * 0.8 * (1 / 3 * 500)) < 1e-6, "expected_daily_cost on the hand example"
