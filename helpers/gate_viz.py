VERSION = "A4-2026.1"

import matplotlib.pyplot as plt
import numpy as np

import gate_data as gd
from gate_train import l2n, pair_scores


def _kinds(flags):
    return ",".join(gd.KINDS[b] for b in range(len(gd.KINDS)) if flags >> b & 1)


def show_pairs(D, split, sel, title, ncol=6):
    s = D[split]
    fig, ax = plt.subplots(2, ncol, figsize=(2.1 * ncol, 4.6))
    for c, k in enumerate(sel[:ncol]):
        for r, q in enumerate((s.pairs.i[k], s.pairs.j[k])):
            ax[r, c].imshow(s.X[q]); ax[r, c].axis("off")
            kinds = _kinds(s.flags[q])
            ax[r, c].set_title(("degraded: " + kinds) if kinds else "clean", fontsize=7)
    fig.suptitle(title); plt.tight_layout(); plt.show()


def plot_hist(HIST, names, keys=("loss", "frac_active")):
    fig, ax = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 3.4))
    for k, a in zip(keys, np.atleast_1d(ax)):
        for n in names:
            a.plot(HIST[n]["step"], HIST[n][k], label=n)
        a.set_title(k); a.set_xlabel("step"); a.legend(fontsize=7); a.grid(alpha=.3)
    plt.tight_layout(); plt.show()


def show_failures(D, E_test, k=6):
    T = D.test
    E = l2n(E_test)
    s, y = pair_scores(E, T.pairs.i, T.pairs.j), T.pairs.y
    fa = np.flatnonzero(y == 0)[np.argsort(-s[y == 0])[:k]]
    fr = np.flatnonzero(y == 1)[np.argsort(s[y == 1])[:k]]
    fig, ax = plt.subplots(4, k, figsize=(2.0 * k, 8.4))
    for r, (sel, ttl) in enumerate(((fa, "FALSE ACCEPT"), (fr, "FALSE REJECT"))):
        for c, q in enumerate(sel):
            for h, p in enumerate((T.pairs.i[q], T.pairs.j[q])):
                a = ax[2 * r + h, c]; a.imshow(T.X[p]); a.axis("off")
                a.set_title((f"{ttl} s={s[q]:.2f}\n" if h == 0 else "") + (_kinds(T.flags[p]) or "clean"), fontsize=6)
    plt.tight_layout(); plt.show()


def plot_splits(D):
    names, cols = ["train", "val", "test"], ["#2b6cb0", "#d69e2e", "#c53030"]
    ids = [int(D.train.y.max()) + 1, D.val.n_ids, D.test.n_ids]
    photos = [len(D.train.y), D.val.n_multi, D.test.n_multi]
    dist = [len(D.train_dist.X), len(D.val.y) - D.val.n_multi, len(D.test.y) - D.test.n_multi]
    degr = [0.0, float((D.val.flags > 0).mean()), float((D.test.flags > 0).mean())]
    fig, ax = plt.subplots(1, 2, figsize=(10.8, 3.5))
    x, w = np.arange(3), 0.38
    for a, (a_vals, b_vals, title) in zip(ax, ((ids, dist, "people"), (photos, dist, "photos"))):
        a.bar(x - w / 2, a_vals, w, color=cols, label="people with 2 or more photos" if title == "people" else "photos of those people")
        a.bar(x + w / 2, b_vals, w, color=cols, alpha=0.35, hatch="//", label="distractors (people with one photo)")
        for xi, v1, v2 in zip(x, a_vals, b_vals):
            a.text(xi - w / 2, v1, f"{v1:,}", ha="center", va="bottom", fontsize=8); a.text(xi + w / 2, v2, f"{v2:,}", ha="center", va="bottom", fontsize=8)
        a.set_xticks(x); a.set_xticklabels(names); a.set_title(f"Number of {title} per split", fontsize=10); a.spines[["top", "right"]].set_visible(False)
        a.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax[1].set_xticklabels([f"{n}\n{int(100 * d)} % degraded" if d else n for n, d in zip(names, degr)])
    fig.suptitle("The benchmark: every person belongs to exactly one split", fontsize=10.5)
    plt.tight_layout(); plt.show()


def show_degradations(D, split="val", seed=0):
    s, rng = D[split], np.random.default_rng(seed)
    picks = [("clean", int(rng.choice(np.flatnonzero((s.flags == 0) & ~s.enroll & (s.y >= 0)))))]
    for k, nm in enumerate(gd.KINDS):
        single = np.flatnonzero(s.flags == (1 << k)); anyk = np.flatnonzero((s.flags >> k) & 1)
        pool = single if len(single) else anyk
        if len(pool):
            picks.append((nm, int(rng.choice(pool))))
    fig, ax = plt.subplots(1, len(picks), figsize=(1.75 * len(picks), 2.3))
    for a, (nm, q) in zip(np.atleast_1d(ax), picks):
        a.imshow(s.X[q]); a.set_title(nm, fontsize=9); a.axis("off")
    fig.suptitle("Degradations applied to 40 % of the val and test photos (1 or 2 per photo)", fontsize=10)
    plt.tight_layout(); plt.show()


def show_protocol(D, split="val", k=6, seed=0):
    s, o = D[split], D[split].os
    rng = np.random.default_rng(seed)
    gal_ids = o.gal_y[:k]
    known = []
    for g in gal_ids:
        cand = np.flatnonzero(o.probe_y == g)
        known.append(o.probe_idx[rng.choice(cand)] if len(cand) else None)
    stranger = rng.choice(o.probe_idx[o.probe_y < 0], k, replace=False)
    rows = [("gallery:\nenrolled photos\n(clean)", list(o.gal_idx[:k])), ("known probes:\nother photos of\nenrolled people", known), ("strangers:\nnot enrolled", list(stranger))]
    fig, ax = plt.subplots(3, k, figsize=(1.7 * k + 1.6, 5.4))
    for r, (label, idx) in enumerate(rows):
        for c in range(k):
            a = ax[r, c]; a.axis("off")
            if idx[c] is not None:
                a.imshow(s.X[idx[c]])
                kinds = _kinds(s.flags[idx[c]])
                if kinds:
                    a.set_title(kinds, fontsize=7, color="#c53030")
        ax[r, 0].text(-0.12, 0.5, label, transform=ax[r, 0].transAxes, ha="right", va="center", fontsize=8.5)
    fig.suptitle("Open-set protocol: a probe is matched to the nearest enrolled photo, or rejected as a stranger", fontsize=10)
    plt.tight_layout(); plt.show()
