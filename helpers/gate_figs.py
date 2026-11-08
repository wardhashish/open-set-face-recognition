"""gate_figs.py -- conceptual diagrams for the assignment (given; do not edit).

Every function draws one figure and shows it. The diagrams contain no data: they illustrate the ideas.
"""
VERSION = "A4-2026.1"

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

BLUE, RED, GREEN, ORANGE, GRAY, INK = "#2b6cb0", "#c53030", "#2f855a", "#dd6b20", "#718096", "#1a202c"
PALE = "#edf2f7"
IDC = ["#2b6cb0", "#d69e2e", "#805ad5", "#319795", "#dd6b20", "#b83280"]


def _canvas(w, h, xmax=None, ymax=None):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xmax or w); ax.set_ylim(0, ymax or h); ax.set_aspect("equal"); ax.axis("off")
    return fig, ax


def _box(ax, x, y, w, h, text, fc=PALE, ec="#4a5568", fs=9, lw=1.2, color=INK, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, fontweight="bold" if bold else "normal")


def _arrow(ax, p, q, color="#4a5568", lw=1.5, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, color=color, lw=lw, linestyle=ls, connectionstyle=f"arc3,rad={rad}"))


def _photo(ax, cx, cy, color, s=0.62, ring=None, lw=1.4):
    sq = FancyBboxPatch((cx - s / 2, cy - s / 2), s, s, boxstyle="round,pad=0,rounding_size=0.05", fc="#f7fafc", ec=ring or "#a0aec0", lw=(2.6 if ring else lw))
    ax.add_patch(sq)
    head = Circle((cx, cy + 0.10 * s), 0.20 * s, fc=color, ec="none")
    body = Ellipse((cx, cy - 0.34 * s), 0.74 * s, 0.56 * s, fc=color, ec="none")
    ax.add_patch(head); ax.add_patch(body); body.set_clip_path(sq)


def _show(fig, show):
    plt.tight_layout()
    if show:
        plt.show()
    return fig


# ------------------------------------------------------------------------------------------------ system
def fig_system(show=True):
    fig, ax = _canvas(11.6, 4.4)
    ax.text(0.1, 4.15, "At the door", fontsize=10, fontweight="bold")
    _photo(ax, 0.6, 3.15, BLUE); ax.text(0.6, 2.62, "probe photo", ha="center", fontsize=8)
    _box(ax, 1.35, 2.75, 1.85, 0.8, "CNN encoder $f$\n(shared weights)", fc="#ebf4ff", ec=BLUE)
    _box(ax, 3.65, 2.75, 1.85, 0.8, "embedding $f(x)$\nunit vector in $\\mathbb{R}^d$")
    _box(ax, 5.95, 2.75, 2.05, 0.8, "nearest gallery vector\n$d_{min}$, member $k$")
    ax.add_patch(Polygon([[8.95, 3.15], [9.55, 3.62], [10.15, 3.15], [9.55, 2.68]], fc="#fffaf0", ec=ORANGE, lw=1.4))
    ax.text(9.55, 3.15, "$d_{min}\\leq\\tau$?", ha="center", va="center", fontsize=8.5)
    _box(ax, 10.45, 3.45, 1.1, 0.55, "admit as\nmember $k$", fc="#f0fff4", ec=GREEN, fs=8)
    _box(ax, 10.45, 2.3, 1.1, 0.55, "reject:\nstranger", fc="#fff5f5", ec=RED, fs=8)
    for p, q in [((0.95, 3.15), (1.35, 3.15)), ((3.2, 3.15), (3.65, 3.15)), ((5.5, 3.15), (5.95, 3.15)), ((8.0, 3.15), (8.95, 3.15))]:
        _arrow(ax, p, q)
    _arrow(ax, (9.9, 3.4), (10.45, 3.7), GREEN); ax.text(10.05, 3.78, "yes", fontsize=8, color=GREEN)
    _arrow(ax, (9.9, 2.9), (10.45, 2.6), RED); ax.text(10.05, 2.45, "no", fontsize=8, color=RED)

    ax.text(0.1, 1.75, "Enrollment (once per member)", fontsize=10, fontweight="bold")
    for k, c in enumerate(IDC[:3]):
        _photo(ax, 0.38 + 0.3 * k, 1.05 + 0.0 * k, c, s=0.48)
    ax.text(0.7, 0.55, "1 clean photo\nper member", ha="center", va="top", fontsize=8)
    _box(ax, 1.65, 0.7, 1.55, 0.7, "same encoder $f$", fc="#ebf4ff", ec=BLUE)
    _arrow(ax, (1.2, 1.05), (1.65, 1.05))
    _box(ax, 3.65, 0.7, 2.6, 0.7, "gallery: one vector per member", fc=PALE)
    _arrow(ax, (3.2, 1.05), (3.65, 1.05))
    for k, c in enumerate(IDC[:5]):
        ax.add_patch(Circle((3.95 + 0.12 * k, 0.48), 0.05, fc=c, ec="none"))
    ax.text(6.55, 1.05, "adding a member = adding one vector,\nno retraining", fontsize=8.5, va="center", color="#2d3748")
    _arrow(ax, (6.95, 1.45), (6.95, 2.72), "#4a5568", ls="--")
    ax.text(7.05, 2.1, "compare", fontsize=8, color="#4a5568")
    return _show(fig, show)


# ------------------------------------------------------------------------------------------------ verification vs identification
def fig_verif_vs_ident(show=True):
    fig, ax = _canvas(11.4, 4.0)
    ax.text(0.1, 3.75, "Verification (1:1): is it the claimed person?", fontsize=10, fontweight="bold")
    for row, (c1, c2, ans, col) in enumerate([(BLUE, BLUE, "same", GREEN), (BLUE, IDC[1], "different", RED)]):
        y = 2.75 - 1.15 * row
        _photo(ax, 0.7, y, c1); _photo(ax, 1.6, y, c2)
        ax.text(1.15, y + 0.5, "score $s$", ha="center", fontsize=8)
        _arrow(ax, (2.05, y), (2.75, y))
        _box(ax, 2.8, y - 0.28, 1.5, 0.56, "$s \\geq t$ ?", fc="#fffaf0", ec=ORANGE)
        _arrow(ax, (4.3, y), (4.8, y), col)
        ax.text(4.9, y, ans, fontsize=9.5, color=col, va="center", fontweight="bold")
    ax.text(6.2, 3.75, "Identification (1:K), open set: who is it, or nobody?", fontsize=10, fontweight="bold")
    ax.text(6.2, 3.3, "gallery (enrolled members)", fontsize=8.5, color="#2d3748")
    for k, c in enumerate(IDC[:4]):
        _photo(ax, 6.55 + 0.85 * k, 2.72, c, s=0.56)
    ax.text(10.1, 2.72, "…  K", fontsize=9, color="#2d3748", va="center")
    for row, (c, res, col, tag) in enumerate([(BLUE, "member 1", GREEN, "known probe"), (IDC[4], "stranger: rejected", RED, "stranger probe")]):
        y = 1.55 - 0.95 * row
        _photo(ax, 6.55, y, c, s=0.56); ax.text(6.55, y - 0.43, tag, ha="center", fontsize=7.5, color=GRAY)
        _arrow(ax, (6.95, y), (7.65, y))
        _box(ax, 7.7, y - 0.28, 1.95, 0.56, "nearest member,\n$d_{min}\\leq\\tau$ ?", fc="#fffaf0", ec=ORANGE, fs=8.5)
        _arrow(ax, (9.65, y), (10.0, y), col)
        ax.text(10.05, y, res, fontsize=8.5, color=col, va="center", fontweight="bold")
    return _show(fig, show)


# ------------------------------------------------------------------------------------------------ softmax vs gallery
def fig_softmax_vs_gallery(show=True):
    fig, ax = _canvas(11.4, 3.9)
    ax.text(0.1, 3.65, "Softmax classifier over the training people", fontsize=10, fontweight="bold")
    _photo(ax, 0.6, 2.35, BLUE); _arrow(ax, (0.95, 2.35), (1.45, 2.35))
    _box(ax, 1.5, 1.95, 1.3, 0.8, "CNN", fc="#ebf4ff", ec=BLUE); _arrow(ax, (2.8, 2.35), (3.3, 2.35))
    for k in range(5):
        _box(ax, 3.35 + 0.0, 3.0 - 0.52 * k, 0.5, 0.4, str(k + 1) if k < 4 else "K", fs=8)
    ax.text(3.6, 0.55, "K output units,\nprobabilities sum to 1", ha="center", fontsize=8.5, color="#2d3748")
    ax.text(4.1, 2.35, "a stranger still\ngets a name;\na new member needs\na new unit and\nretraining", fontsize=8.5, va="center", color=RED)
    ax.plot([5.75, 5.75], [0.2, 3.8], color="#cbd5e0", lw=1)
    ax.text(6.0, 3.65, "Embeddings and a gallery", fontsize=10, fontweight="bold")
    centers = [(7.0, 2.5), (8.4, 1.5), (9.7, 2.3), (7.6, 0.9)]
    for c, (cx, cy) in zip(IDC, centers):
        ax.add_patch(Circle((cx, cy), 0.09, fc=c, ec="none"))
    ax.add_patch(Circle((10.4, 1.0), 0.11, fc=IDC[4], ec=INK, lw=1.4)); ax.text(10.4, 0.62, "new member:\none new vector", ha="center", fontsize=8, color="#2d3748")
    q1 = (7.2, 2.3); ax.plot(*q1, marker="x", color=INK, ms=9, mew=2); ax.add_patch(Circle(q1, 0.5, fc="none", ec=GREEN, ls="--", lw=1.5))
    ax.text(6.0, 3.15, "accept: a member\nis within $\\tau$", fontsize=8, color=GREEN, va="center")
    q2 = (9.3, 3.05); ax.plot(*q2, marker="x", color=INK, ms=9, mew=2); ax.add_patch(Circle(q2, 0.5, fc="none", ec=RED, ls="--", lw=1.5))
    ax.text(9.95, 3.05, "reject: nothing\nwithin $\\tau$", fontsize=8, color=RED, va="center")
    return _show(fig, show)


# ------------------------------------------------------------------------------------------------ triplet loss
def fig_triplet(show=True):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))
    alpha = 0.5
    cfg = [("before training", 1.25, 0.95, "loss > 0"), ("after training", 0.7, 1.9, "loss = 0")]
    for ax, (title, dp, dn, tag) in zip(axes, cfg):
        ax.set_xlim(-2.4, 2.4); ax.set_ylim(-2.4, 2.4); ax.set_aspect("equal"); ax.axis("off")
        ang_p, ang_n = math.radians(35), math.radians(-120)
        a = np.array([0.0, 0.0]); p = dp * np.array([math.cos(ang_p), math.sin(ang_p)]); n = dn * np.array([math.cos(ang_n), math.sin(ang_n)])
        ax.add_patch(Circle(a, dp + alpha, fc="none", ec=GRAY, ls="--", lw=1.4))
        ax.text(0.0, dp + alpha + 0.08, "radius $d(a,p)+\\alpha$", ha="center", fontsize=8, color=GRAY)
        ax.add_patch(Circle(a, dp, fc="#ebf4ff", ec="none", alpha=0.55, zorder=0))
        ax.plot([a[0], p[0]], [a[1], p[1]], color=BLUE, lw=1.6); ax.plot([a[0], n[0]], [a[1], n[1]], color=RED, lw=1.6)
        ax.plot(*a, marker="*", color=INK, ms=17, zorder=3); ax.text(a[0] - 0.12, a[1] + 0.2, "anchor", fontsize=9, ha="right")
        ax.plot(*p, marker="o", color=BLUE, ms=12, zorder=3); ax.text(p[0] + 0.12, p[1] + 0.12, "positive", color=BLUE, fontsize=9)
        ax.plot(*n, marker="s", color=RED, ms=11, zorder=3); ax.text(n[0] + 0.12, n[1] - 0.22, "negative", color=RED, fontsize=9)
        mp, mn = (a + p) / 2, (a + n) / 2
        ax.text(mp[0] - 0.05, mp[1] + 0.12, "$d(a,p)$", color=BLUE, fontsize=8.5, ha="right"); ax.text(mn[0] - 0.08, mn[1], "$d(a,n)$", color=RED, fontsize=8.5, ha="right")
        ax.set_title(f"{title}: {tag}", fontsize=10.5, color=RED if tag == "loss > 0" else GREEN, fontweight="bold")
        if tag == "loss > 0":
            _arrow(ax, p, p * 0.62, BLUE, lw=2.2); _arrow(ax, n, n * 1.35, RED, lw=2.2)
            ax.text(0.0, -2.28, "the negative lies inside the margin ring", ha="center", fontsize=8.5, color="#2d3748")
        else:
            ax.text(0.0, -2.28, "the negative lies outside the margin ring", ha="center", fontsize=8.5, color="#2d3748")
    fig.suptitle("Triplet loss  $\\max(0,\\; d(a,p)-d(a,n)+\\alpha)$ on the unit sphere (2-D sketch)", fontsize=11)
    return _show(fig, show)


# ------------------------------------------------------------------------------------------------ mining
def fig_mining(show=True):
    pos = [0.30, 0.55, 0.85]; neg = [0.45, 0.70, 1.00, 1.35, 1.80]
    rows = [("batch-hard", 0.85, 0.45, "per anchor: the farthest positive\nand the closest negative"),
            ("semi-hard", 0.85, 1.00, "per positive (one shown): the closest\nnegative farther than that positive"),
            ("random", 0.85, 1.35, "per positive (one shown):\na random negative")]
    fig, ax = plt.subplots(figsize=(10.6, 4.4))
    ax.set_xlim(-0.55, 2.55); ax.set_ylim(-0.1, 3.2); ax.axis("off")
    for r, (name, dp, dn, note) in enumerate(rows):
        y = 2.5 - 1.1 * r
        ax.plot([0, 2.0], [y, y], color="#cbd5e0", lw=1.2, zorder=0)
        ax.text(-0.5, y, name, fontsize=10, fontweight="bold", va="center")
        ax.plot(0, y, marker="*", color=INK, ms=14, zorder=3)
        for x in pos:
            ax.plot(x, y, marker="o", color=BLUE, ms=9, zorder=3, mfc="white" if x != dp else BLUE, mew=1.8)
        for x in neg:
            ax.plot(x, y, marker="s", color=RED, ms=8.5, zorder=3, mfc="white" if x != dn else RED, mew=1.8)
        ax.annotate("", xy=(dn, y + 0.30), xytext=(dp, y + 0.30), arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2))
        ax.text((dp + dn) / 2, y + 0.38, "$d(a,p)-d(a,n)$", ha="center", fontsize=8, color=INK)
        ax.text(2.1, y, note, fontsize=8.5, va="center", color="#2d3748")
    ax.text(0, 3.0, "anchor", ha="center", fontsize=8.5); ax.text(1.0, 0.0, "distance to the anchor  →", ha="center", fontsize=9, color=GRAY)
    ax.text(0.55, 3.0, "● positives   ■ negatives   (filled = used in the loss)", fontsize=8.5, color="#2d3748", ha="left")
    ax.set_title("Which triplet does each mining strategy use? (one anchor, one batch)", fontsize=11)
    return _show(fig, show)


# ------------------------------------------------------------------------------------------------ P x K batch
def fig_pk_batch(show=True):
    fig, ax = _canvas(10.6, 4.4)
    P, K = 4, 4
    ax.text(0.1, 4.15, "A $P\\times K$ batch (here $P=4$, $K=4$; the notebook uses $P=24$, $K=4$ and 32 distractors)", fontsize=10, fontweight="bold")
    for r in range(P):
        y = 3.2 - 0.8 * r
        ax.text(0.25, y, f"person {r + 1}", fontsize=8.5, va="center", ha="left")
        for k in range(K):
            ring = INK if (r == 0 and k == 0) else None
            _photo(ax, 1.75 + 0.8 * k, y, IDC[r], s=0.58, ring=ring)
    ax.text(1.75, 3.62, "anchor", ha="center", fontsize=8.5, fontweight="bold")
    ax.text(3.15, 3.62, "its $K-1$ positives", ha="left", fontsize=8.5, color=BLUE)
    ax.plot([1.45, 4.65], [3.55, 3.55], color="#cbd5e0", lw=0)
    for k in range(6):
        _photo(ax, 5.9 + 0.62 * (k % 3), 3.1 - 0.7 * (k // 3), "#a0aec0", s=0.5)
    ax.text(6.75, 1.45, "distractors: people with one photo,\nused as negatives only", ha="center", fontsize=8.5, color="#2d3748")
    ax.text(9.0, 3.2, "negatives of the anchor:", fontsize=8.5, fontweight="bold", color=RED, va="top")
    ax.text(9.0, 2.9, "all photos of other\npeople, distractors\nincluded", fontsize=8.5, color=RED, va="top")
    ax.text(9.0, 1.7, "distractors have no\npositive, so they are\nnever anchors", fontsize=8.5, color=GRAY, va="top")
    return _show(fig, show)


# ------------------------------------------------------------------------------------------------ pair classifier
def fig_pair_head(show=True):
    fig, ax = _canvas(11.4, 3.7)
    ax.text(0.1, 3.5, "Pair classifier: one shared encoder, one small head", fontsize=10, fontweight="bold")
    _photo(ax, 0.6, 2.55, BLUE); ax.text(0.6, 2.05, "$x_1$", ha="center", fontsize=9)
    _photo(ax, 0.6, 0.9, BLUE); ax.text(0.6, 0.4, "$x_2$", ha="center", fontsize=9)
    for y in (2.55, 0.9):
        _arrow(ax, (0.95, y), (1.45, y)); _box(ax, 1.5, y - 0.35, 1.2, 0.7, "CNN $f$", fc="#ebf4ff", ec=BLUE); _arrow(ax, (2.7, y), (3.2, y))
        for i in range(5):
            ax.add_patch(Rectangle((3.25, y - 0.38 + 0.15 * i), 0.38, 0.13, fc=IDC[i % 4], ec="none", alpha=0.8))
    ax.plot([2.1, 2.1], [1.25, 2.2], color=ORANGE, lw=1.6, ls=":"); ax.text(2.2, 1.72, "shared\nweights", fontsize=8, color=ORANGE, va="center")
    ax.text(3.45, 3.0, "$f(x_1)$", ha="center", fontsize=9); ax.text(3.45, 0.2, "$f(x_2)$", ha="center", fontsize=9)
    _arrow(ax, (3.75, 2.45), (4.35, 1.95)); _arrow(ax, (3.75, 1.0), (4.35, 1.5))
    _box(ax, 4.4, 1.35, 1.9, 0.75, "$|f(x_1)-f(x_2)|$\nper dimension", fc=PALE)
    _arrow(ax, (6.3, 1.72), (6.8, 1.72)); _box(ax, 6.85, 1.35, 1.65, 0.75, "$\\sum_i w_i\\,|\\cdot|_i+b$\n(one linear layer)", fc="#fffaf0", ec=ORANGE, fs=8.5)
    _arrow(ax, (8.5, 1.72), (9.0, 1.72)); ax.add_patch(Circle((9.35, 1.72), 0.35, fc=PALE, ec="#4a5568", lw=1.2)); ax.text(9.35, 1.72, "$\\sigma$", ha="center", va="center", fontsize=11)
    _arrow(ax, (9.7, 1.72), (10.2, 1.72)); ax.text(10.3, 1.72, "$y$: same / different", fontsize=9, va="center")
    ax.text(4.4, 0.55, "trained with binary cross-entropy on pairs;\nlearns how much each dimension matters", fontsize=8.5, color="#2d3748")
    return _show(fig, show)


# ------------------------------------------------------------------------------------------------ cost tree
def fig_cost_tree(show=True):
    fig, ax = _canvas(11.2, 4.7)
    _box(ax, 0.2, 1.55, 1.7, 0.7, "probe at\nthe door", fc=PALE, bold=True)
    _box(ax, 2.9, 2.65, 2.2, 0.7, "known member\n(share $1-u$)", fc="#ebf4ff", ec=BLUE)
    _box(ax, 2.9, 0.4, 2.2, 0.7, "stranger\n(share $u$)", fc="#fff5f5", ec=RED)
    _arrow(ax, (1.9, 2.05), (2.9, 3.0)); _arrow(ax, (1.9, 1.75), (2.9, 0.75))
    leaves = [(3.9, 3.0, "admitted as the right member", "cost 0", GREEN, "#f0fff4"), (3.05, 3.0, "admitted under the wrong name", "cost $c_{fa}$", RED, "#fff5f5"),
              (2.2, 3.0, "sent to the front desk", "cost $c_{fr}$", ORANGE, "#fffaf0"), (1.15, 0.75, "admitted", "cost $c_{fa}$", RED, "#fff5f5"),
              (0.35, 0.75, "turned away", "cost 0", GREEN, "#f0fff4")]
    for y, y0, txt, cost, col, fc in leaves:
        _arrow(ax, (5.1, y0), (6.1, y + 0.14)); _box(ax, 6.15, y - 0.17, 3.0, 0.62, txt, fc=fc, ec=col, fs=8.5)
        ax.text(9.35, y + 0.14, cost, fontsize=9.5, color=col, va="center", fontweight="bold")
    ax.text(0.2, 4.55, "Expected cost per entry  $=(1-u)\\,(p_{wrong}\\,c_{fa}+p_{rej}\\,c_{fr})+u\\,p_{stranger}\\,c_{fa}$", fontsize=10, fontweight="bold")
    return _show(fig, show)


ALL = ("fig_system", "fig_verif_vs_ident", "fig_softmax_vs_gallery", "fig_triplet", "fig_mining", "fig_pk_batch",
       "fig_pair_head", "fig_cost_tree")
