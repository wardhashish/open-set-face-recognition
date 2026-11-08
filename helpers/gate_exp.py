"""gate_exp.py -- experiment helpers (given; do not edit in Parts 1-6).

`Lab` bundles the data tensors and settings that every training run shares, so each experiment is one call.
"""
VERSION = "A4-2026.1"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gate_data as gd
from gate_train import EmbeddingNet, embed, train


class Lab:
    def __init__(self, D, cfg, steps, dev, aug, Xtr, ytr, Xdist, Xva, Xte, board, ledger, ckpt, seed, ch_images, final="E_final"):
        self.D, self.cfg, self.steps, self.dev, self.aug = D, cfg, steps, dev, aug
        self.Xtr, self.ytr, self.Xdist, self.Xva, self.Xte = Xtr, ytr, Xdist, Xva, Xte
        self.board, self.ledger, self.ckpt, self.seed, self.ch_images, self.final = board, ledger, ckpt, seed, ch_images, final
        self.hist, self.models, self.chal_log = {}, {}, {}
        self.triplet_loss = None                     # set after you have implemented it:  LAB.triplet_loss = triplet_loss

    def emb_split(self, model):
        s = self.cfg["size"]
        return embed(model, self.Xva, s), embed(model, self.Xte, s)

    def probe(self, steps=25):
        c = self.cfg
        m = EmbeddingNet(c["backbone"], c["emb_dim"]).to(self.dev)
        h = train(m, self.Xtr, self.ytr, lambda e, y: (e[:, 0].mean(), {}), steps, aug=self.aug, P=c["P"], K=c["K"], n_extra=c["n_extra"],
                  X_extra=self.Xdist, warmup=5, log_every=steps, verbose=False, name="probe")
        del m
        return h

    def run_triplet(self, name, mining, margin=0.3, steps=None, normalize=True, aug=None, y=None, keep=False):
        if self.triplet_loss is None:
            raise RuntimeError("set LAB.triplet_loss = triplet_loss first")
        c, tl = self.cfg, self.triplet_loss
        model = EmbeddingNet(c["backbone"], c["emb_dim"], normalize=normalize).to(self.dev)
        h = train(model, self.Xtr, self.ytr if y is None else y, lambda e, yb: tl(e, yb, margin, mining), steps or self.steps["trip"], aug=aug or self.aug,
                  P=c["P"], K=c["K"], n_extra=c["n_extra"], X_extra=self.Xdist, name=name, ledger=self.ledger, ckpt_dir=self.ckpt, seed=self.seed, lr=c["lr"], log_every=200)
        Ev, Et = self.emb_split(model); self.hist[name] = h
        self.board.evaluate(name, Ev, Et, normalize=True)
        if not normalize:
            self.board.evaluate(name + "_rawL2", Ev, Et, normalize=False)
        if keep:
            self.models[name] = model
        else:
            del model
        if self.dev.type == "cuda":
            torch.cuda.empty_cache()
        return h

    def run_challenge(self, name, model, loss_fn, steps, aug=None, extra_modules=(), y=None, P=None, K=None, n_extra=None, lr=None, normalize=True, **kw):
        c = self.cfg
        P_, K_, X_ = P or c["P"], K or c["K"], c["n_extra"] if n_extra is None else n_extra
        batch = P_ * K_ + X_
        if name == self.final and steps * batch > self.ch_images * 1.0001:
            raise ValueError(f"final run: {steps} steps x {batch} images = {steps * batch:,} > budget CH_IMAGES = {self.ch_images:,}")
        print(f"{name}: {steps} steps x {batch} images = {steps * batch / 1e6:.2f} M images (budget {self.ch_images / 1e6:.2f} M)")
        h = train(model, self.Xtr, self.ytr if y is None else y, loss_fn, steps, aug=aug or self.aug, P=P_, K=K_, n_extra=X_, X_extra=self.Xdist, extra_modules=extra_modules,
                  name=name, ledger=self.ledger, ckpt_dir=self.ckpt, seed=self.seed, lr=lr or c["lr"], log_every=200, **kw)
        self.hist[name] = h
        Ev, Et = self.emb_split(model); self.board.evaluate(name, Ev, Et, normalize=normalize)
        self.chal_log[name] = dict(steps=h["steps_done"], images=h["steps_done"] * batch, peak_gb=h.get("peak_gb", 0.0))
        r = self.board.results[name]
        print(f"{name}: S = {r['S']:.4f}  (V-F1 {r['V_F1']:.4f}, O-F1 {r['O_F1']:.4f})  [{h['steps_done']} steps, peak VRAM {h.get('peak_gb', 0):.1f} GB]")
        return h


def make_ce_loss(head):
    def ce_loss(e, y):
        z = head(e)
        return F.cross_entropy(z, y, label_smoothing=0.1), dict(acc=(z.argmax(1) == y).float().mean())
    return ce_loss


def sample_triplets(y, n, rng):
    items = np.flatnonzero(y >= 0)
    a, p, q = rng.choice(items, 4 * n), rng.choice(items, 4 * n), rng.choice(items, 4 * n)
    ok = (y[a] == y[p]) & (a != p) & (y[a] != y[q])
    return a[ok][:n], p[ok][:n], q[ok][:n]


@torch.no_grad()
def head_logits(head, Ea, Eb, dev):
    return head(torch.from_numpy(Ea).float().to(dev), torch.from_numpy(Eb).float().to(dev)).cpu().numpy()


@torch.no_grad()
def head_openset(head, D, E, split, dev):
    o = D[split].os
    G = torch.from_numpy(E[o.gal_idx]).float().to(dev); P_ = torch.from_numpy(E[o.probe_idx]).float().to(dev)
    L = torch.cat([head(P_[s:s + 128, None, :], G[None]) for s in range(0, len(P_), 128)]).cpu().numpy()
    return o.gal_y[L.argmax(1)], -L.max(1), o.probe_y


def scenario_outcomes(D, split, ids, E, open_set_decide):
    g = gd.scenario_split(D, split, ids)
    pred, dmin, _ = open_set_decide(E[g.gal_idx], E[g.probe_idx], np.inf)
    return g.gal_y[pred], dmin, g.probe_y
