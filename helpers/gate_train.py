"""gate_train.py -- training infrastructure (backbones, P x K sampler, resumable trainer, run log).

DO NOT EDIT for Parts 1-6. You may copy `train` into your own cell for the challenge.
Needs a CUDA GPU. CPU is accepted in SMOKE mode only; TPUs are not supported
(PyTorch/XLA has no reliable support for several of the kernels used here).
"""
VERSION = "A4-2026.1"

import importlib.util
import json
import math
import os
import random
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gate_aug import prep_eval

ALLOWED_BACKBONES = ("resnet18", "resnet34", "resnet50", "resnet101", "resnext50_32x4d", "regnet_y_800mf",
                     "regnet_y_3_2gf", "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "convnext_tiny")
# Only torchvision's ImageNet-1k classification weights (weights="DEFAULT") are allowed. No face-specific weights.


# ------------------------------------------------------------------------------------------ device
def get_device(allow_cpu=False):
    if torch.cuda.is_available():
        return torch.device("cuda")
    tpu = os.environ.get("COLAB_TPU_ADDR") or os.environ.get("TPU_NAME") or importlib.util.find_spec("torch_xla")
    if tpu:
        raise RuntimeError("TPU detected. This notebook needs a CUDA GPU.")
    if allow_cpu:
        return torch.device("cpu")
    raise RuntimeError("No CUDA GPU found. Run on a machine with a CUDA GPU, or set SMOKE=True to debug on CPU.")


def gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(Xnhwc, dev):
    return torch.from_numpy(np.ascontiguousarray(Xnhwc)).permute(0, 3, 1, 2).contiguous().to(dev)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ------------------------------------------------------------------------------------------ ledger
class Ledger:

    def __init__(self, path):
        self.path, self.rows = path, []
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if os.path.exists(path):
            self.rows = json.load(open(path))

    def save(self):
        json.dump(self.rows, open(self.path, "w"), indent=1)

    @contextmanager
    def section(self, name, **info):
        yield
        self.rows.append(dict(name=name, gpu=gpu_name(), **info)); self.save()

    def summary(self):
        print(f"{'run':<34}{'steps':>8}{'peakGB':>8}   gpu")
        for r in self.rows:
            print(f"{r['name']:<34}{r.get('steps', ''):>8}{r.get('peak_gb', ''):>8}   {r['gpu']}")


# ------------------------------------------------------------------------------------------ model
def make_backbone(name, pretrained=True):
    import torchvision.models as tvm
    if name not in ALLOWED_BACKBONES:
        raise ValueError(f"backbone must be one of {ALLOWED_BACKBONES}")
    m = tvm.get_model(name, weights="DEFAULT" if pretrained else None)
    if hasattr(m, "fc"):
        feat, m.fc = m.fc.in_features, nn.Identity()
    elif name.startswith("efficientnet"):
        feat, m.classifier = m.classifier[-1].in_features, nn.Identity()
    elif name.startswith("convnext"):
        feat = m.classifier[-1].in_features; m.classifier[-1] = nn.Identity()
    else:
        raise ValueError(name)
    return m, feat


class EmbeddingNet(nn.Module):
    def __init__(self, backbone="resnet50", emb_dim=256, normalize=True, pretrained=True, bn_neck=True):
        super().__init__()
        self.backbone, feat = make_backbone(backbone, pretrained)
        self.neck = nn.Sequential(nn.Linear(feat, emb_dim), nn.BatchNorm1d(emb_dim) if bn_neck else nn.Identity())
        self.normalize, self.emb_dim = normalize, emb_dim

    def forward(self, x):
        e = self.neck(self.backbone(x))
        return F.normalize(e, dim=1) if self.normalize else e


# ------------------------------------------------------------------------------------------ sampling
class PKSampler:

    def __init__(self, y, P, K, n_extra=0, n_pool=0, seed=0):
        self.groups = [np.flatnonzero(y == c) for c in np.unique(y)]
        self.P, self.K, self.n_extra, self.n_pool = P, K, (n_extra if n_pool else 0), n_pool
        self.rng = np.random.default_rng(seed)

    def next(self):
        ids = self.rng.choice(len(self.groups), self.P, replace=len(self.groups) < self.P)
        idx = np.concatenate([self.rng.choice(self.groups[c], self.K, replace=len(self.groups[c]) < self.K) for c in ids])
        ex = self.rng.choice(self.n_pool, self.n_extra, replace=False) if self.n_extra else np.empty(0, np.int64)
        return idx, ex


def flip_labels(y, frac, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); n_cls = int(y.max()) + 1
    mask = np.zeros(len(y), bool)
    mask[rng.choice(len(y), int(round(frac * len(y))), replace=False)] = True
    new = rng.integers(0, n_cls - 1, mask.sum())
    new = new + (new >= y[mask])                                            # guarantees new != old
    y2 = y.copy(); y2[mask] = new
    return y2, mask


# ------------------------------------------------------------------------------------------ training
def train(model, X, y, loss_fn, steps, *, aug, P=24, K=4, n_extra=32, X_extra=None, lr=3e-4, backbone_lr_mult=0.33,
          wd=1e-4, warmup=100, extra_modules=(), name="run", log_every=100, seed=0,
          ledger=None, ckpt_dir=None, resume_every_steps=300, verbose=True):
    ck = os.path.join(ckpt_dir, f"{name}.pt") if ckpt_dir else None
    part = os.path.join(ckpt_dir, f"{name}.partial.pt") if ckpt_dir else None
    if ck and os.path.exists(ck):
        blob = torch.load(ck, map_location="cpu")
        model.load_state_dict(blob["model"])
        for m, sd in zip(extra_modules, blob["extra"]):
            m.load_state_dict(sd)
        print(f"[{name}] loaded cached checkpoint {ck}")
        return dict(blob["hist"], cached=True)

    dev = next(model.parameters()).device
    amp = dev.type == "cuda"
    seed_everything(seed)
    if amp:
        model.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats()
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    bb = [p for p in model.backbone.parameters()]
    rest = [p for n, p in model.named_parameters() if not n.startswith("backbone.")] + \
           [p for m in extra_modules for p in m.parameters()]
    opt = torch.optim.AdamW([dict(params=bb, lr=lr * backbone_lr_mult, base=lr * backbone_lr_mult),
                             dict(params=rest, lr=lr, base=lr)], weight_decay=wd)
    params = bb + rest
    y_np = np.asarray(y)
    y_t = torch.as_tensor(y_np, device=dev, dtype=torch.long)
    hist = dict(step=[], loss=[])
    step = 0
    if part and os.path.exists(part):                                       # resume an interrupted run
        blob = torch.load(part, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"]); opt.load_state_dict(blob["opt"]); scaler.load_state_dict(blob["scaler"])
        for m, sd in zip(extra_modules, blob["extra"]):
            m.load_state_dict(sd)
        step, hist = blob["step"], blob["hist"]
        seed_everything(seed + step)
        print(f"[{name}] RESUMING from step {step}", flush=True)
    sampler = PKSampler(y_np, P, K, n_extra, 0 if X_extra is None else len(X_extra), seed + step)
    neg_lab = -1 - torch.arange(sampler.n_extra, device=dev)
    model.train(); [m.train() for m in extra_modules]
    acc, n_acc = {}, 0
    ctx = ledger.section(name, steps=steps) if ledger else _null()

    def flush():
        nonlocal acc, n_acc
        row = {k: float(v) / n_acc for k, v in acc.items()}
        if not math.isfinite(row["loss"]):
            raise RuntimeError(f"[{name}] loss is {row['loss']} at step {step}: check your loss function.")
        hist["step"].append(step)
        for k, v in row.items():
            hist.setdefault(k, []).append(v)
        acc, n_acc = {}, 0
        return row

    with ctx:
        _sync()
        try:
            while step < steps:
                prog = step / steps
                f = min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * prog))
                for g in opt.param_groups:
                    g["lr"] = g["base"] * f
                idx, ex = sampler.next()
                idx_t = torch.as_tensor(idx, device=dev)
                xb, yb = X[idx_t], y_t[idx_t]
                if sampler.n_extra:
                    xb = torch.cat([xb, X_extra[torch.as_tensor(ex, device=dev)]])
                    yb = torch.cat([yb, neg_lab])
                xb = aug(xb).contiguous(memory_format=torch.channels_last) if dev.type == "cuda" else aug(xb)
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                    emb = model(xb)
                loss, diag = loss_fn(emb.float(), yb)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(params, 5.0)
                scaler.step(opt); scaler.update()
                diag = dict(diag, loss=loss.detach(), emb_norm=emb.detach().float().norm(dim=1).mean())
                for k, v in diag.items():
                    acc[k] = acc.get(k, 0) + torch.as_tensor(v).detach().float()
                n_acc += 1; step += 1
                if step % log_every == 0:
                    row = flush()
                    if verbose:
                        print(f"[{name}] step {step:>5}/{steps} | " + " | ".join(f"{k} {v:.4g}" for k, v in row.items()), flush=True)
                if part and step % resume_every_steps == 0 and step < steps:     # atomic snapshot (never leaves a corrupt file)
                    os.makedirs(ckpt_dir, exist_ok=True)
                    tmp = part + ".tmp"
                    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), step=step,
                                    hist=hist, extra=[m.state_dict() for m in extra_modules]), tmp)
                    os.replace(tmp, part)
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            raise RuntimeError(f"[{name}] CUDA out of memory (batch {P * K + sampler.n_extra} images, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB). "
                               "Reduce P / K / n_extra or the image size, use a smaller backbone, then restart the kernel to free the GPU.") from e
        if n_acc:                                                           # flush the partial logging window
            row = flush()
            if verbose:
                print(f"[{name}] step {step:>5}/{steps} | " + " | ".join(f"{k} {v:.4g}" for k, v in row.items()), flush=True)
        _sync()
    hist["steps_done"] = step
    hist["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if amp else 0.0
    if ledger and ledger.rows:
        ledger.rows[-1].update(steps=step, peak_gb=hist["peak_gb"]); ledger.save()
    if ck:
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(dict(model=model.state_dict(), extra=[m.state_dict() for m in extra_modules], hist=hist), ck)
        if part and os.path.exists(part):
            os.remove(part)
    return hist


@contextmanager
def _null():
    yield


# ------------------------------------------------------------------------------------------ inference
@torch.no_grad()
def embed(model, X, out_size, bs=256, flip=True):
    model.eval()
    dev = next(model.parameters()).device
    amp = dev.type == "cuda"
    outs = []
    for s in range(0, len(X), bs):
        xb = prep_eval(X[s:s + bs], out_size)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
            e = model(xb).float()
            if flip:
                e = e + model(torch.flip(xb, dims=[3])).float()
        outs.append(e.cpu())
    E = torch.cat(outs)
    E = F.normalize(E, dim=1) if getattr(model, "normalize", True) else E / (2 if flip else 1)
    return E.numpy().astype(np.float32)


def l2n(E):
    return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)


def pair_scores(E, i, j, metric="neg_sqeuclid"):
    a, b = E[i].astype(np.float64), E[j].astype(np.float64)
    if metric == "cosine":
        return (l2n(a) * l2n(b)).sum(1)
    return -((a - b) ** 2).sum(1)
