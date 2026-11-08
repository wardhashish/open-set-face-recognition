"""gate_data.py -- data pipeline for CMPS 364, Assignment 4 Part II ("The Gatekeeper").

DO NOT EDIT.  Everything here is deterministic given COURSE_SEED, so the
benchmark (splits, pairs, corruptions, protocols) is IDENTICAL for every student.
Only the *personal scenario* (Part 8) and your fingerprint depend on your AUBNET.
"""
VERSION = "A4-2026.1"

import hashlib
from types import SimpleNamespace

import numpy as np

COURSE_SEED = 36420261
IMG = 144                                   # stored resolution (square crop of the 250x250 LFW-funneled image)
KINDS = ["blur", "lowres", "noise", "shades", "mask", "lighting", "gray"]
P_CORRUPT = 0.40                            # fraction of val/test images that get degraded (enrollment photos never do)
CAP_TRAIN, CAP_EVAL = 40, 20                # max images kept per identity (train / val+test)
SPLIT_CODE = {"train": 0, "val": 1, "test": 2}


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def sha(*parts):
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(np.ascontiguousarray(p).tobytes())
        else:
            h.update(str(p).encode())
        h.update(b"|")
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------
# 1. raw LFW
# ----------------------------------------------------------------------------------------------
def load_lfw_raw(data_home=None):
    import inspect
    from sklearn.datasets import fetch_lfw_people
    kw = dict(n_retries=5, delay=10.0) if "n_retries" in inspect.signature(fetch_lfw_people).parameters else {}   # figshare occasionally drops connections
    for attempt in range(3):
        try:
            lfw = fetch_lfw_people(min_faces_per_person=0, resize=1.0, color=True, funneled=True,
                                   slice_=(slice(53, 197), slice(53, 197)), data_home=data_home, **kw)
            break
        except Exception as e:
            if attempt == 2:
                raise RuntimeError(f"LFW download failed ({e}). Wait briefly and re-run this cell.") from e
            print(f"download attempt {attempt + 1} failed ({type(e).__name__}); retrying ...", flush=True)
    imgs = lfw.images
    names = np.asarray(lfw.target_names)[lfw.target]
    scale = 255.0 if float(imgs[:256].max()) <= 1.0 + 1e-6 else 1.0       # sklearn returns floats in [0,1]
    X = np.empty(imgs.shape, np.uint8)
    for s in range(0, len(imgs), 1024):                                  # chunked: avoid a 2nd 3 GB float copy
        X[s:s + 1024] = np.clip(np.rint(imgs[s:s + 1024] * scale), 0, 255).astype(np.uint8)
    del lfw, imgs
    assert X.shape[1:] == (IMG, IMG, 3), X.shape
    return X, names


# ----------------------------------------------------------------------------------------------
# 2. identity-disjoint splits
# ----------------------------------------------------------------------------------------------
def _identity_table(names):
    uniq, inv, cnt = np.unique(names, return_inverse=True, return_counts=True)
    order = np.argsort(inv, kind="stable")
    members = np.split(order, np.cumsum(cnt)[:-1])
    return uniq, cnt, members


def make_splits(names, seed=COURSE_SEED):
    uniq, cnt, members = _identity_table(names)
    rng = np.random.default_rng([seed, 1])
    multi = np.flatnonzero(cnt >= 2)
    single = np.flatnonzero(cnt == 1)
    multi = multi[np.lexsort((uniq[multi], cnt[multi]))]                  # sort by (count, name)
    pattern = np.array([0] * 7 + [1] + [2] * 2)                          # 0 train, 1 val, 2 test
    assign = np.empty(len(multi), int)
    for s in range(0, len(multi), 10):
        chunk = np.arange(s, min(s + 10, len(multi)))
        assign[chunk[rng.permutation(len(chunk))]] = pattern[:len(chunk)]

    def take(ids, cap):
        idx, y = [], []
        for k, u in enumerate(ids):
            m = members[u]
            if len(m) > cap:
                m = np.sort(rng.choice(m, cap, replace=False))
            idx.append(m)
            y.append(np.full(len(m), k))
        return np.concatenate(idx), np.concatenate(y)

    S = {}
    for code, (nm, cap) in enumerate([("train", CAP_TRAIN), ("val", CAP_EVAL), ("test", CAP_EVAL)]):
        ids = multi[assign == code]
        S[nm] = dict(zip(("idx", "y"), take(ids, cap)))
        S[nm]["names"] = uniq[ids]

    single = single[np.argsort(uniq[single], kind="stable")]
    single = single[rng.permutation(len(single))]
    n_val, n_test = int(round(0.07 * len(single))), int(round(0.37 * len(single)))
    first = lambda arr: np.array([members[u][0] for u in arr], dtype=np.int64)
    S["val_dist"] = first(single[:n_val])
    S["test_dist"] = first(single[n_val:n_val + n_test])
    S["train_dist"] = first(single[n_val + n_test:])
    return S


# ----------------------------------------------------------------------------------------------
# 3. test-time degradations ("faces in the wild", but worse)
# ----------------------------------------------------------------------------------------------
def _blur(x, r):
    from scipy.ndimage import gaussian_filter
    s = r.uniform(1.5, 3.5)
    return gaussian_filter(x, sigma=(s, s, 0))


def _lowres(x, r):
    f = int(r.choice([4, 6, 8]))
    h = IMG // f
    small = x[:h * f, :h * f].reshape(h, f, h, f, 3).mean((1, 3))
    up = np.repeat(np.repeat(small, f, 0), f, 1)
    out = x.copy()
    out[:h * f, :h * f] = up
    return out


def _noise(x, r):
    return x + r.normal(0, r.uniform(10, 28), x.shape)


def _shades(x, r):                                                       # sunglasses band over the eyes
    r0 = int(IMG * r.uniform(0.33, 0.38)); r1 = r0 + int(IMG * r.uniform(0.11, 0.15))
    c0 = int(IMG * r.uniform(0.10, 0.18))
    x = x.copy(); x[r0:r1, c0:IMG - c0] = r.uniform(0, 40)
    return x


def _mask(x, r):                                                         # face mask over nose/mouth
    r0 = int(IMG * r.uniform(0.52, 0.58)); r1 = int(IMG * r.uniform(0.88, 0.96))
    c0 = int(IMG * r.uniform(0.18, 0.25))
    col = np.array([[220, 225, 230], [120, 170, 200], [30, 30, 30]], np.float32)[r.integers(0, 3)]
    x = x.copy(); x[r0:r1, c0:IMG - c0] = col
    return x


def _lighting(x, r):
    g = r.uniform(0.35, 0.6) if r.random() < 0.5 else r.uniform(1.8, 3.0)
    x = 255.0 * np.clip(x / 255.0, 0, 1) ** g
    ramp = np.linspace(-1, 1, IMG, dtype=np.float32)[None, :, None] * r.uniform(-60, 60)   # side lighting
    return x + ramp


def _gray(x, r):
    return np.repeat(x.mean(2, keepdims=True), 3, 2)


_CORR = [_blur, _lowres, _noise, _shades, _mask, _lighting, _gray]


def apply_corruptions(Xraw, idx, exempt, split_code, seed=COURSE_SEED):
    out = Xraw[idx].copy()
    flags = np.zeros(len(idx), np.int32)
    for k, ridx in enumerate(idx):
        if exempt[k]:
            continue
        r = np.random.default_rng([seed, 100 + split_code, int(ridx)])
        if r.random() >= P_CORRUPT:
            continue
        kinds = r.choice(len(KINDS), 1 if r.random() < 0.7 else 2, replace=False)
        x = out[k].astype(np.float32)
        for q in kinds:
            x = _CORR[int(q)](x, r)
            flags[k] |= 1 << int(q)
        out[k] = np.clip(np.rint(x), 0, 255).astype(np.uint8)
    return out, flags


# ----------------------------------------------------------------------------------------------
# 4. hard verification pairs
# ----------------------------------------------------------------------------------------------
def tiny_descriptor(X):
    out = []
    for s in range(0, len(X), 256):
        b = X[s:s + 256].reshape(-1, 8, IMG // 8, 8, IMG // 8, 3).astype(np.int32)
        out.append((b.sum((2, 4)) // ((IMG // 8) ** 2)).reshape(len(b), -1))
    return np.concatenate(out)


def make_pairs(y, desc, n_pos, n_neg, seed, M=100):
    rng = np.random.default_rng(seed)
    n = len(y)
    by = [np.flatnonzero(y == c) for c in np.unique(y)]
    pos = np.array([(a, b) for m in by for i, a in enumerate(m) for b in m[i + 1:]], dtype=np.int64)
    prev = n_pos / (n_pos + n_neg)
    n_pos = min(n_pos, int(0.6 * len(pos)))
    n_neg = int(round(n_pos * (1 - prev) / prev))
    d = ((desc[pos[:, 0]].astype(np.int64) - desc[pos[:, 1]]) ** 2).sum(1)
    hard_pool = np.flatnonzero(d >= np.median(d))
    n_hp = n_pos // 2
    hp = rng.choice(hard_pool, n_hp, replace=False)
    rest = np.setdiff1d(np.arange(len(pos)), hp)
    rp = rng.choice(rest, n_pos - n_hp, replace=False)
    P_i = np.concatenate([pos[hp], pos[rp]])
    P_t = np.r_[np.ones(n_hp, int), np.zeros(n_pos - n_hp, int)]

    seen = set()
    def uniq_add(pairs):
        keep = []
        for a, b in pairs:
            key = (min(a, b), max(a, b))
            if a != b and key not in seen:
                seen.add(key); keep.append(key)
        return keep

    n_hn = n_neg // 2
    hard_neg = []
    while len(hard_neg) < n_hn:
        m = int(1.3 * (n_hn - len(hard_neg))) + 8
        a = rng.integers(0, n, m)
        cand = rng.integers(0, n, (m, M))
        dd = np.empty((m, M), np.int64)
        for s in range(0, m, 256):
            dd[s:s + 256] = ((desc[a[s:s + 256], None, :].astype(np.int64) - desc[cand[s:s + 256]]) ** 2).sum(-1)
        dd[y[cand] == y[a][:, None]] = np.iinfo(np.int64).max
        hard_neg += uniq_add(zip(a, cand[np.arange(m), dd.argmin(1)]))
    hard_neg = hard_neg[:n_hn]
    rnd_neg = []
    while len(rnd_neg) < n_neg - n_hn:
        m = int(1.3 * (n_neg - n_hn - len(rnd_neg))) + 8
        a, b = rng.integers(0, n, m), rng.integers(0, n, m)
        ok = y[a] != y[b]
        rnd_neg += uniq_add(zip(a[ok], b[ok]))
    rnd_neg = rnd_neg[:n_neg - n_hn]

    pairs = np.concatenate([P_i, np.array(hard_neg), np.array(rnd_neg)])
    lab = np.r_[np.ones(n_pos, int), np.zeros(n_neg, int)]
    tier = np.r_[P_t, np.ones(n_hn, int), np.zeros(n_neg - n_hn, int)]
    perm = rng.permutation(len(pairs))
    return pairs[perm, 0], pairs[perm, 1], lab[perm], tier[perm]


# ----------------------------------------------------------------------------------------------
# 5. open-set (gallery / probe) protocol
# ----------------------------------------------------------------------------------------------
def make_openset(y, enroll, n_dist, seed):
    """Positions 0..n_multi-1 are identity images, n_multi.. are distractors (unknown people)."""
    n_multi = len(y)
    ids = np.unique(y)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(ids)
    enrolled, unk = np.sort(perm[:len(ids) // 2]), perm[len(ids) // 2:]
    gal_idx = np.array([np.flatnonzero((y == c) & enroll)[0] for c in enrolled])
    known = np.flatnonzero(np.isin(y, enrolled) & ~enroll)
    unknown = np.concatenate([np.flatnonzero(np.isin(y, unk)), n_multi + np.arange(n_dist)])
    probe_idx = np.concatenate([known, unknown])
    probe_y = np.concatenate([y[known], -np.ones(len(unknown), int)])
    return dict(gal_idx=gal_idx, gal_y=enrolled, probe_idx=probe_idx, probe_y=probe_y)


# ----------------------------------------------------------------------------------------------
# 6. assemble everything
# ----------------------------------------------------------------------------------------------
def build_all(Xraw=None, names=None, data_home=None, verbose=True):
    if Xraw is None:
        Xraw, names = load_lfw_raw(data_home)
    S = make_splits(names)
    D = AttrDict()
    D["train"] = AttrDict(X=Xraw[S["train"]["idx"]], y=S["train"]["y"], raw_idx=S["train"]["idx"], names=S["train"]["names"])
    D["train_dist"] = AttrDict(X=Xraw[S["train_dist"]])
    fp = [COURSE_SEED, len(Xraw), S["train"]["idx"], S["train_dist"]]

    for nm in ("val", "test"):
        code = SPLIT_CODE[nm]
        s, dist = S[nm], S[nm + "_dist"]
        rng = np.random.default_rng([COURSE_SEED, 10 + code])
        y = s["y"]
        enroll = np.zeros(len(y), bool)                                    # one random clean "ID photo" per identity
        for c in np.unique(y):
            m = np.flatnonzero(y == c)
            enroll[m[rng.integers(len(m))]] = True
        idx_all = np.concatenate([s["idx"], dist])
        exempt = np.concatenate([enroll, np.zeros(len(dist), bool)])
        Xc, flags = apply_corruptions(Xraw, idx_all, exempt, code)
        desc = tiny_descriptor(Xraw[s["idx"]])
        n_pos, n_neg = (1000, 1000) if nm == "val" else (2400, 9600)
        pi, pj, pl, pt = make_pairs(y, desc, n_pos, n_neg, [COURSE_SEED, 20 + code])
        os_main = make_openset(y, enroll, len(dist), [COURSE_SEED, 30 + code])
        D[nm] = AttrDict(
            X=Xc, y=np.concatenate([y, -np.ones(len(dist), int)]), enroll=exempt, flags=flags,
            n_multi=len(y), n_ids=int(y.max()) + 1, raw_idx=idx_all, names=s["names"],
            pairs=AttrDict(i=pi, j=pj, y=pl, tier=pt), os=AttrDict(**os_main),
            all_enroll=np.array([np.flatnonzero((y == c) & enroll)[0] for c in range(int(y.max()) + 1)]),
        )
        fp += [idx_all, flags, pi, pj, pl, pt, os_main["gal_idx"], os_main["probe_idx"]]
    D["meta"] = AttrDict(data_fp=sha(*fp), seed=COURSE_SEED, n_raw=len(Xraw), img=IMG,
                         test_prevalence=float(D["test"].pairs.y.mean()), val_prevalence=float(D["val"].pairs.y.mean()))
    if verbose:
        print_summary(D)
    return D


def print_summary(D):
    tr = D.train
    print(f"{'split':<8}{'#ids':>6}{'#imgs':>8}{'#distract.':>12}{'#pairs':>9}{'%same':>7}{'%degraded':>11}")
    print(f"{'train':<8}{int(tr.y.max()) + 1:>6}{len(tr.y):>8}{len(D.train_dist.X):>12}{'-':>9}{'-':>7}{'0%':>11}")
    for nm in ("val", "test"):
        s = D[nm]
        print(f"{nm:<8}{s.n_ids:>6}{s.n_multi:>8}{len(s.y) - s.n_multi:>12}{len(s.pairs.y):>9}"
              f"{100 * s.pairs.y.mean():>6.0f}%{100 * (s.flags > 0).mean():>10.0f}%")
    print("data fingerprint:", D.meta.data_fp[:16], "...")


# ----------------------------------------------------------------------------------------------
# 7. personal fingerprint + personal scenario (Part 8)
# ----------------------------------------------------------------------------------------------
def fingerprint(aubnet, D):
    return sha(aubnet.strip().lower(), D.meta.data_fp)


def make_scenario(aubnet, D):
    h = int(sha(aubnet.strip().lower(), "scenario", COURSE_SEED), 16)
    r = np.random.default_rng(h % (2 ** 63))
    k_max = min(120, D.val.n_ids, D.test.n_ids)
    sc = dict(
        K=int(r.integers(min(40, k_max), k_max + 1)),                       # enrolled members
        u=float(r.choice([0.10, 0.15, 0.20, 0.25, 0.30])),                  # fraction of entries by strangers
        N=int(r.choice(np.arange(1500, 4001, 250))),                        # entries per day
        c_fa=float(r.choice([250, 500, 750, 1000])),                        # $ per stranger / wrong-name admission
        c_fr=float(r.choice([2, 3, 5])),                                    # $ per member sent to the front desk
        days=30,
    )
    for nm in ("val", "test"):
        sc[nm + "_ids"] = np.sort(r.choice(D[nm].n_ids, sc["K"], replace=False))
    return sc


def scenario_split(D, nm, ids):
    s = D[nm]
    gal_idx = s.all_enroll[ids]
    n_all = len(s.y)
    idx = np.arange(n_all)
    is_known = np.isin(s.y, ids) & ~s.enroll
    is_unknown = ~np.isin(s.y, ids)
    probe_idx = np.concatenate([idx[is_known], idx[is_unknown]])
    probe_y = np.concatenate([s.y[is_known], -np.ones(is_unknown.sum(), int)])
    return AttrDict(gal_idx=gal_idx, gal_y=np.asarray(ids), probe_idx=probe_idx, probe_y=probe_y)


def sample_gallery(D, nm, K, rng):
    ids = np.sort(rng.choice(D[nm].n_ids, K, replace=False))
    return scenario_split(D, nm, ids)
