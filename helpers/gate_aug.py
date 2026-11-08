"""gate_aug.py -- batched, GPU-side augmentation (static shapes, no CPU data-loader bottleneck).

Input : uint8 tensor (B,3,144,144) already on the device.
Output: float tensor (B,3,S,S), ImageNet-normalised.
You may copy / subclass GPUAug for the challenge; you do not need to edit this file.
"""
VERSION = "A4-2026.1"

import math

import torch
import torch.nn.functional as F

IM_MEAN = (0.485, 0.456, 0.406)
IM_STD = (0.229, 0.224, 0.225)


def _norm(x):
    m = torch.tensor(IM_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    s = torch.tensor(IM_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - m) / s


def _resample(x, out, theta=None):
    B = x.shape[0]
    if theta is None:
        theta = torch.tensor([[1.0, 0, 0], [0, 1.0, 0]], device=x.device, dtype=x.dtype).expand(B, 2, 3)
    grid = F.affine_grid(theta, (B, 3, out, out), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def prep_eval(xb, out_size):
    return _norm(_resample(xb.float() / 255.0, out_size))


def _gauss_blur(x, sigma, ksize=17):
    B, C, H, W = x.shape
    r = torch.arange(ksize, device=x.device, dtype=x.dtype) - ksize // 2
    k = torch.exp(-(r[None, :] ** 2) / (2 * sigma[:, None].clamp(min=1e-3) ** 2))
    k = k / k.sum(1, keepdim=True)                                                   # (B,K)
    k = k.repeat_interleave(C, 0)                                                    # (B*C,K)
    y = x.reshape(1, B * C, H, W)
    y = F.conv2d(F.pad(y, (ksize // 2, ksize // 2, 0, 0), mode="reflect"), k.view(B * C, 1, 1, ksize), groups=B * C)
    y = F.conv2d(F.pad(y, (0, 0, ksize // 2, ksize // 2), mode="reflect"), k.view(B * C, 1, ksize, 1), groups=B * C)
    return y.reshape(B, C, H, W)


class GPUAug:
    def __init__(self, out_size=128, rot=12.0, zoom=(0.95, 1.20), shift=0.06, flip=True,
                 photometric=0.3, p_blur=0.25, p_lowres=0.20, p_noise=0.25, p_shades=0.15, p_mask=0.15,
                 p_light=0.30, p_gray=0.10, p_erase=0.10):
        self.__dict__.update(locals()); del self.__dict__["self"]

    def clean(self):
        """Geometry + mild photometric only (no degradations) -- the 'naive' augmentation."""
        c = GPUAug(self.out_size, self.rot, self.zoom, self.shift, self.flip, self.photometric,
                   0, 0, 0, 0, 0, 0, 0, 0)
        return c

    @torch.no_grad()
    def __call__(self, xb):
        B, dev, S = xb.shape[0], xb.device, self.out_size
        U = lambda lo=0.0, hi=1.0: torch.rand(B, device=dev) * (hi - lo) + lo
        P = lambda p: (torch.rand(B, device=dev) < p)
        x = xb.float() / 255.0

        # -- geometry: rotation, zoom (>1 zooms in), translation, horizontal flip (one affine grid_sample)
        ang = U(-self.rot, self.rot) * math.pi / 180
        z = U(*self.zoom)
        fl = torch.where(P(0.5), -1.0, 1.0) if self.flip else torch.ones(B, device=dev)
        c, s = torch.cos(ang) / z, torch.sin(ang) / z
        theta = torch.stack([torch.stack([c * fl, -s, U(-self.shift, self.shift)], 1),
                             torch.stack([s * fl, c, U(-self.shift, self.shift)], 1)], 1)
        x = _resample(x, S, theta)

        # -- photometric: brightness / contrast / saturation / gamma
        a = self.photometric
        if a > 0:
            x = x * (1 + U(-a, a))[:, None, None, None]
            m = x.mean((1, 2, 3), keepdim=True)
            x = (x - m) * (1 + U(-a, a))[:, None, None, None] + m
            g = x.mean(1, keepdim=True)
            x = (x - g) * (1 + U(-a, a))[:, None, None, None] + g
            x = x.clamp(1e-4, 1) ** torch.exp(U(-0.35, 0.35))[:, None, None, None]

        if self.p_light > 0:                                                         # side lighting ramp + gamma
            ramp = torch.linspace(-1, 1, S, device=dev)[None, None, None, :] * U(-0.25, 0.25)[:, None, None, None]
            gm = torch.where(P(0.5), U(0.4, 0.7), U(1.6, 2.6))
            lit = x.clamp(1e-4, 1) ** gm[:, None, None, None] + ramp
            x = torch.where(P(self.p_light)[:, None, None, None], lit, x)
        if self.p_blur > 0:
            sig = torch.where(P(self.p_blur), U(1.2, 3.8) * S / 144, torch.full((B,), 1e-3, device=dev))
            x = _gauss_blur(x, sig)
        if self.p_lowres > 0:
            f = torch.randint(4, 9, (B,), device=dev)
            for q in (4, 5, 6, 7, 8):
                low = F.interpolate(F.interpolate(x, scale_factor=1 / q, mode="area"), size=(S, S), mode="nearest")
                sel = (P(self.p_lowres) & (f == q))[:, None, None, None]
                x = torch.where(sel, low, x)
        if self.p_noise > 0:
            sg = torch.where(P(self.p_noise), U(0.04, 0.11), torch.zeros(B, device=dev))
            x = x + torch.randn_like(x) * sg[:, None, None, None]
        if self.p_gray > 0:
            x = torch.where(P(self.p_gray)[:, None, None, None], x.mean(1, keepdim=True).expand_as(x), x)

        rows = torch.arange(S, device=dev).view(1, S, 1) / S
        cols = torch.arange(S, device=dev).view(1, 1, S) / S
        def box(r0, r1, c0, on):
            return (rows >= r0[:, None, None]) & (rows < r1[:, None, None]) & (cols >= c0[:, None, None]) & \
                   (cols < (1 - c0)[:, None, None]) & on[:, None, None]
        if self.p_shades > 0:                                                        # sunglasses
            r0 = U(0.33, 0.38)
            m = box(r0, r0 + U(0.11, 0.15), U(0.10, 0.18), P(self.p_shades))
            x = torch.where(m[:, None], U(0, 0.16)[:, None, None, None], x)
        if self.p_mask > 0:                                                          # face mask
            col = torch.tensor([[.86, .88, .90], [.47, .67, .78], [.12, .12, .12]], device=dev)[torch.randint(0, 3, (B,), device=dev)]
            m = box(U(0.52, 0.58), U(0.88, 0.96), U(0.18, 0.25), P(self.p_mask))
            x = torch.where(m[:, None], col[:, :, None, None], x)
        if self.p_erase > 0:                                                         # random erasing
            r0, c0 = U(0.0, 0.6), U(0.0, 0.6)
            h, w = U(0.15, 0.4), U(0.15, 0.4)
            m = (rows >= r0[:, None, None]) & (rows < (r0 + h)[:, None, None]) & (cols >= c0[:, None, None]) & \
                (cols < (c0 + w)[:, None, None]) & P(self.p_erase)[:, None, None]
            x = torch.where(m[:, None], torch.rand(B, 3, 1, 1, device=dev), x)
        return _norm(x.clamp(0, 1))
