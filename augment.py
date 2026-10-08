import math

import torch
import torch.nn.functional as F

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
MASK_KEYS = ("mask", "inner", "outer", "mid")


def uniform(n, low, high, generator):
    return low + (high - low) * torch.rand(n, generator=generator)


def random_affine(n, size, rot, scale, shift, flip, generator):
    angle = uniform(n, -math.radians(rot), math.radians(rot), generator)
    s = uniform(n, 1.0 - scale, 1.0 + scale, generator)
    tx = uniform(n, -shift * size, shift * size, generator)
    ty = uniform(n, -shift * size, shift * size, generator)
    fx = torch.where(torch.rand(n, generator=generator) < flip, -1.0, 1.0)
    fy = torch.where(torch.rand(n, generator=generator) < flip, -1.0, 1.0)
    c = (size - 1) / 2.0
    cos, sin = torch.cos(angle), torch.sin(angle)
    m = torch.zeros(n, 3, 3)
    m[:, 0, 0] = s * cos * fx
    m[:, 0, 1] = -s * sin * fy
    m[:, 1, 0] = s * sin * fx
    m[:, 1, 1] = s * cos * fy
    m[:, 0, 2] = c + tx - (m[:, 0, 0] + m[:, 0, 1]) * c
    m[:, 1, 2] = c + ty - (m[:, 1, 0] + m[:, 1, 1]) * c
    m[:, 2, 2] = 1.0
    return m


def sampling_grid(m, size):
    inv = torch.linalg.inv(m)
    r = torch.arange(size, dtype=torch.float32, device=m.device)
    ys, xs = torch.meshgrid(r, r, indexing="ij")
    p = torch.stack([xs, ys, torch.ones_like(xs)], -1).view(-1, 3)
    src = torch.einsum("bij,nj->bni", inv, p)
    g = 2.0 * src[..., :2] / (size - 1) - 1.0
    return g.view(-1, size, size, 2)


def color_jitter(x, strength, generator):
    if strength <= 0:
        return x
    n = x.shape[0]

    def factor():
        return uniform(n, 1.0 - strength, 1.0 + strength, generator).view(n, 1, 1, 1).to(x.device)

    x = x * factor()
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * factor() + mean
    gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    x = (x - gray) * factor() + gray
    return x.clamp(0.0, 1.0)


def make_batch(raw, device, size, train, cfg, generator):
    rgb = raw["image"].to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
    keys = [k for k in MASK_KEYS if k in raw]
    masks = torch.stack([raw[k].to(device, non_blocking=True) for k in keys], 1).float()
    fov = torch.ones_like(masks[:, :1])
    out = {}
    coords = raw["coords"].to(device, non_blocking=True).float() if "coords" in raw else None
    valid = raw["valid"].to(device, non_blocking=True) if "valid" in raw else None
    if train and cfg.augment:
        n = rgb.shape[0]
        m = random_affine(n, size, cfg.rot, cfg.scale, cfg.shift, cfg.flip, generator).to(device)
        grid = sampling_grid(m, size)
        rgb = F.grid_sample(rgb, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        masks = F.grid_sample(masks, grid, mode="nearest", padding_mode="zeros", align_corners=True)
        fov = F.grid_sample(fov, grid, mode="nearest", padding_mode="zeros", align_corners=True)
        if coords is not None:
            coords = torch.einsum("bij,bskj->bski", m[:, :2, :2], coords) + m[:, None, None, :2, 2]
            inside = (coords >= 0).all(-1) & (coords <= size - 1).all(-1)
            valid = valid & inside.all(-1)
        rgb = color_jitter(rgb, cfg.color, generator)
    for i, k in enumerate(keys):
        out[k] = masks[:, i : i + 1] > 0.5
    out["fov"] = fov > 0.5
    out["rgb"] = rgb
    out["image"] = (rgb - MEAN.to(device)) / STD.to(device)
    if coords is not None:
        out["coords"] = coords
        out["valid"] = valid
        out["breaks"] = raw["breaks"].to(device, non_blocking=True)
        out["closed"] = raw["closed"].to(device, non_blocking=True)
    if "dense" in raw:
        out["dense"] = raw["dense"].to(device, non_blocking=True)
    return out


def normalize_uint8(images, device):
    x = images.to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
    return (x - MEAN.to(device)) / STD.to(device)
