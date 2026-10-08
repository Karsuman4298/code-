import math

import torch
import torch.nn.functional as F

RGB2XYZ = torch.tensor([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]])
WHITE = torch.tensor([0.950456, 1.0, 1.088754])


def to_grid(coords, height, width):
    g = coords.clone()
    g[..., 0] = 2.0 * g[..., 0] / (width - 1) - 1.0
    g[..., 1] = 2.0 * g[..., 1] / (height - 1) - 1.0
    return g


def streamline_loglik(logits, grid):
    z = F.grid_sample(logits, grid, mode="bilinear", padding_mode="border", align_corners=True)[:, 0]
    lf = F.logsigmoid(z)
    lb = F.logsigmoid(-z)
    zero = torch.zeros_like(z[..., :1])
    head = torch.cat([zero, lf.cumsum(-1)], -1)
    tail = torch.cat([lb.flip(-1).cumsum(-1).flip(-1), zero], -1)
    return head + tail


def masked_image_mean(per_image, counts):
    has = (counts > 0).to(per_image.dtype)
    return (per_image * has).sum() / has.sum().clamp_min(1.0)


def interval_nll(ll, valid):
    v = valid.to(ll.dtype)
    nll = -torch.logsumexp(ll, -1)
    counts = v.sum(1)
    per_image = (nll * v).sum(1) / counts.clamp_min(1.0)
    return masked_image_mean(per_image, counts) / (ll.shape[-1] - 1)


def transition_logits(bins, beta, device=None):
    k = torch.arange(bins, dtype=torch.float32, device=device)
    return -(k[None, :] - k[:, None]).abs() / beta


def compact(ll, valid, breaks, closed):
    b, s, k1 = ll.shape
    comp = torch.cumsum(breaks.long(), 1)
    order = torch.argsort((~valid).long() * s + torch.arange(s, device=ll.device), dim=1)
    ll = torch.gather(ll, 1, order.unsqueeze(-1).expand(-1, -1, k1))
    valid = torch.gather(valid, 1, order)
    comp = torch.gather(comp, 1, order)
    closed = torch.gather(closed, 1, order) & valid
    prev_comp = torch.cat([torch.full_like(comp[:, :1], -1), comp[:, :-1]], 1)
    prev_valid = torch.cat([~valid[:, :1], valid[:, :-1]], 1)
    breaks = (comp != prev_comp) | (valid != prev_valid)
    breaks[:, 0] = True
    return ll * valid.unsqueeze(-1).to(ll.dtype), valid, breaks, closed


def chain_nll(ll, log_t, valid, breaks, closed, cyclic=True):
    ll, valid, breaks, closed = compact(ll, valid, breaks, closed)
    b, s_count, k1 = ll.shape
    if cyclic:
        pin = torch.full((k1, k1), -1e9, dtype=ll.dtype, device=ll.device)
        pin.fill_diagonal_(0.0)
        closing = log_t.t()

        def finish(alpha, ring):
            loop = torch.logsumexp((alpha + closing).flatten(1), -1)
            line = torch.logsumexp(alpha.flatten(1), -1)
            return torch.where(ring, loop, line)

        alpha = ll[:, 0].unsqueeze(1) + pin
        ring = closed[:, 0]
        log_z = ll.new_zeros(b)
        for s in range(1, s_count):
            step = ll[:, s].unsqueeze(1) + torch.logsumexp(alpha.unsqueeze(-1) + log_t, dim=2)
            cut = breaks[:, s]
            log_z = log_z + torch.where(cut, finish(alpha, ring), torch.zeros_like(log_z))
            alpha = torch.where(cut[:, None, None], ll[:, s].unsqueeze(1) + pin, step)
            ring = torch.where(cut, closed[:, s], ring)
        log_z = log_z + finish(alpha, ring)
    else:
        alpha = ll[:, 0]
        for s in range(1, s_count):
            prop = torch.logsumexp(alpha.unsqueeze(-1) + log_t, dim=1)
            free = torch.logsumexp(alpha, -1, keepdim=True).expand_as(prop)
            prop = torch.where(breaks[:, s : s + 1], free, prop)
            alpha = ll[:, s] + prop
        log_z = torch.logsumexp(alpha, -1)
    counts = valid.to(ll.dtype).sum(1)
    return masked_image_mean(-log_z / counts.clamp_min(1.0), counts) / (k1 - 1)


def partial_bce(logits, inner, outer, fov, ring):
    fg = (inner & fov).float()
    bg = (~outer & fov).float()
    known = fg + bg
    loss = F.binary_cross_entropy_with_logits(logits, fg, reduction="none")
    total = (loss * known).sum() / known.sum().clamp_min(1.0)
    if ring > 0:
        near = F.max_pool2d((outer & ~inner).float(), 2 * ring + 1, stride=1, padding=ring)
        f, g = fg * near, bg * near
        total = total + 0.5 * ((loss * f).sum() / f.sum().clamp_min(1.0) + (loss * g).sum() / g.sum().clamp_min(1.0))
    return total


def shift_pair(t, dy, dx):
    h, w = t.shape[-2:]
    ya, yb = slice(0, h - dy), slice(dy, h)
    xa, xb = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    return t[..., ya, xa], t[..., yb, xb]


def pairwise_loss(logits, feats, region, theta, tau, dilation):
    lf, lb = F.logsigmoid(logits), F.logsigmoid(-logits)
    total, count = logits.new_zeros(()), logits.new_zeros(())
    d = dilation
    for dy, dx in ((0, d), (d, 0), (d, d), (d, -d)):
        fa, fb = shift_pair(feats, dy, dx)
        sim = torch.exp(-(fa - fb).norm(dim=1, keepdim=True) / theta)
        ra, rb = shift_pair(region, dy, dx)
        mask = ((sim >= tau) & ra & rb).float()
        a1, b1 = shift_pair(lf, dy, dx)
        a0, b0 = shift_pair(lb, dy, dx)
        same = torch.logaddexp(a1 + b1, a0 + b0)
        total = total - (same * mask).sum()
        count = count + mask.sum()
    return total / count.clamp_min(1.0)


def rgb_to_lab(rgb):
    lin = torch.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    xyz = torch.einsum("ij,bjhw->bihw", RGB2XYZ.to(rgb), lin) / WHITE.to(rgb).view(1, 3, 1, 1)
    f = torch.where(xyz > 0.008856, xyz.clamp_min(1e-12) ** (1.0 / 3.0), 7.787 * xyz + 16.0 / 116.0)
    return torch.stack([116.0 * f[:, 1] - 16.0, 500.0 * (f[:, 0] - f[:, 1]), 200.0 * (f[:, 1] - f[:, 2])], 1)


def gaussian_blur(x, sigma):
    if sigma <= 0:
        return x
    r = int(math.ceil(2 * sigma))
    k = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
    g = torch.exp(-0.5 * (k / sigma) ** 2)
    g = g / g.sum()
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), g.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), g.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)


def soft_dice(prob, target, fov):
    p = (prob * fov).flatten(1)
    t = (target * fov).flatten(1)
    inter = (p * t).sum(1)
    return (1.0 - (2.0 * inter + 1.0) / (p.sum(1) + t.sum(1) + 1.0)).mean()


def bce_dice(logits, target, fov):
    t, f = target.float(), fov.float()
    bce = (F.binary_cross_entropy_with_logits(logits, t, reduction="none") * f).sum() / f.sum().clamp_min(1.0)
    return bce + soft_dice(torch.sigmoid(logits), t, f)


def two_mask_loss(logits, inner, outer, fov):
    p, f = torch.sigmoid(logits), fov.float()
    return soft_dice(p, inner.float(), f) + soft_dice(p, outer.float(), f)
