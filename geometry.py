import numpy as np
import scipy.sparse as sp
from scipy import ndimage as ndi
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from skimage import draw, measure

HABITS = {
    "centred": (0.5, 0.05, 0.0),
    "tight_inner": (0.2, 0.05, 0.0),
    "tight_outer": (0.8, 0.05, 0.0),
    "mixed": (0.5, 0.25, 0.0),
    "annotators": (0.5, 0.08, 0.3),
}


def deepest_disc(region, radius=2):
    dist = ndi.distance_transform_edt(region)
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    yy, xx = np.ogrid[: region.shape[0], : region.shape[1]]
    return region & ((yy - y) ** 2 + (xx - x) ** 2 <= radius**2)


def seed_empty_components(inner, outer):
    inner = inner & outer
    labels, count = ndi.label(outer)
    for i in range(1, count + 1):
        comp = labels == i
        if not (inner & comp).any():
            inner = inner | deepest_disc(comp)
    return inner


def band_laplace(inner, outer):
    inner = seed_empty_components(inner, outer)
    band = outer & ~inner
    u = np.where(outer, 0.0, 1.0)
    if not band.any() or outer.all():
        return u, inner, False
    labels, _ = ndi.label(band)
    fixed = inner | ~outer
    touching = np.unique(labels[band & ndi.binary_dilation(fixed)])
    solvable = band & np.isin(labels, touching[touching > 0])
    u[band & ~solvable] = 1.0
    ys, xs = np.nonzero(solvable)
    n = ys.size
    if n == 0:
        return u, inner, False
    h, w = band.shape
    index = np.full((h, w), -1, dtype=np.int64)
    index[ys, xs] = np.arange(n)
    diag = np.zeros(n)
    rhs = np.zeros(n)
    rows, cols = [], []
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        yy, xx = ys + dy, xs + dx
        ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        src, yy, xx = np.nonzero(ok)[0], yy[ok], xx[ok]
        diag[src] += 1.0
        inside = solvable[yy, xx]
        rows.append(src[inside])
        cols.append(index[yy[inside], xx[inside]])
        rhs[src[~inside]] += u[yy[~inside], xx[~inside]]
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    values = np.concatenate([diag, -np.ones(rows.size)])
    a = sp.csc_matrix((values, (np.concatenate([np.arange(n), rows]), np.concatenate([np.arange(n), cols]))), shape=(n, n))
    u[ys, xs] = spsolve(a, rhs)
    return u, inner, True


def contour_seeds(inner, n_lines, min_length=4.0):
    contours = []
    for c in measure.find_contours(inner.astype(np.float32), 0.5):
        if len(c) < 3:
            continue
        seg = np.sqrt((np.diff(c, axis=0) ** 2).sum(1))
        length = float(seg.sum())
        if length < min_length:
            continue
        contours.append((c, seg, length, bool(np.allclose(c[0], c[-1]))))
    empty = np.zeros((0, 2)), np.zeros(0, np.int64), np.zeros(0, bool)
    if not contours or n_lines <= 0:
        return empty
    lengths = np.array([c[2] for c in contours])
    counts = np.floor(n_lines * lengths / lengths.sum()).astype(int)
    counts[int(np.argmax(lengths))] += n_lines - int(counts.sum())
    points, comp, closed = [], [], []
    for ci, ((c, seg, length, is_closed), m) in enumerate(zip(contours, counts)):
        if m <= 0:
            continue
        s = np.concatenate([[0.0], np.cumsum(seg)])
        t = (np.arange(m) + 0.5) * length / m
        points.append(np.stack([np.interp(t, s, c[:, 0]), np.interp(t, s, c[:, 1])], 1))
        comp.append(np.full(m, ci))
        closed.append(np.full(m, is_closed))
    if not points:
        return empty
    return np.concatenate(points), np.concatenate(comp), np.concatenate(closed)


def trace_streamlines(u, inner, n_lines, n_levels, step=0.5, border=1.0, max_steps=None):
    h, w = u.shape
    coords = np.zeros((n_lines, n_levels, 2), np.float32)
    valid = np.zeros(n_lines, bool)
    breaks = np.zeros(n_lines, bool)
    closed = np.zeros(n_lines, bool)
    if n_lines > 0:
        breaks[0] = True
    pts, comp, is_closed = contour_seeds(inner, n_lines)
    m = len(pts)
    if m == 0:
        return coords, valid, breaks, closed
    if max_steps is None:
        max_steps = int(4 * (h + w) / step)
    gy, gx = np.gradient(u)

    def direction(yq, xq):
        vy = ndi.map_coordinates(gy, [yq, xq], order=1, mode="nearest")
        vx = ndi.map_coordinates(gx, [yq, xq], order=1, mode="nearest")
        norm = np.sqrt(vx * vx + vy * vy)
        moving = norm > 1e-9
        norm = np.where(moving, norm, 1.0)
        return vy / norm * moving, vx / norm * moving, moving

    y, x = pts[:, 0].copy(), pts[:, 1].copy()
    path_x, path_y = [x.copy()], [y.copy()]
    path_u = [ndi.map_coordinates(u, [y, x], order=1, mode="nearest")]
    alive = path_u[0] < 1.0 - 1e-6
    hit = np.zeros(m, bool)
    for _ in range(max_steps):
        vy, vx, mov1 = direction(y, x)
        vy, vx, mov2 = direction(y + 0.5 * step * vy, x + 0.5 * step * vx)
        alive &= mov1 & mov2
        y = np.where(alive, y + step * vy, y)
        x = np.where(alive, x + step * vx, x)
        outside = (x < border) | (x > w - 1 - border) | (y < border) | (y > h - 1 - border)
        hit |= alive & outside
        y = np.clip(y, 0, h - 1)
        x = np.clip(x, 0, w - 1)
        uu = ndi.map_coordinates(u, [y, x], order=1, mode="nearest")
        path_x.append(x.copy())
        path_y.append(y.copy())
        path_u.append(uu)
        alive &= (uu < 1.0 - 1e-6) & ~hit
        if not alive.any():
            break
    px = np.stack(path_x, 1)
    py = np.stack(path_y, 1)
    pu = np.maximum.accumulate(np.stack(path_u, 1), axis=1)
    pu = pu + 1e-9 * np.arange(pu.shape[1])
    levels = (np.arange(n_levels) + 0.5) / n_levels
    for i in range(m):
        coords[i, :, 0] = np.interp(levels, pu[i], px[i])
        coords[i, :, 1] = np.interp(levels, pu[i], py[i])
    valid[:m] = (pu[:, -1] >= levels[-1]) & ~hit
    breaks[:m] = np.concatenate([[True], comp[1:] != comp[:-1]])
    closed[:m] = is_closed
    if m < n_lines:
        breaks[m] = True
    return coords, valid, breaks, closed


def midline(u, inner, outer):
    return inner | (outer & ~inner & (u < 0.5))


def smooth_periodic(rng, modes=3):
    a = rng.normal(size=modes) / np.arange(1, modes + 1)
    b = rng.normal(size=modes) / np.arange(1, modes + 1)
    scale = np.sqrt(0.5 * (a**2 + b**2).sum()) + 1e-12
    k = 2 * np.pi * np.arange(1, modes + 1)
    return lambda s: (a * np.cos(np.outer(s, k)) + b * np.sin(np.outer(s, k))).sum(1) / scale


def fill_polygons(mask, tolerance):
    out = np.zeros_like(mask)
    for c in measure.find_contours(np.pad(mask, 1).astype(np.float32), 0.5):
        if len(c) < 3:
            continue
        p = measure.approximate_polygon(c, tolerance=tolerance) if tolerance > 0 else c
        p = p - 1.0
        rr, cc = draw.polygon(p[:, 0], p[:, 1], shape=mask.shape)
        out[rr, cc] = True
    return out


def simulate_band(gt, rng, rho, rho_jitter, width, width_jitter=0.2, tolerance=1.0, contain=True):
    if not gt.any():
        empty = np.zeros_like(gt)
        return empty, empty.copy()
    contours = [c for c in measure.find_contours(gt.astype(np.float32), 0.5) if len(c) > 2]
    if not contours:
        return gt.copy(), gt.copy()
    pts, svals = [], []
    for c in contours:
        seg = np.sqrt((np.diff(c, axis=0) ** 2).sum(1))
        s = np.concatenate([[0.0], np.cumsum(seg)])
        pts.append(c)
        svals.append(s / max(s[-1], 1e-9))
    pts = np.concatenate(pts)
    svals = np.concatenate(svals)
    h, w = gt.shape
    yy, xx = np.mgrid[:h, :w]
    _, nearest = cKDTree(pts).query(np.stack([yy.ravel(), xx.ravel()], 1))
    s_map = svals[nearest]
    f_w, f_r = smooth_periodic(rng), smooth_periodic(rng)
    wmap = (width * np.exp(width_jitter * f_w(s_map))).reshape(h, w)
    rmap = np.clip(rho + rho_jitter * f_r(s_map), 0.02, 0.98).reshape(h, w)
    sdf = ndi.distance_transform_edt(~gt) - ndi.distance_transform_edt(gt)
    inner = fill_polygons(sdf < -(rmap * wmap), tolerance)
    outer = fill_polygons(sdf < (1.0 - rmap) * wmap, tolerance)
    if contain:
        inner = inner & gt
        outer = outer | gt
    return inner & outer, outer | inner


def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2 * (a & b).sum() / s)


def make_band(gt, rng, habit, width_frac, width_min, width_max, tolerance, contain, n_lines, n_levels):
    rho, jitter, spread = HABITS[habit]
    if spread > 0:
        rho = rho + rng.uniform(-spread, spread)
    r_eq = np.sqrt(gt.sum() / np.pi)
    width = float(np.clip(width_frac * r_eq, width_min, width_max))
    inner, outer = simulate_band(gt, rng, rho, jitter, width, 0.2, tolerance, contain)
    u, seeded, ok = band_laplace(inner, outer)
    if ok:
        coords, valid, breaks, closed = trace_streamlines(u, seeded, n_lines, n_levels)
    else:
        coords = np.zeros((n_lines, n_levels, 2), np.float32)
        valid = np.zeros(n_lines, bool)
        breaks = np.zeros(n_lines, bool)
        breaks[0] = True
        closed = np.zeros(n_lines, bool)
    mid = midline(u, seeded, outer)
    edge = gt & ~ndi.binary_erosion(gt) & outer & ~inner
    position = float(u[edge].mean()) if edge.any() else float("nan")
    stats = {"midline_dice": dice(mid, gt), "position": position, "valid": float(valid.mean()), "width": width}
    return {"inner": inner, "outer": outer, "mid": mid, "coords": coords, "valid": valid, "breaks": breaks, "closed": closed}, stats
