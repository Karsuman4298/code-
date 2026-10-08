import numpy as np
from scipy import ndimage as ndi

COLUMNS = ("dice", "iou", "acc", "sen", "spe", "hd95", "assd", "bf", "shift")


def surface(mask):
    return mask & ~ndi.binary_erosion(mask)


def image_metrics(pred, gt, tolerance=2.0):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = float((pred & gt).sum())
    fp = float((pred & ~gt).sum())
    fn = float((~pred & gt).sum())
    tn = float((~pred & ~gt).sum())
    out = {
        "dice": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 1.0,
        "iou": tp / (tp + fp + fn) if tp + fp + fn > 0 else 1.0,
        "acc": (tp + tn) / pred.size,
        "sen": tp / (tp + fn) if tp + fn > 0 else 1.0,
        "spe": tn / (tn + fp) if tn + fp > 0 else 1.0,
    }
    ps, gs = surface(pred), surface(gt)
    diag = float(np.hypot(*pred.shape))
    if ps.any() and gs.any():
        d_pg = ndi.distance_transform_edt(~gs)[ps]
        d_gp = ndi.distance_transform_edt(~ps)[gs]
        out["hd95"] = float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)))
        out["assd"] = float((d_pg.mean() + d_gp.mean()) / 2)
        precision = float((d_pg <= tolerance).mean())
        recall = float((d_gp <= tolerance).mean())
        out["bf"] = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    elif not ps.any() and not gs.any():
        out["hd95"], out["assd"], out["bf"] = 0.0, 0.0, 1.0
    else:
        out["hd95"], out["assd"], out["bf"] = diag, diag, 0.0
    out["shift"] = (float(pred.sum()) - float(gt.sum())) / max(float(gs.sum()), 1.0)
    return out
