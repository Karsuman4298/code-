import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from augment import normalize_uint8
from datasets import base_dir, load_mask
from metrics import COLUMNS, image_metrics
from models import build_model


@torch.no_grad()
def predict(model, images, device, batch, amp):
    model.eval()
    for start in range(0, len(images), batch):
        x = normalize_uint8(torch.from_numpy(np.array(images[start : start + batch])), device)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x)[-1]
        yield start, logits.float()


def evaluate_split(model, cache, name, size, split, device, batch=16, eval_size="train", tolerance=2.0, amp=False):
    base = base_dir(cache, name, size, split)
    with open(os.path.join(base, "meta.json")) as fh:
        meta = json.load(fh)
    images = np.load(os.path.join(base, "images.npy"), mmap_mode="r")
    masks = np.load(os.path.join(base, "masks.npy"), mmap_mode="r")
    rows = []
    for start, logits in predict(model, images, device, batch, amp):
        for j in range(logits.shape[0]):
            i = start + j
            if eval_size == "original":
                gt = load_mask(meta["masks"][i])
                h, w = gt.shape
                pred = F.interpolate(logits[j : j + 1], size=(h, w), mode="bilinear", align_corners=False)[0, 0] > 0
                tol = tolerance * max(h, w) / size
            else:
                gt = np.array(masks[i])
                pred = logits[j, 0] > 0
                tol = tolerance
            row = image_metrics(pred.cpu().numpy(), gt, tol)
            row["id"] = meta["ids"][i]
            rows.append(row)
    return rows


def summarize(rows):
    return {k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))] for k in COLUMNS}


def write_rows(rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", *COLUMNS])
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in ["id", *COLUMNS]})


def format_summary(summary):
    return "  ".join(f"{k} {v[0]:.4f}±{v[1]:.4f}" for k, v in summary.items())


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["args"]["model"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device), ckpt["args"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--eval-size", default="train", choices=["train", "original"])
    p.add_argument("--tolerance", type=float, default=2.0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--out", default="")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true")
    args = p.parse_args()
    device = torch.device(args.device)
    model, train_args = load_checkpoint(args.ckpt, device)
    rows = evaluate_split(model, args.cache, args.data, train_args["size"], args.split, device, args.batch, args.eval_size, args.tolerance, args.amp)
    summary = summarize(rows)
    print(f"{args.data} {args.split} ({args.eval_size}, n={len(rows)}): {format_summary(summary)}")
    out = args.out or os.path.join(os.path.dirname(args.ckpt), f"{args.data}_{args.split}_{args.eval_size}.csv")
    write_rows(rows, out)
    with open(out[:-4] + "_summary.json", "w") as fh:
        json.dump({"n": len(rows), "summary": summary}, fh, indent=1)


if __name__ == "__main__":
    main()
