import argparse
import csv
import json
import math
import os
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from augment import make_batch
from datasets import SegmentationSet, band_dir, band_tag, base_dir
from evaluate import evaluate_split, format_summary, summarize, write_rows
from losses import (
    bce_dice,
    chain_nll,
    gaussian_blur,
    interval_nll,
    pairwise_loss,
    partial_bce,
    rgb_to_lab,
    streamline_loglik,
    to_grid,
    transition_logits,
    two_mask_loss,
)
from models import build_model

LOSSES = ("full", "two_mask", "midline", "pce", "pce_pair", "interval", "interval_pair", "chain", "chain_pair")
LOG_FIELDS = ("epoch", "lr", "seconds", "loss", "certain", "band", "pair", "dense", "val_dice")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mean_of(values):
    values = list(values)
    return sum(values) / len(values)


def compute_loss(outputs, batch, args, lam, log_t):
    outputs = [o.float() for o in outputs]
    heads = outputs if len(outputs) == 1 else outputs + [sum(outputs)]
    final = outputs[-1]
    fov = batch["fov"]
    parts = {}
    if args.loss == "full":
        loss = mean_of(bce_dice(h, batch["mask"], fov) for h in heads)
    elif args.loss == "two_mask":
        loss = mean_of(two_mask_loss(h, batch["inner"], batch["outer"], fov) for h in heads)
    elif args.loss == "midline":
        loss = mean_of(bce_dice(h, batch["mid"], fov) for h in heads)
    else:
        loss = mean_of(partial_bce(h, batch["inner"], batch["outer"], fov, args.ring) for h in heads)
        parts["certain"] = loss.detach()
        extra = final.new_zeros(())
        if args.loss in ("interval", "interval_pair", "chain", "chain_pair"):
            ll = streamline_loglik(final, to_grid(batch["coords"], args.size, args.size))
            if args.loss.startswith("chain"):
                band = chain_nll(ll, log_t, batch["valid"], batch["breaks"], batch["closed"], cyclic=not args.open_chain)
            else:
                band = interval_nll(ll, batch["valid"])
            parts["band"] = band.detach()
            extra = extra + args.w_band * band
        if args.loss.endswith("_pair"):
            with torch.no_grad():
                feats = gaussian_blur(rgb_to_lab(batch["rgb"]), args.pair_sigma)
                d = args.pair_dilation
                region = (F.max_pool2d((batch["outer"] & ~batch["inner"]).float(), 2 * d + 1, stride=1, padding=d) > 0.5) & fov
            pair = pairwise_loss(final, feats, region, args.pair_theta, args.pair_tau, d)
            parts["pair"] = pair.detach()
            extra = extra + args.w_pair * pair
        loss = loss + lam * extra
    if args.loss != "full" and "dense" in batch and bool(batch["dense"].any()):
        sel = batch["dense"]
        dense = mean_of(bce_dice(h[sel], batch["mask"][sel], fov[sel]) for h in heads)
        parts["dense"] = dense.detach()
        loss = loss + args.w_dense * dense
    parts["loss"] = loss.detach()
    return loss, parts


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--habit", default="centred")
    p.add_argument("--width-frac", type=float, default=0.15)
    p.add_argument("--width-min", type=float, default=3.0)
    p.add_argument("--width-max", type=float, default=24.0)
    p.add_argument("--simplify", type=float, default=1.0)
    p.add_argument("--no-contain", action="store_true")
    p.add_argument("--band-seed", type=int, default=0)
    p.add_argument("--lines", type=int, default=128)
    p.add_argument("--levels", type=int, default=24)
    p.add_argument("--model", default="emcad_b2", choices=["emcad_b2", "unet_r50"])
    p.add_argument("--loss", default="chain_pair", choices=LOSSES)
    p.add_argument("--open-chain", action="store_true")
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--ring", type=int, default=8)
    p.add_argument("--pair-theta", type=float, default=2.0)
    p.add_argument("--pair-tau", type=float, default=0.3)
    p.add_argument("--pair-dilation", type=int, default=2)
    p.add_argument("--pair-sigma", type=float, default=1.0)
    p.add_argument("--w-band", type=float, default=1.0)
    p.add_argument("--w-pair", type=float, default=1.0)
    p.add_argument("--w-dense", type=float, default=1.0)
    p.add_argument("--band-warmup", type=float, default=0.1)
    p.add_argument("--dense-frac", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=-1.0)
    p.add_argument("--optimizer", default="auto", choices=["auto", "adamw", "adam"])
    p.add_argument("--schedule", default="cosine", choices=["cosine", "constant"])
    p.add_argument("--clip", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--rot", type=float, default=30.0)
    p.add_argument("--scale", type=float, default=0.15)
    p.add_argument("--shift", type=float, default=0.05)
    p.add_argument("--flip", type=float, default=0.5)
    p.add_argument("--color", type=float, default=0.1)
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--no-val", action="store_true")
    p.add_argument("--eval-batch", type=int, default=16)
    p.add_argument("--test", default="")
    p.add_argument("--eval-size", default="train", choices=["train", "original"])
    p.add_argument("--bf-tol", type=float, default=2.0)
    args = p.parse_args()
    if args.optimizer == "auto":
        args.optimizer = "adamw" if args.model == "emcad_b2" else "adam"
    if args.weight_decay < 0:
        args.weight_decay = 1e-4 if args.optimizer == "adamw" else 0.0
    if not args.test:
        args.test = f"{args.data}:test"
    return args


def save_checkpoint(model, args, epoch, score, path):
    torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch, "val_dice": score}, path)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    torch.backends.cudnn.benchmark = True
    tag = band_tag(args.habit, args.width_frac, args.width_min, args.width_max, args.simplify, not args.no_contain, args.band_seed, args.lines, args.levels)
    base_train = base_dir(args.cache, args.data, args.size, "train")
    if not os.path.exists(os.path.join(base_train, "meta.json")):
        raise SystemExit(f"missing {base_train}; run prepare.py first")
    band_train = None
    if args.loss != "full":
        band_train = band_dir(args.cache, args.data, args.size, tag, "train")
        if not os.path.exists(os.path.join(band_train, "stats.json")):
            raise SystemExit(f"missing {band_train}; run prepare.py with the same band settings")
    with open(os.path.join(base_train, "meta.json")) as fh:
        n_train = len(json.load(fh)["ids"])
    dense = None
    if args.dense_frac > 0 and args.loss != "full":
        dense = np.zeros(n_train, bool)
        dense[np.random.default_rng(args.seed + 7919).choice(n_train, int(round(args.dense_frac * n_train)), replace=False)] = True
    train_set = SegmentationSet(base_train, band_train, dense)
    loader = DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
        persistent_workers=args.workers > 0,
    )
    has_val = not args.no_val and os.path.exists(os.path.join(base_dir(args.cache, args.data, args.size, "val"), "meta.json"))
    model = build_model(args.model, pretrained=not args.no_pretrained).to(device)
    optim_cls = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    optimizer = optim_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = max(1, args.epochs * len(loader))
    if args.schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda it: 0.5 * (1.0 + math.cos(math.pi * min(it, total) / total)))
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda it: 1.0)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    log_t = transition_logits(args.levels + 1, args.beta, device)
    aug = SimpleNamespace(augment=not args.no_augment, rot=args.rot, scale=args.scale, shift=args.shift, flip=args.flip, color=args.color)
    generator = torch.Generator().manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "args.json"), "w") as fh:
        json.dump(vars(args), fh, indent=1)
    log_path = os.path.join(args.out, "log.csv")
    with open(log_path, "w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=LOG_FIELDS).writeheader()
    best, step = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        sums, count, start = {}, 0, time.time()
        for raw in loader:
            batch = make_batch(raw, device, args.size, True, aug, generator)
            lam = min(1.0, step / (args.band_warmup * total)) if args.band_warmup > 0 else 1.0
            with torch.autocast(device_type=device.type, enabled=amp):
                outputs = model(batch["image"])
            loss, parts = compute_loss(outputs, batch, args, lam, log_t)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v
            count += 1
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "seconds": round(time.time() - start, 1)}
        row.update({k: round(float(v) / max(count, 1), 5) for k, v in sums.items()})
        if has_val and (epoch % args.val_every == 0 or epoch == args.epochs):
            rows = evaluate_split(model, args.cache, args.data, args.size, "val", device, args.eval_batch, "train", args.bf_tol, amp)
            row["val_dice"] = round(float(np.mean([r["dice"] for r in rows])), 5)
            if row["val_dice"] > best:
                best = row["val_dice"]
                save_checkpoint(model, args, epoch, best, os.path.join(args.out, "best.pt"))
        save_checkpoint(model, args, epoch, best, os.path.join(args.out, "last.pt"))
        with open(log_path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=LOG_FIELDS, restval="", extrasaction="ignore").writerow(row)
        print(json.dumps(row), flush=True)
    chosen = os.path.join(args.out, "best.pt" if has_val and best >= 0 else "last.pt")
    model.load_state_dict(torch.load(chosen, map_location="cpu", weights_only=False)["model"])
    print(f"testing {os.path.basename(chosen)}")
    for target in [t for t in args.test.split(",") if t]:
        name, split = target.split(":")
        if not os.path.exists(os.path.join(base_dir(args.cache, name, args.size, split), "meta.json")):
            print(f"skip {target}: not prepared")
            continue
        rows = evaluate_split(model, args.cache, name, args.size, split, device, args.eval_batch, args.eval_size, args.bf_tol, amp)
        summary = summarize(rows)
        out = os.path.join(args.out, f"{name}_{split}_{args.eval_size}.csv")
        write_rows(rows, out)
        with open(out[:-4] + "_summary.json", "w") as fh:
            json.dump({"n": len(rows), "checkpoint": os.path.basename(chosen), "summary": summary}, fh, indent=1)
        print(f"{name} {split} ({args.eval_size}, n={len(rows)}): {format_summary(summary)}")


if __name__ == "__main__":
    main()
