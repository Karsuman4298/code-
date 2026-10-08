import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

from datasets import BAND_KEYS, band_dir, band_tag, base_dir, cache_name, discover, load_resized
from geometry import HABITS, make_band

WORKER = {}


def init_band_worker(masks_path, cfg):
    WORKER["masks"] = np.load(masks_path, mmap_mode="r")
    WORKER["cfg"] = cfg


def band_job(i):
    cfg = WORKER["cfg"]
    gt = np.array(WORKER["masks"][i])
    rng = np.random.default_rng([cfg["seed"], i])
    return i, make_band(gt, rng, cfg["habit"], cfg["width_frac"], cfg["width_min"], cfg["width_max"], cfg["simplify"], cfg["contain"], cfg["lines"], cfg["levels"])


def build_base(items, out, size, workers, overwrite):
    if os.path.exists(os.path.join(out, "meta.json")) and not overwrite:
        print(f"exists: {out}")
        return
    os.makedirs(out, exist_ok=True)
    n = len(items)
    images = np.lib.format.open_memmap(os.path.join(out, "images.npy"), mode="w+", dtype=np.uint8, shape=(n, size, size, 3))
    masks = np.lib.format.open_memmap(os.path.join(out, "masks.npy"), mode="w+", dtype=bool, shape=(n, size, size))
    sizes = [None] * n
    jobs = [(ip, mp, size) for _, ip, mp in items]
    with Pool(workers) as pool:
        for i, (image, mask, hw) in enumerate(pool.imap(load_resized, jobs, chunksize=8)):
            images[i] = image
            masks[i] = mask
            sizes[i] = list(hw)
    images.flush()
    masks.flush()
    meta = {"ids": [it[0] for it in items], "images": [it[1] for it in items], "masks": [it[2] for it in items], "sizes": sizes}
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump(meta, fh)
    print(f"wrote {n} images to {out}")


def build_band(base, out, cfg, workers, overwrite):
    if os.path.exists(os.path.join(out, "stats.json")) and not overwrite:
        print(f"exists: {out}")
        return
    os.makedirs(out, exist_ok=True)
    masks_path = os.path.join(base, "masks.npy")
    masks = np.load(masks_path, mmap_mode="r")
    n, h, w = masks.shape
    shapes = {"inner": (n, h, w), "outer": (n, h, w), "mid": (n, h, w), "coords": (n, cfg["lines"], cfg["levels"], 2), "valid": (n, cfg["lines"]), "breaks": (n, cfg["lines"]), "closed": (n, cfg["lines"])}
    dtypes = {"coords": np.float32}
    arrays = {k: np.lib.format.open_memmap(os.path.join(out, k + ".npy"), mode="w+", dtype=dtypes.get(k, bool), shape=shapes[k]) for k in BAND_KEYS}
    stats = [None] * n
    with Pool(workers, initializer=init_band_worker, initargs=(masks_path, cfg)) as pool:
        for i, (record, st) in pool.imap_unordered(band_job, range(n), chunksize=4):
            for k in BAND_KEYS:
                arrays[k][i] = record[k]
            stats[i] = st
    for a in arrays.values():
        a.flush()
    summary = {k: float(np.nanmean([s[k] for s in stats])) for k in stats[0]}
    summary["images_without_valid_streamlines"] = int(sum(s["valid"] == 0 for s in stats))
    summary["config"] = cfg
    with open(os.path.join(out, "stats.json"), "w") as fh:
        json.dump({"summary": summary, "per_image": stats}, fh)
    print(f"{cfg['habit']}: midline-rule Dice {summary['midline_dice']:.4f}, true boundary at u = {summary['position']:.3f}, valid streamlines {summary['valid']:.3f}, images without streamlines {summary['images_without_valid_streamlines']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["isic2017", "isic2018", "kvasir", "folder"])
    p.add_argument("--root", default="")
    p.add_argument("--cache", required=True)
    p.add_argument("--name", default="")
    p.add_argument("--image-dir", default="")
    p.add_argument("--mask-dir", default="")
    p.add_argument("--mask-suffix", default="")
    p.add_argument("--split-file", default="")
    p.add_argument("--kvasir-train", type=int, default=800)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--splits", default="")
    p.add_argument("--band-splits", default="train")
    p.add_argument("--habits", default="centred")
    p.add_argument("--width-frac", type=float, default=0.15)
    p.add_argument("--width-min", type=float, default=3.0)
    p.add_argument("--width-max", type=float, default=24.0)
    p.add_argument("--simplify", type=float, default=1.0)
    p.add_argument("--no-contain", action="store_true")
    p.add_argument("--lines", type=int, default=128)
    p.add_argument("--levels", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    name = cache_name(args)
    found = discover(args)
    splits = [s for s in args.splits.split(",") if s] or sorted(found)
    for split in splits:
        if split not in found or not found[split]:
            print(f"skip {split}: no image/mask pairs found")
            continue
        print(f"{name} {split}: {len(found[split])} pairs")
        build_base(found[split], base_dir(args.cache, name, args.size, split), args.size, args.workers, args.overwrite)
    contain = not args.no_contain
    for habit in [h for h in args.habits.split(",") if h]:
        if habit not in HABITS:
            raise SystemExit(f"unknown habit {habit}; choose from {sorted(HABITS)}")
        tag = band_tag(habit, args.width_frac, args.width_min, args.width_max, args.simplify, contain, args.seed, args.lines, args.levels)
        cfg = {"habit": habit, "width_frac": args.width_frac, "width_min": args.width_min, "width_max": args.width_max, "simplify": args.simplify, "contain": contain, "seed": args.seed, "lines": args.lines, "levels": args.levels}
        for split in [s for s in args.band_splits.split(",") if s]:
            base = base_dir(args.cache, name, args.size, split)
            if not os.path.exists(os.path.join(base, "meta.json")):
                print(f"skip bands for {split}: base arrays missing")
                continue
            build_band(base, band_dir(args.cache, name, args.size, tag, split), cfg, args.workers, args.overwrite)
            print(f"band tag: {tag}")


if __name__ == "__main__":
    main()
