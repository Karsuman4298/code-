import json
import os
import re

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".npy"}
ISIC_IMAGE = re.compile(r"^(ISIC_\d{7})$")
ISIC_MASK = re.compile(r"^(ISIC_\d{7})_segmentation$")
BAND_KEYS = ("inner", "outer", "mid", "coords", "valid", "breaks", "closed")
EXPECTED = {"isic2017": {"train": 2000, "val": 150, "test": 600}, "isic2018": {"train": 2594, "val": 100, "test": 1000}}
RESAMPLE = getattr(Image, "Resampling", Image).BILINEAR


def split_of_dir(path):
    name = os.path.basename(os.path.dirname(path)).lower()
    if "test" in name:
        return "test"
    if "validation" in name or name in ("val", "valid") or name.endswith("_val") or "_val_" in name:
        return "val"
    if "train" in name:
        return "train"
    return None


def discover_isic(root):
    images, masks = {}, {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            stem, ext = os.path.splitext(f)
            if ext.lower() not in IMAGE_EXT:
                continue
            path = os.path.join(dirpath, f)
            split = split_of_dir(path)
            if split is None:
                continue
            hit = ISIC_MASK.match(stem)
            if hit:
                masks[(split, hit.group(1))] = path
                continue
            hit = ISIC_IMAGE.match(stem)
            if hit:
                images[(split, hit.group(1))] = path
    out = {}
    for (split, iid), image_path in images.items():
        mask_path = masks.get((split, iid))
        if mask_path is not None:
            out.setdefault(split, []).append((iid, image_path, mask_path))
    return {k: sorted(v) for k, v in out.items()}


def list_images(folder, suffix=""):
    found = {}
    for f in sorted(os.listdir(folder)):
        stem, ext = os.path.splitext(f)
        if ext.lower() not in IMAGE_EXT:
            continue
        if suffix and stem.endswith(suffix):
            stem = stem[: -len(suffix)]
        found[stem] = os.path.join(folder, f)
    return found


def find_dir(root, name):
    for dirpath, dirnames, _ in os.walk(root):
        for d in sorted(dirnames):
            if d.lower() == name:
                return os.path.join(dirpath, d)
    return None


def pair_folders(image_dir, mask_dir, mask_suffix=""):
    images = list_images(image_dir)
    masks = list_images(mask_dir, mask_suffix)
    return [(k, images[k], masks[k]) for k in sorted(images) if k in masks]


def random_split(items, n_train, seed, path):
    if os.path.exists(path):
        with open(path) as fh:
            chosen = json.load(fh)
    else:
        order = np.random.default_rng(seed).permutation(len(items))
        chosen = {"train": [items[i][0] for i in sorted(order[:n_train])], "test": [items[i][0] for i in sorted(order[n_train:])]}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(chosen, fh, indent=1)
    lookup = {it[0]: it for it in items}
    return {split: [lookup[k] for k in keys if k in lookup] for split, keys in chosen.items()}


def discover(args):
    # Preserve existing splits in exported datasets, including NumPy slices.
    if args.dataset == "folder" and args.root:
        found = {}
        for split in ("train", "val", "test"):
            image_dir = os.path.join(args.root, split, "images")
            mask_dir = os.path.join(args.root, split, "masks")
            if os.path.isdir(image_dir) and os.path.isdir(mask_dir):
                found[split] = pair_folders(image_dir, mask_dir, args.mask_suffix)
        if not found:
            raise SystemExit(f"no split/images and split/masks directories under {args.root}")
        return found
    if args.dataset in ("isic2017", "isic2018"):
        found = discover_isic(args.root)
        for split, n in EXPECTED[args.dataset].items():
            got = len(found.get(split, []))
            if got != n:
                print(f"warning: {args.dataset} {split}: found {got} image/mask pairs, expected {n}")
        return found
    if args.dataset == "kvasir":
        image_dir = args.image_dir or find_dir(args.root, "images")
        mask_dir = args.mask_dir or find_dir(args.root, "masks")
        if image_dir is None or mask_dir is None:
            raise SystemExit(f"could not find images/ and masks/ under {args.root}")
        items = pair_folders(image_dir, mask_dir, args.mask_suffix)
        if len(items) != 1000:
            print(f"warning: kvasir: found {len(items)} image/mask pairs, expected 1000")
        path = os.path.join(args.cache, cache_name(args), f"split_seed{args.split_seed}_n{args.kvasir_train}.json")
        return random_split(items, args.kvasir_train, args.split_seed, path)
    items = pair_folders(args.image_dir, args.mask_dir, args.mask_suffix)
    if args.split_file:
        with open(args.split_file) as fh:
            chosen = json.load(fh)
        lookup = {it[0]: it for it in items}
        return {split: [lookup[k] for k in keys if k in lookup] for split, keys in chosen.items()}
    return {"test": items}


def cache_name(args):
    if args.name:
        return args.name
    if args.dataset == "kvasir" and args.split_seed != 0:
        return f"kvasir_s{args.split_seed}"
    return args.dataset


def band_tag(habit, width_frac, width_min, width_max, simplify, contain, seed, lines, levels):
    flag = "c" if contain else "n"
    return f"{habit}_w{width_frac:g}-{width_min:g}-{width_max:g}_t{simplify:g}{flag}_s{seed}_L{lines}_K{levels}"


def base_dir(cache, name, size, split):
    return os.path.join(cache, name, str(size), "base", split)


def band_dir(cache, name, size, tag, split):
    return os.path.join(cache, name, str(size), "bands", tag, split)


def load_image(path):
    if os.path.splitext(path)[1].lower() != ".npy":
        return Image.open(path).convert("RGB")
    image = np.load(path, allow_pickle=False)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and image.size and image.min() >= 0 and image.max() <= 1:
            image = image * 255
        image = np.clip(image, 0, 255).astype(np.uint8)
    return Image.fromarray(image).convert("RGB")


def load_resized(job):
    image_path, mask_path, size = job
    image = load_image(image_path)
    w, h = image.size
    image = np.asarray(image.resize((size, size), RESAMPLE), dtype=np.uint8)
    mask = Image.fromarray(load_mask(mask_path).astype(np.uint8) * 255).resize((size, size), RESAMPLE)
    return image, np.asarray(mask) > 127, (h, w)


def load_mask(path):
    if os.path.splitext(path)[1].lower() == ".npy":
        mask = np.load(path, allow_pickle=False)
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        elif mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        if mask.ndim != 2:
            raise ValueError(f"expected a 2D binary mask in {path}, got {mask.shape}")
        return mask > 0
    return np.asarray(Image.open(path).convert("L")) > 127


class SegmentationSet(Dataset):
    def __init__(self, base, band=None, dense=None):
        self.base = base
        self.band = band
        self.dense = dense
        self.arrays = None
        with open(os.path.join(base, "meta.json")) as fh:
            self.meta = json.load(fh)

    def open(self):
        arrays = {"image": np.load(os.path.join(self.base, "images.npy"), mmap_mode="r"), "mask": np.load(os.path.join(self.base, "masks.npy"), mmap_mode="r")}
        if self.band is not None:
            for k in BAND_KEYS:
                arrays[k] = np.load(os.path.join(self.band, k + ".npy"), mmap_mode="r")
        self.arrays = arrays

    def __len__(self):
        return len(self.meta["ids"])

    def __getitem__(self, i):
        if self.arrays is None:
            self.open()
        item = {k: np.array(v[i]) for k, v in self.arrays.items()}
        item["index"] = i
        if self.dense is not None:
            item["dense"] = bool(self.dense[i])
        return item
