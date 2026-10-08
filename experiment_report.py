"""Dataset checks and Markdown reporting for run_training.sh."""

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re


EXTENSIONS = {".npy", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
METRICS = (
    ("dice", "Dice ↑"), ("iou", "IoU ↑"), ("acc", "Accuracy ↑"),
    ("sen", "Sensitivity ↑"), ("spe", "Specificity ↑"),
    ("hd95", "HD95 ↓"), ("assd", "ASSD ↓"),
    ("bf", "Boundary F1 ↑"), ("shift", "Signed shift"),
)


def pairs(root, split):
    folders = [root / split / name for name in ("images", "masks")]
    if not all(p.is_dir() for p in folders):
        return {}
    maps = [
        {p.stem: p for p in sorted(folder.iterdir())
         if p.is_file() and p.suffix.lower() in EXTENSIONS}
        for folder in folders
    ]
    return {key: (maps[0][key], maps[1][key]) for key in maps[0].keys() & maps[1].keys()}


def identical_pairs(a, b):
    # Same exported slice IDs are checked by content, not just by their names.
    def digest(path):
        return hashlib.sha256(path.read_bytes()).digest()
    return sum(
        all(digest(x) == digest(y) for x, y in zip(a[key], b[key]))
        for key in a.keys() & b.keys()
    )


def write_state(path, state):
    path.write_text(json.dumps(state, indent=2) + "\n")


def plan(args):
    rows = []
    for name in dict.fromkeys(args.datasets):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError(f"invalid dataset name: {name!r}; use directory names, not paths")
        root = args.data_dir / name
        splits = {s: pairs(root, s) for s in ("train", "val", "test")}
        counts = {s: len(p) for s, p in splits.items()}
        reason = ""
        if not root.is_dir():
            reason = "Dataset directory missing"
        elif counts["train"] < 2:
            reason = "Need at least two paired training images/masks"
        elif counts["test"] == 0:
            reason = "No paired test images/masks"
        notes = []
        use_val = counts["val"] > 0
        if not reason:
            for a, b in (("val", "test"), ("train", "test"), ("train", "val")):
                overlap = identical_pairs(splits[a], splits[b])
                if overlap:
                    notes.append(f"{overlap} identical {a}/{b} image-mask pairs with matching IDs")
                    if a == "val" or b == "val":
                        use_val = False
            if not use_val:
                notes.append("Validation selection disabled; final checkpoint used")
        run_dir = args.output_dir / name
        rows.append({
            "dataset": name, "status": "skipped" if reason else "pending",
            "reason": reason, "counts": counts, "notes": notes, "use_val": use_val,
            "run_dir": str(run_dir),
            "summary_path": str(run_dir / f"{name}_test_{args.eval_size}_summary.json"),
        })
    config = {k: getattr(args, k) for k in
              ("model", "loss", "epochs", "size", "batch", "device", "seed", "eval_size")}
    write_state(args.state, {"started": datetime.now().astimezone().isoformat(timespec="seconds"),
                             "config": config, "datasets": rows})


def load_summary(row):
    result = json.loads(Path(row["summary_path"]).read_text())
    if result["n"] <= 0:
        raise ValueError("test summary has no images")
    for key, _ in METRICS:
        values = result["summary"][key]
        if len(values) != 2 or not all(math.isfinite(float(v)) for v in values):
            raise ValueError(f"invalid test metric: {key}")
    return result


def update(args):
    state = json.loads(args.state.read_text())
    row = next(r for r in state["datasets"] if r["dataset"] == args.dataset)
    if args.status == "completed":
        row["result"] = load_summary(row)
    row.update(status=args.status, reason=args.reason)
    write_state(args.state, state)


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def report(args):
    state = json.loads(args.state.read_text())
    cfg = state["config"]
    lines = ["# Segmentation training results", "",
             f"Started: {state['started']}", "",
             ", ".join(f"**{key}**: {cell(value)}" for key, value in cfg.items()), "",
             "Scores are test-set mean ± standard deviation across images, on a 0–1 scale "
             "for overlap/classification metrics. HD95 and ASSD are in pixels. "
             "Signed shift is predicted area minus true area, divided by true boundary length.", "",
             "| Dataset | Status | Train / Val / Test | Test N | Checkpoint | "
             + " | ".join(label for _, label in METRICS) + " | Notes |",
             "| " + " | ".join(["---"] * (6 + len(METRICS))) + " |"]
    failed = 0
    for row in state["datasets"]:
        status = row["status"]
        result = row.get("result", {}) if status == "completed" else {}
        metrics = [f"{result['summary'][key][0]:.4f} ± {result['summary'][key][1]:.4f}"
                   if result else "—" for key, _ in METRICS]
        notes = row["notes"] + ([row["reason"]] if row["reason"] else [])
        if status in ("pending", "preparing", "training"):
            notes.append("No completed result yet; run may have been interrupted")
        if status == "failed":
            failed += 1
        counts = " / ".join(str(row["counts"][s]) for s in ("train", "val", "test"))
        fields = [row["dataset"], status, counts, result.get("n", "—"),
                  result.get("checkpoint", "—"), *metrics, "; ".join(notes)]
        lines.append("| " + " | ".join(cell(v) for v in fields) + " |")
    lines.extend(["", "Each dataset has its own trained model. Logs, checkpoints, per-image CSVs, "
                  "and summary JSON files are stored in its run directory.", "",
                  "Overlap checks compare file contents for matching slice IDs; they do not "
                  "establish patient-level independence or detect renamed duplicates.", ""])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(f"Report: {args.output} ({failed} failed datasets)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    for key in ("model", "loss", "epochs", "size", "batch", "device", "seed", "eval-size"):
        p.add_argument("--" + key, required=True)
    p.add_argument("datasets", nargs="+")
    p.set_defaults(func=plan)
    p = sub.add_parser("update")
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--status", choices=["preparing", "training", "completed", "failed"], required=True)
    p.add_argument("--reason", default="")
    p.set_defaults(func=update)
    p = sub.add_parser("report")
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=report)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
