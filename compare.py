import argparse
import csv

import numpy as np
from scipy import stats


def read_csv(path):
    with open(path) as fh:
        return {row["id"]: row for row in csv.DictReader(fh)}


def holm(pvalues):
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def bootstrap_ci(diff, n_boot, rng):
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", action="append", required=True)
    p.add_argument("--metrics", default="dice,iou,hd95,bf")
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="")
    args = p.parse_args()
    methods = []
    for spec in args.method:
        name, paths = spec.split("=", 1)
        methods.append((name, [read_csv(x) for x in paths.split(",") if x]))
    ids = sorted(set.intersection(*[set(run) for _, runs in methods for run in runs]))
    if not ids:
        raise SystemExit("no image ids shared by all files")
    rng = np.random.default_rng(args.seed)
    records = []
    for metric in [m for m in args.metrics.split(",") if m]:
        per_method = []
        for name, runs in methods:
            table = np.array([[float(run[i][metric]) for i in ids] for run in runs])
            per_method.append((name, table))
        ref_name, ref_table = per_method[0]
        ref_image = ref_table.mean(0)
        tests = []
        for name, table in per_method:
            run_means = table.mean(1)
            row = {"metric": metric, "method": name, "runs": table.shape[0], "images": len(ids), "mean": float(run_means.mean()), "std_over_runs": float(run_means.std(ddof=1)) if len(run_means) > 1 else 0.0}
            if name != ref_name:
                image_mean = table.mean(0)
                diff = image_mean - ref_image
                if np.allclose(diff, 0):
                    pvalue = 1.0
                else:
                    pvalue = float(stats.wilcoxon(image_mean, ref_image, zero_method="wilcox", alternative="two-sided").pvalue)
                low, high = bootstrap_ci(diff, args.boot, rng)
                row.update({"diff_vs_" + ref_name: float(diff.mean()), "ci_low": low, "ci_high": high, "p": pvalue})
                tests.append(row)
            records.append(row)
        if tests:
            for row, adj in zip(tests, holm(np.array([r["p"] for r in tests]))):
                row["p_holm"] = float(adj)
    keys = []
    for r in records:
        for k in r:
            if k not in keys:
                keys.append(k)
    for r in records:
        text = f"{r['metric']:6s} {r['method']:20s} runs={r['runs']} mean={r['mean']:.4f} sd={r['std_over_runs']:.4f}"
        if "p" in r:
            diff_key = [k for k in r if k.startswith("diff_vs_")][0]
            text += f"  {diff_key}={r[diff_key]:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  p={r['p']:.2e}  p_holm={r['p_holm']:.2e}"
        print(text)
    if args.out:
        with open(args.out, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys, restval="")
            writer.writeheader()
            writer.writerows(records)


if __name__ == "__main__":
    main()
