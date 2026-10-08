#!/usr/bin/env bash
# Usage: bash run_training.sh [dataset_directory_name ...]
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/.venv}"
INSTALL_REQUIREMENTS="${INSTALL_REQUIREMENTS:-1}"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/data}"
CACHE_DIR="${CACHE_DIR:-$SCRIPT_DIR/cache}"
MODEL="${MODEL:-emcad_b2}"
LOSS="${LOSS:-chain_pair}"
EPOCHS="${EPOCHS:-100}"
SIZE="${SIZE:-256}"
LR="${LR:-1e-4}"
SEED="${SEED:-0}"
WORKERS="${WORKERS:-4}"
PREP_WORKERS="${PREP_WORKERS:-4}"
LINES="${LINES:-128}"
LEVELS="${LEVELS:-24}"
HABIT="${HABIT:-centred}"
PRETRAINED="${PRETRAINED:-1}"
EVAL_SIZE="${EVAL_SIZE:-train}"
# A fresh output directory keeps old results out of the new score table.
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/runs/batch_$(date +%Y%m%d_%H%M%S)_$$}"
REPORT_PATH="${REPORT_PATH:-$OUTPUT_DIR/results.md}"

if [[ "$INSTALL_REQUIREMENTS" == 1 ]]; then
    if [[ ! -x "$VENV_DIR/bin/python" ]]; then
        "$PYTHON" -m venv "$VENV_DIR"
    fi
    PYTHON="$VENV_DIR/bin/python"
    "$PYTHON" -m pip install --upgrade pip
    # Optional index for a specific CUDA/CPU PyTorch wheel distribution.
    if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
        "$PYTHON" -m pip install 'torch>=2.8,<3' 'torchvision>=0.23,<1' --index-url "$TORCH_INDEX_URL"
    fi
    "$PYTHON" -m pip install -r "$SCRIPT_DIR/requirements.txt"
elif [[ -x "$VENV_DIR/bin/python" ]]; then
    PYTHON="$VENV_DIR/bin/python"
fi
"$PYTHON" -c 'import torch, torchvision, timm, numpy, scipy, skimage, PIL'

DEVICE="${DEVICE:-$("$PYTHON" -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')}"
if [[ "$DEVICE" == cuda* ]]; then
    BATCH="${BATCH:-16}"
else
    BATCH="${BATCH:-2}"
fi
EVAL_BATCH="${EVAL_BATCH:-$BATCH}"
"$PYTHON" - "$EPOCHS" "$SIZE" "$BATCH" "$EVAL_BATCH" "$PREP_WORKERS" "$WORKERS" "$LINES" "$LEVELS" <<'PY'
import sys
epochs, size, batch, eval_batch, prep_workers, workers, lines, levels = map(int, sys.argv[1:])
if min(epochs, batch, eval_batch, prep_workers, lines, levels) < 1 or workers < 0 or size < 32 or size % 32:
    raise SystemExit('Use positive epochs/batches/prepare workers/lines/levels, nonnegative workers, and SIZE divisible by 32 (at least 32).')
PY

if [[ $# -gt 0 ]]; then
    DATASETS=("$@")
else
    DATASETS=(kvasir clinicdb_polyp colondb_polyp etis_polyp etis_root isic_2d isic_2018 lung liver covid19 drive)
    # Include additional directories with the supported split structure.
    while IFS= read -r dataset; do
        DATASETS+=("$dataset")
    done < <("$PYTHON" - "$DATA_DIR" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
if root.is_dir():
    for p in sorted(root.iterdir()):
        if (p / 'train' / 'images').is_dir():
            print(p.name)
PY
    )
fi

if [[ -e "$OUTPUT_DIR/experiment_state.json" ]]; then
    echo "Output directory already contains an experiment. Choose a new OUTPUT_DIR." >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"
STATE="$OUTPUT_DIR/experiment_state.json"
"$PYTHON" experiment_report.py plan --state "$STATE" --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" --model "$MODEL" --loss "$LOSS" \
    --epochs "$EPOCHS" --size "$SIZE" --batch "$BATCH" --device "$DEVICE" \
    --seed "$SEED" --eval-size "$EVAL_SIZE" "${DATASETS[@]}"

make_report() {
    "$PYTHON" experiment_report.py report --state "$STATE" --output "$REPORT_PATH"
}
finish() {
    local code=$?
    trap - EXIT
    make_report || code=1
    exit "$code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
update() {
    "$PYTHON" experiment_report.py update --state "$STATE" --dataset "$dataset" --status "$1" --reason "${2:-}"
}
make_report
echo "Device: $DEVICE; model: $MODEL; loss: $LOSS; epochs: $EPOCHS"
echo "Outputs: $OUTPUT_DIR"

# Use a file rather than a pipe so loop state and failure status persist.
"$PYTHON" - "$STATE" > "$OUTPUT_DIR/dataset_plan.tsv" <<'PY'
import json, sys
for row in json.load(open(sys.argv[1]))['datasets']:
    print(row['dataset'], row['status'], row['counts']['train'], int(row['use_val']), sep='\t')
PY
failures=0
while IFS=$'\t' read -r dataset status train_count use_val; do
    if [[ "$status" == skipped ]]; then
        echo "Skipping $dataset: missing or insufficient image/mask pairs (see report)."
        continue
    fi
    run_dir="$OUTPUT_DIR/$dataset"
    mkdir -p "$run_dir"
    batch="$BATCH"
    if (( batch > train_count )); then batch="$train_count"; fi
    prep_args=(--dataset folder --root "$DATA_DIR/$dataset" --name "$dataset"
        --cache "$CACHE_DIR" --size "$SIZE" --habits "$HABIT" --seed "$SEED"
        --lines "$LINES" --levels "$LEVELS" --workers "$PREP_WORKERS")
    if [[ "$LOSS" == full ]]; then prep_args+=(--band-splits ""); fi
    echo "Preparing $dataset"
    update preparing
    if ! "$PYTHON" -u prepare.py "${prep_args[@]}" 2>&1 | tee "$run_dir/prepare.log"; then
        update failed "Preprocessing failed; see $run_dir/prepare.log"
        failures=$((failures + 1))
        make_report
        continue
    fi
    train_args=(--cache "$CACHE_DIR" --data "$dataset" --out "$run_dir"
        --model "$MODEL" --loss "$LOSS" --epochs "$EPOCHS" --size "$SIZE"
        --batch "$batch" --eval-batch "$EVAL_BATCH" --lr "$LR" --seed "$SEED"
        --workers "$WORKERS" --device "$DEVICE" --habit "$HABIT" --band-seed "$SEED"
        --lines "$LINES" --levels "$LEVELS" --test "$dataset:test" --eval-size "$EVAL_SIZE")
    if [[ "$PRETRAINED" != 1 ]]; then train_args+=(--no-pretrained); fi
    if [[ "$use_val" != 1 ]]; then train_args+=(--no-val); fi
    echo "Training $dataset (batch=$batch)"
    update training
    if "$PYTHON" -u train.py "${train_args[@]}" 2>&1 | tee "$run_dir/train.log"; then
        if ! update completed; then
            update failed "Training exited successfully but its test summary is missing or invalid; see $run_dir/train.log"
            failures=$((failures + 1))
        fi
    else
        update failed "Training or evaluation failed; see $run_dir/train.log"
        failures=$((failures + 1))
    fi
    make_report
done < "$OUTPUT_DIR/dataset_plan.tsv"

echo "Finished. Markdown scores: $REPORT_PATH"
if (( failures > 0 )); then exit 1; fi
