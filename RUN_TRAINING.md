# Automated training

Run from a terminal:

```bash
bash run_training.sh
```

The script creates `.venv`, installs `requirements.txt`, prepares the datasets,
and trains a separate EMCAD B2 model with `chain_pair` loss for each dataset.
It uses CUDA when available and otherwise CPU. Pretrained encoder weights are
downloaded on first use. Dataset files are not downloaded or modified.

Known datasets are attempted even if absent; missing datasets are marked
`skipped`. Additional immediate subdirectories of `data/` containing
`train/images/` are discovered automatically. Supported layout:

```text
data/<dataset>/
  train/images/    train/masks/
  val/images/      val/masks/       # optional
  test/images/     test/masks/
```

Images and masks must have matching filename stems. NumPy `.npy` slices and
standard image formats are supported. At least two training pairs and one test
pair are required. Masks are binary; all positive NumPy mask values are treated
as foreground.

Each run creates a new `runs/batch_<timestamp>_<pid>/` directory. Its `results.md`
contains a table with dataset names, status, split counts, checkpoint, and all
nine test metrics as mean ± standard deviation across images. Each dataset's
subdirectory contains checkpoints, training/preparation logs, per-image test
CSV, and summary JSON. Failures are recorded and other datasets continue;
the script exits with status 1 if any dataset failed. Skips are not failures.
The report is also updated during the run and on ordinary interruption.

## Examples

Only selected datasets:

```bash
bash run_training.sh kvasir clinicdb_polyp lung
```

One-epoch pipeline check using CPU and random initialization:

```bash
EPOCHS=1 BATCH=2 EVAL_BATCH=2 PRETRAINED=0 DEVICE=cpu \
  bash run_training.sh kvasir
```

Full training on a CUDA machine:

```bash
DEVICE=cuda EPOCHS=100 BATCH=16 bash run_training.sh
```

Fully supervised baseline:

```bash
LOSS=full bash run_training.sh kvasir
```

Use dependencies already installed in the selected Python environment:

```bash
INSTALL_REQUIREMENTS=0 PYTHON=python3 bash run_training.sh kvasir
```

When `.venv/bin/python` exists it is preferred. Set `VENV_DIR` to another path
to choose a different environment. `PYTHON` supplies the interpreter used to
create a new virtual environment.

Choose data and output paths:

```bash
DATA_DIR=/path/to/data OUTPUT_DIR=runs/my_experiment \
  REPORT_PATH=runs/my_experiment/scores.md bash run_training.sh
```

Other settings: `MODEL` (`emcad_b2` or `unet_r50`), `SIZE` (default 256, divisible
by 32), `LR` (default 1e-4), `SEED`, `WORKERS`, `PREP_WORKERS`, `CACHE_DIR`,
`HABIT`, `LINES`, `LEVELS`, and `EVAL_SIZE` (`train` or `original`). Reduce
`BATCH` if GPU memory runs out. CPU training is substantially slower.

`TORCH_INDEX_URL` optionally selects a PyTorch wheel index when installing
dependencies. An existing experiment state in `OUTPUT_DIR` is rejected to avoid
mixing results; choose a fresh directory to rerun. Cached preprocessing is
reused, so remove the corresponding cache if source data changed.

## Split overlap

Matching slice IDs are checked for identical image and mask file contents
between splits. Validation selection is disabled when matching validation
pairs overlap train or test, and the final checkpoint is evaluated instead.
Train/test overlap is explicitly noted in the report: those scores are not an
independent test assessment. The checks do not detect renamed duplicates or
establish patient-level independence.
