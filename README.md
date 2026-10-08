# Segmentation with uncertain boundaries

PyTorch binary segmentation using EMCAD B2 or a ResNet-50 U-Net. Supports full
mask supervision and simulated uncertain-boundary losses, including
`chain_pair`.

## Train on an NVIDIA GPU server

Clone this repository and transfer your datasets into `data/`:

```bash
git clone https://github.com/Karsuman4298/code-.git
cd code-
```

Each dataset uses `data/<dataset>/{train,val,test}/{images,masks}/`, with
matching image/mask filename stems. NumPy `.npy` slices are supported.

Run all available datasets and write the final score table:

```bash
DEVICE=cuda REPORT_PATH=training_scores.md bash run_training.sh
```

The script creates `.venv`, installs dependencies, and trains one model per
dataset. Missing datasets are skipped, and dataset failures are logged while
the remaining datasets continue. Defaults are EMCAD B2, `chain_pair`, 100
epochs, and batch size 16 on CUDA. Reduce `BATCH` if GPU memory is insufficient.
Pretrained weights may download on first use. Your NVIDIA driver must support
the installed PyTorch CUDA build; `TORCH_INDEX_URL` can select a specific wheel
index.

Check one dataset before a full run:

```bash
DEVICE=cuda EPOCHS=1 BATCH=2 bash run_training.sh kvasir
```

Checkpoints and detailed results are saved under `runs/`. Datasets, virtual
environments, caches, and generated model files are excluded from Git.

See [RUN_TRAINING.md](RUN_TRAINING.md) for configuration, report metrics, and
split-overlap handling.
