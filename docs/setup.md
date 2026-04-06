# Setup Guide

## Requirements

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.9+ | 3.10 or 3.11 recommended |
| PyTorch | 2.x | Platform-specific install required |
| CUDA (optional) | 11.8+ | For GPU-accelerated training |
| RAM | 8 GB+ | 16 GB recommended for full dataset |
| Disk space | ~5 GB | For raw .mat files and processed .npz files |

---

## Step 1: Clone and Create Environment

```bash
git clone https://github.com/shreyabalki/eeg-card-game.git
cd eeg-card-game

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows PowerShell
```

---

## Step 2: Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Contents of `requirements.txt`:
```
numpy
scipy
pandas
matplotlib
scikit-learn
tensorflow==2.12
h5py
mne
seaborn
tqdm
```

---

## Step 3: Install PyTorch

PyTorch installation depends on your hardware. Visit [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for the exact command.

**CPU only (works everywhere):**
```bash
pip install torch torchvision
```

**CUDA 11.8 (NVIDIA GPU):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Step 4: Verify Installation

```bash
# Check Python and TensorFlow
python check_tf.py

# Check PyTorch
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Check scikit-learn and NumPy
python -c "import sklearn, numpy; print('sklearn:', sklearn.__version__, '| numpy:', numpy.__version__)"
```

Expected output:
```
Python executable: /path/to/.venv/bin/python
TensorFlow version: 2.12.x
PyTorch: 2.x.x
CUDA available: True/False
sklearn: 1.x.x | numpy: 1.x.x
```

---

## Step 5: Prepare Raw Data

The raw MATLAB files are **not included** in the repository (data is excluded via `.gitignore`).

```bash
# Create the data directory
mkdir -p data/raw

# Place your session files here:
# data/raw/sessionevents01.mat
# data/raw/sessionevents02.mat
# ...
# data/raw/sessionevents22.mat
```

Expected MATLAB file structure per session:
- `data`: array of shape `(801, 32, 3, E)` — time × channels × players × events
- `labels`: array of shape `(E, 8)` — metadata per event
- `t`: array of shape `(801,)` — time axis

---

## Step 6: Run the Full Pipeline

### 6a. Extract data and run QC
```bash
python scripts/mat_to_csv_all_sessions.py
```
Outputs:
- `reports/qc_mat_extract.csv` — per-session shape/NaN/timing validation
- `reports/labels_uniques_top30.xlsx` — label value frequencies

### 6b. Build training datasets
```bash
python scripts/prepare_dataset_v3.py
```
Outputs (in `data/processed/`):
- `mixed_train.npz`, `mixed_test.npz`
- `type0_train.npz`, `type0_test.npz`
- `type1_train.npz`, `type1_test.npz`
- `type2_train.npz`, `type2_test.npz`
- `type3_train.npz`, `type3_test.npz`

### 6c. Train a model
```bash
# CNN on the mixed dataset
python scripts/train_eval_v3.py --model cnn --dataset mixed --epochs 30

# Transformer with custom learning rate
python scripts/train_eval_v3.py --model transformer --dataset type1 --epochs 50 --lr 5e-4 --patience 10
```

### 6d. View results
```bash
# Check results CSV
python -c "import pandas as pd; print(pd.read_csv('data/processed/train_eval_v4_eventsplit_results.csv').to_string())"

# Plot CNN vs Transformer comparison
python plot_results.py
# → saves figure_results.png
```

---

## Common Issues

### ImportError: No module named 'torch'
PyTorch is not in requirements.txt because installation is hardware-dependent. Install it manually (see Step 3).

### FileNotFoundError: data/raw/sessionevents01.mat
Raw data files must be placed in `data/raw/` manually. They are not distributed with the repository.

### CUDA out of memory
Reduce batch size: `--batch-size 32` or `--batch-size 16`. Alternatively, train on CPU (slower but reliable).

### Shape mismatch errors during training
Verify datasets were built with the current version of `prepare_dataset_v3.py`. Delete old `.npz` files and rebuild.

```bash
rm data/processed/*.npz
python scripts/prepare_dataset_v3.py
```

### NaN loss during training
Try reducing the learning rate: `--lr 1e-4`. If NaN persists, check QC report for sessions with data anomalies.

---

## Optional: Run Sequence Analysis Tools

```bash
# Inspect label column distributions
python scripts/inspect_labels.py

# Find timing anomalies in events
python scripts/detect_delayed_prompts.py
# → reports/delay_flag_summary.csv

# Analyze post-"table full" event sequences
python scripts/inspect_table_full_sequences.py
# → reports/table_full_following_sequences.csv

# Top-50 event sequence patterns
python scripts/summarize_sequences.py
# → reports/table_full_sequence_patterns_top50.csv
```

---

*Continue reading: [Technical Decisions](technical-decisions.md) | [Results & Impact](results-and-impact.md)*
