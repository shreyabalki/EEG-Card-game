# EEG Card Game — Neural Role Classification from Brainwave Signals

> **Classifying player intent in a multiplayer card game using EEG deep learning.**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![GitHub Pages](https://img.shields.io/badge/Portfolio-Live-blueviolet)](https://shreyabalki.github.io/eeg-card-game)

---

## What This Project Does

This project builds an **end-to-end EEG machine learning pipeline** that classifies a player's real-time role in a 3-player card game — using only raw brain signals. No game state. No button presses. Just EEG.

Given 32-channel EEG recordings from 22 experimental sessions, the system learns to distinguish three cognitive states:

| Label | Role | Description |
|-------|------|-------------|
| `0` | Played | Player who just took a turn |
| `1` | Current / Next | Player whose turn is active or upcoming |
| `2` | Observer | Player watching but not acting |

Two neural architectures are trained and compared: a **CNN baseline** and a **Patch Transformer**, evaluated on held-out event-based test splits.

---

## Key Highlights

- **Full pipeline ownership**: MATLAB → tensor extraction → dataset engineering → training → evaluation
- **Two model architectures**: Conv1D CNN and Patch-based Transformer
- **Leakage-safe evaluation**: event-wise stratified splits per session and event type
- **22 EEG sessions** parsed from raw `.mat` files into training-ready tensors
- **32-channel, 801-sample EEG** per player per game event
- **5 dataset variants** generated: mixed + per-event-type subsets

---

## Problem Statement

In cognitive neuroscience and BCI (Brain-Computer Interface) research, a key challenge is **decoding intent and role from passive brain activity** — without relying on explicit user input. During a card game, each player has a distinct cognitive load depending on their role (active player vs. passive observer). This project asks: **can a neural network learn those cognitive signatures from raw EEG?**

This has applications in:
- Adaptive brain-computer interfaces
- Passive mental state monitoring
- Cognitive load detection in real-world tasks

---

## Architecture

```
Raw MATLAB Files (sessionevents01.mat … sessionevents22.mat)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  mat_to_csv_all_sessions.py                             │
│  • Load (T=801, C=32, P=3, E=events) tensors            │
│  • Extract 8-column label metadata                      │
│  • QC reports: shape checks, NaN detection, timing      │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  prepare_dataset_v3.py                                  │
│  • Reshape to (N, C=32, T=801) per player-event         │
│  • Assign role labels (0/1/2)                           │
│  • Event-wise stratified train/test split               │
│  • Output: 5 NPZ datasets (mixed, type0, type1, type2, type3) │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  train_eval_v3.py  (--model cnn | transformer)          │
│  • SimpleEEGCNN: Conv1d(32→64→128→256) + MLP head       │
│  • PatchTransformer: d_model=128, nhead=4, layers=2     │
│  • AdamW + CrossEntropyLoss + early stopping            │
│  • Metrics: Accuracy, Balanced-Acc, Macro-F1, W-F1      │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
        results CSV + model checkpoints (.pt)
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Data I/O** | `scipy.io.loadmat`, NumPy, Pandas |
| **Deep Learning** | PyTorch — Conv1d, TransformerEncoder, BatchNorm, ELU |
| **ML Evaluation** | scikit-learn — balanced accuracy, macro-F1, confusion matrix |
| **Visualization** | Matplotlib, Seaborn |
| **EEG Tooling** | MNE-Python |
| **Environment** | Python 3.9+, argparse CLI, pathlib, logging |

---

## Repository Structure

```
EEG-Card-game/
├── scripts/
│   ├── mat_to_csv_all_sessions.py   # Step 1: MATLAB extraction + QC
│   ├── prepare_dataset_v3.py        # Step 2: Dataset builder (event-wise splits)
│   ├── train_eval_v3.py             # Step 3: CNN + Transformer trainer (unified)
│   ├── train_baseline.py            # Legacy: CNN-only trainer
│   ├── train_transformer.py         # Legacy: Transformer-only trainer
│   ├── build_dataset.py             # Legacy: single-NPZ dataset builder
│   ├── config.py                    # Label column index mappings
│   ├── inspect_labels.py            # Label distribution analysis
│   ├── detect_delayed_prompts.py    # Event timing anomaly detection
│   ├── inspect_table_full_sequences.py  # Post-event sequence analysis
│   ├── summarize_sequences.py       # Sequence pattern frequency
│   └── guess_label_mapping.py       # Heuristic label validation
├── src/
│   └── data_loading.py              # Shared MATLAB loader utility
├── notebooks/
│   ├── 02_cnn_baseline.ipynb        # CNN exploration notebook
│   └── 03_eegnet.ipynb              # EEGNet architecture experiments
├── docs/
│   ├── index.html                   # GitHub Pages portfolio site
│   ├── styles.css                   # Site styling
│   ├── overview.md                  # Project summary
│   ├── architecture.md              # Technical architecture deep-dive
│   ├── features.md                  # Feature documentation
│   ├── setup.md                     # Detailed setup guide
│   ├── technical-decisions.md       # Design rationale
│   ├── results-and-impact.md        # Results, metrics, impact
│   └── roadmap.md                   # Future improvements
├── reports/
│   ├── qc_mat_extract.csv           # Session QC results
│   ├── labels_uniques_top30.xlsx    # Label value distributions
│   ├── delay_flag_summary.csv       # Delayed prompt statistics
│   └── table_full_sequence_patterns_top50.csv
├── plot_results.py                  # CNN vs Transformer bar chart
├── requirements.txt                 # Python dependencies
└── .gitignore
```

---

## Setup

### Prerequisites

- Python 3.9+
- PyTorch 2.x (install separately for your hardware)
- Raw MATLAB data files: `sessionevents01.mat` … `sessionevents22.mat`

### Installation

```bash
git clone https://github.com/shreyabalki/eeg-card-game.git
cd eeg-card-game

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Install PyTorch — choose your platform at https://pytorch.org/get-started
pip install torch torchvision
```

### Verify Environment

```bash
python check_tf.py
python -c "import torch; print(torch.__version__)"
```

---

## Usage

### Step 1 — Extract raw data and run QC

```bash
# Place raw .mat files in data/raw/
mkdir -p data/raw
cp /path/to/sessionevents*.mat data/raw/

python scripts/mat_to_csv_all_sessions.py
# Output: reports/qc_mat_extract.csv, reports/labels_uniques_top30.xlsx
```

### Step 2 — Build training datasets

```bash
python scripts/prepare_dataset_v3.py
# Output: data/processed/*.npz (5 datasets: mixed, type0, type1, type2, type3)
```

### Step 3 — Train and evaluate

```bash
# CNN on mixed dataset
python scripts/train_eval_v3.py --model cnn --dataset mixed --epochs 30

# Transformer on type-1 event dataset
python scripts/train_eval_v3.py --model transformer --dataset type1 --epochs 30 --lr 5e-4

# All results saved to:
# data/processed/train_eval_v4_eventsplit_results.csv
```

### Step 4 — Visualize results

```bash
python plot_results.py
# Output: figure_results.png (CNN vs Transformer Macro-F1 bar chart)
```

### Optional: Data analysis utilities

```bash
python scripts/inspect_labels.py               # Label column distributions
python scripts/detect_delayed_prompts.py       # Timing anomalies
python scripts/summarize_sequences.py          # Event sequence patterns
```

---

## Model Architectures

### SimpleEEGCNN

Three convolutional layers process temporal EEG features, progressively increasing channel depth while reducing sequence length, followed by adaptive pooling and a linear classification head.

```
Input: (B, 32 channels, 801 time steps)
Conv1d(32→64, k=7) → BatchNorm → ELU → MaxPool(2)
Conv1d(64→128, k=5) → BatchNorm → ELU → MaxPool(2)
Conv1d(128→256, k=3) → BatchNorm → ELU → AdaptiveAvgPool(1)
Linear(256 → n_classes)
```

### PatchTransformer

The EEG signal is divided into temporal patches; each patch is linearly projected into a `d_model`-dimensional embedding, then processed by a standard Transformer encoder. Mean pooling over patch tokens feeds a classification head.

```
Input: (B, 32 channels, 801 time steps)
Patch Embedding: patches of (C × patch_len) → Linear → d_model=128
TransformerEncoder: nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1
Mean Pool over patches
Linear(128 → n_classes)
```

**Training**: AdamW, weight_decay=1e-4, CrossEntropyLoss, early stopping on Macro-F1 (patience=6).

---

## Dataset Design

A key engineering decision is **event-wise splitting** to prevent data leakage.

Rather than randomly splitting individual EEG samples (which would allow the same card-game event to appear in both train and test), the pipeline groups events by session and event type, then holds out 20% of entire events per group. This ensures the model evaluates on genuinely unseen game moments.

Five NPZ datasets are produced:

| Dataset | Classes | Description |
|---------|---------|-------------|
| `mixed` | 3 | All event types combined |
| `type0` | 2 | 2-class event subset |
| `type1` | 3 | Event type 1 only |
| `type2` | 3 | Event type 2 only |
| `type3` | 3 | Event type 3 only |

---

## Results

> **Note**: The table below uses placeholder values. Replace with your actual results from `train_eval_v4_eventsplit_results.csv` after running training.

| Model | Dataset | Accuracy | Balanced Acc | Macro-F1 |
|-------|---------|---------|--------------|---------|
| CNN | mixed | [replace]% | [replace]% | [replace] |
| Transformer | mixed | [replace]% | [replace]% | [replace] |
| CNN | type1 | [replace]% | [replace]% | [replace] |
| Transformer | type1 | [replace]% | [replace]% | [replace] |

Confusion matrices and per-class F1 scores are saved with each run.

---

## Challenges Solved

| Challenge | Solution |
|-----------|---------|
| Raw data is MATLAB `.mat` format | Custom loader using `scipy.io.loadmat` with shape validation |
| 3D data tensor (T, C, P, E) must become 2D samples | Reshape: expand player axis, flatten to (N, C, T) |
| Unknown label column semantics | Heuristic analysis scripts + manual inspection workflow |
| Data leakage risk from random splitting | Event-wise stratified split (20% held-out events per session/type) |
| Class imbalance across roles | Balanced accuracy and macro-F1 as primary metrics; optional class weights |
| Multiple dataset variants needed | Parameterized dataset builder outputting 5 NPZ files |

---

## Future Improvements

- [ ] **Real-time inference**: WebSocket server for live EEG stream classification
- [ ] **EEGNet integration**: Add depthwise-separable CNN (EEGNet) as third architecture
- [ ] **Cross-subject generalization**: Leave-one-subject-out cross-validation
- [ ] **Frequency-domain features**: FFT / wavelet preprocessing as alternative input
- [ ] **Hyperparameter search**: Optuna sweep over architecture and training parameters
- [ ] **CI/CD pipeline**: GitHub Actions for automated testing and linting
- [ ] **Docker containerization**: Reproducible execution environment

---



## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Portfolio site: (https://shreyabalki.github.io/EEG-Card-game/#problem)*
