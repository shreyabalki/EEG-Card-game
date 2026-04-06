# Technical Architecture

## Pipeline Overview

The system is organized as a sequential, modular pipeline. Each stage produces well-defined output artifacts that serve as inputs to the next stage.

```
┌──────────────────────────────────────────────────────────────────────┐
│  RAW DATA                                                            │
│  sessionevents01.mat … sessionevents22.mat                           │
│  Each: data(T=801, C=32, P=3, E) + labels(E, 8) + t(T)              │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 1: DATA EXTRACTION & QC                                       │
│  mat_to_csv_all_sessions.py                                          │
│                                                                      │
│  • scipy.io.loadmat → NumPy arrays                                   │
│  • Shape validation (T, C, P, E checks)                              │
│  • NaN/Inf detection                                                 │
│  • Time vector sanity check                                          │
│  • Label column frequency analysis                                   │
│                                                                      │
│  Outputs:                                                            │
│    reports/qc_mat_extract.csv                                        │
│    reports/labels_uniques_top30.xlsx                                 │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 2: DATASET ENGINEERING                                        │
│  prepare_dataset_v3.py                                               │
│                                                                      │
│  Per session:                                                        │
│  1. Load data tensor: (T=801, C=32, P=3, E)                          │
│  2. Transpose to event-first: (E, P, C, T)                          │
│  3. Flatten player axis: N = E × P samples → (N, C=32, T=801)       │
│  4. Assign role labels from label metadata                           │
│  5. Stratified event-wise split: 80% train / 20% test               │
│     (split on event IDs, not sample IDs — prevents leakage)         │
│                                                                      │
│  5 NPZ outputs:                                                      │
│    mixed_train.npz / mixed_test.npz      (all event types)          │
│    type0_train.npz / type0_test.npz      (2-class subset)           │
│    type1_train.npz / type1_test.npz      (3-class, type 1)          │
│    type2_train.npz / type2_test.npz      (3-class, type 2)          │
│    type3_train.npz / type3_test.npz      (3-class, type 3)          │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 3: MODEL TRAINING & EVALUATION                                │
│  train_eval_v3.py                                                    │
│                                                                      │
│  Models:                                                             │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  SimpleEEGCNN                                               │    │
│  │  (B,32,801) → Conv1d×3 → BatchNorm → ELU → Pool → Linear   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  PatchTransformer                                           │    │
│  │  (B,32,801) → PatchEmbed → TransformerEncoder → MeanPool    │    │
│  │            → Linear head                                    │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Training:                                                           │
│    Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)                     │
│    Loss: CrossEntropyLoss (optional class weights)                   │
│    Early stopping: patience=6 on Macro-F1                           │
│                                                                      │
│  Metrics per run:                                                    │
│    Accuracy, Balanced Accuracy, Macro-F1, Weighted-F1               │
│    Per-class precision/recall, Confusion matrix                      │
│                                                                      │
│  Outputs:                                                            │
│    data/processed/train_eval_v4_eventsplit_results.csv              │
│    checkpoints/best_model_{run_id}.pt                               │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Data Shape Transformations

Understanding the data tensor transformations is central to this pipeline.

### Raw MATLAB Structure
Each session file contains:
- `data`: shape `(T=801, C=32, P=3, E)` — time × channels × players × events
- `labels`: shape `(E, 8)` — 8 metadata columns per event
- `t`: shape `(T,)` — time axis in seconds

### Transformation to Training Samples

```
Step 1: Load session
  data: (801, 32, 3, E_session)
  labels: (E_session, 8)

Step 2: Identify event types and assign role labels
  For each event e in 0..E_session-1:
    For each player p in {0, 1, 2}:
      role_label = f(labels[e, :], p)   # 0=played, 1=current, 2=observer

Step 3: Reshape to sample-level
  X[n] = data[:, :, p, e].T  →  shape (C=32, T=801)
  y[n] = role_label
  Where n = e * 3 + p  (event-major ordering)

Step 4: Split on events (not samples)
  test_event_ids = stratified_sample(event_ids, test_size=0.20, stratify=event_type)
  X_train = X[sample is in train_event_ids]
  X_test  = X[sample is in test_event_ids]
```

This event-wise split means all three player samples from the same event land in the same split — preventing any label or temporal leakage.

---

## Neural Architecture Details

### SimpleEEGCNN

```python
class SimpleEEGCNN(nn.Module):
    def __init__(self, n_channels=32, n_times=801, n_classes=3):
        # Block 1: temporal feature extraction
        Conv1d(32, 64, kernel_size=7, padding=3)
        BatchNorm1d(64)
        ELU()
        MaxPool1d(2)           # T: 801 → 400

        # Block 2: higher-level features
        Conv1d(64, 128, kernel_size=5, padding=2)
        BatchNorm1d(128)
        ELU()
        MaxPool1d(2)           # T: 400 → 200

        # Block 3: abstract representation
        Conv1d(128, 256, kernel_size=3, padding=1)
        BatchNorm1d(256)
        ELU()
        AdaptiveAvgPool1d(1)   # T: 200 → 1

        # Classification head
        Linear(256, n_classes)
```

### PatchTransformer

```python
class PatchTransformer(nn.Module):
    def __init__(self, n_channels=32, n_times=801, patch_len=50,
                 d_model=128, nhead=4, num_layers=2, n_classes=3):
        # Patch creation
        n_patches = n_times // patch_len          # ~16 patches
        patch_dim = n_channels * patch_len        # 32 × 50 = 1600

        # Linear patch embedding
        Linear(patch_dim, d_model)                # 1600 → 128

        # Positional encoding (learned)
        Embedding(n_patches, d_model)

        # Transformer encoder
        TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256,
            dropout=0.1, batch_first=True
        ) × num_layers=2

        # Classification
        Mean pool over patches
        Linear(128, n_classes)
```

---

## Module Dependency Map

```
src/data_loading.py
    └─ used by: scripts/mat_to_csv_all_sessions.py
                scripts/build_dataset.py
                scripts/prepare_dataset_v3.py

scripts/config.py
    └─ used by: scripts/mat_to_csv_all_sessions.py
                scripts/build_dataset.py
                scripts/prepare_dataset_v3.py
                scripts/inspect_labels.py

scripts/prepare_dataset_v3.py
    └─ produces: data/processed/*.npz
    └─ consumed by: scripts/train_eval_v3.py

scripts/train_eval_v3.py
    └─ produces: results CSV + checkpoints
    └─ consumed by: plot_results.py
```

---

## Key Engineering Decisions

See [Technical Decisions](technical-decisions.md) for a full discussion of:
- Why event-wise splits over random splits
- Why PyTorch over TensorFlow
- Why Conv1D CNN as baseline
- Why Patch Transformer over standard sequence Transformer
- Why macro-F1 over accuracy as the primary metric

---

*Continue reading: [Features](features.md) | [Technical Decisions](technical-decisions.md)*
