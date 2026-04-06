# Features & Capabilities

## Core Capabilities

### 1. MATLAB Data Ingestion
- Loads raw `.mat` session files using `scipy.io.loadmat`
- Handles variable-length event arrays across sessions
- Validates tensor shapes: `(T=801, C=32, P=3, E)`
- Detects NaN and Inf values with per-session reporting
- Checks time vector consistency

### 2. Automated Quality Control
- Generates `qc_mat_extract.csv` with per-session status, shapes, and anomaly flags
- Produces `labels_uniques_top30.xlsx` — Excel workbook with value frequencies per label column
- Flags sessions with missing data, unexpected dimensions, or timing irregularities

### 3. Event-Wise Stratified Dataset Splitting
- Groups samples by event ID rather than splitting randomly
- Stratifies splits by event type to maintain class distribution in both train and test sets
- Prevents data leakage: samples from the same game event always stay in the same partition
- Configurable split ratio (default: 80% train / 20% test)

### 4. Multi-Variant Dataset Generation
Generates 5 distinct NPZ datasets from a single run:

| Dataset | Classes | Use Case |
|---------|---------|---------|
| `mixed` | 3 | General model trained on all event types |
| `type0` | 2 | Simpler 2-class problem for baseline |
| `type1` | 3 | Model specialized for event type 1 dynamics |
| `type2` | 3 | Model specialized for event type 2 dynamics |
| `type3` | 3 | Model specialized for event type 3 dynamics |

### 5. Conv1D CNN Classifier
- 3-block temporal convolutional architecture
- Progressive feature abstraction: 32 → 64 → 128 → 256 channels
- BatchNorm + ELU activations for training stability
- Adaptive pooling for fixed-dimension representation
- Configurable number of output classes (2 or 3)

### 6. Patch Transformer Classifier
- Divides EEG time series into fixed-size temporal patches
- Learns patch embeddings via linear projection
- Learned positional encodings per patch position
- Standard TransformerEncoder: 4 attention heads, 2 layers, 256-dim feedforward
- Mean pooling over patch tokens → classification head

### 7. Unified Training CLI
`train_eval_v3.py` exposes a full argparse interface:

```
--model       {cnn, transformer}        Architecture to train
--dataset     {mixed, type0, type1, type2, type3}  Dataset variant
--epochs      int                        Max training epochs (default 30)
--lr          float                      Learning rate (default 1e-3)
--batch-size  int                        Batch size (default 64)
--patience    int                        Early stopping patience (default 6)
--class-weights                          Enable class-balanced loss weights
--seed        int                        Random seed for reproducibility
--output-dir  path                       Where to save results and checkpoints
```

### 8. Comprehensive Evaluation Metrics
Each training run records:
- **Accuracy** — standard classification accuracy
- **Balanced Accuracy** — unweighted mean of per-class recall (handles class imbalance)
- **Macro-F1** — primary optimization target (treats all classes equally)
- **Weighted F1** — accounts for class support
- **Per-class Precision / Recall / F1**
- **Confusion Matrix** (saved per run)

### 9. Event Sequence Analysis Tools
- `detect_delayed_prompts.py` — Identifies events with atypical inter-event timing; outputs per-session delay flag statistics
- `inspect_table_full_sequences.py` — Extracts and analyzes the 80 events following "table full" game transitions
- `summarize_sequences.py` — Ranks the top-N most frequent event sequence patterns across all sessions
- `inspect_labels.py` — Shows label column value distributions from any session sample

### 10. Results Visualization
- `plot_results.py` generates a bar chart comparing CNN vs Transformer Macro-F1 across dataset variants
- All model checkpoints saved as `.pt` files for later inference or fine-tuning

---

## Reproducibility Features

| Feature | Implementation |
|---------|---------------|
| Random seed control | `--seed` argument propagated to NumPy, PyTorch, and scikit-learn |
| Deterministic training | `torch.manual_seed` + `torch.backends.cudnn.deterministic` |
| Pinned dependencies | `requirements.txt` with explicit version constraints |
| Metrics persistence | Results CSV accumulates across runs; each row includes run metadata |
| Checkpoint naming | Model files include run ID, dataset, and architecture for unambiguous identification |

---

## Limitations

- **No real-time inference layer**: pipeline is batch-only; serving requires additional engineering
- **Data not included**: raw `.mat` files and processed `.npz` datasets are excluded from the repository (gitignored)
- **No automated tests**: scripts lack unit/integration tests
- **Label semantics partially inferred**: some label column interpretations are heuristic (see `guess_label_mapping.py`)
- **Single-machine training only**: no distributed training support

---

*Continue reading: [Architecture](architecture.md) | [Setup Guide](setup.md)*
