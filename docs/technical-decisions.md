# Technical Decisions

This document explains the key engineering and design decisions made throughout the project, with reasoning.

---

## 1. Event-Wise Splits Over Random Splits

**Decision**: Split train/test on event IDs, not individual samples.

**Why it matters**: Each card game event generates 3 samples (one per player). If we split randomly at the sample level, samples from the same event will appear in both train and test sets. Since these samples share the same game context (same moment in time, same cards on the table, same EEG recording window), this creates **temporal leakage** — the model can effectively memorize training events and get artificially high test performance.

**Our approach**: Identify unique event IDs per session and event type, then hold out 20% of entire events. All 3 player samples from a held-out event are moved to the test set together.

**Trade-off**: Slightly less training data per split, but the evaluation reflects true generalization.

---

## 2. PyTorch Over TensorFlow

**Decision**: PyTorch is the primary deep learning framework for model definition and training.

**Why**: PyTorch's imperative (eager) execution model makes debugging neural architectures significantly easier — `print` statements, breakpoints, and arbitrary Python logic inside the training loop work naturally. For research-oriented EEG work where architecture experimentation is central, this flexibility outweighs TensorFlow's deployment ecosystem advantages.

Note: `requirements.txt` includes `tensorflow==2.12` for environment verification (`check_tf.py`), but the core models are PyTorch-only.

---

## 3. Conv1D CNN as Baseline Architecture

**Decision**: Use a simple 3-block Conv1D CNN rather than a 2D CNN or EEGNet.

**Why Conv1D**: EEG data has shape `(channels, time)`. While 2D convolutions could process this as a 2D image, the temporal dimension has clear directional semantics (time progresses left to right), while the channel dimension does not have the same spatial structure as pixels. Conv1D operates along the temporal axis, which better matches EEG signal structure.

**Why not EEGNet immediately**: EEGNet (Lawhern et al., 2018) uses depthwise-separable convolutions optimized for EEG, but has more hyperparameters to tune. Starting with a simple Conv1D baseline establishes a lower bound on performance and provides a faster debugging cycle. EEGNet is listed in the roadmap.

**Why BatchNorm + ELU**: BatchNorm stabilizes training for batch-normalized EEG distributions. ELU (Exponential Linear Unit) avoids the "dying ReLU" problem while maintaining negative outputs, which helps with gradient flow on irregular EEG signals.

---

## 4. Patch Transformer Architecture

**Decision**: Divide the EEG time series into fixed patches and apply a Transformer encoder, rather than using a sequence Transformer on individual time steps.

**Why patches**: Applying attention over 801 time steps would be computationally expensive (O(T²) attention complexity) and might not learn meaningful long-range dependencies for 1-second EEG windows. Grouping into ~16 patches (801 / ~50 steps per patch) reduces sequence length, makes attention computationally tractable, and encourages the model to learn coarser temporal structure.

**Why this over vanilla LSTM/GRU**: Transformers generalize better to variable-length inputs and are more parallelizable. For cross-architecture comparison, CNN vs Transformer is a more informative contrast than CNN vs RNN.

---

## 5. Macro-F1 as Primary Early Stopping Metric

**Decision**: Stop training based on validation Macro-F1, not accuracy.

**Why**: Class imbalance is likely in multi-player EEG data — not all roles appear equally often. Accuracy can be misleadingly high when the model simply predicts the majority class. Macro-F1 averages F1 equally across all classes regardless of support, providing a more honest estimate of whether the model has learned all three roles.

**Balanced Accuracy** is also tracked as a complementary metric — it is the unweighted mean of per-class recall, which has a similar imbalance-robustness property.

---

## 6. AdamW Over Adam

**Decision**: Use AdamW (Adam with decoupled weight decay) rather than standard Adam.

**Why**: Standard Adam applies L2 regularization within the gradient update, which interacts with the adaptive learning rates in suboptimal ways. AdamW decouples weight decay from the gradient update, resulting in more stable regularization and generally better generalization. For EEG models prone to overfitting on small neurophysiology datasets, proper regularization matters.

---

## 7. Separate Dataset Variants (5 NPZ files)

**Decision**: Generate 5 separate NPZ files (mixed + per-event-type) rather than one file with an event type filter.

**Why**: Pre-generating datasets makes training runs faster (no filtering at load time) and more reproducible (the exact train/test split is frozen on disk). Separate files also make it easy to run multiple training jobs in parallel, each reading a different dataset variant.

**Trade-off**: Disk storage is multiplied by 5. Given that raw `.mat` files are ~several GB and processed `.npz` files are a fraction of that, this is acceptable.

---

## 8. Argparse CLI for All Scripts

**Decision**: Expose all configurable parameters through argparse rather than hardcoded constants or config files.

**Why**: CLI parameterization makes it straightforward to run training sweeps (loop over arguments in a shell script), log exactly what configuration was used, and reproduce any specific run. It is also more portable than config file formats like YAML or TOML for small-to-medium projects.

---

## 9. `.gitignore` for Data and Models

**Decision**: Exclude all data files (`.npy`, `.npz`, `.mat`) and model checkpoints (`.pt`, `.h5`) from version control.

**Why**: 
- Data files are large (MATLAB session files can be hundreds of MB each)
- Model checkpoints are reproducible artifacts — they can be regenerated by running the training script
- Including large binaries in git history causes repository bloat that is hard to undo
- Some data may contain participant information that should not be in a public repository

**Trade-off**: Reproducibility requires the original raw data files. The README documents exactly where to place them.

---

*See also: [Architecture](architecture.md) | [Results & Impact](results-and-impact.md)*
