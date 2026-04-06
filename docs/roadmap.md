# Roadmap

## Current Status: v1.0 (Research Prototype)

The core pipeline — data extraction, dataset engineering, CNN/Transformer training, and evaluation — is complete and functional. The project is a research-grade prototype suitable for EEG decoding experiments.

---

## Near-Term Improvements (v1.1)

### Code Quality
- [ ] **Resolve empty placeholder**: `scripts/metrics_v3.py` is currently empty — implement shared metric computation utilities used by all trainers
- [ ] **Add unit tests**: pytest-based tests for data loading, dataset building, and metric computation
- [ ] **Type annotations**: Add full type hints to all public functions
- [ ] **Linting CI**: GitHub Actions workflow running `flake8` + `black` on every push

### Data Pipeline
- [ ] **Per-channel Z-score normalization**: Normalize EEG per channel per session before training; likely to improve model performance
- [ ] **Artifact rejection**: Flag and optionally exclude epochs with extreme amplitudes (|signal| > threshold × std)
- [ ] **Configurable split ratio**: Expose train/test split ratio as a CLI argument in `prepare_dataset_v3.py`

### Training
- [ ] **EEGNet architecture**: Implement the Lawhern et al. (2018) depthwise-separable CNN as a third architecture option
- [ ] **Learning rate scheduling**: Add cosine annealing or reduce-on-plateau scheduler
- [ ] **Class weight auto-computation**: Calculate class frequencies from training data and automatically weight the loss

---

## Medium-Term Improvements (v1.2)

### Evaluation & Analysis
- [ ] **Leave-one-session-out cross-validation**: Evaluate model generalization across subjects rather than within-session splits
- [ ] **Ablation study**: Systematically vary number of CNN layers, transformer heads, and patch sizes; log all results
- [ ] **Frequency-domain features**: Add a preprocessing option to convert raw EEG to power spectral density (via FFT or wavelet) as an alternative input modality
- [ ] **Attention visualization**: For the Transformer model, visualize attention weights over time to understand which temporal segments drive classification

### Hyperparameter Optimization
- [ ] **Optuna integration**: Automated hyperparameter search over learning rate, batch size, architecture depth, patch size
- [ ] **Results dashboard**: Weights & Biases or MLflow integration for experiment tracking

---

## Long-Term Goals (v2.0)

### Deployment
- [ ] **Real-time inference server**: FastAPI WebSocket endpoint that accepts streaming EEG frames and returns role predictions with confidence scores
- [ ] **Docker containerization**: `Dockerfile` for reproducible, portable execution environment
- [ ] **Model export**: ONNX export for platform-independent inference

### Research Extensions
- [ ] **Transfer learning**: Pre-train on pooled sessions, fine-tune on held-out subject
- [ ] **Multi-modal fusion**: Combine EEG features with game state metadata (e.g., card counts, turn number) for richer predictions
- [ ] **Temporal modeling**: Replace static event windows with rolling window inference for continuous monitoring
- [ ] **Cross-dataset validation**: Test model on publicly available EEG datasets (e.g., BCICIV, SEED)

### Documentation
- [ ] **Jupyter walkthrough notebook**: End-to-end notebook from raw data to trained model, suitable for teaching
- [ ] **Results screenshots**: Add actual confusion matrices and metric plots to the GitHub Pages portfolio
- [ ] **Video demo**: Record screen demo of the full pipeline execution

---

## Contributing

This project is a personal research portfolio. If you find a bug or want to suggest an improvement, please open an issue on GitHub.

---

*See also: [Technical Decisions](technical-decisions.md) | [Results & Impact](results-and-impact.md)*
