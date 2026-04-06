# Project Overview

## What Is This Project?

**EEG Card Game** is an end-to-end machine learning pipeline that decodes player roles in a multiplayer card game from raw EEG (electroencephalography) brain signals.

Three players participate in a card game while wearing EEG headsets recording 32 channels of neural activity. At each game event, the system captures a 801-sample brainwave window per player and asks: **which role is this player in right now?**

| Role | Label | Cognitive State |
|------|-------|----------------|
| Played | 0 | Player who just acted — post-decision, reduced active attention |
| Current / Next | 1 | Player whose turn is active or imminent — heightened anticipation |
| Observer | 2 | Passive player — monitoring but not engaged |

These subtle cognitive differences produce detectable EEG signatures. Two deep learning architectures — a **Conv1D CNN** and a **Patch Transformer** — are trained to recognize these patterns.

---

## Motivation

This project sits at the intersection of **neuroscience, signal processing, and deep learning**. The core research question — *can a neural network decode passive cognitive roles from EEG?* — is directly relevant to:

- **Brain-Computer Interfaces (BCI)**: enabling devices that respond to mental state without explicit input
- **Adaptive systems**: game environments or productivity tools that adjust to user cognitive load
- **Neuroergonomics**: monitoring attention and fatigue in operators, drivers, or medical personnel
- **Affective computing**: systems that understand human state to improve interaction quality

---

## What Was Built

1. **Data ingestion layer** — Parses 22 raw MATLAB session files into structured NumPy tensors with full QC reporting
2. **Dataset engineering layer** — Transforms 4D tensors (time × channels × players × events) into flat (samples × channels × time) training arrays with event-wise stratified splits
3. **Model training layer** — Unified trainer for CNN and Transformer architectures with reproducible configuration, metrics logging, and early stopping
4. **Analysis utilities** — Scripts for label inspection, event sequence analysis, and timing anomaly detection
5. **Portfolio presentation** — GitHub Pages site documenting the project as a professional case study

---

## Project Scale

| Metric | Value |
|--------|-------|
| EEG sessions processed | 22 |
| EEG channels per player | 32 |
| Time samples per event | 801 |
| Players per session | 3 |
| Dataset variants generated | 5 |
| Python scripts authored | 14 |
| Neural architectures implemented | 2 (CNN, Transformer) |
| Total lines of source code | ~4,000+ |

---

## Technologies at a Glance

- **PyTorch** — Model definition, training loop, optimization
- **NumPy / SciPy** — Array operations, MATLAB file loading
- **Pandas** — Tabular data, QC reports
- **scikit-learn** — Evaluation metrics (balanced accuracy, macro-F1, confusion matrix)
- **MNE-Python** — EEG processing utilities
- **Matplotlib / Seaborn** — Visualization

---

*Continue reading: [Architecture](architecture.md) | [Features](features.md) | [Setup Guide](setup.md)*
