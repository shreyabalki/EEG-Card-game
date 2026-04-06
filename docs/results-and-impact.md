# Results and Impact

## Evaluation Methodology

All models are evaluated on a **held-out test split** constructed via event-wise stratified sampling — 20% of events per session per event type are withheld from training. This prevents temporal leakage and ensures metrics reflect true generalization to unseen game moments.

Primary metric: **Macro-F1** (equal weight per class, robust to class imbalance)  
Secondary metrics: Balanced Accuracy, Weighted F1, per-class Precision/Recall

---

## Results Table

> **Note**: The values below are placeholders. Run `scripts/train_eval_v3.py` and replace with actual results from `data/processed/train_eval_v4_eventsplit_results.csv`.

| Model | Dataset | Accuracy | Balanced Acc | Macro-F1 | Weighted F1 |
|-------|---------|---------|--------------|---------|------------|
| CNN | mixed | [replace]% | [replace]% | [replace] | [replace] |
| Transformer | mixed | [replace]% | [replace]% | [replace] | [replace] |
| CNN | type0 | [replace]% | [replace]% | [replace] | [replace] |
| CNN | type1 | [replace]% | [replace]% | [replace] | [replace] |
| CNN | type2 | [replace]% | [replace]% | [replace] | [replace] |
| CNN | type3 | [replace]% | [replace]% | [replace] | [replace] |
| Transformer | type1 | [replace]% | [replace]% | [replace] | [replace] |
| Transformer | type2 | [replace]% | [replace]% | [replace] | [replace] |
| Transformer | type3 | [replace]% | [replace]% | [replace] | [replace] |

**Chance level**: 33.3% accuracy for 3-class problems, 50% for 2-class.

---

## Metrics to Fill In After Training

When you run training, collect and record these values:

### CNN on Mixed Dataset
```bash
python scripts/train_eval_v3.py --model cnn --dataset mixed --epochs 30 --seed 42
```
Record from output:
- Best epoch (early stopping)
- Test Accuracy: ____%
- Test Balanced Accuracy: ____%
- Test Macro-F1: ____
- Test Weighted F1: ____
- Confusion matrix (copy from run log)

### Transformer on Mixed Dataset
```bash
python scripts/train_eval_v3.py --model transformer --dataset mixed --epochs 30 --seed 42
```
Record same metrics as above.

---

## QC Report Summary

The pipeline generates automated QC across all 22 sessions. Key statistics:
- **Sessions processed**: 22 of 22 (verify from `reports/qc_mat_extract.csv`)
- **Sessions with NaN**: [replace — check qc report]
- **Sessions with timing irregularities**: [replace — check qc report]
- **Delayed prompt events flagged**: [replace — check `reports/delay_flag_summary.csv`]

---

## Dataset Statistics

After running `prepare_dataset_v3.py`:

| Dataset | Total Samples | Train Samples | Test Samples | Class Distribution |
|---------|--------------|--------------|-------------|-------------------|
| mixed | [replace] | [replace] | [replace] | [replace] |
| type0 | [replace] | [replace] | [replace] | [replace] |
| type1 | [replace] | [replace] | [replace] | [replace] |
| type2 | [replace] | [replace] | [replace] | [replace] |
| type3 | [replace] | [replace] | [replace] | [replace] |

Estimate: 22 sessions × ~E events/session × 3 players per event → total sample count.

---

## Engineering Impact

| Metric | Value |
|--------|-------|
| Manual MATLAB inspection hours eliminated | [replace — estimate hours saved by automated QC] |
| Sessions automatically QC'd | 22 |
| Dataset variants generated per pipeline run | 5 |
| Training configurations comparable per run | All tracked in results CSV |
| Reproducibility | Fully reproducible via seed + pinned dependencies |

---

## Research Significance

**Baseline performance context**:
- Chance level (3-class): 33.3%
- A model achieving >50% Macro-F1 demonstrates meaningful EEG decoding
- A model achieving >60% Macro-F1 would be a strong result for passive role classification from single-trial EEG

**Why this matters**:
1. Single-trial decoding (no averaging across trials) is harder than averaged EEG analysis
2. The task is passive — players are not explicitly performing a BCI task
3. Natural behavioral context (card game) rather than controlled laboratory paradigm
4. Multi-player setup is rare in EEG literature

---

## What to Do With Your Results

### If CNN > Transformer:
- Local temporal features (captured by convolutions) are more discriminative than global patterns
- Consider trying a larger CNN or adding temporal attention on top of CNN features

### If Transformer > CNN:
- Long-range temporal dependencies in the EEG signal matter for role classification
- Consider increasing number of transformer layers or attention heads

### If Both Are Near Chance:
- Check QC report for data quality issues
- Verify label assignment logic in `prepare_dataset_v3.py`
- Consider per-session normalization (z-score per channel)
- Check that test events are genuinely held out (no leakage)

---

*See also: [Architecture](architecture.md) | [Roadmap](roadmap.md)*
