from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
REPORT_DIR = Path("reports")

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MAT_RE = re.compile(r"sessionevents(\d{2})\.mat$", re.IGNORECASE)


def load_session(mat_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Returns:
      trials_df: 249 x 8 labels table (plus session + trial index)
      meta: shapes + basic info for QC
    """
    m = MAT_RE.search(mat_path.name)
    if not m:
        raise ValueError(f"Not a sessioneventsXX.mat file: {mat_path.name}")
    session = int(m.group(1))

    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    data = mat["data"]      # (801, 32, 3, Ntrials)
    labels = mat["labels"]  # (Ntrials, 8)
    t = mat["t"]            # (801,)

    # Basic sanity checks
    if data.ndim != 4:
        raise ValueError(f"{mat_path.name}: data expected 4D, got shape {getattr(data,'shape',None)}")
    if labels.ndim != 2:
        raise ValueError(f"{mat_path.name}: labels expected 2D, got shape {getattr(labels,'shape',None)}")
    if t.ndim != 1:
        raise ValueError(f"{mat_path.name}: t expected 1D, got shape {getattr(t,'shape',None)}")

    n_time, n_chan, n_third, n_trials = data.shape
    if labels.shape[0] != n_trials:
        raise ValueError(
            f"{mat_path.name}: labels rows ({labels.shape[0]}) != trials in data ({n_trials})"
        )

    # Convert labels to a clean dataframe
    # labels may be numeric or strings; force object then clean
    labels_arr = np.array(labels, dtype=object)

    colnames = [f"label_{i+1}" for i in range(labels_arr.shape[1])]
    df = pd.DataFrame(labels_arr, columns=colnames)
    df.insert(0, "trial", np.arange(1, n_trials + 1))
    df.insert(0, "session", session)

    # Add derived meta
    df["n_time"] = n_time
    df["n_chan"] = n_chan
    df["n_third_dim"] = n_third

    meta = {
        "session": session,
        "file": str(mat_path),
        "data_shape": tuple(data.shape),
        "labels_shape": tuple(labels.shape),
        "t_shape": tuple(t.shape),
        "t_min": float(np.min(t)),
        "t_max": float(np.max(t)),
        "t_monotonic": bool(np.all(np.diff(t) > 0)),
        "n_trials": int(n_trials),
        "n_time": int(n_time),
        "n_chan": int(n_chan),
        "n_third_dim": int(n_third),
        "labels_nan_count": int(pd.isna(df).sum().sum()),
    }

    return df, meta


def main() -> None:
    mat_files = sorted(RAW_DIR.glob("sessionevents*.mat"))
    mat_files = [p for p in mat_files if MAT_RE.search(p.name)]

    if not mat_files:
        raise SystemExit(f"No sessionevents*.mat found in {RAW_DIR.resolve()}")

    all_trials = []
    qc_rows = []

    for p in mat_files:
        try:
            trials_df, meta = load_session(p)
            all_trials.append(trials_df)
            qc_rows.append({"status": "ok", **meta})

            out_csv = OUT_DIR / f"session{meta['session']:02d}_labels.csv"
            trials_df.to_csv(out_csv, index=False)

        except Exception as e:
            # Keep going even if one session is broken/incomplete (e.g., session 22)
            m = MAT_RE.search(p.name)
            session = int(m.group(1)) if m else None
            qc_rows.append({
                "status": "error",
                "session": session,
                "file": str(p),
                "error": str(e),
            })

    qc = pd.DataFrame(qc_rows).sort_values(["status", "session"])
    qc.to_csv(REPORT_DIR / "qc_mat_extract.csv", index=False)

    if all_trials:
        combined = pd.concat(all_trials, ignore_index=True)
        combined.to_csv(OUT_DIR / "all_sessions_labels.csv", index=False)

        # Quick “uniques” report per label column to help decode what each label means
        uniq_report = {}
        for c in [c for c in combined.columns if c.startswith("label_")]:
            # show top values (stringified)
            vc = combined[c].astype(str).value_counts().head(30)
            uniq_report[c] = vc

        with pd.ExcelWriter(REPORT_DIR / "labels_uniques_top30.xlsx") as xw:
            for c, vc in uniq_report.items():
                vc.to_frame("count").to_excel(xw, sheet_name=c[:31])

    print("Done.")
    print(f"- Per-session label CSVs: {OUT_DIR}")
    print(f"- Combined labels: {OUT_DIR / 'all_sessions_labels.csv'}")
    print(f"- QC report: {REPORT_DIR / 'qc_mat_extract.csv'}")
    print(f"- Label value report: {REPORT_DIR / 'labels_uniques_top30.xlsx'}")


if __name__ == "__main__":
    main()
