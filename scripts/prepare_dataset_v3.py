<<<<<<< HEAD
from pathlib import Path
from collections import Counter
import numpy as np
import scipy.io as sio

# ============================================================
# CONFIG
# ============================================================
# Verified from your earlier label inspection:
# col 1 -> player ids
# col 3 -> player ids
# col 2 -> event type (0,1,2,3)
LAST_PLAYER_COL = 1
CURRENT_PLAYER_COL = 3
EVENT_TYPE_COL = 2

ALL_SESSIONS = list(range(1, 23))   # sessions 01..22
TEST_RATIO = 0.20                   # 20% held-out events per session
RANDOM_SEED = 42
SKIP_BROKEN_SESSIONS = True

# Output names (new event-wise split version)
OUT_MIXED = "v4_eventsplit_mixed.npz"
OUT_TYPE0_2C = "v4_eventsplit_type0_2class.npz"
OUT_TYPE1 = "v4_eventsplit_type1.npz"
OUT_TYPE2 = "v4_eventsplit_type2.npz"
OUT_TYPE3 = "v4_eventsplit_type3.npz"


# ============================================================
# PATH SETUP
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_CANDIDATES = [
    PROJECT_ROOT / "data" / "raw",   # preferred
    PROJECT_ROOT / "data",           # fallback
]

raw_dir = None
raw_files = []
for cand in RAW_CANDIDATES:
    raw_files = sorted(cand.glob("sessionevents*.mat"))
    if raw_files:
        raw_dir = cand
        break

if raw_dir is None:
    raise FileNotFoundError(
        f"No sessionevents*.mat found in: {[str(p) for p in RAW_CANDIDATES]}"
    )

OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[prepare_dataset_v3] RAW_DIR = {raw_dir}")
print(f"[prepare_dataset_v3] found {len(raw_files)} sessionevents*.mat")
print("first files:", [f.name for f in raw_files[:5]])
print(f"[prepare_dataset_v3] OUT_DIR = {OUT_DIR}")
print(f"[CONFIG] CURRENT={CURRENT_PLAYER_COL} LAST={LAST_PLAYER_COL} EVENT_TYPE={EVENT_TYPE_COL}")
print(f"[CONFIG] EVENT-WISE SPLIT across ALL sessions, TEST_RATIO={TEST_RATIO}, SEED={RANDOM_SEED}")


# ============================================================
# HELPERS
# ============================================================
def _safe_int(x) -> int:
    return int(np.asarray(x).squeeze())


def _debug_label_columns(labels: np.ndarray) -> None:
    print("[debug] labels shape:", labels.shape)
    print("[debug] label columns uniques (preview):")
    for i in range(labels.shape[1]):
        col = labels[:, i]
        if col.dtype.kind == "f":
            col = col[~np.isnan(col)]
        u = np.unique(col)
        preview = u[:12]
        suffix = " ..." if len(u) > 12 else ""
        print(f"  col {i}: {preview}{suffix}")


def _load_session(mat_path: Path):
    m = sio.loadmat(mat_path, squeeze_me=False)
    if "data" not in m or "labels" not in m:
        raise KeyError(f"{mat_path.name}: missing 'data' or 'labels' keys. keys={list(m.keys())}")
    return m["data"], m["labels"]


def _split_event_indices_stratified_by_event_type(labels: np.ndarray, seed: int):
    """
    Event-wise split inside one session.
    Stratifies by event type so each session preserves event-type proportions
    in train and test as much as possible.

    Returns:
        train_idx, test_idx (sorted numpy arrays of event indices)
    """
    rng = np.random.RandomState(seed)
    event_types = labels[:, EVENT_TYPE_COL].astype(int).reshape(-1)

    train_parts = []
    test_parts = []

    for et in np.unique(event_types):
        idx = np.where(event_types == et)[0]
        idx = idx.copy()
        rng.shuffle(idx)

        # ensure at least 1 test sample when possible
        n_test = int(np.round(len(idx) * TEST_RATIO))
        if len(idx) >= 2:
            n_test = max(1, min(len(idx) - 1, n_test))
        else:
            n_test = 0

        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

        train_parts.append(train_idx)
        test_parts.append(test_idx)

    train_idx = np.sort(np.concatenate(train_parts)) if train_parts else np.array([], dtype=int)
    test_idx = np.sort(np.concatenate(test_parts)) if test_parts else np.array([], dtype=int)

    return train_idx, test_idx


def _append_event_samples(data, labels, event_indices, sid, X_list, y_list, et_list, sid_list):
    """
    Append one sample per player for each selected event index.

    data shape: (T, C, P, E)
    labels shape: (E, L)
    """
    T, C, P, E = data.shape

    for e in event_indices:
        current = _safe_int(labels[e, CURRENT_PLAYER_COL])
        last = _safe_int(labels[e, LAST_PLAYER_COL])
        evt = _safe_int(labels[e, EVENT_TYPE_COL])

        for p in range(1, P + 1):
            eeg = data[:, :, p - 1, e].T  # (C, T)
            eeg = np.nan_to_num(eeg, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            if p == last:
                lbl = 0  # played
            elif p == current:
                lbl = 1  # next/current
            else:
                lbl = 2  # observer

            X_list.append(eeg)
            y_list.append(lbl)
            et_list.append(evt)
            sid_list.append(sid)


def _build_eventwise_samples(session_ids):
    """
    Build train/test by using ALL sessions, but splitting event indices inside each session.

    Returns:
      Xtr, ytr, ettr, sidtr
      Xte, yte, ette, sidte
    """
    Xtr_list, ytr_list, ettr_list, sidtr_list = [], [], [], []
    Xte_list, yte_list, ette_list, sidte_list = [], [], [], []

    for sid in session_ids:
        mat_path = raw_dir / f"sessionevents{sid:02d}.mat"
        if not mat_path.exists():
            msg = f"[prepare_dataset_v3] missing {mat_path}"
            if SKIP_BROKEN_SESSIONS:
                print(msg + " (skipping)")
                continue
            raise FileNotFoundError(msg)

        try:
            data, labels = _load_session(mat_path)
        except Exception as e:
            msg = f"[prepare_dataset_v3] failed loading {mat_path.name}: {e}"
            if SKIP_BROKEN_SESSIONS:
                print(msg + " (skipping)")
                continue
            raise

        if data.ndim != 4:
            raise ValueError(f"{mat_path.name}: expected data.ndim==4, got {data.shape}")

        T, C, P, E = data.shape
        if labels.shape[0] != E:
            raise ValueError(f"{mat_path.name}: labels.shape[0]={labels.shape[0]} != E={E}")

        if sid == 1:
            print("[debug] session 01 data shape:", data.shape)
            _debug_label_columns(labels)

        # per-session event split; vary seed by session so splits are reproducible but not identical
        train_idx, test_idx = _split_event_indices_stratified_by_event_type(labels, seed=RANDOM_SEED + sid)

        print(
            f"[prepare_dataset_v3] session {sid:02d}: "
            f"train_events={len(train_idx)}, test_events={len(test_idx)}, total={E}"
        )

        _append_event_samples(
            data, labels, train_idx, sid,
            Xtr_list, ytr_list, ettr_list, sidtr_list
        )
        _append_event_samples(
            data, labels, test_idx, sid,
            Xte_list, yte_list, ette_list, sidte_list
        )

    if not Xtr_list or not Xte_list:
        raise RuntimeError("No train/test samples created. Check label columns and split logic.")

    Xtr = np.stack(Xtr_list, axis=0).astype(np.float32, copy=False)
    ytr = np.asarray(ytr_list, dtype=np.int64)
    ettr = np.asarray(ettr_list, dtype=np.int64)
    sidtr = np.asarray(sidtr_list, dtype=np.int64)

    Xte = np.stack(Xte_list, axis=0).astype(np.float32, copy=False)
    yte = np.asarray(yte_list, dtype=np.int64)
    ette = np.asarray(ette_list, dtype=np.int64)
    sidte = np.asarray(sidte_list, dtype=np.int64)

    return Xtr, ytr, ettr, sidtr, Xte, yte, ette, sidte


def _save_npz(filename: str, **arrays) -> None:
    path = OUT_DIR / filename
    np.savez(path, **arrays)
    print(f"[prepare_dataset_v3] saved: {path}")


def _summarize(name: str, y: np.ndarray) -> None:
    print(f"[summary] {name} counts:", dict(Counter(y.tolist())))


def _event_type_distribution(et: np.ndarray, label: str) -> None:
    u, c = np.unique(et, return_counts=True)
    pairs = list(zip(u.tolist(), c.tolist()))
    print(f"[debug] event_type {label} uniques+counts: {pairs}")


def _filter_event_type(X, y, et, sid, et_value: int):
    m = (et == et_value)
    return X[m], y[m], et[m], sid[m]


def _type0_to_2class(X, y):
    """
    Type0 is treated as 2-class:
      drop class0 (played)
      remap class1 -> 0, class2 -> 1
    """
    keep = (y != 0)
    X2 = X[keep]
    y2 = y[keep]
    y2 = np.where(y2 == 1, 0, 1).astype(np.int64)
    return X2, y2


# ============================================================
# MAIN
# ============================================================
def main():
    Xtr, ytr, ettr, sidtr, Xte, yte, ette, sidte = _build_eventwise_samples(ALL_SESSIONS)

    print()
    _summarize("TRAIN mixed", ytr)
    _summarize("TEST mixed", yte)
    _event_type_distribution(ettr, "train")
    _event_type_distribution(ette, "test")
    print()

    # Save mixed
    _save_npz(
        OUT_MIXED,
        X_train=Xtr, y_train=ytr, event_type_train=ettr, session_id_train=sidtr,
        X_test=Xte, y_test=yte, event_type_test=ette, session_id_test=sidte,
    )

    # Type0 -> 2-class
    X0tr, y0tr, et0tr, sid0tr = _filter_event_type(Xtr, ytr, ettr, sidtr, 0)
    X0te, y0te, et0te, sid0te = _filter_event_type(Xte, yte, ette, sidte, 0)
    X0tr2, y0tr2 = _type0_to_2class(X0tr, y0tr)
    X0te2, y0te2 = _type0_to_2class(X0te, y0te)

    _summarize("TRAIN type0 (2-class)", y0tr2)
    _summarize("TEST type0 (2-class)", y0te2)

    _save_npz(
        OUT_TYPE0_2C,
        X_train=X0tr2, y_train=y0tr2,
        X_test=X0te2, y_test=y0te2,
    )

    # Type1
    X1tr, y1tr, et1tr, sid1tr = _filter_event_type(Xtr, ytr, ettr, sidtr, 1)
    X1te, y1te, et1te, sid1te = _filter_event_type(Xte, yte, ette, sidte, 1)
    _summarize("TRAIN type1", y1tr)
    _summarize("TEST type1", y1te)
    _save_npz(OUT_TYPE1, X_train=X1tr, y_train=y1tr, X_test=X1te, y_test=y1te)

    # Type2
    X2tr, y2tr, et2tr, sid2tr = _filter_event_type(Xtr, ytr, ettr, sidtr, 2)
    X2te, y2te, et2te, sid2te = _filter_event_type(Xte, yte, ette, sidte, 2)
    _summarize("TRAIN type2", y2tr)
    _summarize("TEST type2", y2te)
    _save_npz(OUT_TYPE2, X_train=X2tr, y_train=y2tr, X_test=X2te, y_test=y2te)

    # Type3
    X3tr, y3tr, et3tr, sid3tr = _filter_event_type(Xtr, ytr, ettr, sidtr, 3)
    X3te, y3te, et3te, sid3te = _filter_event_type(Xte, yte, ette, sidte, 3)
    _summarize("TRAIN type3", y3tr)
    _summarize("TEST type3", y3te)
    _save_npz(OUT_TYPE3, X_train=X3tr, y_train=y3tr, X_test=X3te, y_test=y3te)

    print()
    print("[prepare_dataset_v3] DONE.")
    print("[prepare_dataset_v3] New files created:")
    for name in [OUT_MIXED, OUT_TYPE0_2C, OUT_TYPE1, OUT_TYPE2, OUT_TYPE3]:
        print("  -", OUT_DIR / name)


if __name__ == "__main__":
=======
from pathlib import Path
from collections import Counter
import numpy as np
import scipy.io as sio

# ============================================================
# CONFIG
# ============================================================
# Verified from your earlier label inspection:
# col 1 -> player ids
# col 3 -> player ids
# col 2 -> event type (0,1,2,3)
LAST_PLAYER_COL = 1
CURRENT_PLAYER_COL = 3
EVENT_TYPE_COL = 2

ALL_SESSIONS = list(range(1, 23))   # sessions 01..22
TEST_RATIO = 0.20                   # 20% held-out events per session
RANDOM_SEED = 42
SKIP_BROKEN_SESSIONS = True

# Output names (new event-wise split version)
OUT_MIXED = "v4_eventsplit_mixed.npz"
OUT_TYPE0_2C = "v4_eventsplit_type0_2class.npz"
OUT_TYPE1 = "v4_eventsplit_type1.npz"
OUT_TYPE2 = "v4_eventsplit_type2.npz"
OUT_TYPE3 = "v4_eventsplit_type3.npz"


# ============================================================
# PATH SETUP
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_CANDIDATES = [
    PROJECT_ROOT / "data" / "raw",   # preferred
    PROJECT_ROOT / "data",           # fallback
]

raw_dir = None
raw_files = []
for cand in RAW_CANDIDATES:
    raw_files = sorted(cand.glob("sessionevents*.mat"))
    if raw_files:
        raw_dir = cand
        break

if raw_dir is None:
    raise FileNotFoundError(
        f"No sessionevents*.mat found in: {[str(p) for p in RAW_CANDIDATES]}"
    )

OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[prepare_dataset_v3] RAW_DIR = {raw_dir}")
print(f"[prepare_dataset_v3] found {len(raw_files)} sessionevents*.mat")
print("first files:", [f.name for f in raw_files[:5]])
print(f"[prepare_dataset_v3] OUT_DIR = {OUT_DIR}")
print(f"[CONFIG] CURRENT={CURRENT_PLAYER_COL} LAST={LAST_PLAYER_COL} EVENT_TYPE={EVENT_TYPE_COL}")
print(f"[CONFIG] EVENT-WISE SPLIT across ALL sessions, TEST_RATIO={TEST_RATIO}, SEED={RANDOM_SEED}")


# ============================================================
# HELPERS
# ============================================================
def _safe_int(x) -> int:
    return int(np.asarray(x).squeeze())


def _debug_label_columns(labels: np.ndarray) -> None:
    print("[debug] labels shape:", labels.shape)
    print("[debug] label columns uniques (preview):")
    for i in range(labels.shape[1]):
        col = labels[:, i]
        if col.dtype.kind == "f":
            col = col[~np.isnan(col)]
        u = np.unique(col)
        preview = u[:12]
        suffix = " ..." if len(u) > 12 else ""
        print(f"  col {i}: {preview}{suffix}")


def _load_session(mat_path: Path):
    m = sio.loadmat(mat_path, squeeze_me=False)
    if "data" not in m or "labels" not in m:
        raise KeyError(f"{mat_path.name}: missing 'data' or 'labels' keys. keys={list(m.keys())}")
    return m["data"], m["labels"]


def _split_event_indices_stratified_by_event_type(labels: np.ndarray, seed: int):
    """
    Event-wise split inside one session.
    Stratifies by event type so each session preserves event-type proportions
    in train and test as much as possible.

    Returns:
        train_idx, test_idx (sorted numpy arrays of event indices)
    """
    rng = np.random.RandomState(seed)
    event_types = labels[:, EVENT_TYPE_COL].astype(int).reshape(-1)

    train_parts = []
    test_parts = []

    for et in np.unique(event_types):
        idx = np.where(event_types == et)[0]
        idx = idx.copy()
        rng.shuffle(idx)

        # ensure at least 1 test sample when possible
        n_test = int(np.round(len(idx) * TEST_RATIO))
        if len(idx) >= 2:
            n_test = max(1, min(len(idx) - 1, n_test))
        else:
            n_test = 0

        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

        train_parts.append(train_idx)
        test_parts.append(test_idx)

    train_idx = np.sort(np.concatenate(train_parts)) if train_parts else np.array([], dtype=int)
    test_idx = np.sort(np.concatenate(test_parts)) if test_parts else np.array([], dtype=int)

    return train_idx, test_idx


def _append_event_samples(data, labels, event_indices, sid, X_list, y_list, et_list, sid_list):
    """
    Append one sample per player for each selected event index.

    data shape: (T, C, P, E)
    labels shape: (E, L)
    """
    T, C, P, E = data.shape

    for e in event_indices:
        current = _safe_int(labels[e, CURRENT_PLAYER_COL])
        last = _safe_int(labels[e, LAST_PLAYER_COL])
        evt = _safe_int(labels[e, EVENT_TYPE_COL])

        for p in range(1, P + 1):
            eeg = data[:, :, p - 1, e].T  # (C, T)
            eeg = np.nan_to_num(eeg, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            if p == last:
                lbl = 0  # played
            elif p == current:
                lbl = 1  # next/current
            else:
                lbl = 2  # observer

            X_list.append(eeg)
            y_list.append(lbl)
            et_list.append(evt)
            sid_list.append(sid)


def _build_eventwise_samples(session_ids):
    """
    Build train/test by using ALL sessions, but splitting event indices inside each session.

    Returns:
      Xtr, ytr, ettr, sidtr
      Xte, yte, ette, sidte
    """
    Xtr_list, ytr_list, ettr_list, sidtr_list = [], [], [], []
    Xte_list, yte_list, ette_list, sidte_list = [], [], [], []

    for sid in session_ids:
        mat_path = raw_dir / f"sessionevents{sid:02d}.mat"
        if not mat_path.exists():
            msg = f"[prepare_dataset_v3] missing {mat_path}"
            if SKIP_BROKEN_SESSIONS:
                print(msg + " (skipping)")
                continue
            raise FileNotFoundError(msg)

        try:
            data, labels = _load_session(mat_path)
        except Exception as e:
            msg = f"[prepare_dataset_v3] failed loading {mat_path.name}: {e}"
            if SKIP_BROKEN_SESSIONS:
                print(msg + " (skipping)")
                continue
            raise

        if data.ndim != 4:
            raise ValueError(f"{mat_path.name}: expected data.ndim==4, got {data.shape}")

        T, C, P, E = data.shape
        if labels.shape[0] != E:
            raise ValueError(f"{mat_path.name}: labels.shape[0]={labels.shape[0]} != E={E}")

        if sid == 1:
            print("[debug] session 01 data shape:", data.shape)
            _debug_label_columns(labels)

        # per-session event split; vary seed by session so splits are reproducible but not identical
        train_idx, test_idx = _split_event_indices_stratified_by_event_type(labels, seed=RANDOM_SEED + sid)

        print(
            f"[prepare_dataset_v3] session {sid:02d}: "
            f"train_events={len(train_idx)}, test_events={len(test_idx)}, total={E}"
        )

        _append_event_samples(
            data, labels, train_idx, sid,
            Xtr_list, ytr_list, ettr_list, sidtr_list
        )
        _append_event_samples(
            data, labels, test_idx, sid,
            Xte_list, yte_list, ette_list, sidte_list
        )

    if not Xtr_list or not Xte_list:
        raise RuntimeError("No train/test samples created. Check label columns and split logic.")

    Xtr = np.stack(Xtr_list, axis=0).astype(np.float32, copy=False)
    ytr = np.asarray(ytr_list, dtype=np.int64)
    ettr = np.asarray(ettr_list, dtype=np.int64)
    sidtr = np.asarray(sidtr_list, dtype=np.int64)

    Xte = np.stack(Xte_list, axis=0).astype(np.float32, copy=False)
    yte = np.asarray(yte_list, dtype=np.int64)
    ette = np.asarray(ette_list, dtype=np.int64)
    sidte = np.asarray(sidte_list, dtype=np.int64)

    return Xtr, ytr, ettr, sidtr, Xte, yte, ette, sidte


def _save_npz(filename: str, **arrays) -> None:
    path = OUT_DIR / filename
    np.savez(path, **arrays)
    print(f"[prepare_dataset_v3] saved: {path}")


def _summarize(name: str, y: np.ndarray) -> None:
    print(f"[summary] {name} counts:", dict(Counter(y.tolist())))


def _event_type_distribution(et: np.ndarray, label: str) -> None:
    u, c = np.unique(et, return_counts=True)
    pairs = list(zip(u.tolist(), c.tolist()))
    print(f"[debug] event_type {label} uniques+counts: {pairs}")


def _filter_event_type(X, y, et, sid, et_value: int):
    m = (et == et_value)
    return X[m], y[m], et[m], sid[m]


def _type0_to_2class(X, y):
    """
    Type0 is treated as 2-class:
      drop class0 (played)
      remap class1 -> 0, class2 -> 1
    """
    keep = (y != 0)
    X2 = X[keep]
    y2 = y[keep]
    y2 = np.where(y2 == 1, 0, 1).astype(np.int64)
    return X2, y2


# ============================================================
# MAIN
# ============================================================
def main():
    Xtr, ytr, ettr, sidtr, Xte, yte, ette, sidte = _build_eventwise_samples(ALL_SESSIONS)

    print()
    _summarize("TRAIN mixed", ytr)
    _summarize("TEST mixed", yte)
    _event_type_distribution(ettr, "train")
    _event_type_distribution(ette, "test")
    print()

    # Save mixed
    _save_npz(
        OUT_MIXED,
        X_train=Xtr, y_train=ytr, event_type_train=ettr, session_id_train=sidtr,
        X_test=Xte, y_test=yte, event_type_test=ette, session_id_test=sidte,
    )

    # Type0 -> 2-class
    X0tr, y0tr, et0tr, sid0tr = _filter_event_type(Xtr, ytr, ettr, sidtr, 0)
    X0te, y0te, et0te, sid0te = _filter_event_type(Xte, yte, ette, sidte, 0)
    X0tr2, y0tr2 = _type0_to_2class(X0tr, y0tr)
    X0te2, y0te2 = _type0_to_2class(X0te, y0te)

    _summarize("TRAIN type0 (2-class)", y0tr2)
    _summarize("TEST type0 (2-class)", y0te2)

    _save_npz(
        OUT_TYPE0_2C,
        X_train=X0tr2, y_train=y0tr2,
        X_test=X0te2, y_test=y0te2,
    )

    # Type1
    X1tr, y1tr, et1tr, sid1tr = _filter_event_type(Xtr, ytr, ettr, sidtr, 1)
    X1te, y1te, et1te, sid1te = _filter_event_type(Xte, yte, ette, sidte, 1)
    _summarize("TRAIN type1", y1tr)
    _summarize("TEST type1", y1te)
    _save_npz(OUT_TYPE1, X_train=X1tr, y_train=y1tr, X_test=X1te, y_test=y1te)

    # Type2
    X2tr, y2tr, et2tr, sid2tr = _filter_event_type(Xtr, ytr, ettr, sidtr, 2)
    X2te, y2te, et2te, sid2te = _filter_event_type(Xte, yte, ette, sidte, 2)
    _summarize("TRAIN type2", y2tr)
    _summarize("TEST type2", y2te)
    _save_npz(OUT_TYPE2, X_train=X2tr, y_train=y2tr, X_test=X2te, y_test=y2te)

    # Type3
    X3tr, y3tr, et3tr, sid3tr = _filter_event_type(Xtr, ytr, ettr, sidtr, 3)
    X3te, y3te, et3te, sid3te = _filter_event_type(Xte, yte, ette, sidte, 3)
    _summarize("TRAIN type3", y3tr)
    _summarize("TEST type3", y3te)
    _save_npz(OUT_TYPE3, X_train=X3tr, y_train=y3tr, X_test=X3te, y_test=y3te)

    print()
    print("[prepare_dataset_v3] DONE.")
    print("[prepare_dataset_v3] New files created:")
    for name in [OUT_MIXED, OUT_TYPE0_2C, OUT_TYPE1, OUT_TYPE2, OUT_TYPE3]:
        print("  -", OUT_DIR / name)


if __name__ == "__main__":
>>>>>>> 1733f74 (first commit)
    main()