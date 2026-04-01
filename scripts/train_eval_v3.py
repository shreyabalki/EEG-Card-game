<<<<<<< HEAD
# scripts/train_eval_v3.py
# ------------------------------------------------------------
# Trains + evaluates a simple CNN or a small Transformer on
# v4 event-wise split EEG datasets.
#
# Supports datasets:
#   mixed, type0, type1, type2, type3
#
# Notes:
# - type0 is 2-class (from v4_eventsplit_type0_2class.npz)
# - others are 3-class (labels 0/1/2)
# - train/test split is EVENT-WISE within every session
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
CKPT_DIR = DATA_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = DATA_DIR / "train_eval_v4_eventsplit_results.csv"

DATASET_TO_FILE: Dict[str, str] = {
    "mixed": "v4_eventsplit_mixed.npz",
    "type0": "v4_eventsplit_type0_2class.npz",
    "type1": "v4_eventsplit_type1.npz",
    "type2": "v4_eventsplit_type2.npz",
    "type3": "v4_eventsplit_type3.npz",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NpzEEGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class SimpleEEGCNN(nn.Module):
    def __init__(self, n_ch: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_ch, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.net(x)
        x = x.squeeze(-1)
        return self.fc(x)


class PatchTransformer(nn.Module):
    def __init__(
        self,
        n_ch: int,
        n_classes: int,
        patch_len: int = 25,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        if patch_len <= 0:
            raise ValueError("patch_len must be > 0")
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")

        self.n_ch = n_ch
        self.patch_len = patch_len
        self.d_model = d_model

        self.proj = nn.Linear(n_ch * patch_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        B, C, T = x.shape
        P = self.patch_len

        n_patches = T // P
        T_trim = n_patches * P

        if T_trim == 0:
            raise ValueError(f"Time length T={T} is too short for patch_len={P}")

        x = x[:, :, :T_trim]
        x = x.view(B, C, n_patches, P)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, n_patches, C * P)
        x = self.proj(x)

        x = self.encoder(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


def compute_class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    w = inv / inv.sum() * n_classes
    return torch.tensor(w, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        n += X.size(0)

    return total_loss / max(n, 1)


@torch.no_grad()
def eval_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys = []
    ps = []

    for X, y in loader:
        X = X.to(device)
        logits = model(X)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        ys.append(y.numpy())
        ps.append(pred)

    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(ps, axis=0)
    return y_true, y_pred


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def append_csv_row(path: Path, row: Dict[str, object]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _optional_array(d: np.lib.npyio.NpzFile, key: str) -> Optional[np.ndarray]:
    return d[key] if key in d.files else None


def load_eventsplit_dataset(
    dataset_name: str,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    Optional[np.ndarray], Optional[np.ndarray],
    Optional[np.ndarray], Optional[np.ndarray],
    Optional[np.ndarray], Optional[np.ndarray]
]:
    if dataset_name not in DATASET_TO_FILE:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Choose from {list(DATASET_TO_FILE.keys())}"
        )

    npz_path = DATA_DIR / DATASET_TO_FILE[dataset_name]
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {npz_path}")

    d = np.load(npz_path)

    X_train = d["X_train"].astype(np.float32)
    y_train = d["y_train"].astype(np.int64)
    X_test = d["X_test"].astype(np.float32)
    y_test = d["y_test"].astype(np.int64)

    event_type_train = _optional_array(d, "event_type_train")
    event_type_test = _optional_array(d, "event_type_test")
    session_id_train = _optional_array(d, "session_id_train")
    session_id_test = _optional_array(d, "session_id_test")
    event_id_train = _optional_array(d, "event_id_train")
    event_id_test = _optional_array(d, "event_id_test")

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        event_type_train,
        event_type_test,
        session_id_train,
        session_id_test,
        event_id_train,
        event_id_test,
    )


def print_optional_metadata_summary(
    name: str,
    event_type: Optional[np.ndarray],
    session_id: Optional[np.ndarray],
    event_id: Optional[np.ndarray],
) -> None:
    print(f"[train_eval_v3] {name} metadata summary:")

    if event_type is not None:
        u, c = np.unique(event_type, return_counts=True)
        print(f"  event_type counts: {dict(zip(u.tolist(), c.tolist()))}")
    else:
        print("  event_type counts: not available")

    if session_id is not None:
        u = np.unique(session_id)
        preview = u[:10].tolist()
        suffix = " ..." if len(u) > 10 else ""
        print(f"  unique sessions ({len(u)}): {preview}{suffix}")
    else:
        print("  unique sessions: not available")

    if event_id is not None:
        print(f"  event_id array present: yes, n={len(event_id)}")
    else:
        print("  event_id array present: no")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(
        description="Train/eval CNN or Transformer on v4 event-wise EEG datasets."
    )
    parser.add_argument("--model", choices=["cnn", "transformer"], required=True)
    parser.add_argument("--dataset", choices=list(DATASET_TO_FILE.keys()), required=True)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_class_weights", action="store_true")

    parser.add_argument("--patch_len", type=int, default=25)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    args = parser.parse_args()

    print("[train_eval_v3] starting...")
    print(f"[train_eval_v3] model={args.model} dataset={args.dataset}")
    print("[train_eval_v3] split_type=event-wise within sessions")

    seed_everything(args.seed)

    (
        X_train,
        y_train,
        X_test,
        y_test,
        event_type_train,
        event_type_test,
        session_id_train,
        session_id_test,
        event_id_train,
        event_id_test,
    ) = load_eventsplit_dataset(args.dataset)

    n_ch = X_train.shape[1]
    n_classes = int(np.max(y_train)) + 1

    print("[train_eval_v3] X_train:", X_train.shape, "X_test:", X_test.shape)
    print("[train_eval_v3] y_train counts:", dict(Counter(y_train.tolist())))
    print("[train_eval_v3] y_test  counts:", dict(Counter(y_test.tolist())))
    print("[train_eval_v3] n_classes:", n_classes)

    print_optional_metadata_summary("TRAIN", event_type_train, session_id_train, event_id_train)
    print_optional_metadata_summary("TEST", event_type_test, session_id_test, event_id_test)

    if (
        session_id_train is not None
        and session_id_test is not None
        and event_id_train is not None
        and event_id_test is not None
    ):
        train_pairs = set(zip(session_id_train.tolist(), event_id_train.tolist()))
        test_pairs = set(zip(session_id_test.tolist(), event_id_test.tolist()))
        overlap = train_pairs.intersection(test_pairs)
        print(f"[train_eval_v3] leakage check overlap pairs: {len(overlap)}")
        if overlap:
            raise RuntimeError(f"Leakage detected: {list(overlap)[:10]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[train_eval_v3] device:", device)

    train_ds = NpzEEGDataset(X_train, y_train)
    test_ds = NpzEEGDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.model == "cnn":
        model = SimpleEEGCNN(n_ch=n_ch, n_classes=n_classes)
        model_tag = "CNN"
    else:
        model = PatchTransformer(
            n_ch=n_ch,
            n_classes=n_classes,
            patch_len=args.patch_len,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_ff=args.dim_ff,
            dropout=args.dropout,
        )
        model_tag = f"Transformer_patch{args.patch_len}"

    model = model.to(device)

    n_params = count_parameters(model)
    print(f"[train_eval_v3] trainable_parameters={n_params}")

    if args.use_class_weights:
        cw = compute_class_weights(y_train, n_classes).to(device)
        print("[train_eval_v3] class_weights:", cw.detach().cpu().numpy().round(4).tolist())
        criterion = nn.CrossEntropyLoss(weight=cw)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_macro_f1 = -1.0
    best_epoch = -1
    epochs_no_improve = 0

    ckpt_path = CKPT_DIR / f"best_eventsplit_{model_tag}_{args.dataset}.pt"

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        y_true, y_pred = eval_model(model, test_loader, device)
        m = metrics_dict(y_true, y_pred)

        print(
            f"[train_eval_v3] epoch {epoch:03d} "
            f"loss={tr_loss:.4f} "
            f"acc={m['accuracy']:.3f} "
            f"bal_acc={m['balanced_accuracy']:.3f} "
            f"macro_f1={m['macro_f1']:.3f}"
        )

        if m["macro_f1"] > best_macro_f1:
            best_macro_f1 = m["macro_f1"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "n_classes": n_classes,
                    "n_ch": n_ch,
                },
                ckpt_path,
            )
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f"[train_eval_v3] early stopping at epoch {epoch} (best epoch {best_epoch})")
            break

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    y_true, y_pred = eval_model(model, test_loader, device)
    m = metrics_dict(y_true, y_pred)

    print("\n[train_eval_v3] FINAL METRICS")
    for k, v in m.items():
        print(f"  {k}: {v:.4f}")

    print("\n[train_eval_v3] CONFUSION MATRIX")
    print(confusion_matrix(y_true, y_pred))

    print("\n[train_eval_v3] CLASSIFICATION REPORT")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    row = {
        "dataset": args.dataset,
        "split_type": "eventwise",
        "model": args.model,
        "model_tag": model_tag,
        "patch_len": args.patch_len if args.model == "transformer" else "",
        "d_model": args.d_model if args.model == "transformer" else "",
        "nhead": args.nhead if args.model == "transformer" else "",
        "num_layers": args.num_layers if args.model == "transformer" else "",
        "dim_ff": args.dim_ff if args.model == "transformer" else "",
        "dropout": args.dropout if args.model == "transformer" else "",
        "use_class_weights": args.use_class_weights,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1,
        "accuracy": m["accuracy"],
        "balanced_accuracy": m["balanced_accuracy"],
        "macro_f1": m["macro_f1"],
        "weighted_f1": m["weighted_f1"],
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_classes": n_classes,
        "n_parameters": n_params,
        "ckpt_path": str(ckpt_path),
    }
    append_csv_row(RESULTS_CSV, row)

    print(f"\n[train_eval_v3] logged to: {RESULTS_CSV}")
    print(f"[train_eval_v3] best checkpoint: {ckpt_path}")


if __name__ == "__main__":
=======
# scripts/train_eval_v3.py
# ------------------------------------------------------------
# Trains + evaluates a simple CNN or a small Transformer on
# v4 event-wise split EEG datasets.
#
# Supports datasets:
#   mixed, type0, type1, type2, type3
#
# Notes:
# - type0 is 2-class (from v4_eventsplit_type0_2class.npz)
# - others are 3-class (labels 0/1/2)
# - train/test split is EVENT-WISE within every session
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
CKPT_DIR = DATA_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = DATA_DIR / "train_eval_v4_eventsplit_results.csv"

DATASET_TO_FILE: Dict[str, str] = {
    "mixed": "v4_eventsplit_mixed.npz",
    "type0": "v4_eventsplit_type0_2class.npz",
    "type1": "v4_eventsplit_type1.npz",
    "type2": "v4_eventsplit_type2.npz",
    "type3": "v4_eventsplit_type3.npz",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NpzEEGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class SimpleEEGCNN(nn.Module):
    def __init__(self, n_ch: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_ch, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.net(x)
        x = x.squeeze(-1)
        return self.fc(x)


class PatchTransformer(nn.Module):
    def __init__(
        self,
        n_ch: int,
        n_classes: int,
        patch_len: int = 25,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        if patch_len <= 0:
            raise ValueError("patch_len must be > 0")
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")

        self.n_ch = n_ch
        self.patch_len = patch_len
        self.d_model = d_model

        self.proj = nn.Linear(n_ch * patch_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        B, C, T = x.shape
        P = self.patch_len

        n_patches = T // P
        T_trim = n_patches * P

        if T_trim == 0:
            raise ValueError(f"Time length T={T} is too short for patch_len={P}")

        x = x[:, :, :T_trim]
        x = x.view(B, C, n_patches, P)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, n_patches, C * P)
        x = self.proj(x)

        x = self.encoder(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


def compute_class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    w = inv / inv.sum() * n_classes
    return torch.tensor(w, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        n += X.size(0)

    return total_loss / max(n, 1)


@torch.no_grad()
def eval_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys = []
    ps = []

    for X, y in loader:
        X = X.to(device)
        logits = model(X)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        ys.append(y.numpy())
        ps.append(pred)

    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(ps, axis=0)
    return y_true, y_pred


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def append_csv_row(path: Path, row: Dict[str, object]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _optional_array(d: np.lib.npyio.NpzFile, key: str) -> Optional[np.ndarray]:
    return d[key] if key in d.files else None


def load_eventsplit_dataset(
    dataset_name: str,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    Optional[np.ndarray], Optional[np.ndarray],
    Optional[np.ndarray], Optional[np.ndarray],
    Optional[np.ndarray], Optional[np.ndarray]
]:
    if dataset_name not in DATASET_TO_FILE:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Choose from {list(DATASET_TO_FILE.keys())}"
        )

    npz_path = DATA_DIR / DATASET_TO_FILE[dataset_name]
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {npz_path}")

    d = np.load(npz_path)

    X_train = d["X_train"].astype(np.float32)
    y_train = d["y_train"].astype(np.int64)
    X_test = d["X_test"].astype(np.float32)
    y_test = d["y_test"].astype(np.int64)

    event_type_train = _optional_array(d, "event_type_train")
    event_type_test = _optional_array(d, "event_type_test")
    session_id_train = _optional_array(d, "session_id_train")
    session_id_test = _optional_array(d, "session_id_test")
    event_id_train = _optional_array(d, "event_id_train")
    event_id_test = _optional_array(d, "event_id_test")

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        event_type_train,
        event_type_test,
        session_id_train,
        session_id_test,
        event_id_train,
        event_id_test,
    )


def print_optional_metadata_summary(
    name: str,
    event_type: Optional[np.ndarray],
    session_id: Optional[np.ndarray],
    event_id: Optional[np.ndarray],
) -> None:
    print(f"[train_eval_v3] {name} metadata summary:")

    if event_type is not None:
        u, c = np.unique(event_type, return_counts=True)
        print(f"  event_type counts: {dict(zip(u.tolist(), c.tolist()))}")
    else:
        print("  event_type counts: not available")

    if session_id is not None:
        u = np.unique(session_id)
        preview = u[:10].tolist()
        suffix = " ..." if len(u) > 10 else ""
        print(f"  unique sessions ({len(u)}): {preview}{suffix}")
    else:
        print("  unique sessions: not available")

    if event_id is not None:
        print(f"  event_id array present: yes, n={len(event_id)}")
    else:
        print("  event_id array present: no")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(
        description="Train/eval CNN or Transformer on v4 event-wise EEG datasets."
    )
    parser.add_argument("--model", choices=["cnn", "transformer"], required=True)
    parser.add_argument("--dataset", choices=list(DATASET_TO_FILE.keys()), required=True)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_class_weights", action="store_true")

    parser.add_argument("--patch_len", type=int, default=25)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    args = parser.parse_args()

    print("[train_eval_v3] starting...")
    print(f"[train_eval_v3] model={args.model} dataset={args.dataset}")
    print("[train_eval_v3] split_type=event-wise within sessions")

    seed_everything(args.seed)

    (
        X_train,
        y_train,
        X_test,
        y_test,
        event_type_train,
        event_type_test,
        session_id_train,
        session_id_test,
        event_id_train,
        event_id_test,
    ) = load_eventsplit_dataset(args.dataset)

    n_ch = X_train.shape[1]
    n_classes = int(np.max(y_train)) + 1

    print("[train_eval_v3] X_train:", X_train.shape, "X_test:", X_test.shape)
    print("[train_eval_v3] y_train counts:", dict(Counter(y_train.tolist())))
    print("[train_eval_v3] y_test  counts:", dict(Counter(y_test.tolist())))
    print("[train_eval_v3] n_classes:", n_classes)

    print_optional_metadata_summary("TRAIN", event_type_train, session_id_train, event_id_train)
    print_optional_metadata_summary("TEST", event_type_test, session_id_test, event_id_test)

    if (
        session_id_train is not None
        and session_id_test is not None
        and event_id_train is not None
        and event_id_test is not None
    ):
        train_pairs = set(zip(session_id_train.tolist(), event_id_train.tolist()))
        test_pairs = set(zip(session_id_test.tolist(), event_id_test.tolist()))
        overlap = train_pairs.intersection(test_pairs)
        print(f"[train_eval_v3] leakage check overlap pairs: {len(overlap)}")
        if overlap:
            raise RuntimeError(f"Leakage detected: {list(overlap)[:10]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[train_eval_v3] device:", device)

    train_ds = NpzEEGDataset(X_train, y_train)
    test_ds = NpzEEGDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.model == "cnn":
        model = SimpleEEGCNN(n_ch=n_ch, n_classes=n_classes)
        model_tag = "CNN"
    else:
        model = PatchTransformer(
            n_ch=n_ch,
            n_classes=n_classes,
            patch_len=args.patch_len,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_ff=args.dim_ff,
            dropout=args.dropout,
        )
        model_tag = f"Transformer_patch{args.patch_len}"

    model = model.to(device)

    n_params = count_parameters(model)
    print(f"[train_eval_v3] trainable_parameters={n_params}")

    if args.use_class_weights:
        cw = compute_class_weights(y_train, n_classes).to(device)
        print("[train_eval_v3] class_weights:", cw.detach().cpu().numpy().round(4).tolist())
        criterion = nn.CrossEntropyLoss(weight=cw)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_macro_f1 = -1.0
    best_epoch = -1
    epochs_no_improve = 0

    ckpt_path = CKPT_DIR / f"best_eventsplit_{model_tag}_{args.dataset}.pt"

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        y_true, y_pred = eval_model(model, test_loader, device)
        m = metrics_dict(y_true, y_pred)

        print(
            f"[train_eval_v3] epoch {epoch:03d} "
            f"loss={tr_loss:.4f} "
            f"acc={m['accuracy']:.3f} "
            f"bal_acc={m['balanced_accuracy']:.3f} "
            f"macro_f1={m['macro_f1']:.3f}"
        )

        if m["macro_f1"] > best_macro_f1:
            best_macro_f1 = m["macro_f1"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "n_classes": n_classes,
                    "n_ch": n_ch,
                },
                ckpt_path,
            )
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f"[train_eval_v3] early stopping at epoch {epoch} (best epoch {best_epoch})")
            break

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    y_true, y_pred = eval_model(model, test_loader, device)
    m = metrics_dict(y_true, y_pred)

    print("\n[train_eval_v3] FINAL METRICS")
    for k, v in m.items():
        print(f"  {k}: {v:.4f}")

    print("\n[train_eval_v3] CONFUSION MATRIX")
    print(confusion_matrix(y_true, y_pred))

    print("\n[train_eval_v3] CLASSIFICATION REPORT")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    row = {
        "dataset": args.dataset,
        "split_type": "eventwise",
        "model": args.model,
        "model_tag": model_tag,
        "patch_len": args.patch_len if args.model == "transformer" else "",
        "d_model": args.d_model if args.model == "transformer" else "",
        "nhead": args.nhead if args.model == "transformer" else "",
        "num_layers": args.num_layers if args.model == "transformer" else "",
        "dim_ff": args.dim_ff if args.model == "transformer" else "",
        "dropout": args.dropout if args.model == "transformer" else "",
        "use_class_weights": args.use_class_weights,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1,
        "accuracy": m["accuracy"],
        "balanced_accuracy": m["balanced_accuracy"],
        "macro_f1": m["macro_f1"],
        "weighted_f1": m["weighted_f1"],
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_classes": n_classes,
        "n_parameters": n_params,
        "ckpt_path": str(ckpt_path),
    }
    append_csv_row(RESULTS_CSV, row)

    print(f"\n[train_eval_v3] logged to: {RESULTS_CSV}")
    print(f"[train_eval_v3] best checkpoint: {ckpt_path}")


if __name__ == "__main__":
>>>>>>> 1733f74 (first commit)
    main()