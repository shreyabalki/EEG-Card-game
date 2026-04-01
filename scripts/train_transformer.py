# scripts/train_transformer_v2.py
# Patch-based Temporal Transformer for EEG (v2)
#
# Improvements vs v1:
# 1) Prints class distributions (train/val/test) per run
# 2) Uses class-weighted CrossEntropyLoss (computed from TRAIN labels only)
# 3) Uses a smaller default model (CPU-friendly, less overfitting)
# 4) Early stopping on validation accuracy + saves best checkpoint per run
# 5) Logs results to CSV: data/processed/transformer_v2_results.csv
#
# Run:
#   C:\Users\91735\anaconda3\envs\eeg\python.exe C:\Users\91735\Documents\eeg-cardgame\scripts\train_transformer_v2.py

import os
import math
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


DATA_PATH = r"data\processed\dataset_role_eventtype.npz"
OUT_DIR = r"data\processed"
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_CSV = os.path.join(OUT_DIR, "transformer_v2_results.csv")


# -----------------------------
# Metrics (no sklearn)
# -----------------------------
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def macro_f1_from_cm(cm: np.ndarray) -> float:
    f1s = []
    for k in range(cm.shape[0]):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(f1s))

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())

def class_counts(y: np.ndarray, n_classes: int = 3) -> np.ndarray:
    return np.bincount(y.astype(np.int64), minlength=n_classes)

def pretty_counts(name: str, counts: np.ndarray) -> str:
    total = int(counts.sum())
    parts = [f"class{i}={int(c)}" for i, c in enumerate(counts)]
    return f"{name}: total={total} | " + ", ".join(parts)

def compute_class_weights_from_train(y_train: np.ndarray, n_classes: int = 3) -> np.ndarray:
    # Inverse frequency weights, normalized to average=1
    counts = np.bincount(y_train.astype(np.int64), minlength=n_classes).astype(np.float32)
    weights = counts.sum() / (counts + 1e-6)
    weights = weights / weights.mean()
    return weights.astype(np.float32)


# -----------------------------
# Dataset
# -----------------------------
class EegDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))  # (N, C, T)
        self.y = torch.from_numpy(y.astype(np.int64))    # (N,)

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, i: int):
        return self.X[i], self.y[i]


# -----------------------------
# Model: Patch Transformer
# -----------------------------
class PatchEmbed(nn.Module):
    def __init__(self, C: int, patch_len: int, d_model: int):
        super().__init__()
        self.C = C
        self.patch_len = patch_len
        self.proj = nn.Linear(C * patch_len, d_model)

    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.shape
        assert C == self.C, f"Expected C={self.C}, got C={C}"
        n_patches = T // self.patch_len
        T_used = n_patches * self.patch_len
        x = x[:, :, :T_used]                         # (B, C, T_used)
        x = x.view(B, C, n_patches, self.patch_len)  # (B, C, N, P)
        x = x.permute(0, 2, 1, 3).contiguous()       # (B, N, C, P)
        x = x.view(B, n_patches, C * self.patch_len) # (B, N, C*P)
        return self.proj(x)                          # (B, N, d_model)

class EEGTransformer(nn.Module):
    def __init__(
        self,
        C: int,
        T: int,
        n_classes: int = 3,
        patch_len: int = 25,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        norm_first: bool = False,  # set False to reduce nested tensor warning
    ):
        super().__init__()
        self.embed = PatchEmbed(C=C, patch_len=patch_len, d_model=d_model)
        n_tokens = (T // patch_len)

        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        z = self.embed(x)                          # (B, N, d)
        z = z + self.pos[:, : z.size(1), :]        # (B, N, d)
        z = self.encoder(z)                        # (B, N, d)
        z = self.norm(z)
        z = z.mean(dim=1)                          # token average
        return self.head(z)                        # (B, n_classes)


# -----------------------------
# Splits
# -----------------------------
def split_by_session(session: np.ndarray, test_sessions=(20, 21, 22)):
    test_mask = np.isin(session, np.array(test_sessions))
    train_mask = ~test_mask
    return train_mask, test_mask

def make_train_val_indices(idx_train: np.ndarray, val_frac: float = 0.15, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = idx_train.copy()
    rng.shuffle(idx)
    n_val = max(1, int(val_frac * len(idx)))
    idx_val = idx[:n_val]
    idx_tr = idx[n_val:]
    return idx_tr, idx_val


# -----------------------------
# Training / Evaluation
# -----------------------------
def train_one_model(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    patch_len: int = 25,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    dropout: float = 0.1,
    patience: int = 5,
    seed: int = 42,
    checkpoint_path: str | None = None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    C = X_tr.shape[1]
    T = X_tr.shape[2]

    model = EEGTransformer(
        C=C, T=T,
        n_classes=3,
        patch_len=patch_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        norm_first=False,  # reduces nested tensor warning
    ).to(device)

    # Weighted loss from TRAIN labels only
    w = compute_class_weights_from_train(y_tr, n_classes=3)
    w_t = torch.tensor(w, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w_t)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(EegDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(EegDataset(X_val, y_val), batch_size=batch_size, shuffle=False, drop_last=False)

    print(pretty_counts("TRAIN", class_counts(y_tr)))
    print(pretty_counts("VAL  ", class_counts(y_val)))
    print("Class weights (from TRAIN):", w)

    best_val_acc = -1.0
    best_state = None
    bad_epochs = 0

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total_loss += float(loss.item()) * xb.size(0)
            n += xb.size(0)

        # validation
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                pred = logits.argmax(dim=1).cpu().numpy()
                y_pred.append(pred)
                y_true.append(yb.numpy())

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        val_acc = accuracy_score(y_true, y_pred)

        print(f"epoch {ep:02d} train_loss {total_loss/n:.4f} val_acc {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
            if checkpoint_path is not None:
                torch.save(best_state, checkpoint_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping: no val improvement for {patience} epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_acc

@torch.no_grad()
def eval_model(model: nn.Module, X: np.ndarray, y: np.ndarray):
    device = next(model.parameters()).device
    model.eval()
    loader = DataLoader(EegDataset(X, y), batch_size=128, shuffle=False, drop_last=False)
    preds, ys = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        pred = logits.argmax(dim=1).cpu().numpy()
        preds.append(pred)
        ys.append(yb.numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, n_classes=3)
    mf1 = macro_f1_from_cm(cm)
    return acc, mf1, cm, y_true, y_pred


# -----------------------------
# Experiment runner
# -----------------------------
def append_results_csv(row: dict):
    header = [
        "run_name", "n_train", "n_val", "n_test",
        "train_counts", "val_counts", "test_counts",
        "best_val_acc", "test_acc", "test_macroF1",
        "cm_00","cm_01","cm_02","cm_10","cm_11","cm_12","cm_20","cm_21","cm_22",
        "patch_len","d_model","n_heads","n_layers","dropout","epochs","batch_size","lr","weight_decay","patience",
        "device",
    ]
    file_exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            w.writeheader()
        w.writerow(row)

def run_one(name: str, X: np.ndarray, y: np.ndarray, event_type: np.ndarray, session: np.ndarray, mask: np.ndarray):
    train_mask, test_mask = split_by_session(session, test_sessions=(20, 21, 22))
    train_mask &= mask
    test_mask &= mask

    idx_train_all = np.where(train_mask)[0]
    idx_test = np.where(test_mask)[0]

    if len(idx_train_all) < 300 or len(idx_test) < 80:
        print(f"\n===== {name} =====")
        print(f"Too few samples (train={len(idx_train_all)}, test={len(idx_test)}). Skipping.")
        return

    idx_tr, idx_val = make_train_val_indices(idx_train_all, val_frac=0.15, seed=42)

    X_tr, y_tr = X[idx_tr], y[idx_tr]
    X_val, y_val = X[idx_val], y[idx_val]
    X_te, y_te = X[idx_test], y[idx_test]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n===== {name} =====")
    print(f"train {len(idx_tr)} | val {len(idx_val)} | test {len(idx_test)}")
    print(pretty_counts("TEST ", class_counts(y_te)))

    ckpt_path = os.path.join(OUT_DIR, f"transformer_v2_{name.replace(' ', '_').replace('(', '').replace(')', '').replace(':','').replace('/','_')}.pt")

    # CPU-friendly defaults (good starting point)
    cfg = dict(
        epochs=25,
        batch_size=64,
        lr=3e-4,
        weight_decay=1e-2,
        patch_len=25,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
        patience=5,
    )

    model, best_val_acc = train_one_model(
        X_tr, y_tr,
        X_val, y_val,
        checkpoint_path=ckpt_path,
        **cfg
    )

    test_acc, test_mf1, cm, y_true, y_pred = eval_model(model, X_te, y_te)
    print(f"TEST acc={test_acc:.3f} | macroF1={test_mf1:.3f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    row = {
        "run_name": name,
        "n_train": len(idx_tr),
        "n_val": len(idx_val),
        "n_test": len(idx_test),
        "train_counts": class_counts(y_tr).tolist(),
        "val_counts": class_counts(y_val).tolist(),
        "test_counts": class_counts(y_te).tolist(),
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "test_macroF1": float(test_mf1),
        "cm_00": int(cm[0,0]), "cm_01": int(cm[0,1]), "cm_02": int(cm[0,2]),
        "cm_10": int(cm[1,0]), "cm_11": int(cm[1,1]), "cm_12": int(cm[1,2]),
        "cm_20": int(cm[2,0]), "cm_21": int(cm[2,1]), "cm_22": int(cm[2,2]),
        "patch_len": cfg["patch_len"],
        "d_model": cfg["d_model"],
        "n_heads": cfg["n_heads"],
        "n_layers": cfg["n_layers"],
        "dropout": cfg["dropout"],
        "epochs": cfg["epochs"],
        "batch_size": cfg["batch_size"],
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "patience": cfg["patience"],
        "device": device,
    }
    append_results_csv(row)
    print(f"Logged to: {RESULTS_CSV}")
    print(f"Saved best checkpoint to: {ckpt_path}")


def main():
    data = np.load(DATA_PATH)
    X = data["X"].astype(np.float32)          # (N, C, T)
    y = data["y"].astype(np.int64)            # (N,)
    event_type = data["event_type"].astype(np.int64)
    session = data["session"].astype(np.int64)

    print("Loaded:", X.shape, y.shape)
    print("Event counts:", dict(zip(*np.unique(event_type, return_counts=True))))
    print("Using test sessions:", [20, 21, 22])
    print("Device:", "cuda" if torch.cuda.is_available() else "cpu")

    # Mixed
    run_one(
        name="TransformerV2 Mixed",
        X=X, y=y, event_type=event_type, session=session,
        mask=np.ones(len(y), dtype=bool)
    )

    # Per event type
    for et in [0, 1, 2]:
        run_one(
            name=f"TransformerV2 EventType {et}",
            X=X, y=y, event_type=event_type, session=session,
            mask=(event_type == et)
        )


if __name__ == "__main__":
    main()
