# scripts/train_baseline.py
# Baseline CNN training + evaluation overall and per event_type
# Run:
#   C:/Users/91735/anaconda3/envs/eeg/python.exe scripts/train_baseline.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_PATH = r"data\processed\dataset_role_eventtype.npz"


class EegDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (N, C, T) float32, y: (N,) int64
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, i: int):
        return self.X[i], self.y[i]


class SmallCNN(nn.Module):
    def __init__(self, C: int, n_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(C, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.MaxPool1d(2),

            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        # x: (B, C, T)
        z = self.net(x).squeeze(-1)  # (B, 128)
        return self.fc(z)            # (B, n_classes)


def split_by_session(session: np.ndarray, test_sessions=(20, 21, 22)):
    test_mask = np.isin(session, np.array(test_sessions))
    train_mask = ~test_mask
    return train_mask, test_mask


def train_model(X: np.ndarray, y: np.ndarray, epochs: int = 10, lr: float = 1e-3, batch: int = 64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallCNN(C=int(X.shape[1])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    dl = DataLoader(EegDataset(X, y), batch_size=batch, shuffle=True, drop_last=False)

    model.train()
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        n = 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        print(f"epoch {ep:02d} loss {total_loss/n:.4f}")

    return model


@torch.no_grad()
def accuracy(model: nn.Module, X: np.ndarray, y: np.ndarray) -> float:
    device = next(model.parameters()).device
    model.eval()
    xb = torch.from_numpy(X).to(device)
    logits = model(xb)
    pred = logits.argmax(dim=1).cpu().numpy()
    return float((pred == y).mean())


def main():
    data = np.load(DATA_PATH)
    X = data["X"].astype(np.float32)         # (N, C, T)
    y = data["y"].astype(np.int64)           # (N,)
    event_type = data["event_type"].astype(np.int64)
    session = data["session"].astype(np.int64)

    train_mask, test_mask = split_by_session(session, test_sessions=(20, 21, 22))
    print("Train samples:", int(train_mask.sum()), "Test samples:", int(test_mask.sum()))
    print("Test sessions:", [20, 21, 22])

    # Overall baseline (all event types mixed)
    print("\n=== Training baseline on ALL event types (mixed) ===")
    model_all = train_model(X[train_mask], y[train_mask], epochs=10, lr=1e-3, batch=64)
    acc_all = accuracy(model_all, X[test_mask], y[test_mask])
    print(f"\nOverall test accuracy (mixed types): {acc_all:.3f}")

    # Per event type
    print("\n=== Training baseline PER event_type ===")
    for et in [0, 1, 2]:
        tr = train_mask & (event_type == et)
        te = test_mask & (event_type == et)

        ntr, nte = int(tr.sum()), int(te.sum())
        print(f"\nEventType {et}: train {ntr} test {nte}")

        # Updated thresholds (IMPORTANT)
        if ntr < 200 or nte < 50:
            print("  too few samples, skipping")
            continue

        model = train_model(X[tr], y[tr], epochs=10, lr=1e-3, batch=64)
        acc = accuracy(model, X[te], y[te])
        print(f"  test accuracy: {acc:.3f}")


if __name__ == "__main__":
    main()
