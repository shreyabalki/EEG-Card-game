<<<<<<< HEAD
import os, glob
import numpy as np
from scipy.io import loadmat

RAW_DIR = r"data\raw"

def main():
    path = sorted(glob.glob(os.path.join(RAW_DIR, "sessionevents01.mat")))[0]
    labels = loadmat(path)["labels"]

    # Heuristic: player and last_player should mostly be in {0,1,2,3} (or 1..3)
    candidates = []
    for c in range(labels.shape[1]):
        u = np.unique(labels[:, c])
        if u.min() >= 0 and u.max() <= 3 and len(u) <= 4:
            candidates.append((c, u))
    print("Candidate columns for player-like ids (0..3):")
    for c,u in candidates:
        print(f"col {c}: {u}")

    # Heuristic: cardsplayed columns are binary {0,1}
    print("\nCandidate columns for binary flags {0,1}:")
    for c in range(labels.shape[1]):
        u = np.unique(labels[:, c])
        if set(u.tolist()).issubset({0, 1}):
            print(f"col {c}: uniques={u}")

if __name__ == "__main__":
    main()
=======
import os, glob
import numpy as np
from scipy.io import loadmat

RAW_DIR = r"data\raw"

def main():
    path = sorted(glob.glob(os.path.join(RAW_DIR, "sessionevents01.mat")))[0]
    labels = loadmat(path)["labels"]

    # Heuristic: player and last_player should mostly be in {0,1,2,3} (or 1..3)
    candidates = []
    for c in range(labels.shape[1]):
        u = np.unique(labels[:, c])
        if u.min() >= 0 and u.max() <= 3 and len(u) <= 4:
            candidates.append((c, u))
    print("Candidate columns for player-like ids (0..3):")
    for c,u in candidates:
        print(f"col {c}: {u}")

    # Heuristic: cardsplayed columns are binary {0,1}
    print("\nCandidate columns for binary flags {0,1}:")
    for c in range(labels.shape[1]):
        u = np.unique(labels[:, c])
        if set(u.tolist()).issubset({0, 1}):
            print(f"col {c}: uniques={u}")

if __name__ == "__main__":
    main()
>>>>>>> 1733f74 (first commit)
