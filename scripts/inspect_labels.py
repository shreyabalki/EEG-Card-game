<<<<<<< HEAD
import os, glob
import numpy as np
from scipy.io import loadmat

RAW_DIR = r"data\raw"

def main():
    path = sorted(glob.glob(os.path.join(RAW_DIR, "sessionevents01.mat")))[0]
    mat = loadmat(path)
    labels = mat["labels"]  # (E, 8)

    print("labels shape:", labels.shape)
    print("\nFirst 10 rows:")
    print(labels[:10])

    print("\nPer-column unique sample (up to 20 uniques each):")
    for c in range(labels.shape[1]):
        u = np.unique(labels[:, c])
        print(f"col {c}: min={labels[:,c].min()} max={labels[:,c].max()} uniques={u[:20]}{' ...' if len(u)>20 else ''}")

if __name__ == "__main__":
    main()
=======
import os, glob
import numpy as np
from scipy.io import loadmat

RAW_DIR = r"data\raw"

def main():
    path = sorted(glob.glob(os.path.join(RAW_DIR, "sessionevents01.mat")))[0]
    mat = loadmat(path)
    labels = mat["labels"]  # (E, 8)

    print("labels shape:", labels.shape)
    print("\nFirst 10 rows:")
    print(labels[:10])

    print("\nPer-column unique sample (up to 20 uniques each):")
    for c in range(labels.shape[1]):
        u = np.unique(labels[:, c])
        print(f"col {c}: min={labels[:,c].min()} max={labels[:,c].max()} uniques={u[:20]}{' ...' if len(u)>20 else ''}")

if __name__ == "__main__":
    main()
>>>>>>> 1733f74 (first commit)
