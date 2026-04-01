import os, glob
import numpy as np
from scipy.io import loadmat
from config import LABEL_IDX

RAW_DIR = r"data\raw"
OUT_DIR = r"data\processed"
os.makedirs(OUT_DIR, exist_ok=True)

ROLE_TO_Y = {"played": 0, "next": 1, "observer": 2}

def compute_event_type(last_player: int, cardsplayed: int) -> int:
    if last_player == 0:
        return 0
    if cardsplayed == 1:
        return 1
    if cardsplayed == 2:
        return 2
    return -1

def main():
    paths = sorted(glob.glob(os.path.join(RAW_DIR, "sessionevents*.mat")))
    X_list, y_list, et_list, sess_list = [], [], [], []

    for path in paths:
        fname = os.path.basename(path)
        session_id = int(fname.replace("sessionevents", "").replace(".mat", ""))
        mat = loadmat(path)

        data = mat["data"]      # (T, C, P, E)
        labels = mat["labels"]  # (E, 8)
        T, C, P, E = data.shape

        kept = 0
        for e in range(E):
            row = labels[e, :]
            current_player = int(row[LABEL_IDX["player"]])
            last_player = int(row[LABEL_IDX["last_player"]])

            cp1 = int(row[LABEL_IDX["cardsplayed1"]])
            cp2 = int(row[LABEL_IDX["cardsplayed2"]])
            cp3 = int(row[LABEL_IDX["cardsplayed3"]])
            cardsplayed = cp1 + cp2 + cp3

            et = compute_event_type(last_player, cardsplayed)
            if et == -1:
                continue

            for p in range(1, P + 1):
                eeg = data[:, :, p - 1, e]     # (T, C)
                eeg = np.nan_to_num(eeg).T     # (C, T)

                if p == last_player:
                    role = "played"
                elif p == current_player:
                    role = "next"
                else:
                    role = "observer"

                X_list.append(eeg.astype(np.float32))
                y_list.append(ROLE_TO_Y[role])
                et_list.append(et)
                sess_list.append(session_id)
                kept += 1

        print(f"Session {session_id:02d}: kept samples {kept}")

    X = np.stack(X_list)  # (N, C, T)
    y = np.array(y_list, dtype=np.int64)
    event_type = np.array(et_list, dtype=np.int64)
    session = np.array(sess_list, dtype=np.int64)

    out_path = os.path.join(OUT_DIR, "dataset_role_eventtype.npz")
    np.savez_compressed(out_path, X=X, y=y, event_type=event_type, session=session)

    print("\nSaved:", out_path)
    print("X:", X.shape, "y:", y.shape)
    print("Event type counts:", dict(zip(*np.unique(event_type, return_counts=True))))

if __name__ == "__main__":
    main()
