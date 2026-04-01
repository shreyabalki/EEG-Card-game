from scipy.io import loadmat
import numpy as np
import pandas as pd

def load_session(path):
    mat = loadmat(path)
    t = mat["t"]                      # (time, 1)
    data = mat["data"]               # (time, chan, players, events)
    labels = mat["labels"]           # (events, 8)

    cols = ["session", "current", "last", "solo",
            "cardValue", "countP1", "countP2", "countP3"]
    labels_df = pd.DataFrame(labels, columns=cols)
    return t, data, labels_df
