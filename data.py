# data: shape (T, C, P, E)
# labels_df has columns: current, last (player ids 1..P)

import numpy as np

X_list = []
y_list = []

for e in range(E):
    current = labels_df.iloc[e]["current"]
    last    = labels_df.iloc[e]["last"]

    for p in range(1, P + 1):
        eeg = data[:, :, p - 1, e]          # (T, C)
        eeg = np.nan_to_num(eeg)            # replace NaN/Inf safely
        eeg = np.transpose(eeg)             # (C, T)

        if p == last:
            role = "played"
        elif p == current:
            role = "next"
        else:
            role = "observer"

        X_list.append(eeg)
        y_list.append(role)

X = np.stack(X_list, axis=0)                # (N, C, T)
y = np.array(y_list)                        # (N,)
X_dl = X[..., None]                         # (N, C, T, 1) for CNN/EEGNet
