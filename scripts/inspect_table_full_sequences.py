from pathlib import Path
import pandas as pd

IN_FILE = Path("data/processed/all_sessions_labels.csv")
OUT_DIR = Path("reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# From your manual check:
EVENT_TABLE_FULL = 1      # label_1 when label_3 == 3 (3rd card)
EVENT_PROMPT = 3          # prompt event code you found

LOOKAHEAD = 80            # how many events to inspect after table-full

df = pd.read_csv(IN_FILE)
df = df.sort_values(["session", "trial"]).reset_index(drop=True)

rows = []

# Find all "table full" events (3rd card)
table_full = df[(df["label_1"] == EVENT_TABLE_FULL) & (df["label_3"] == 3)].copy()

for _, r in table_full.iterrows():
    sess = int(r["session"])
    trial0 = int(r["trial"])

    s = df[df["session"] == sess]
    # get rows after trial0
    after = s[s["trial"] > trial0].head(LOOKAHEAD)

    seq = after["label_1"].tolist()
    seq_trials = after["trial"].tolist()

    # where does prompt appear in this window?
    try:
        prompt_index = seq.index(EVENT_PROMPT) + 1  # 1-based offset
    except ValueError:
        prompt_index = None

    rows.append({
        "session": sess,
        "table_full_trial": trial0,
        "next_trials": seq_trials,
        "next_label_1_sequence": seq,
        "prompt_offset_within_lookahead": prompt_index
    })

out = pd.DataFrame(rows)
out_file = OUT_DIR / "table_full_following_sequences.csv"
out.to_csv(out_file, index=False)

print("Saved:", out_file)
print("How often prompt appears within lookahead:")
print(out["prompt_offset_within_lookahead"].value_counts(dropna=False).sort_index())
