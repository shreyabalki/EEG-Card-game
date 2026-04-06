from pathlib import Path
import pandas as pd

IN_FILE = Path("data/processed/all_sessions_labels.csv")
OUT_FILE = Path("data/processed/all_sessions_with_delay_flag.csv")
REPORT_FILE = Path("reports/delay_flag_summary.csv")

EVENT_TABLE_FULL = 1
EVENT_PROMPT = 3

df = pd.read_csv(IN_FILE)
df = df.sort_values(["session", "trial"]).reset_index(drop=True)

df["delayed_prompt"] = False

# Track last table-full event per session
last_table_full_trial = {}

for idx, row in df.iterrows():
    sess = row["session"]

    if sess not in last_table_full_trial:
        last_table_full_trial[sess] = None

    # register table-full event
    if row["label_1"] == EVENT_TABLE_FULL and row["label_3"] == 3:
        last_table_full_trial[sess] = row["trial"]

    # flag delayed prompt if it occurs after table-full
    if (
        row["label_1"] == EVENT_PROMPT
        and last_table_full_trial[sess] is not None
        and row["trial"] > last_table_full_trial[sess]
    ):
        df.loc[idx, "delayed_prompt"] = True
        # reset so we don’t flag multiple prompts
        last_table_full_trial[sess] = None

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_FILE, index=False)

summary = (
    df.groupby("session")["delayed_prompt"]
      .agg(total_delayed="sum", total_trials="count")
      .reset_index()
)
summary["pct_delayed"] = 100 * summary["total_delayed"] / summary["total_trials"]

REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(REPORT_FILE, index=False)

print("Saved:", OUT_FILE)
print("Saved:", REPORT_FILE)
print(summary.head(10))
