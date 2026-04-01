from pathlib import Path
import pandas as pd
from collections import Counter

IN_FILE = Path("reports/table_full_following_sequences.csv")
OUT_FILE = Path("reports/table_full_sequence_patterns_top50.csv")

df = pd.read_csv(IN_FILE)

# Convert list-like strings back to tuples safely
# They were written as Python-like lists in CSV.
def parse_list(s):
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
        if not s:
            return []
        return [int(x.strip()) for x in s.split(",")]
    return []

seqs = [tuple(parse_list(x)) for x in df["next_label_1_sequence"]]

counts = Counter(seqs)
top = counts.most_common(50)

rows = []
for seq, c in top:
    rows.append({
        "count": c,
        "sequence_next10_label1": list(seq),
        "starts_with": seq[0] if len(seq) > 0 else None,
        "second": seq[1] if len(seq) > 1 else None,
        "third": seq[2] if len(seq) > 2 else None,
    })

out = pd.DataFrame(rows)
out.to_csv(OUT_FILE, index=False)
print("Saved:", OUT_FILE)
print(out.head(10))
