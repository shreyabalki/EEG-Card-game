from pathlib import Path
from scipy.io import loadmat

# Find sessionevents01.mat under data/raw
hits = list(Path("data/raw").rglob("sessionevents01.mat"))
if not hits:
    raise FileNotFoundError("Could not find sessionevents01.mat under data/raw")

mat_file = hits[0]
print("Loading:", mat_file)

mat = loadmat(mat_file, squeeze_me=True, struct_as_record=False)

print("\nKeys in the .mat file:")
for k in sorted(mat.keys()):
    if k.startswith("__"):
        continue
    v = mat[k]
    shape = getattr(v, "shape", None)
    print(f"- {k}: type={type(v).__name__}, shape={shape}")
