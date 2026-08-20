import pandas as pd
import re

harmful_path = "dataset/jbb_behaviors/harmful-behaviors.csv"
benign_path = "dataset/jbb_behaviors/benign-behaviors.csv"

h = pd.read_csv(harmful_path)
b = pd.read_csv(benign_path)


def normalize_text(text):
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


h["Behavior_normalized"] = h["Behavior"].apply(normalize_text)
b["Behavior_normalized"] = b["Behavior"].apply(normalize_text)

merged = h.merge(
    b,
    on="Index",
    suffixes=("_harmful", "_benign")
)

mismatch = merged[
    merged["Behavior_normalized_harmful"]
    != merged["Behavior_normalized_benign"]
]

print("Total paired rows:", len(merged))
print("Mismatch after normalization:", len(mismatch))

if len(mismatch) == 0:
    print("PASS: all 100 harmful/benign pairs match after normalization.")
else:
    print(mismatch[
        [
            "Index",
            "Behavior_harmful",
            "Behavior_benign",
        ]
    ].to_string(index=False))
