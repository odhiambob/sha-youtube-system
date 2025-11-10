# prepare_labels.py
import pandas as pd

SRC = "labels_log.csv"          # appended by your app
DST = "new_training_data.csv"   # clean file for fine-tuning

df = pd.read_csv(SRC)

# keep only rows where you picked a human label
df = df[df["human_label"].isin(["negative", "neutral", "positive"])].copy()

# minimal schema: text, label
out = df[["text", "human_label"]].rename(columns={"human_label": "label"})

# drop empties/duplicates
out = out.dropna()
out = out.drop_duplicates(subset=["text", "label"])

out.to_csv(DST, index=False, encoding="utf-8")
print(f"✅ Saved {len(out)} rows to {DST}")
