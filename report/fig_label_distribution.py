import numpy as np
import json
import pandas as pd

# Paths to files
label_map_file = "data/processed/labels_6class.json"
splits = ["train", "val", "test"]

# Load label mapping
with open(label_map_file) as f:
    label_map = json.load(f)
inv_label_map = {v: k for k, v in label_map.items()}

# Count samples per class in each split
table = {}
for split in splits:
    y = np.load(f"data/processed/{split}_labels_6class.npy")
    unique, counts = np.unique(y, return_counts=True)
    for idx, cnt in zip(unique, counts):
        cls = inv_label_map[int(idx)]
        table.setdefault(cls, {})[split] = int(cnt)

# Compute totals per class
for cls in table:
    table[cls]['total'] = sum(table[cls][s] for s in splits)

# Create DataFrame
df = pd.DataFrame(table).T.fillna(0).astype(int)
df = df[['train', 'val', 'test', 'total']]
df = df.sort_values('total', ascending=False)
print(df)

# Save as CSV (optional)
df.to_csv("data/processed/class_distribution_6class.csv")

# For LaTeX report, print as LaTeX table:
print(df.to_latex(index=True, caption="Samples per class (after mapping)", label="tab:classdist"))
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt

# Paths to files
label_map_file = "data/processed/labels_6class.json"
splits = ["train", "val", "test"]

# Load label mapping
with open(label_map_file) as f:
    label_map = json.load(f)
inv_label_map = {v: k for k, v in label_map.items()}

# Count samples per class in each split
table = {}
for split in splits:
    y = np.load(f"data/processed/{split}_labels_6class.npy")
    unique, counts = np.unique(y, return_counts=True)
    for idx, cnt in zip(unique, counts):
        cls = inv_label_map[int(idx)]
        table.setdefault(cls, {})[split] = int(cnt)

# Compute totals per class
for cls in table:
    table[cls]['total'] = sum(table[cls][s] for s in splits)

# Create DataFrame
df = pd.DataFrame(table).T.fillna(0).astype(int)
df = df[['train', 'val', 'test', 'total']]
df = df.sort_values('total', ascending=False)
print(df)

# Save as CSV (optional)
df.to_csv("data/processed/class_distribution_6class.csv")

# Print as LaTeX table (for report)
print(df.to_latex(index=True, caption="Samples per class (after mapping)", label="tab:classdist"))

# --- Plotting ---
df_plot = df[['train', 'val', 'test']]
ax = df_plot.plot(kind='bar', stacked=False, figsize=(10, 6))
plt.title("Class Distribution by Split (6-class, MIT-BIH)")
plt.ylabel("Number of samples")
plt.xlabel("Class")
plt.xticks(rotation=0)
plt.legend(title="Split")
plt.tight_layout()
plt.savefig("class_distribution_6class.png", dpi=300)
plt.show()

print("Saved plot as class_distribution_6class.png")
