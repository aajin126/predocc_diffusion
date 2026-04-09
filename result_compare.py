import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


MODEL_FOLDER = os.path.join(ROOT, "predocc_diffusion", "output", "ldm2.0_compare") 


csv_files = sorted(glob.glob(os.path.join(MODEL_FOLDER, "eval_table_iou_0.*.csv")))

if not csv_files:
    print(f"No eval_table_iou_0.*.csv files found in {MODEL_FOLDER}")
    exit(1)

print(f"Found {len(csv_files)} files:")
for f in csv_files:
    print(f"  {os.path.basename(f)}")

plt.figure(figsize=(12, 6))

for csv_path in csv_files:

    basename = os.path.basename(csv_path)
    threshold = basename.replace("eval_table_iou_", "").replace(".csv", "")
    
    df = pd.read_csv(csv_path)

    step_cols = sorted([c for c in df.columns if c.startswith("n=")],
                       key=lambda s: int(s.split("=")[1]))
    
    x = np.arange(1, len(step_cols) + 1)
    mean = df[step_cols].mean(axis=0).to_numpy()
    
    plt.plot(x, mean, marker="o", linewidth=2, label=f"{threshold}")

plt.xlabel("Prediction time steps", fontsize=12)
plt.ylabel("Average IoU", fontsize=12)
plt.xticks(x)
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()

out_png = os.path.join(MODEL_FOLDER, "iou_threshold_comparison.png")
plt.savefig(out_png, dpi=500)
plt.show()
print("Saved:", out_png)
