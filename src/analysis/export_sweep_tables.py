"""
Export sweep_thirds results to CSV tables.
Output: results/analysis/tables/<alias>_sweep.csv + all_models_sweep.csv
"""

import json
import csv
from pathlib import Path

SWEEP_DIR = Path("results/sweep_thirds")
OUT_DIR = Path("results/analysis/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIELDS = [
    "alias", "layer_idx", "n_layers",
    "overall_accuracy",
    "ambig_accuracy", "ambig_bias_score", "ambig_bias_direction",
    "disambig_accuracy", "disambig_bias_score", "disambig_bias_direction",
]

all_rows = []

for model_dir in sorted(SWEEP_DIR.iterdir()):
    if not model_dir.is_dir():
        continue
    alias = model_dir.name
    rows = []

    for f in sorted(model_dir.glob("sweep_L*_bbq.json")):
        d = json.loads(f.read_text())
        m = d["metrics"]
        n_layers = d.get("n_layers", None)
        layer_idx = d.get("layer_idx", None)

        row = {
            "alias": alias,
            "layer_idx": layer_idx,
            "n_layers": n_layers,
            "overall_accuracy": m["overall"]["accuracy"],
            "ambig_accuracy": m["ambig"]["accuracy"],
            "ambig_bias_score": m["ambig"]["bias_score"],
            "ambig_bias_direction": m["ambig"]["bias_direction"],
            "disambig_accuracy": m["disambig"]["accuracy"],
            "disambig_bias_score": m["disambig"]["bias_score"],
            "disambig_bias_direction": m["disambig"]["bias_direction"],
        }
        rows.append(row)

    if not rows:
        print(f"  [skip] {alias} — no files")
        continue

    rows.sort(key=lambda r: r["layer_idx"])

    out_path = OUT_DIR / f"{alias}_sweep.csv"
    with open(out_path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  {alias}: {len(rows)} layers → {out_path.name}")
    all_rows.extend(rows)

# Combined
all_path = OUT_DIR / "all_models_sweep.csv"
with open(all_path, "w", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(all_rows)

print(f"\nDone. {len(all_rows)} total rows → {all_path}")
