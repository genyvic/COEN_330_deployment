"""
verify.py - checks that the deployed tool gives the same results as the report.

  1. The 23 features computed by features.py from the raw test measurements
     are the same as in dataset_B_test_final.csv (made by the preprocessing notebook).
  2. The Random Forest gives the same test MAE as in the report (3.85 / 6.64).

Run:  python verify.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from features import FEATURES, compute_features, raw_from_collected_row  # noqa: E402
import model as m  # noqa: E402

raw = pd.read_csv(os.path.join(HERE, "tests", "combined_test.csv"))
final = pd.read_csv(os.path.join(HERE, "tests", "dataset_B_test_final.csv"))

# ---- 1. features
ok = True
for i, row in raw.iterrows():
    mine = compute_features(raw_from_collected_row(row.to_dict()))
    for k in FEATURES:
        a, b = mine[k], final.loc[i, k]
        if a is None or not np.isclose(a, b, rtol=1e-9, atol=1e-9):
            ok = False
            print(f"  different: row {i} ({row['workload']}) {k}: {a} vs {b}")
print("1. Features match the preprocessing notebook:", "YES" if ok else "NO")

# ---- 2. model
model = m.load_model()
pred = model.predict(final[FEATURES])
mae_e = mean_absolute_error(final["energy_saved_pct"], pred[:, 0])
mae_t = mean_absolute_error(final["extra_time_pct"], pred[:, 1])
# Another scikit-learn version builds slightly different random trees, so allow 0.1
same = abs(mae_e - m.TEST_MAE_ENERGY) <= 0.1 and abs(mae_t - m.TEST_MAE_TIME) <= 0.1
print(f"2. Test MAE: energy {mae_e:.2f}, extra time {mae_t:.2f} "
      f"(report: {m.TEST_MAE_ENERGY} / {m.TEST_MAE_TIME}, within 0.1):", "YES" if same else "NO")

# ---- the answers on the 15 test runs
print("\nPredictions on the test apps:")
for i, row in final.iterrows():
    e, t = pred[i]
    label, emoji, _ = m.verdict(e, t)
    real, _, _ = m.verdict(row["energy_saved_pct"], row["extra_time_pct"])
    print(f"  {row['workload']:20s} predicted {e:6.1f}% / {t:6.1f}%  {emoji} {label:34s} (real: {real})")

sys.exit(0 if ok and same else 1)
