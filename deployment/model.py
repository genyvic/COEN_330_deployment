"""
model.py - the final model (Random Forest) and the "is it worth it?" rule.

The model is the one chosen in the report: one Random Forest that predicts both
outputs, with the hyperparameters from the Random Forest notebook, trained on
all 252 training rows (dataset_B_final.csv).

The trained model is saved in rf_model.joblib. If that file is missing or was
made with another scikit-learn version, the model is simply trained again from
dataset_B_final.csv (about 2 seconds, same result thanks to random_state=42).
"""
import os

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor

from features import FEATURES

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(HERE, "dataset_B_final.csv")
MODEL_FILE = os.path.join(HERE, "rf_model.joblib")
TARGETS = ["energy_saved_pct", "extra_time_pct"]

# Test MAE of this model in the report (used to show how precise a prediction is)
TEST_MAE_ENERGY = 3.85
TEST_MAE_TIME = 6.64


def train_model():
    df = pd.read_csv(TRAIN_CSV)
    model = RandomForestRegressor(n_estimators=400, random_state=42, oob_score=True,
                                  max_depth=5, min_samples_leaf=5)
    model.fit(df[FEATURES], df[TARGETS])
    return model


def load_model():
    """Loads rf_model.joblib, or trains the model again if needed."""
    try:
        saved = joblib.load(MODEL_FILE)
        if saved["sklearn_version"] == sklearn.__version__ and saved["features"] == FEATURES:
            return saved["model"]
    except Exception:
        pass
    model = train_model()
    try:
        joblib.dump({"model": model, "features": FEATURES,
                     "sklearn_version": sklearn.__version__}, MODEL_FILE)
    except OSError:
        pass                      # read-only folder (e.g. some hosts): just keep it in memory
    return model


def training_means():
    """Average of each input in the training data. Used when a value can't be
    measured (e.g. power and temperature inside a virtual machine)."""
    return pd.read_csv(TRAIN_CSV)[FEATURES].mean().to_dict()


def fill_missing(features, means=None):
    """Replaces missing values by the training average. Returns (filled, names of missing)."""
    means = means or training_means()
    missing = [k for k in FEATURES if features.get(k) is None or not np.isfinite(features[k])]
    filled = {k: (means[k] if k in missing else float(features[k])) for k in FEATURES}
    return filled, missing


def predict(model, features):
    """features = {name: value} for the 23 inputs -> (energy saved %, extra time %)."""
    x = pd.DataFrame([[features[k] for k in FEATURES]], columns=FEATURES)
    energy, time = model.predict(x)[0]
    return float(energy), float(time)


# ------------------------------------------------------------------ the decision
ENERGY_MIN = 5      # below this, the saving could just be the model's error
TIME_FREE = 15      # below this, the program is basically not slower
TIME_MAX = 100      # above this, the program takes more than twice as long


def verdict(energy, time):
    """3-class answer: (short label, emoji, one-sentence explanation)."""
    if energy < ENERGY_MIN:
        return ("Not worth it", "❌",
                f"It saves almost no energy (less than {ENERGY_MIN}%).")
    if time > TIME_MAX:
        return ("Not worth it", "❌",
                f"It saves energy, but the program takes {1 + time / 100:.1f}× longer.")
    if time <= TIME_FREE:
        return ("Worth it", "✅",
                "It saves energy and the program is basically not slower.")
    return ("Worth it if you're not in a hurry", "⚖️",
            f"It saves energy, but the program takes {1 + time / 100:.1f}× longer.")
