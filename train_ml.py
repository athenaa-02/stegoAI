from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier


def base_id_from_path(p: str) -> str:
    """
    Convert:
      cover/00005.png -> 00005
      stego/00005_stego.png -> 00005
    """
    name = Path(p).name
    name = re.sub(r"_stego$", "", Path(name).stem)  # remove suffix if present
    return name

def pairwise_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """
    Split by base image id, so cover+stego stay together.
    """
    groups = df["base_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    n_test = int(round(len(groups) * test_size))
    test_groups = set(groups[:n_test])

    test_mask = df["base_id"].isin(test_groups)
    train_df = df[~test_mask].copy()
    test_df = df[test_mask].copy()
    return train_df, test_df

def main() -> None:
    base = Path(__file__).resolve().parent
    features_path = base / "dataset" / "features.csv"
    if not features_path.exists():
        raise SystemExit(f"Missing: {features_path}")

    df = pd.read_csv(features_path)

    # If your CSV ever misses headers, stop here and tell me.
    if "path" not in df.columns or "label" not in df.columns:
        raise SystemExit("features.csv must have headers including 'path' and 'label'.")

    df["base_id"] = df["path"].astype(str).apply(base_id_from_path)

    train_df, test_df = pairwise_split(df, test_size=0.2, seed=42)

    y_train = train_df["label"].astype(int).values
    y_test = test_df["label"].astype(int).values

    X_train = train_df.drop(columns=[c for c in ["path", "label", "base_id"] if c in train_df.columns]).astype(float).values
    X_test  = test_df.drop(columns=[c for c in ["path", "label", "base_id"] if c in test_df.columns]).astype(float).values

    model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print("=== Random Forest (pair-wise split) ===")

    print(f"Rows: total={len(df)}, train={len(train_df)}, test={len(test_df)}")
    print(f"Unique base images: total={df['base_id'].nunique()}, test={test_df['base_id'].nunique()}")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")
    print("\nConfusion matrix [ [TN FP]\n                  [FN TP] ]")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Feature weights
    # feature_names = list(train_df.drop(columns=[c for c in ["path", "label", "base_id"] if c in train_df.columns]).columns)
    # coefs = model.named_steps["clf"].coef_[0]
    # order = np.argsort(np.abs(coefs))[::-1]
    # # print("\nTop 12 features by |weight|:")
    # for i in order[:12]:
    #     print(f"{feature_names[i]:>26s}  weight={coefs[i]: .6f}")

if __name__ == "__main__":
    main()
