import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

df = pd.read_parquet("/home/pleum/projects/fraud-detection/sherlock/data/processed/predictions.parquet")
y_all = df["is_mule"].values
indices = np.arange(len(df))
train_idx, val_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=y_all)
y_val = y_all[val_idx]

models = {
    "XGBoost": df["xgb_prob"].values[val_idx],
    "GraphSAGE": df["sage_prob"].values[val_idx],
    "Hybrid (0.60 SAGE + 0.40 XGB)": 0.60 * df["sage_prob"].values[val_idx] + 0.40 * df["xgb_prob"].values[val_idx],
}

for name, probs in models.items():
    preds = (probs >= 0.5).astype(int)
    pr_auc = average_precision_score(y_val, probs) * 100
    roc_auc = roc_auc_score(y_val, probs) * 100
    f1 = f1_score(y_val, preds) * 100
    acc = accuracy_score(y_val, preds) * 100
    prec = precision_score(y_val, preds) * 100
    rec = recall_score(y_val, preds) * 100
    top100 = y_val[np.argsort(probs)[::-1][:100]].mean() * 100
    print(f"=== {name} ===")
    print(f"  PR-AUC:        {pr_auc:.2f}%")
    print(f"  ROC-AUC:       {roc_auc:.2f}%")
    print(f"  F1-Score:      {f1:.2f}%")
    print(f"  Accuracy:      {acc:.2f}%")
    print(f"  Precision:     {prec:.2f}%")
    print(f"  Recall:        {rec:.2f}%")
    print(f"  Precision@100: {top100:.2f}%")
