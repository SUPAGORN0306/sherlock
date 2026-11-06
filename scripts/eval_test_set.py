import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)

DATA_DIR = "/home/pleum/projects/fraud-detection/sherlock/data/processed"
df = pd.read_parquet(f"{DATA_DIR}/predictions.parquet")
y_all = df["is_mule"].values
n_nodes = len(df)
indices = np.arange(n_nodes)

# 80% Train (35,329) | 10% Validation (4,416) | 10% Unseen Test (4,417)
train_idx, temp_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=y_all)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=42, stratify=y_all[temp_idx])


def get_metrics(idx, probs):
    y_true = y_all[idx]
    preds = (probs >= 0.5).astype(int)
    top100 = y_true[np.argsort(probs)[::-1][:100]].mean() * 100
    return {
        "pr_auc": round(float(average_precision_score(y_true, probs) * 100), 2),
        "roc_auc": round(float(roc_auc_score(y_true, probs) * 100), 2),
        "f1_score": round(float(f1_score(y_true, preds) * 100), 2),
        "accuracy": round(float(accuracy_score(y_true, preds) * 100), 2),
        "precision": round(float(precision_score(y_true, preds, zero_division=0) * 100), 2),
        "recall": round(float(recall_score(y_true, preds, zero_division=0) * 100), 2),
        "precision_at_100": round(float(top100), 2),
    }


xgb_p = df["xgb_prob"].values
sage_p = df["sage_prob"].values
ens_p = 0.60 * sage_p + 0.40 * xgb_p

val_res = {
    "xgboost": get_metrics(val_idx, xgb_p[val_idx]),
    "graphsage": get_metrics(val_idx, sage_p[val_idx]),
    "ensemble": get_metrics(val_idx, ens_p[val_idx]),
}

test_res = {
    "xgboost": get_metrics(test_idx, xgb_p[test_idx]),
    "graphsage": get_metrics(test_idx, sage_p[test_idx]),
    "ensemble": get_metrics(test_idx, ens_p[test_idx]),
}

benchmark = {
    "split_info": {
        "total_nodes": int(n_nodes),
        "train_nodes": int(len(train_idx)),
        "val_nodes": int(len(val_idx)),
        "test_nodes": int(len(test_idx)),
    },
    "validation_set": val_res,
    "test_set": test_res,
    "xgboost": test_res["xgboost"],
    "graphsage": test_res["graphsage"],
    "ensemble": {
        "weights": "0.60 GraphSAGE + 0.40 XGBoost",
        **test_res["ensemble"],
        "tier_distribution": df["mule_tier"].value_counts().to_dict(),
    },
}

with open(f"{DATA_DIR}/model_benchmark.json", "w", encoding="utf-8") as f:
    json.dump(benchmark, f, indent=2, ensure_ascii=False)

print("=== VALIDATION SET (4,416 accounts) ===")
for k, v in val_res.items():
    print(k, v)

print("\n=== UNSEEN TEST SET (4,417 accounts) ===")
for k, v in test_res.items():
    print(k, v)
