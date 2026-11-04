"""
Sherlock - Dual Model Training & 5-Tier Mule Calibration
Models:
1. Baseline: XGBoost with Class-Weight Scaling & Feature Importance
2. Champion: GraphSAGE (PyTorch Geometric SAGEConv on GPU/CPU)

Outputs:
- data/processed/predictions.parquet (All accounts scored + 5-tier mule classification)
- data/processed/model_benchmark.json (Benchmark metrics)
- models/graphsage.pt (Saved PyG model weights)
"""

import logging
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("ModelTrainer")

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

class GraphSAGEClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=1, dropout=0.2):
        super().__init__()
        # SAGEConv with mean aggregator and self-concatenation
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
        self.conv2 = SAGEConv(hidden_channels, hidden_channels // 2, aggr="mean")
        self.fc = nn.Linear(hidden_channels // 2, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        out = self.fc(x)
        return out.squeeze(-1)

def main():
    logger.info("Starting Dual Model Training & Benchmark Pipeline...")
    
    # 1. Load Preprocessed Features and Graph Edges
    logger.info("Loading node features and transactions...")
    node_df = pd.read_parquet(DATA_DIR / "node_features.parquet")
    train_tx = pd.read_parquet(DATA_DIR / "splits" / "train.parquet")
    
    accounts = node_df["account"].values
    account_to_idx = {acc: i for i, acc in enumerate(accounts)}
    y_all = node_df["is_mule"].values
    
    feature_cols = [c for c in node_df.columns if c not in ("account", "is_mule")]
    X_all = node_df[feature_cols].values
    n_nodes, in_dim = X_all.shape
    n_mules = int(y_all.sum())
    logger.info(f"Nodes: {n_nodes:,} | Features: {in_dim} | Mules: {n_mules:,} ({n_mules/n_nodes*100:.2f}%)")
    
    # Stratified Train/Val split
    indices = np.arange(n_nodes)
    train_idx, val_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=y_all)
    logger.info(f"Train Nodes: {len(train_idx):,} | Val Nodes: {len(val_idx):,}")
    
    # =======================================================================
    # MODEL 1: XGBoost Baseline
    # =======================================================================
    logger.info("\n" + "="*50)
    logger.info("TRAINING MODEL 1: XGBoost Baseline")
    logger.info("="*50)
    
    scale_pos = (len(train_idx) - y_all[train_idx].sum()) / (y_all[train_idx].sum() + 1e-6)
    logger.info(f"XGBoost scale_pos_weight: {scale_pos:.2f}")
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        scale_pos_weight=scale_pos,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )
    
    xgb_model.fit(
        X_all[train_idx], y_all[train_idx],
        eval_set=[(X_all[val_idx], y_all[val_idx])],
        verbose=False
    )
    
    xgb_val_probs = xgb_model.predict_proba(X_all[val_idx])[:, 1]
    xgb_pr_auc = average_precision_score(y_all[val_idx], xgb_val_probs)
    xgb_roc_auc = roc_auc_score(y_all[val_idx], xgb_val_probs)
    xgb_f1 = f1_score(y_all[val_idx], (xgb_val_probs >= 0.5).astype(int))
    top100_idx = np.argsort(xgb_val_probs)[::-1][:100]
    xgb_p100 = y_all[val_idx][top100_idx].mean() * 100
    
    logger.info(f"XGBoost Validation PR-AUC:  {xgb_pr_auc*100:.2f}%")
    logger.info(f"XGBoost Validation ROC-AUC: {xgb_roc_auc*100:.2f}%")
    logger.info(f"XGBoost Validation F1-Score: {xgb_f1*100:.2f}%")
    logger.info(f"XGBoost Precision@Top-100:  {xgb_p100:.2f}%")
    
    # =======================================================================
    # MODEL 2: GraphSAGE Champion (PyTorch Geometric)
    # =======================================================================
    logger.info("\n" + "="*50)
    logger.info("TRAINING MODEL 2: GraphSAGE Champion (PyTorch Geometric)")
    logger.info("="*50)
    
    src = train_tx["From Account"].map(account_to_idx)
    dst = train_tx["To Account"].map(account_to_idx)
    valid_mask = src.notna() & dst.notna()
    edge_index = torch.tensor(
        np.array([src[valid_mask].astype(int).values, dst[valid_mask].astype(int).values]),
        dtype=torch.long
    )
    logger.info(f"Graph Edge Index constructed: {edge_index.shape[1]:,} edges")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using Compute Device: {device}")
    
    x_tensor = torch.tensor(X_all, dtype=torch.float, device=device)
    y_tensor = torch.tensor(y_all, dtype=torch.float, device=device)
    edge_index = edge_index.to(device)
    
    train_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    
    sage_model = GraphSAGEClassifier(in_channels=in_dim, hidden_channels=64, out_channels=1, dropout=0.2).to(device)
    optimizer = torch.optim.AdamW(sage_model.parameters(), lr=0.01, weight_decay=1e-4)
    pos_weight = torch.tensor([scale_pos], dtype=torch.float, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    epochs = 60
    logger.info(f"Training GraphSAGE on {device} for {epochs} epochs...")
    
    best_val_pr = 0.0
    for epoch in range(1, epochs + 1):
        sage_model.train()
        optimizer.zero_grad()
        out = sage_model(x_tensor, edge_index)
        loss = criterion(out[train_mask], y_tensor[train_mask])
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == epochs:
            sage_model.eval()
            with torch.no_grad():
                val_out = sage_model(x_tensor, edge_index)
                val_probs = torch.sigmoid(val_out[val_mask]).cpu().numpy()
                val_pr = average_precision_score(y_all[val_idx], val_probs)
                logger.info(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {loss.item():.4f} | Val PR-AUC: {val_pr*100:.2f}%")
                
    # Final GraphSAGE Evaluation on Validation
    sage_model.eval()
    with torch.no_grad():
        all_logits = sage_model(x_tensor, edge_index)
        sage_all_probs = torch.sigmoid(all_logits).cpu().numpy()
        
    sage_val_probs = sage_all_probs[val_idx]
    sage_pr_auc = average_precision_score(y_all[val_idx], sage_val_probs)
    sage_roc_auc = roc_auc_score(y_all[val_idx], sage_val_probs)
    sage_f1 = f1_score(y_all[val_idx], (sage_val_probs >= 0.5).astype(int))
    sage_top100_idx = np.argsort(sage_val_probs)[::-1][:100]
    sage_p100 = y_all[val_idx][sage_top100_idx].mean() * 100
    
    logger.info(f"GraphSAGE Validation PR-AUC:  {sage_pr_auc*100:.2f}%")
    logger.info(f"GraphSAGE Validation ROC-AUC: {sage_roc_auc*100:.2f}%")
    logger.info(f"GraphSAGE Validation F1-Score: {sage_f1*100:.2f}%")
    logger.info(f"GraphSAGE Precision@Top-100:  {sage_p100:.2f}%")
    
    # Save GraphSAGE Weights
    torch.save(sage_model.state_dict(), MODEL_DIR / "graphsage.pt")
    logger.info(f"Saved GraphSAGE model weights to {MODEL_DIR / 'graphsage.pt'}")
    
    # =======================================================================
    # 3. Ensemble Scoring & 5-Tier Mule Risk Calibration
    # =======================================================================
    logger.info("\n" + "="*50)
    logger.info("ENSEMBLE SCORING & 5-TIER MULE CALIBRATION")
    logger.info("="*50)
    
    xgb_all_probs = xgb_model.predict_proba(X_all)[:, 1]
    
    # Weighted ensemble: 0.60 GraphSAGE (Graph Structural Power) + 0.40 XGBoost (Tabular Precision)
    ensemble_probs = 0.60 * sage_all_probs + 0.40 * xgb_all_probs
    risk_score_100 = np.clip(np.round(ensemble_probs * 100, 1), 0, 100)
    
    def assign_tier(row):
        score = row["risk_score"]
        actual_mule = row["is_mule"]
        
        if actual_mule == 1 and score >= 80:
            return "ม้าดำ (Blacklist ปปง.)", "black"
        elif score >= 70:
            return "ม้าเทาเข้ม (มีผู้เสียหาย แจ้งความแล้ว)", "dark_grey"
        elif score >= 50:
            return "ม้าเทาอ่อน (มีผู้เสียหาย ยังไม่แจ้งความ)", "light_grey"
        elif score >= 35:
            return "ม้าน้ำตาลเข้ม (สงสัยระดับสูง)", "dark_brown"
        elif score >= 20:
            return "ม้าน้ำตาลอ่อน (เริ่มสงสัย พฤติกรรมแปลก)", "light_brown"
        else:
            return "ปกติ (ความเสี่ยงต่ำ)", "normal"
            
    pred_df = pd.DataFrame({
        "account": accounts,
        "risk_score": risk_score_100,
        "is_mule": y_all,
        "sage_prob": np.round(sage_all_probs, 4),
        "xgb_prob": np.round(xgb_all_probs, 4)
    })
    
    tiers = pred_df.apply(assign_tier, axis=1)
    pred_df["mule_tier"] = [t[0] for t in tiers]
    pred_df["mule_color"] = [t[1] for t in tiers]
    
    logger.info("\n5-Tier Mule Distribution across All Accounts:")
    print(pred_df["mule_tier"].value_counts())
    
    pred_df.to_parquet(DATA_DIR / "predictions.parquet", index=False)
    logger.info(f"Saved predictions to {DATA_DIR / 'predictions.parquet'}")
    
    # Save Benchmark JSON
    benchmark = {
        "xgboost": {
            "pr_auc": round(float(xgb_pr_auc) * 100, 2),
            "roc_auc": round(float(xgb_roc_auc) * 100, 2),
            "f1_score": round(float(xgb_f1) * 100, 2),
            "precision_at_100": round(float(xgb_p100), 2)
        },
        "graphsage": {
            "pr_auc": round(float(sage_pr_auc) * 100, 2),
            "roc_auc": round(float(sage_roc_auc) * 100, 2),
            "f1_score": round(float(sage_f1) * 100, 2),
            "precision_at_100": round(float(sage_p100), 2)
        },
        "ensemble": {
            "weights": "0.60 GraphSAGE + 0.40 XGBoost",
            "tier_distribution": pred_df["mule_tier"].value_counts().to_dict()
        }
    }
    with open(DATA_DIR / "model_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Saved benchmark metrics to {DATA_DIR / 'model_benchmark.json'}")
    logger.info("ALL TRAINING & BENCHMARK TASKS COMPLETE!")

if __name__ == "__main__":
    main()
