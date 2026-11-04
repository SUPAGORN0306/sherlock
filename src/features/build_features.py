"""
Sherlock - All-in-One Node Feature Engineering Pipeline.
Combines:
1. AML Rule Typologies (S1, S3, S5, S6, S7, S8)
2. Behavioral Drift (Recent 30d vs Previous 30d in Train)
3. Temporal & Frequency Features (Night transactions, Weekend, Cyclic Hours)
4. Demographic & Merchant Profiles (Age, Occupation, Capital, Industry)
Outputs a unified, normalized table: data/processed/node_features.parquet
"""

import logging
from pathlib import Path
import json
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("FeatureBuilder")

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
CONFIG_PATH = ROOT / "configs" / "split.yaml"
OUTPUT_PARQUET = DATA_DIR / "node_features.parquet"
OUTPUT_META = DATA_DIR / "node_features_meta.json"

def main():
    logger.info(f"Starting Unified Feature Engineering Pipeline. ROOT: {ROOT}")
    
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["temporal_split"]
    train_end = pd.Timestamp(cfg["train_end"])
    
    # 1. Load Data
    logger.info("Loading cleaned train transactions, person, and merchant data...")
    train = pd.read_parquet(DATA_DIR / "splits" / "train.parquet")
    person = pd.read_parquet(DATA_DIR / "person_clean.parquet")
    merchant = pd.read_parquet(DATA_DIR / "merchant_clean.parquet")
    
    # Target accounts
    from_acc = train["From Account"]
    to_acc = train["To Account"]
    all_accounts = pd.Index(pd.concat([from_acc, to_acc]).unique(), name="account")
    logger.info(f"Total Unique Accounts in Train: {len(all_accounts):,}")
    
    # Compute account open time (first seen in train)
    from_side = train[["From Account", "Timestamp"]].rename(columns={"From Account": "account"})
    to_side = train[["To Account", "Timestamp"]].rename(columns={"To Account": "account"})
    both_tx = pd.concat([from_side, to_side], ignore_index=True)
    open_time = both_tx.groupby("account")["Timestamp"].min()
    
    # ==========================================
    # 2. AML Rule Features (S1, S3, S5, S6, S7, S8)
    # ==========================================
    logger.info("Computing AML Rule Typologies (S1 - S8)...")
    
    # S1: New account + immediate large outflow
    t_s1 = train.copy()
    t_s1["open_time"] = t_s1["From Account"].map(open_time)
    t_s1["is_new"] = (t_s1["Timestamp"] - t_s1["open_time"]).dt.days == 0
    t_s1["date"] = t_s1["Timestamp"].dt.date
    daily_new = t_s1.loc[t_s1["is_new"]].groupby("date")["From Account"].nunique()
    daily_out = t_s1.loc[t_s1["is_new"]].groupby("date")["Amount Paid"].sum()
    s1_mask = t_s1["is_new"] & (t_s1["date"].map(daily_new).fillna(0) >= 10) & (t_s1["date"].map(daily_out).fillna(0) >= 500_000)
    s1 = t_s1.loc[s1_mask].groupby("From Account").size().rename("s1_hit_count").to_frame()
    s1["s1_hit"] = 1
    s1.index.name = "account"
    
    # S3: Rapid in-then-out (pass-through)
    t_s3 = train[["Timestamp", "To Account", "Amount Received"]].sort_values(["To Account", "Timestamp"]).set_index("Timestamp")
    rolled_in = t_s3.groupby("To Account")["Amount Received"].rolling("10D").count().reset_index().rename(columns={"Amount Received": "in_cnt_10d"})
    t_s3_reset = t_s3.reset_index().merge(rolled_in, on=["To Account", "Timestamp"], how="left")
    last_in = t_s3_reset.groupby("To Account")["Timestamp"].max().rename("last_in_ts")
    out_s3 = train[["Timestamp", "From Account"]].rename(columns={"From Account": "To Account"}).merge(last_in, left_on="To Account", right_index=True)
    out_s3["gap"] = (out_s3["Timestamp"] - out_s3["last_in_ts"]).dt.days
    out_after = out_s3[(out_s3["gap"] >= 0) & (out_s3["gap"] <= 2)].groupby("To Account").size().rename("out_within_2d_after_in")
    s3 = t_s3_reset.groupby("To Account").agg(s3_max_in_cnt_10d=("in_cnt_10d", "max")).join(out_after, how="left").fillna(0)
    s3["s3_hit"] = ((s3["s3_max_in_cnt_10d"] >= 20) & (s3["out_within_2d_after_in"] > 0)).astype(int)
    s3.index.name = "account"
    
    # S6: Fast in-out with low end balance (3-day rolling)
    t_s6 = train[["Timestamp", "From Account", "Amount Paid", "Amount Received"]].sort_values(["From Account", "Timestamp"]).set_index("Timestamp")
    g_s6 = t_s6.groupby("From Account")
    amt_in_3d = g_s6["Amount Received"].rolling("3D").sum().reset_index(drop=True)
    amt_out_3d = g_s6["Amount Paid"].rolling("3D").sum().reset_index(drop=True)
    max_3d = g_s6[["Amount Received", "Amount Paid"]].rolling("3D").max().max(axis=1).reset_index(drop=True)
    df_s6 = pd.DataFrame({"account": t_s6["From Account"].values, "in_3d": amt_in_3d.values, "out_3d": amt_out_3d.values, "max_3d": max_3d.values})
    df_s6["ratio_3d"] = df_s6["in_3d"] / (df_s6["out_3d"] + 1e-6)
    df_s6["bal"] = (df_s6["in_3d"] - df_s6["out_3d"]).abs()
    df_s6["hit"] = df_s6["ratio_3d"].between(0.9, 1.1) & (df_s6["max_3d"] >= 200_000) & (df_s6["bal"] <= 0.01 * df_s6["max_3d"])
    s6 = df_s6.groupby("account").agg(s6_max_ratio=("ratio_3d", "max"), s6_hit=("hit", "max"))
    s6["s6_hit"] = s6["s6_hit"].astype(int)
    
    # S7: Structuring signals
    t_s7_in = train[["Timestamp", "To Account", "Amount Received"]].sort_values(["To Account", "Timestamp"])
    s7_same_in = t_s7_in.groupby(["To Account", "Amount Received"]).size().groupby(level=0).max().rename("s7_same_amount_cnt")
    t_s7_out = train[["Timestamp", "From Account", "Amount Paid"]].sort_values(["From Account", "Timestamp"])
    s7_same_out = t_s7_out.groupby(["From Account", "Amount Paid"]).size().groupby(level=0).max().rename("s7_same_out_cnt")
    s7 = pd.concat([s7_same_in, s7_same_out], axis=1).fillna(0)
    s7.index.name = "account"

    # S8: In/out counts & amounts & counterparties
    in_stats = train.groupby("To Account").agg(in_tx_cnt=("Amount Received", "count"), in_amt_sum=("Amount Received", "sum"), in_amt_mean=("Amount Received", "mean"), in_amt_max=("Amount Received", "max"), unique_senders=("From Account", "nunique"))
    out_stats = train.groupby("From Account").agg(out_tx_cnt=("Amount Paid", "count"), out_amt_sum=("Amount Paid", "sum"), out_amt_mean=("Amount Paid", "mean"), out_amt_max=("Amount Paid", "max"), unique_receivers=("To Account", "nunique"))
    s8 = pd.concat([in_stats, out_stats], axis=1).fillna(0)
    s8["in_out_cnt_ratio"] = s8["in_tx_cnt"] / (s8["out_tx_cnt"] + 1e-6)
    s8["in_out_amt_ratio"] = s8["in_amt_sum"] / (s8["out_amt_sum"] + 1e-6)
    s8["total_unique_counterparties"] = s8["unique_senders"] + s8["unique_receivers"]
    s8.index.name = "account"
    
    # ==========================================
    # 3. Behavioral Drift (Recent 30d vs Prev 30d)
    # ==========================================
    logger.info("Computing Behavioral Drift Features...")
    recent_start = train_end - pd.Timedelta(days=30)
    prev_start = train_end - pd.Timedelta(days=60)
    
    rec_tx = train[train["Timestamp"] > recent_start]
    prev_tx = train[(train["Timestamp"] > prev_start) & (train["Timestamp"] <= recent_start)]
    
    rec_cnt = rec_tx.groupby("From Account").size().add(rec_tx.groupby("To Account").size(), fill_value=0)
    prev_cnt = prev_tx.groupby("From Account").size().add(prev_tx.groupby("To Account").size(), fill_value=0)
    rec_amt = rec_tx.groupby("From Account")["Amount Paid"].sum().add(rec_tx.groupby("To Account")["Amount Received"].sum(), fill_value=0)
    prev_amt = prev_tx.groupby("From Account")["Amount Paid"].sum().add(prev_tx.groupby("To Account")["Amount Received"].sum(), fill_value=0)
    
    drift = pd.DataFrame(index=all_accounts)
    drift["drift_txn_cnt"] = np.log1p((rec_cnt + 1) / (prev_cnt + 1)).fillna(0)
    drift["drift_amt_sum"] = np.log1p((rec_amt + 1) / (prev_amt + 1)).fillna(0)

    # ==========================================
    # 4. Temporal & Frequency Features
    # ==========================================
    logger.info("Computing Temporal & Frequency Features (Night, Weekend, Cycles)...")
    long_tx = both_tx.copy()
    long_tx["hour"] = long_tx["Timestamp"].dt.hour
    long_tx["dow"] = long_tx["Timestamp"].dt.dayofweek
    long_tx["is_night"] = (long_tx["hour"] < 6).astype(int)
    long_tx["is_weekend"] = (long_tx["dow"] >= 5).astype(int)
    long_tx["hour_sin"] = np.sin(2 * np.pi * long_tx["hour"] / 24.0)
    long_tx["hour_cos"] = np.cos(2 * np.pi * long_tx["hour"] / 24.0)
    long_tx["date"] = long_tx["Timestamp"].dt.date
    
    g_time = long_tx.groupby("account")
    time_feats = pd.DataFrame({
        "night_txn_ratio": g_time["is_night"].mean(),
        "weekend_txn_ratio": g_time["is_weekend"].mean(),
        "hour_sin_mean": g_time["hour_sin"].mean(),
        "hour_cos_mean": g_time["hour_cos"].mean(),
        "active_days": g_time["date"].nunique(),
        "days_since_open": (g_time["Timestamp"].max() - g_time["Timestamp"].min()).dt.days
    })
    time_feats["days_since_open_norm"] = time_feats["days_since_open"] / 365.0
    time_feats["txn_per_active_day"] = g_time.size() / (time_feats["active_days"] + 1e-6)

    # ==========================================
    # 5. Entity Demographics (Person & Merchant)
    # ==========================================
    logger.info("Enriching with Person & Merchant profiles...")
    p = person.copy().rename(columns={"bank_account_number": "account"})
    p["person_age_norm"] = ((p["person_age"] - 18) / (78 - 18)).clip(0, 1)
    p["is_senior"] = (p["person_age"] >= 60).astype(int)
    p["is_minor"] = (p["person_age"] < 20).astype(int)
    p_occ = pd.get_dummies(p["person_occupation"], prefix="occ", dtype=float)
    p_feats = pd.concat([p[["account", "person_age_norm", "is_senior", "is_minor"]], p_occ], axis=1).set_index("account")
    p_feats["is_person"] = 1.0
    p_feats["is_merchant"] = 0.0

    m = merchant.copy().rename(columns={"bank_account_number": "account"})
    log_cap = np.log1p(m["registered_capital"])
    m["capital_norm"] = (log_cap - log_cap.mean()) / (log_cap.std() + 1e-6)
    m_typ = pd.get_dummies(m["type"], prefix="mtype", dtype=float)
    m_ind = pd.get_dummies(m["industry"], prefix="ind", dtype=float)
    m_feats = pd.concat([m[["account", "capital_norm"]], m_typ, m_ind], axis=1).set_index("account")
    m_feats["is_person"] = 0.0
    m_feats["is_merchant"] = 1.0

    entity_feats = pd.concat([p_feats, m_feats], axis=0).fillna(0)
    # Handle multi-account per merchant if any
    entity_feats = entity_feats[~entity_feats.index.duplicated(keep="first")]

    # ==========================================
    # 6. Master Merge & Normalization
    # ==========================================
    logger.info("Master merging all features onto account universe...")
    feat_df = pd.DataFrame(index=all_accounts)
    for block in [s1, s3, s6, s7, s8, drift, time_feats, entity_feats]:
        feat_df = feat_df.join(block, how="left")
    feat_df = feat_df.fillna(0)
    
    # Target label: 1 if account was ever in laundering in train
    fraud_accounts = set(train[train["Is Laundering"] == 1]["From Account"]) | set(train[train["Is Laundering"] == 1]["To Account"])
    feat_df["is_mule"] = feat_df.index.isin(fraud_accounts).astype(int)
    
    logger.info(f"Normalizing heavy monetary & count features... (Total accounts: {len(feat_df):,})")
    num_cols = [c for c in feat_df.columns if c != "is_mule"]
    for col in num_cols:
        if feat_df[col].max() > 10:
            signed_log = np.sign(feat_df[col]) * np.log1p(np.abs(feat_df[col]))
            mu, sigma = signed_log.mean(), signed_log.std()
            feat_df[col] = (signed_log - mu) / (sigma + 1e-6)
            
    # Verification
    assert not feat_df.isna().any().any(), "Found NaNs in features!"
    finite_mask = np.isfinite(feat_df.to_numpy(dtype=float))
    assert finite_mask.all(), "Found infinite values in features!"
    
    feat_df = feat_df.reset_index()
    feat_df.to_parquet(OUTPUT_PARQUET, index=False)
    
    meta = {
        "n_accounts": int(len(feat_df)),
        "n_features": int(len(feat_df.columns) - 2),  # excluding account and is_mule
        "n_mules": int(feat_df["is_mule"].sum()),
        "feature_columns": [c for c in feat_df.columns if c not in ("account", "is_mule")]
    }
    with OUTPUT_META.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        
    logger.info(f"SUCCESS! Saved unified node features to {OUTPUT_PARQUET}")
    logger.info(f"Accounts: {len(feat_df):,} | Features: {len(meta['feature_columns'])} | Mules in Train: {meta['n_mules']:,}")

if __name__ == "__main__":
    main()
