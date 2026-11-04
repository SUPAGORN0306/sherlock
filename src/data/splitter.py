"""
Temporal data splitting module for Sherlock.
Reads configs/split.yaml and splits tx_clean.parquet into:
- train.parquet
- val.parquet
- test.parquet
Ensures zero data leakage and validates temporal boundaries and fraud rates.
"""

import logging
from pathlib import Path
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("SherlockSplitter")

def main():
    base_dir = Path(__file__).resolve().parents[2]
    config_path = base_dir / "configs" / "split.yaml"
    processed_dir = base_dir / "data" / "processed"
    splits_dir = processed_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading split config from {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cfg = config["temporal_split"]
    train_end = pd.Timestamp(cfg["train_end"])
    val_start = pd.Timestamp(cfg["val_start"])
    val_end = pd.Timestamp(cfg["val_end"])
    test_start = pd.Timestamp(cfg["test_start"])
    test_end = pd.Timestamp(cfg["test_end"])

    tx_path = processed_dir / "tx_clean.parquet"
    logger.info(f"Loading cleaned transactions from {tx_path}")
    tx = pd.read_parquet(tx_path)

    logger.info("Executing temporal split...")
    train = tx[tx["Timestamp"] <= train_end].copy()
    val = tx[(tx["Timestamp"] >= val_start) & (tx["Timestamp"] <= val_end)].copy()
    test = tx[(tx["Timestamp"] >= test_start) & (tx["Timestamp"] <= test_end)].copy()

    # Integrity assertions (Leakage Prevention)
    assert train["Timestamp"].max() < val["Timestamp"].min(), "Data leakage: Train overlaps Validation!"
    assert val["Timestamp"].max() < test["Timestamp"].min(), "Data leakage: Validation overlaps Test!"
    assert len(train) + len(val) + len(test) == len(tx), "Transaction count mismatch across splits!"

    logger.info(f"Train size: {len(train):,} | Fraud: {train['Is Laundering'].sum():,} ({train['Is Laundering'].mean()*100:.4f}%)")
    logger.info(f"Val size:   {len(val):,} | Fraud: {val['Is Laundering'].sum():,} ({val['Is Laundering'].mean()*100:.4f}%)")
    logger.info(f"Test size:  {len(test):,} | Fraud: {test['Is Laundering'].sum():,} ({test['Is Laundering'].mean()*100:.4f}%)")

    # Save to parquet
    train.to_parquet(splits_dir / "train.parquet", index=False)
    val.to_parquet(splits_dir / "val.parquet", index=False)
    test.to_parquet(splits_dir / "test.parquet", index=False)
    logger.info(f"Splits saved successfully to {splits_dir}")

if __name__ == "__main__":
    main()
