"""
Data cleaning pipeline for Sherlock.
Cleans raw TransXion tables (person, merchant, tx) based on audited rules:
1. person: map marital_status '2' -> 'widowed', occupations '8', '9', 'other' -> 'unknown'.
2. merchant: parse establishment_date, retain 1-to-many merchant-account relationships.
3. tx: parse datetime, sort chronologically for temporal integrity.
Saves outputs as high-performance Parquet and CSV in data/processed/.
"""

import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("SherlockCleaner")

def clean_person(raw_path: Path) -> pd.DataFrame:
    logger.info(f"Loading person from {raw_path}")
    df = pd.read_csv(raw_path)
    
    # Fix marital status '2' -> 'widowed' (median age 61)
    df["person_marital_status"] = df["person_marital_status"].replace({"2": "widowed"})
    
    # Fix unknown occupations
    df["person_occupation"] = df["person_occupation"].replace({
        "8": "unknown",
        "9": "unknown",
        "other": "unknown"
    })
    
    logger.info(f"Cleaned person table: {df.shape}")
    return df

def clean_merchant(raw_path: Path) -> pd.DataFrame:
    logger.info(f"Loading merchant from {raw_path}")
    df = pd.read_csv(raw_path)
    
    # Parse establishment_date
    df["establishment_date"] = pd.to_datetime(df["establishment_date"], errors="coerce")
    
    # Note: merchant_id c000060 has 2 distinct bank accounts (Utilities and E-commerce).
    # We intentionally preserve merchant_id to reflect 1 business entity owning multiple accounts.
    logger.info(f"Cleaned merchant table: {df.shape}")
    return df

def clean_transactions(raw_path: Path) -> pd.DataFrame:
    logger.info(f"Loading transactions from {raw_path}")
    df = pd.read_csv(raw_path)
    
    logger.info("Parsing timestamps...")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    
    logger.info("Sorting chronologically...")
    df = df.sort_values("Timestamp").reset_index(drop=True)
    
    logger.info(f"Cleaned transactions table: {df.shape}")
    return df

def main():
    base_dir = Path(__file__).resolve().parents[2]
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting Data Cleaning Pipeline. Base dir: {base_dir}")
    
    # 1. Clean Person
    person_clean = clean_person(raw_dir / "person.csv")
    person_clean.to_parquet(processed_dir / "person_clean.parquet", index=False)
    person_clean.to_csv(processed_dir / "person_clean.csv", index=False)
    logger.info("Saved person_clean (.parquet and .csv)")
    
    # 2. Clean Merchant
    merchant_clean = clean_merchant(raw_dir / "merchant.csv")
    merchant_clean.to_parquet(processed_dir / "merchant_clean.parquet", index=False)
    merchant_clean.to_csv(processed_dir / "merchant_clean.csv", index=False)
    logger.info("Saved merchant_clean (.parquet and .csv)")
    
    # 3. Clean Transactions
    tx_clean = clean_transactions(raw_dir / "tx.csv")
    tx_clean.to_parquet(processed_dir / "tx_clean.parquet", index=False)
    logger.info("Saved tx_clean.parquet")
    
    logger.info("All raw data successfully cleaned and saved to Parquet!")

if __name__ == "__main__":
    main()
