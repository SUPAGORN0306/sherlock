"""
Sherlock - Part 1: Overview, Behavioral, Demographic & Graph Hub EDA
Analyzes:
1. Target Distribution & Mule Roles (Sender, Receiver, Bridge Relay)
2. Payment Format Breakdown (Fraud vs Normal)
3. Cross-Bank vs Intra-Bank Transfers
4. Amount Comparison & Smurfing Patterns
5. Demographics & Occupations (Person Mules)
6. Merchant & Corporate Mules
7. Network Hubs (Top Inbound / Outbound Fan-in & Fan-out)
"""

import pandas as pd
from pathlib import Path

base_dir = Path("/home/pleum/projects/fraud-detection/sherlock")
train_path = base_dir / "data/processed/splits/train.parquet"
person_path = base_dir / "data/processed/person_clean.parquet"
merchant_path = base_dir / "data/processed/merchant_clean.parquet"

print("Loading cleaned datasets...")
train = pd.read_parquet(train_path)
person = pd.read_parquet(person_path)
merchant = pd.read_parquet(merchant_path)

fraud_tx = train[train["Is Laundering"] == 1]
norm_tx = train[train["Is Laundering"] == 0]

print("\n" + "="*50)
print("1. FRAUD OVERVIEW & MULE ACCOUNT ROLES")
print("="*50)
total_tx = len(train)
total_fraud = len(fraud_tx)
print(f"Total Transactions: {total_tx:,}")
print(f"Fraud Transactions: {total_fraud:,} ({total_fraud/total_tx*100:.4f}%)")

fraud_senders = set(fraud_tx["From Account"])
fraud_receivers = set(fraud_tx["To Account"])
both_mules = fraud_senders & fraud_receivers
all_mules = fraud_senders | fraud_receivers

print(f"Unique Sender Mules: {len(fraud_senders):,}")
print(f"Unique Receiver Mules: {len(fraud_receivers):,}")
print(f"Bridge / Relay Mules (Both send & receive): {len(both_mules):,}")
print(f"Total Unique Accounts in Fraud: {len(all_mules):,}")

print("\n" + "="*50)
print("2. PAYMENT FORMATS (%)")
print("="*50)
fmt_df = pd.DataFrame({
    "Normal (%)": norm_tx["Payment Format"].value_counts(normalize=True) * 100,
    "Fraud (%)": fraud_tx["Payment Format"].value_counts(normalize=True) * 100
})
print(fmt_df.round(2))

print("\n" + "="*50)
print("3. CROSS-BANK FLOW")
print("="*50)
train["is_cross_bank"] = train["From Bank"] != train["To Bank"]
cross_df = train.groupby("Is Laundering")["is_cross_bank"].mean() * 100
print(f"Normal Cross-Bank: {cross_df[0]:.2f}%")
print(f"Fraud Cross-Bank:  {cross_df[1]:.2f}%")

print("\n" + "="*50)
print("4. AMOUNTS COMPARISON (SMURFING DETECTION)")
print("="*50)
print("Normal Amount Paid:")
print(norm_tx["Amount Paid"].describe()[["mean", "50%", "max"]].round(2))
print("\nFraud Amount Paid:")
print(fraud_tx["Amount Paid"].describe()[["mean", "50%", "max"]].round(2))

print("\n" + "="*50)
print("5. PERSON DEMOGRAPHICS & OCCUPATIONS")
print("="*50)
person_mules = person[person["bank_account_number"].isin(all_mules)]
person_normal = person[~person["bank_account_number"].isin(all_mules)]

print(f"Person Mules: {len(person_mules):,} / {len(person):,} ({len(person_mules)/len(person)*100:.2f}%)")
print(f"Median Age - Mules: {person_mules['person_age'].median():.1f} | Normal: {person_normal['person_age'].median():.1f}")
print(f"Minors (<20) in Mules: {(person_mules['person_age'] < 20).sum()} ({(person_mules['person_age'] < 20).mean()*100:.2f}%)")
print(f"Elderly (>60) in Mules: {(person_mules['person_age'] > 60).sum()} ({(person_mules['person_age'] > 60).mean()*100:.2f}%)")

print("\nTop 7 Occupations in Mules:")
print(person_mules["person_occupation"].value_counts().head(7))

print("\n" + "="*50)
print("6. MERCHANTS & CORPORATE MULES")
print("="*50)
merch_mules = merchant[merchant["bank_account_number"].isin(all_mules)]
print(f"Merchants touched by fraud: {len(merch_mules):,} / {len(merchant):,}")
print("Top Industries among Merchant Mules:")
print(merch_mules["industry"].value_counts().head(5))

print("\n" + "="*50)
print("7. NETWORK HUBS (TOP FAN-IN & FAN-OUT)")
print("="*50)
in_counts = fraud_tx.groupby("To Account").size().sort_values(ascending=False).head(5)
out_counts = fraud_tx.groupby("From Account").size().sort_values(ascending=False).head(5)
print("Top 5 Inbound Mules (Fan-in):")
for acc, cnt in in_counts.items():
    print(f"  {acc}: {cnt} inbound fraud transfers")

print("\nTop 5 Outbound Mules (Fan-out):")
for acc, cnt in out_counts.items():
    print(f"  {acc}: {cnt} outbound fraud transfers")

print("\nOverview EDA completed successfully!")
