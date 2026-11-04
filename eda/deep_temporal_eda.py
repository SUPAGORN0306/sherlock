"""
Deep Temporal and Behavioral EDA for Sherlock
Focuses on:
1. Hour-of-day dynamics (Off-hours, Business hours, Night-time)
2. Day-of-week and Weekend patterns
3. Monthly trend & seasonality across 2025
4. Money Velocity & Pass-through Duration (Delta-T between In and Out for Bridge Mules)
5. Transaction Burstiness & Structuring
"""

import pandas as pd
import numpy as np
from pathlib import Path

base_dir = Path("/home/pleum/projects/fraud-detection/sherlock")
train_path = base_dir / "data/processed/splits/train.parquet"
person_path = base_dir / "data/processed/person_clean.parquet"
merchant_path = base_dir / "data/processed/merchant_clean.parquet"
output_dir = base_dir / "eda"

print(f"Loading data from {train_path}...")
train = pd.read_parquet(train_path)
person = pd.read_parquet(person_path)
merchant = pd.read_parquet(merchant_path)

# Ensure datetime components
train["hour"] = train["Timestamp"].dt.hour
train["dow"] = train["Timestamp"].dt.day_name()
train["month"] = train["Timestamp"].dt.month
train["date"] = train["Timestamp"].dt.date

fraud = train[train["Is Laundering"] == 1].copy()
normal = train[train["Is Laundering"] == 0].copy()

total_fraud = len(fraud)
total_normal = len(normal)

print("\n" + "="*50)
print("1. HOURLY TEMPORAL DYNAMICS")
print("="*50)

# Categorize time windows
def get_time_window(h):
    if 0 <= h < 6:
        return "Late Night (00:00 - 05:59)"
    elif 6 <= h < 12:
        return "Morning (06:00 - 11:59)"
    elif 12 <= h < 18:
        return "Afternoon (12:00 - 17:59)"
    else:
        return "Evening (18:00 - 23:59)"

train["time_window"] = train["hour"].apply(get_time_window)

tw_summary = train.groupby("time_window").agg(
    total_tx=("Is Laundering", "count"),
    fraud_tx=("Is Laundering", "sum"),
    fraud_rate=("Is Laundering", lambda x: x.mean() * 100)
).loc[[
    "Late Night (00:00 - 05:59)",
    "Morning (06:00 - 11:59)",
    "Afternoon (12:00 - 17:59)",
    "Evening (18:00 - 23:59)"
]]

tw_summary["pct_of_all_fraud"] = (tw_summary["fraud_tx"] / total_fraud) * 100
print(tw_summary.round(3))

print("\n" + "="*50)
print("2. DAY-OF-WEEK & WEEKEND PATTERNS")
print("="*50)

day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
dow_summary = train.groupby("dow").agg(
    total_tx=("Is Laundering", "count"),
    fraud_tx=("Is Laundering", "sum"),
    fraud_rate=("Is Laundering", lambda x: x.mean() * 100)
).reindex(day_order)

dow_summary["pct_of_all_fraud"] = (dow_summary["fraud_tx"] / total_fraud) * 100
print(dow_summary.round(3))

train["is_weekend"] = train["Timestamp"].dt.dayofweek >= 5
wk_summary = train.groupby("is_weekend").agg(
    total_tx=("Is Laundering", "count"),
    fraud_tx=("Is Laundering", "sum"),
    fraud_rate=("Is Laundering", lambda x: x.mean() * 100)
)
wk_summary.index = ["Weekday (Mon-Fri)", "Weekend (Sat-Sun)"]
print("\nWeekday vs Weekend:")
print(wk_summary.round(3))

print("\n" + "="*50)
print("3. MONTHLY TIMELINE & DRIFT (Jan - Nov 2025)")
print("="*50)

month_names = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun",
               7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov"}
month_summary = train.groupby("month").agg(
    total_tx=("Is Laundering", "count"),
    fraud_tx=("Is Laundering", "sum"),
    fraud_rate=("Is Laundering", lambda x: x.mean() * 100)
)
month_summary.index = [month_names.get(m, str(m)) for m in month_summary.index]
print(month_summary.round(3))

print("\n" + "="*50)
print("4. VELOCITY & PASS-THROUGH DURATION (THE SMOKING GUN)")
print("="*50)

# Analyze the 779 Bridge Mules
fraud_senders = set(fraud["From Account"])
fraud_receivers = set(fraud["To Account"])
bridge_mules = fraud_senders & fraud_receivers
print(f"Analyzing {len(bridge_mules)} Bridge Mules for IN -> OUT time lag...")

# Collect time deltas
lags_hours = []
for acc in bridge_mules:
    in_times = fraud[fraud["To Account"] == acc]["Timestamp"].sort_values()
    out_times = fraud[fraud["From Account"] == acc]["Timestamp"].sort_values()
    
    # For each incoming transfer, find the earliest subsequent outgoing transfer
    for in_t in in_times:
        subsequent_out = out_times[out_times >= in_t]
        if len(subsequent_out) > 0:
            first_out = subsequent_out.iloc[0]
            delta = (first_out - in_t).total_seconds() / 3600.0  # in hours
            lags_hours.append(delta)

lags_series = pd.Series(lags_hours)
print(f"Total In->Out Forwarding Pairs Detected: {len(lags_series):,}")
print("Pass-through Time Summary (Hours):")
print(lags_series.describe())

pct_under_1h = (lags_series <= 1.0).mean() * 100
pct_under_6h = (lags_series <= 6.0).mean() * 100
pct_under_24h = (lags_series <= 24.0).mean() * 100
pct_under_48h = (lags_series <= 48.0).mean() * 100

print(f"\nPass-through Speed Breakdown:")
print(f"  <= 1 hour   : {pct_under_1h:.1f}%")
print(f"  <= 6 hours  : {pct_under_6h:.1f}%")
print(f"  <= 24 hours : {pct_under_24h:.1f}%")
print(f"  <= 48 hours : {pct_under_48h:.1f}%")

print("\n" + "="*50)
print("5. TIME GAP BETWEEN CONSECUTIVE TRANSFERS (BURSTINESS)")
print("="*50)

# Check consecutive transaction intervals per account
fraud_sorted = fraud.sort_values(["From Account", "Timestamp"])
fraud_sorted["prev_time"] = fraud_sorted.groupby("From Account")["Timestamp"].shift(1)
fraud_sorted["time_diff_min"] = (fraud_sorted["Timestamp"] - fraud_sorted["prev_time"]).dt.total_seconds() / 60.0
fraud_bursts = fraud_sorted["time_diff_min"].dropna()

normal_sample = normal.sample(n=100000, random_state=42).sort_values(["From Account", "Timestamp"])
normal_sample["prev_time"] = normal_sample.groupby("From Account")["Timestamp"].shift(1)
normal_sample["time_diff_min"] = (normal_sample["Timestamp"] - normal_sample["prev_time"]).dt.total_seconds() / 60.0
normal_bursts = normal_sample["time_diff_min"].dropna()

print("Median time between consecutive transfers (Minutes):")
print(f"  Fraud Accounts : {fraud_bursts.median():.1f} mins (Mean: {fraud_bursts.mean():.1f} mins)")
print(f"  Normal Accounts: {normal_bursts.median():.1f} mins (Mean: {normal_bursts.mean():.1f} mins)")

print("\n" + "="*50)
print("6. NIGHT-TIME DEMOGRAPHICS ANOMALY")
print("="*50)

# Check people operating between 00:00 - 05:59
late_night_tx = train[train["time_window"] == "Late Night (00:00 - 05:59)"]
late_fraud = late_night_tx[late_night_tx["Is Laundering"] == 1]
late_mules = set(late_fraud["From Account"]) | set(late_fraud["To Account"])
late_persons = person[person["bank_account_number"].isin(late_mules)]

print(f"Persons involved in Late-Night Fraud: {len(late_persons):,}")
print("Top Occupations in Late-Night Fraud:")
print(late_persons["person_occupation"].value_counts().head(5))
print(f"Elderly (>60) in Late-Night Fraud: {(late_persons['person_age'] > 60).sum()} ({(late_persons['person_age'] > 60).mean()*100:.1f}%)")

print("\nEDA script completed successfully!")
