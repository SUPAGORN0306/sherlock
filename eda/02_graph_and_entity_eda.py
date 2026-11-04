"""
Sherlock - Part 2: Multi-Hop Chains, Nominee Directors, Cross-Currency, and Net Flow
Analyzes:
1. Multi-Hop Chains & Graph Structure (Path lengths, components, cycles)
2. Nominee Directors & Shell Company Networks (legal_representative_id -> merchants)
3. Cross-Currency / FX Hopping (Currency mismatches in laundering)
4. Net Flow & Pass-Through Balance (Inbound vs Outbound volume per account)
"""

import pandas as pd
import numpy as np
import networkx as nx
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
print("1. MULTI-HOP GRAPH STRUCTURE & MONEY LAUNDERING CHAINS")
print("="*50)

# Build directed graph of laundering transactions
G_fraud = nx.DiGraph()
for _, row in fraud_tx.iterrows():
    G_fraud.add_edge(row["From Account"], row["To Account"], amount=row["Amount Paid"])

print(f"Laundering Graph Nodes (Accounts): {G_fraud.number_of_nodes():,}")
print(f"Laundering Graph Edges (Transfers): {G_fraud.number_of_edges():,}")

# Connected components (Laundering Rings / Syndicates)
wcc = list(nx.weakly_connected_components(G_fraud))
wcc_sizes = [len(c) for c in wcc]
print(f"\nNumber of Distinct Laundering Clusters / Syndicates: {len(wcc):,}")
print(f"Largest Syndicate Size (Accounts involved): {max(wcc_sizes):,} accounts")
print(f"Top 5 Syndicate Sizes: {sorted(wcc_sizes, reverse=True)[:5]}")

# Path Length / Hops Analysis (in DAG components)
is_dag = nx.is_directed_acyclic_graph(G_fraud)
print(f"\nIs Laundering Graph a DAG (Acyclic): {is_dag}")

if is_dag:
    longest_path = nx.dag_longest_path(G_fraud)
    print(f"Longest Sequential Laundering Chain: {len(longest_path) - 1} Hops")
    print(f"Longest Path Account Trail: {' -> '.join(longest_path[:6])} ... ({len(longest_path)} nodes)")
else:
    cycles = list(nx.simple_cycles(G_fraud))
    print(f"Closed Cycles / Loops Detected: {len(cycles)}")
    if cycles:
        print(f"Sample Cycle: {' -> '.join(cycles[0])}")

print("\n" + "="*50)
print("2. NOMINEE DIRECTORS & SHELL COMPANY NETWORKS")
print("="*50)

# Multi-merchant directors
rep_counts = merchant.groupby("legal_representative_id").size()
multi_reps = rep_counts[rep_counts > 1]
print(f"Total Unique Legal Representatives: {len(rep_counts):,}")
print(f"Representatives managing >1 Merchant: {len(multi_reps):,}")
print("Top 5 Nominee Directors (Number of Merchants managed):")
print(multi_reps.sort_values(ascending=False).head(5))

# Check if multi-merchant directors are linked to fraud
fraud_accounts = set(fraud_tx["From Account"]) | set(fraud_tx["To Account"])
merchants_in_fraud = merchant[merchant["bank_account_number"].isin(fraud_accounts)]
fraud_reps = set(merchants_in_fraud["legal_representative_id"])

multi_fraud_reps = set(multi_reps.index) & fraud_reps
print(f"\nRepresentatives managing >1 Merchant touched by Laundering: {len(multi_fraud_reps):,}")

# Demographics of these multi-merchant directors
nominee_persons = person[person["person_id"].isin(multi_fraud_reps)]
print("\nDemographics of Top Nominee Directors:")
print(f"Median Age: {nominee_persons['person_age'].median():.1f} years")
print(f"Age Range: {nominee_persons['person_age'].min()} - {nominee_persons['person_age'].max()} years")
print("Occupations of Nominee Directors:")
print(nominee_persons["person_occupation"].value_counts().head(5))

print("\n" + "="*50)
print("3. CROSS-CURRENCY & FX HOPPING")
print("="*50)

train["is_fx_hop"] = train["Payment Currency"] != train["Receiving Currency"]
fx_normal = train[train["Is Laundering"] == 0]["is_fx_hop"].mean() * 100
fx_fraud = train[train["Is Laundering"] == 1]["is_fx_hop"].mean() * 100

print(f"Normal Cross-Currency Rate: {fx_normal:.2f}%")
print(f"Fraud Cross-Currency Rate:  {fx_fraud:.2f}%")

fraud_currencies = fraud_tx.groupby(["Payment Currency", "Receiving Currency"]).size().sort_values(ascending=False)
print("\nTop 5 Currency Pairs used in Laundering:")
print(fraud_currencies.head(5))

print("\n" + "="*50)
print("4. NET FLOW & PASS-THROUGH BALANCE RETENTION")
print("="*50)

# Compute in/out volume per account in fraud
in_vol = train.groupby("To Account")["Amount Received"].sum()
out_vol = train.groupby("From Account")["Amount Paid"].sum()

flow_df = pd.DataFrame({"in_vol": in_vol, "out_vol": out_vol}).fillna(0)
flow_df["total_vol"] = flow_df["in_vol"] + flow_df["out_vol"]
flow_df["net_balance"] = flow_df["in_vol"] - flow_df["out_vol"]
flow_df["retention_ratio"] = flow_df["net_balance"] / (flow_df["total_vol"] + 1e-5)
flow_df["is_fraud_account"] = flow_df.index.isin(fraud_accounts)

fraud_flow = flow_df[flow_df["is_fraud_account"] & (flow_df["total_vol"] > 100)]
normal_flow = flow_df[~flow_df["is_fraud_account"] & (flow_df["total_vol"] > 100)]

print("Retention Ratio ((In - Out) / Total Volume):")
print(f"  Mule Accounts (Mean):   {fraud_flow['retention_ratio'].mean():.4f} (Median: {fraud_flow['retention_ratio'].median():.4f})")
print(f"  Normal Accounts (Mean): {normal_flow['retention_ratio'].mean():.4f} (Median: {normal_flow['retention_ratio'].median():.4f})")

near_zero_mules = (fraud_flow["retention_ratio"].abs() < 0.1).mean() * 100
print(f"\nPercentage of Mule Accounts with near-zero retention (|In-Out| < 10%): {near_zero_mules:.2f}%")
print("High-throughput conduits forward almost 100% of received money.")

print("\nPart 2 EDA completed successfully!")
