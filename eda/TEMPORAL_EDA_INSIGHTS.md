# Sherlock: Deep Temporal & Behavioral EDA Insights Report
**Dataset:** TransXion Training Split (2,714,741 Transactions | 4,213 Laundering Cases)  
**Target:** Mule Account & Financial Crime Identification  

---

## 1. Executive Summary & Key Red Flags
1. **The Late-Night Danger Zone (00:00 - 05:59):**  
   - Fraud rate is **0.334%**, which is **nearly 3x higher** than daytime business hours (0.122%).
   - Over 220 retiree accounts (`retired`) conducted transfers between midnight and 5 AM.
2. **Extreme Transfer Burstiness (54x Faster):**  
   - Normal accounts have a median inter-transaction interval of **12.4 days** (17,906 minutes).
   - Mule accounts conduct consecutive transactions with a median interval of **just 5.5 hours** (329 minutes).
3. **The Pass-Through Pipeline (Velocity):**  
   - In 779 Bridge Mules, **40.1% of received funds are forwarded within 24 hours**, and **61.9% within 48 hours**.
   - These accounts act as conduits/pipes rather than end-user balance holders.
4. **May 2025 Fraud Spike:**  
   - Laundering peaked in May 2025 with **662 fraud transactions (0.270% fraud rate)**, representing a coordinated campaign wave.

---

## 2. Temporal Metrics Table

| Metric | Normal Accounts | Fraud / Mule Accounts | Insight |
| :--- | :---: | :---: | :--- |
| **Late Night Fraud Rate (00:00-06:00)** | Base (0.12%) | **0.334%** | Criminals exploit off-hours when live monitoring staff is low |
| **Consecutive Transfer Interval** | 12.4 days | **5.5 hours** | Mules exhibit heavy transaction burstiness |
| **Funds Retention (Pass-through)** | Long-term | **40.1% exited in <24h** | Strong evidence of money laundering layering |
| **Peak Day of Week** | Saturday (Shopping) | **Thursday (695 tx)** | Smurfing dispersals peak mid-to-late week |
