# Sherlock: Graph Diagnostics & Model Selection Report
**Dataset:** TransXion Training Split Graph  
**Scope:** 44,162 Accounts | 2,714,741 Transaction Edges | 5,431 Mule Accounts  
**Diagnostic Script:** `src/models/graph_diagnostics.py`  

---

## 1. Homophily Audit (The Core Finding)
* **Overall Edge Homophily:** 35.38%
* **Edge Type Breakdown:**
  * `Normal -> Normal`: 774,674 edges (28.5%)
  * `Normal -> Mule` (Victims/Normal to Mule): **1,676,632 edges (61.8%)**
  * `Mule -> Normal` (Cashing out / Merchant off-ramp): **77,683 edges (2.9%)**
  * `Mule -> Mule` (Relay / Layering): **185,752 edges (6.8%)**
* 🚨 **Mule-Specific Homophily:** **9.57%**
* 🚨 **Mule Heterophily Rate:** **90.43%**
  * **Critical Insight:** 90.4% of edges connected to a mule account lead to a *normal* person or merchant.

---

## 2. Degree Skewness & Hub Analysis
* **0th Percentile (Min):** 1.0
* **25th Percentile:** 46.0
* **50th Percentile (Median):** 76.0
* **75th Percentile:** 113.0
* **90th Percentile:** 150.0
* **95th Percentile:** 176.0
* **99th Percentile:** 2,142.0
* **100th Percentile (Max):** 5,413.0
* **Largest Hub:** Account `A010300` (5,413 total transactions: 5,411 In, 580 Out)
* **Isolated Nodes (Degree 0):** **0 (0.00%)** — All accounts participate in at least one transaction.

---

## 3. Directionality & Reciprocity
* **Unique Directed Account Pairs:** 1,605,059
* **Edge Reciprocity:** 98.96% (Money flows strictly one-way downstream from victim to mule to cash-out).

---

## 4. Architectural Verdict (Why GraphSAGE?)

| Architecture | Suitability | Justification |
| :--- | :---: | :--- |
| **GCN (Graph Convolutional Network)** | ❌ **FAIL** | GCN relies on symmetric normalized averaging. With 90.4% heterophily, normal neighbors will dilute and drown out the mule signal (Severe Over-smoothing). |
| **GraphSAGE** | ✅ **OPTIMAL** | 1. **Concatenation $[h_{self} \| h_{neighbors}]$**: Never lets neighbors overwrite the node's own AML rules and demographics.<br/>2. **Neighbor Sampling**: Handles hub nodes (up to 5,413 degree) with fixed-size mini-batches (e.g. 15 at hop 1, 10 at hop 2). |
| **GATv2 (Graph Attention Network)** | ⭐ **COMPLEMENT** | Self-attention weights downweight the 90% normal neighbors, providing edge importance for Explainability on the Dashboard. |
