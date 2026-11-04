# Sherlock: Graph Structure & Corporate Entity EDA Insights Report
**Focus:** Multi-Hop Chains, Laundering Cycles, Nominee Directors, and Currency Mechanics  

---

## 1. Multi-Hop Clusters & Criminal Syndicates
* **Total Laundering Graph:** 5,431 accounts and 4,033 transfer edges.
* **1,645 Distinct Criminal Clusters (Syndicates):**
  * **Largest Syndicate:** Encompasses **242 coordinated accounts**!
  * **Second Largest:** Encompasses **191 coordinated accounts**.
* **Circular Laundering (Cycles):**
  * Detected **75 closed loops / cycles** where money flows back to previous accounts (`A -> B -> ... -> A`).
  * Demonstrates wash-trading and artificial volume inflation, proving why **Graph Neural Networks (GraphSAGE)** are essential over tabular models.

---

## 2. Nominee Directors & Shell Company Networks
* **28 Legal Representatives** control multiple merchants.
* 🚨 **100% of these 28 multi-merchant directors (28/28) are linked to accounts touched by laundering!**
* **The Nominee Profile (Textbook Red Flag):**
  * **Students (Age 19):** 3 students serve as directors of multiple commercial enterprises involved in laundering flows.
  * **Retirees (Age 60-74):** 4 senior citizens serve as directors of multiple corporate entities.
  * Verifies the "Corporate Mule" evolution outlined in Slide 2.

---

## 3. Currency Mechanics
* **100% of Laundering Transfers** occur within the primary domestic currency (`CNY -> CNY`).
* Fraudsters avoid cross-currency conversion delays to maximize speed and bypass international FX clearing friction.
