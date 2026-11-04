# Sherlock: Overview & Behavioral EDA Insights Report
**Focus:** Class Distribution, Smurfing, Demographics, Corporate Mules, and Network Hubs  

---

## 1. Fraud Overview & Mule Account Roles
* **Total Transactions in Train:** 2,714,741
* **Laundering Transactions:** 4,213 (0.1552%)
* **Unique Accounts Involved in Laundering:** 5,431 accounts
  * **Sender-Only Mules:** 2,861 accounts
  * **Receiver-Only Mules:** 3,349 accounts
  * **Bridge / Relay Mules (Both Inbound & Outbound):** 779 accounts (Multi-hop Layering nodes)

---

## 2. Amount Distribution & Smurfing Patterns
* **Median Amount:** **50.75** (Fraud) vs **51.97** (Normal)  
  * **Insight:** Criminals deliberately split laundering flows into small, ordinary amounts (~50) to mimic everyday grocery and retail spending.
* **Mean Amount:** **2,636.81** (Fraud) vs **1,576.63** (Normal)  
  * **Insight:** Fraud transactions have high positive skewness; occasional high-value consolidations drive the average up by 67%.

---

## 3. Demographics of Person Mules
* Out of 46,400 individuals, **4,256 accounts** engaged in laundering transfers.
* **Top Occupation:** **`retired` (1,017 accounts)** — Retirees are disproportionately targeted or rented as mule accounts, verifying Slide 2.
* **Secondary Occupations:** `unknown` (878), `blue-collar worker` (506), `business owner` (435), `freelancer` (417), `unemployed` (415), `student` (230).

---

## 4. Merchant & Corporate Mules
* **1,175 merchants** (out of 1,330) had at least one incoming or outgoing laundering transaction in the train split.
* **Key Sectors:** Travel & Hospitality, Telecommunications, Food & Beverage, Utilities, Healthcare.
* **Insight:** Merchants serve as commercial sinks and off-ramps with higher limits.

---

## 5. Network Hubs (Fan-in / Fan-out)
* **Top Fan-in:** `A041744` (28 in), `A040519` (22 in), `A033040` (16 in)
* **Top Fan-out:** `A010186` (243 out), `A010069` (158 out), `A040519` (29 out)
* **Hybrid Hub:** `A040519` received from 22 distinct accounts and distributed to 29 distinct accounts.
