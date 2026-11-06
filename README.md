# Sherlock: Mule Account Risk & Financial Crime Intelligence Workbench

Enterprise-grade Financial Fraud & Mule Account Detection Decision Support System compliant with:
- **Bank of Thailand (BOT)** 5-Tier Mule Risk Classification Framework
- **Anti-Technology Crime Decree B.E. 2566 (พ.ร.ก. 2566)** Section 6 (72-Hour Freeze) & Section 9 (3-Year Penalty)
- **AMLO STR Form 1-03** Reporting Standards

---

## 1. System Overview

**Sherlock** is an end-to-end intelligence platform built for financial institutions and central bank analysts to proactively detect, triage, investigate, and legally action suspicious mule accounts. It bridges advanced Machine Learning (Graph Neural Networks + Gradient Boosting) with legal compliance and Human-in-the-Loop decision governance.

```
+---------------------------------------------------------------------------------------------+
|                                    SHERLOCK WORKBENCH                                       |
+---------------------------------------------------------------------------------------------+
|                                                                                             |
|   [ Tab 1: Executive Overview ]   ---> Real-time KPIs, 5-tier distribution, night surges    |
|   [ Tab 2: Triage Queue ]         ---> Unified sorting by risk score / date, audit status   |
|   [ Tab 3: Deep Investigation ]   ---> Ego-network graph, 3 behavioral cards, AI Copilot    |
|   [ Tab 4: AMLO Form 1-03 ]       ---> Official STR document generator, 1-click PDF print   |
|   [ Tab 5: Tamper-Evident Audit ] ---> SHA-256 hash-chained immutable officer audit trail    |
|                                                                                             |
+---------------------------------------------------------------------------------------------+
```

---

## 2. Regulatory Alignment: BOT 5-Tier Framework

Sherlock calibrates a single **Risk Score (0–100)** and combines it with legal evidence proxies into the Bank of Thailand's 5-tier taxonomy:

| ระดับความเสี่ยง (Risk Tier) | ช่วงคะแนน (Score) | ที่มาในระบบจริง (Data Source / Proxy) | มาตรการตอบสนอง (Officer Action) |
| :--- | :---: | :--- | :--- |
| **ม้าดำ (Black Mule)** | $\ge$ 80 | รายชื่อบัญชีดำ ปปง. (AMLO HR-03) | อายัดถาวร และปิดบัญชีทุกธนาคาร |
| **ม้าเทาเข้ม (Dark Grey)** | 70 – 79.9 | มีใบแจ้งความดำเนินคดีอาญา | ระงับทันที และแกะรอยเส้นเงินต่อ |
| **ม้าเทาอ่อน (Light Grey)** | 50 – 69.9 | เส้นเงินเชื่อมต่อจากม้าเทาเข้ม (ทอดถัดไป) | ระงับธุรกรรมช่องทางอิเล็กทรอนิกส์ |
| **ม้าน้ำตาลเข้ม (Dark Brown)** | 35 – 49.9 | ระบบตรวจจับเชิงรุก (พฤติกรรมเสี่ยงสูง) | ระงับชั่วคราว 72 ชม. ให้เจ้าหน้าที่ตรวจ (พ.ร.ก. ม.6) |
| **ม้าน้ำตาลอ่อน (Light Brown)** | 20 – 34.9 | ระบบตรวจจับเชิงรุก (พฤติกรรมเริ่มผิดปกติ) | เพิ่มความเข้มงวด สแกนใบหน้า ลดเพดานวงเงินโอน |
| **บัญชีปกติ (Normal)** | < 20 | พฤติกรรมทั่วไป | ใช้งานได้ตามปกติ |

> **หมายเหตุ:** ในโมเดลทดสอบยังไม่มีฐานข้อมูลคดีจริงจากตำรวจ ระบบใช้ Proxy Mapping จากคะแนนโมเดลร่วมกับสถานะประวัติธุรกรรมฟอกเงินเดิม (`is_mule`) เพื่อจำลองการตัดเกรด 5 ระดับอย่างสมจริง

---

## 3. Model Architecture & Benchmark

Sherlock utilizes a **Dual-Architecture Hybrid Ensemble** combining tabular behavioral features with relational graph topology:

1. **Baseline Model: XGBoost (40% Weight)**
   - Captures tabular typologies: Rapid pass-through, structuring, balance exhaustion within 48h, late-night transaction ratio, 30-day velocity drift, and entity demographics.
2. **Champion Model: GraphSAGE (60% Weight)**
   - PyTorch Geometric 2-layer inductive GNN with mean aggregation and feature concatenation.
   - Accurately captures multi-hop money laundering chains and 90.4% heterophilic graph structures without over-smoothing.

### Evaluation on Strictly Unseen Test Set (10% Split, 4,417 Nodes)

| Metric | XGBoost | GraphSAGE | Hybrid Ensemble (Sherlock) | Business Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Precision@100** | 100.0% | 99.0% | **100.0%** | 100 เคสแรกที่ส่งให้เจ้าหน้าที่ตรวจ เป็นม้าจริงครบ 100% |
| **Precision** | 93.92% | 86.83% | **93.63%** | ความแม่นยำสูง ไม่รบกวนหรืออายัดบัญชีผู้บริสุทธิ์ |
| **Recall** | 82.50% | 86.19% | **89.32%** | ดักจับบัญชีม้าได้เกือบ 90% ของม้าทั้งหมด |
| **F1-Score** | 87.84% | 86.51% | **91.42%** | ความสมดุลสูงสุดระหว่าง Precision และ Recall |
| **PR-AUC** | 89.83% | 91.69% | **93.60%** | ตัวชี้วัดหลักสำหรับข้อมูล Imbalanced Fraud |
| **ROC-AUC** | 95.46% | 96.20% | **97.31%** | ความสามารถในการแยกคนดีออกจากคนร้าย |

---

## 4. Project Directory Structure

```text
sherlock/
├── .gitignore                   # Ignore caches, DB locks, and virtual environments
├── README.md                    # Project documentation & reference manual
├── requirements.txt             # Dashboard & ML training dependencies
├── run_dashboard.sh             # One-click dashboard startup script
│
├── configs/                     # System & pipeline configurations
│   └── split.yaml               # Strict temporal split timestamp (train_end: 2025-11-25)
│
├── data/                        # Tiered data store
│   ├── raw/                     # Raw datasets (tx.csv, person.csv, merchant.csv)
│   └── processed/               # Cleaned & feature-engineered artifacts
│       ├── predictions.parquet  # Risk scores, 5-tier classification & model probabilities
│       ├── model_benchmark.json # Persisted evaluation metrics across splits
│       └── audit_trail_v2.db    # Tamper-evident SQLite database with SHA-256 hash chains
│
├── eda/                         # Exploratory data analysis scripts & research reports
│   ├── 01_overview_and_behavioral_eda.py
│   ├── 02_graph_and_entity_eda.py
│   ├── deep_temporal_eda.py
│   ├── OVERVIEW_EDA_INSIGHTS.md
│   ├── GRAPH_AND_ENTITY_INSIGHTS.md
│   ├── TEMPORAL_EDA_INSIGHTS.md
│   └── 03_GRAPH_DIAGNOSTICS_REPORT.md
│
├── models/                      # Trained model weights
│   └── graphsage.pt             # PyTorch Geometric GraphSAGE checkpoint
│
├── scripts/                     # Standalone verification & evaluation scripts
│   ├── eval_test_set.py         # Test set evaluator
│   └── eval_hybrid.py           # Ensemble weight tuning script
│
└── src/                         # Core modular source code
    ├── data/                    # Ingestion & cleaning
    │   ├── cleaner.py
    │   └── splitter.py
    ├── features/                # AML typology & behavioral feature engineering
    │   └── build_features.py
    ├── models/                  # ML & GNN model definitions & training
    │   ├── train_models.py
    │   └── graph_diagnostics.py
    └── dashboard/               # Streamlit workbench & intelligence UI
        ├── app.py               # Main 5-tab application
        ├── narrative.py         # Automated bilingual case narrative generator
        ├── copilot_rag.py       # Legal & investigation AI Copilot
        ├── graph_viz.py         # Interactive Ego-network Pyvis renderer
        ├── audit_log.py         # Tamper-evident audit trail engine
        └── i18n.py              # Bilingual localization (TH/EN)
```

---

## 5. Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/SUPAGORN0306/sherlock.git
cd sherlock

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Dashboard
```bash
streamlit run src/dashboard/app.py
```
Open your browser and navigate to `http://localhost:8501`.

### 3. Re-train Models (Optional)
```bash
python src/features/build_features.py
python src/models/train_models.py
```
Outputs will update `data/processed/predictions.parquet` and `data/processed/model_benchmark.json`.

---

## 6. Key Governance & Security Features
- **Zero Temporal Data Leakage:** Features and graph structures are constructed strictly from transactions on or before `2025-11-25`.
- **Tamper-Evident Audit Trail:** Every officer action (Normal, Watchlist, 72h Freeze) is stored with SHA-256 cryptographic linkage to the preceding log entry.
- **Victim Protection Protocol:** Counterparty risk scoring differentiates innocent fraud victims from willing money mules, preventing unfair account freezing.
- **Full Bilingual Localization:** Seamless one-click toggle between Thai and English across all UI labels, narratives, tooltips, and AMLO forms.
