# Sherlock: Mule Account Risk & Financial Crime Intelligence Workbench

Enterprise-grade Financial Fraud / Mule Account Detection & Decision Support System compliant with:
- **Bank of Thailand (BOT)** 5-Tier Mule Risk Classification Framework
- **Anti-Technology Crime Decree B.E. 2566 (พ.ร.ก. 2566)** Section 6 (72-Hour Freeze) & Sections 9-10
- **AMLO STR Form 1-03** Reporting Standards

## Features
- **BOT 5-Tier Mule Classification:** Black Mule, Dark Grey, Light Grey, Dark Brown, Light Brown, and Normal.
- **Single Unified Risk Score:** Calibrated 0–100 risk score combining behavioral anomalies & graph network topology.
- **Interactive Ego-Network Visualization:** Pyvis-powered directional network graph with Radial and Left-to-Right Flow layouts.
- **Full Bilingual Support:** 100% Thai and 100% English toggle.
- **Official AMLO Form 1-03:** Auto-generated print-ready official report with one-click PDF print.
- **Tamper-Evident Audit Trail:** SQLite append-only log with SHA-256 hash chaining.

## Quick Start
```bash
pip install -r requirements.txt
streamlit run src/dashboard/app.py
```
