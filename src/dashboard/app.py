"""
Sherlock: Enterprise Mule Account & Financial Crime Intelligence Workbench
Complete Edition featuring:
1. Full Bilingual Toggle (100% Thai / 100% English)
2. Light / Dark Theme Switcher & Custom Font Selector
3. 6 Interactive Executive Charts with Multi-Dimensional Filtering
4. Case Triage Queue with Pending/Reviewed Workflow Filter
5. Nominee Ring & Linked Account Detection
6. Top 5 Upstream Senders & Downstream Beneficiaries + 1-Click Chain Investigation
7. Account Statement Table & Inflow/Outflow Timeline Chart
8. Printable Official HTML Form (AMLO Form 1-03) with PDF Print Support
"""

import sys
import importlib
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import src.dashboard.i18n as _i18n_mod
import src.dashboard.graph_viz as _gviz_mod
import src.dashboard.narrative as _narr_mod
import src.dashboard.copilot_rag as _cop_mod
importlib.reload(_i18n_mod)
importlib.reload(_gviz_mod)
importlib.reload(_narr_mod)
importlib.reload(_cop_mod)

from src.dashboard.i18n import (
    translate_tier, translate_val,
    TIER_MAP_TH, TIER_MAP_EN, OCCUPATION_MAP, CHANNEL_MAP
)
from src.dashboard.graph_viz import build_ego_network_html
from src.dashboard.narrative import generate_case_narrative
from src.dashboard.copilot_rag import CopilotRAG, generate_aml_str_report, generate_aml_str_html
from src.dashboard.audit_log import (
    init_db, log_decision, get_audit_history, get_account_last_action,
    verify_chain, recommended_action, ACTION_CODES
)
from src.dashboard.i18n import translate_action, translate_signals

st.set_page_config(
    page_title="Sherlock Financial Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# Cached Data Loaders
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_datasets():
    data_dir = PROJECT_ROOT / "data" / "processed"
    pred_df = pd.read_parquet(data_dir / "predictions.parquet")
    pred_df["risk_score"] = pred_df["risk_score"].astype(np.float32)
    
    # Load only necessary rule features to save memory on cloud
    feat_cols = ["account", "s1_hit", "s3_hit", "s6_hit", "s7_same_amount_cnt", "night_txn_ratio", "drift_txn_cnt"]
    feat_df = pd.read_parquet(data_dir / "node_features.parquet", columns=feat_cols)
    
    tx_df = pd.read_parquet(
        data_dir / "tx_clean.parquet",
        columns=["Timestamp", "From Account", "To Account", "Amount Received", "Payment Format", "Is Laundering"]
    )
    # Downcast datatypes to reduce RAM usage by >70%
    tx_df["hour"] = tx_df["Timestamp"].dt.hour.astype(np.int8)
    tx_df["Amount Received"] = tx_df["Amount Received"].astype(np.float32)
    tx_df["Is Laundering"] = tx_df["Is Laundering"].astype(np.int8)
    tx_df["Payment Format"] = tx_df["Payment Format"].astype("category")

    person_df = pd.read_parquet(data_dir / "person_clean.parquet")
    merchant_df = pd.read_parquet(data_dir / "merchant_clean.parquet")
    
    p_sub = person_df[["bank_account_number", "person_age", "person_occupation"]].drop_duplicates("bank_account_number")
    m_sub = merchant_df[["bank_account_number", "industry"]].drop_duplicates("bank_account_number")
    
    pred_enriched = pred_df.merge(p_sub, left_on="account", right_on="bank_account_number", how="left")
    pred_enriched = pred_enriched.merge(m_sub, left_on="account", right_on="bank_account_number", how="left", suffixes=("", "_m"))
    
    pred_enriched["entity_type"] = np.where(
        pred_enriched["person_age"].notna(), "individual",
        np.where(pred_enriched["industry"].notna(), "corporate", "external")
    )
    # Precompute last active transaction date per account
    f_max = tx_df.groupby("From Account")["Timestamp"].max()
    t_max = tx_df.groupby("To Account")["Timestamp"].max()
    acc_max = pd.concat([f_max, t_max]).groupby(level=0).max().reset_index()
    acc_max.columns = ["account", "last_active_date"]
    pred_enriched = pred_enriched.merge(acc_max, on="account", how="left")
    
    return pred_enriched, feat_df, tx_df, person_df, merchant_df

@st.cache_resource(show_spinner=False)
def load_copilot_service_v4():
    return CopilotRAG()

pred_df, feat_df, tx_df, person_df, merchant_df = load_datasets()
copilot = load_copilot_service_v4()
init_db()

# Account -> bank id lookup (7-digit bank code from the person / merchant master data)
bank_lookup = {
    **dict(zip(person_df["bank_account_number"], person_df["bank"].astype(str))),
    **dict(zip(merchant_df["bank_account_number"], merchant_df["bank"].astype(str))),
}
person_by_id_lookup = person_df.drop_duplicates("person_id").set_index("person_id").to_dict(orient="index")

if "selected_account" not in st.session_state:
    high_risk_pool = pred_df[pred_df["mule_tier"].str.contains("ม้าน้ำตาลเข้ม")].sort_values("risk_score", ascending=False)
    st.session_state["selected_account"] = high_risk_pool.iloc[0]["account"] if not high_risk_pool.empty else pred_df.iloc[0]["account"]

# Helper to find linked accounts under same person_id or legal_representative_id
def find_linked_accounts(target_acc: str):
    owner_id = None
    p_rows = person_df[person_df["bank_account_number"] == target_acc]
    m_rows = merchant_df[merchant_df["bank_account_number"] == target_acc]
    if not p_rows.empty:
        owner_id = p_rows.iloc[0]["person_id"]
    elif not m_rows.empty:
        owner_id = m_rows.iloc[0]["legal_representative_id"]
        
    if not owner_id:
        return None, []
        
    linked_p = person_df[person_df["person_id"] == owner_id]["bank_account_number"].tolist()
    linked_m = merchant_df[merchant_df["legal_representative_id"] == owner_id]["bank_account_number"].tolist()
    all_linked = sorted(list(set(linked_p + linked_m) - {target_acc}))
    return owner_id, all_linked

# -------------------------------------------------------------
# Sidebar: Display Settings (Language, Theme, Font) & Summary
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("#### SHERLOCK COMPLIANCE")
    
    lang_choice = st.radio(
        "ภาษา / Language",
        options=["ภาษาไทย", "English"],
        horizontal=True,
        index=0
    )
    lang = "th" if lang_choice == "ภาษาไทย" else "en"
    
    theme_label = "โหมดสี" if lang == "th" else "Theme Mode"
    theme_opts = ["มืด", "สว่าง"] if lang == "th" else ["Dark", "Light"]
    theme_sel = st.radio(theme_label, options=theme_opts, horizontal=True, index=0)
    is_dark = ("Dark" in theme_sel or "มืด" in theme_sel)
    current_theme = "dark" if is_dark else "light"
    
    font_label = "รูปแบบอักษร" if lang == "th" else "Font Family"
    font_sel = st.selectbox(
        font_label,
        options=["IBM Plex Sans Thai", "Prompt", "Sarabun", "Noto Sans Thai"],
        index=0
    )
    
    st.markdown("---")
    officer_lbl = "รหัสเจ้าหน้าที่ผู้ตรวจ" if lang == "th" else "Compliance Officer ID"
    officer_default = "เจ้าหน้าที่ตรวจสอบ (INV-042)" if lang == "th" else "Compliance Officer (INV-042)"
    officer_name = st.text_input(officer_lbl, value=officer_default, key="officer_id")
    
    st.markdown("---")
    summary_lbl = "สรุปบัญชีทั้งหมด (44,162)" if lang == "th" else "Monitored Accounts (44,162)"
    st.markdown(f"**{summary_lbl}**")
    t_counts = pred_df["mule_tier"].value_counts()
    for raw_t, cnt in t_counts.items():
        st.caption(f"• {translate_tier(raw_t, lang)}: **{cnt:,}**")

# -------------------------------------------------------------
# Dynamic CSS: Compact Typography, Custom Fonts & Light/Dark Theme
# -------------------------------------------------------------
bg_main = "#0B0F17" if is_dark else "#F8FAFC"
bg_card = "#151C28" if is_dark else "#FFFFFF"
border_col = "#263244" if is_dark else "#E2E8F0"
text_main = "#E2E8F0" if is_dark else "#0F172A"
text_sub = "#94A3B8" if is_dark else "#475569"
chart_font_col = "#CBD5E1" if is_dark else "#1E293B"
grid_col = "rgba(148, 163, 184, 0.12)" if is_dark else "rgba(15, 23, 42, 0.08)"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@300;400;500;600;700&family=Noto+Sans+Thai:wght@300;400;500;600;700&family=Prompt:wght@300;400;500;600&family=Sarabun:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"], .stMarkdown, .stText, .stDataFrame, button, input, select, textarea {{
        font-family: '{font_sel}', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 12.5px !important;
    }}

    .stApp {{
        background-color: {bg_main};
        color: {text_main};
    }}

    .main .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 1.5rem;
        max-width: 98%;
    }}

    .sys-header {{
        font-size: 18px !important;
        font-weight: 600;
        color: {text_main};
        margin-bottom: 2px;
    }}
    .sys-sub {{
        font-size: 11.5px !important;
        color: {text_sub};
        margin-bottom: 14px;
        border-bottom: 1px solid {border_col};
        padding-bottom: 8px;
    }}

    .kpi-box {{
        background-color: {bg_card};
        border: 1px solid {border_col};
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }}
    .kpi-label {{
        font-size: 11px !important;
        font-weight: 500;
        color: {text_sub};
    }}
    .kpi-num {{
        font-size: 20px !important;
        font-weight: 700;
        color: {text_main};
        margin-top: 2px;
    }}
    .kpi-desc {{
        font-size: 10.5px !important;
        color: {text_sub};
        margin-top: 2px;
    }}

    .chart-title {{
        font-size: 12.5px !important;
        font-weight: 600;
        color: {text_main};
        margin-bottom: 4px;
    }}

    h1, h2, h3, h4, h5, h6 {{
        font-family: '{font_sel}', sans-serif !important;
        color: {text_main} !important;
    }}
    h4 {{ font-size: 15px !important; }}
    h5 {{ font-size: 13.5px !important; }}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# App Title & Tabs
# -------------------------------------------------------------
if lang == "th":
    st.markdown("<div class='sys-header'>ระบบบริหารจัดการความเสี่ยงบัญชีม้าและสืบสวนอาชญากรรมทางการเงิน</div>", unsafe_allow_html=True)
    st.markdown("<div class='sys-sub'>การคัดกรองความเสี่ยง 5 ระดับตามเกณฑ์ธนาคารแห่งประเทศไทย และ พ.ร.ก. ป้องกันอาชญากรรมทางเทคโนโลยี พ.ศ. 2566</div>", unsafe_allow_html=True)
    tab_names = [
        "แดชบอร์ดภาพรวมความเสี่ยง",
        "คิวงานคัดกรองบัญชีต้องสงสัย",
        "แฟ้มคดีและการสืบสวนเครือข่าย",
        "คู่มือกฎหมายและเกณฑ์กำกับดูแล",
        "ประวัติการปฏิบัติงานย้อนหลัง"
    ]
else:
    st.markdown("<div class='sys-header'>SHERLOCK: MULE ACCOUNT RISK & FINANCIAL CRIME INTELLIGENCE</div>", unsafe_allow_html=True)
    st.markdown("<div class='sys-sub'>5-Tier Mule Account Classification & Network Investigation Workbench (BOT & AMLO Compliant)</div>", unsafe_allow_html=True)
    tab_names = [
        "Executive Risk Dashboard",
        "Case Triage Queue",
        "Case Investigation & Network",
        "Regulatory Knowledge Base",
        "Compliance Audit Trail"
    ]

tab_overview, tab_queue, tab_case, tab_kb, tab_audit = st.tabs(tab_names)

# =============================================================
# TAB 1: Executive Risk Dashboard (With Rich Filters & 6 Charts)
# =============================================================
with tab_overview:
    with st.expander("ตัวกรองข้อมูลแดชบอร์ด" if lang == "th" else "Dashboard Interactive Filters", expanded=True):
        df_c1, df_c2, df_c3, df_c4 = st.columns([2.2, 1.5, 1.8, 1.5])
        
        all_raw_tiers = list(pred_df["mule_tier"].unique())
        tier_disp_to_raw = {translate_tier(t, lang): t for t in all_raw_tiers}
        
        with df_c1:
            sel_tiers_disp = st.multiselect(
                "ระดับความเสี่ยงบัญชีม้า" if lang == "th" else "Mule Risk Tiers",
                options=list(tier_disp_to_raw.keys()),
                default=list(tier_disp_to_raw.keys()),
                key="dash_tier_filter"
            )
            sel_raw_tiers = [tier_disp_to_raw[d] for d in sel_tiers_disp] if sel_tiers_disp else all_raw_tiers

        with df_c2:
            ent_opts = ["ทั้งหมด", "บุคคลธรรมดา", "นิติบุคคล"] if lang == "th" else ["All", "Individual", "Corporate"]
            sel_ent = st.selectbox("ประเภทบัญชี" if lang == "th" else "Account Type", options=ent_opts, index=0)

        with df_c3:
            raw_chans = ["Mobile", "Cash", "Card", "Transfer"]
            chan_disp_to_raw = {translate_val(c, "chan", lang): c for c in raw_chans}
            sel_chans_disp = st.multiselect(
                "ช่องทางธุรกรรม" if lang == "th" else "Payment Channels",
                options=list(chan_disp_to_raw.keys()),
                default=list(chan_disp_to_raw.keys())
            )
            sel_raw_chans = [chan_disp_to_raw[c] for c in sel_chans_disp] if sel_chans_disp else raw_chans

        with df_c4:
            score_range = st.slider(
                "ช่วงคะแนนความเสี่ยง" if lang == "th" else "Risk Score Range",
                0.0, 100.0, (0.0, 100.0), 5.0
            )

    dash_pred = pred_df[
        (pred_df["mule_tier"].isin(sel_raw_tiers)) &
        (pred_df["risk_score"] >= score_range[0]) &
        (pred_df["risk_score"] <= score_range[1])
    ]
    if sel_ent in ["บุคคลธรรมดา", "Individual"]:
        dash_pred = dash_pred[dash_pred["entity_type"] == "individual"]
    elif sel_ent in ["นิติบุคคล", "Corporate"]:
        dash_pred = dash_pred[dash_pred["entity_type"] == "corporate"]

    dash_tx = tx_df[tx_df["Payment Format"].isin(sel_raw_chans)]
    dash_laundering_tx = dash_tx[dash_tx["Is Laundering"] == 1]

    f_total_acc = len(dash_pred)
    f_tier_counts = dash_pred["mule_tier"].value_counts()
    f_freeze = f_tier_counts.get("ม้าดำ (Blacklist ปปง.)", 0) + f_tier_counts.get("ม้าเทาเข้ม (มีผู้เสียหาย แจ้งความแล้ว)", 0)
    f_watch = f_tier_counts.get("ม้าน้ำตาลเข้ม (สงสัยระดับสูง)", 0) + f_tier_counts.get("ม้าน้ำตาลอ่อน (เริ่มสงสัย พฤติกรรมแปลก)", 0)
    f_susp_vol = dash_laundering_tx["Amount Received"].sum()

    k1, k2, k3, k4 = st.columns(4)
    if lang == "th":
        with k1:
            st.markdown(f"<div class='kpi-box'><div class='kpi-label'>บัญชีที่คัดกรองทั้งหมด</div><div class='kpi-num'>{f_total_acc:,}</div><div class='kpi-desc'>บัญชีที่ตรงตามเงื่อนไขตัวกรอง</div></div>", unsafe_allow_html=True)
        with k2:
            st.markdown(f"<div class='kpi-box' style='border-left: 3px solid #EF4444;'><div class='kpi-label'>ต้องระงับทันที (ม้าดำ / ม้าเทาเข้ม)</div><div class='kpi-num'>{f_freeze:,}</div><div class='kpi-desc'>บัญชีดำ ปปง. และมีผู้เสียหายแจ้งความ</div></div>", unsafe_allow_html=True)
        with k3:
            st.markdown(f"<div class='kpi-box' style='border-left: 3px solid #F59E0B;'><div class='kpi-label'>พฤติกรรมผิดปกติ (ม้าน้ำตาลเข้ม / อ่อน)</div><div class='kpi-num'>{f_watch:,}</div><div class='kpi-desc'>ตรวจพบความเสี่ยงเชิงรุก ต้องตรวจสอบ</div></div>", unsafe_allow_html=True)
        with k4:
            st.markdown(f"<div class='kpi-box' style='border-left: 3px solid #3B82F6;'><div class='kpi-label'>มูลค่าธุรกรรมฟอกเงินสะสม</div><div class='kpi-num'>฿{f_susp_vol/1_000_000:,.2f}M บาท</div><div class='kpi-desc'>จาก {len(dash_laundering_tx):,} รายการที่ตรวจพบ</div></div>", unsafe_allow_html=True)
    else:
        with k1:
            st.markdown(f"<div class='kpi-box'><div class='kpi-label'>Filtered Accounts</div><div class='kpi-num'>{f_total_acc:,}</div><div class='kpi-desc'>Accounts matching active criteria</div></div>", unsafe_allow_html=True)
        with k2:
            st.markdown(f"<div class='kpi-box' style='border-left: 3px solid #EF4444;'><div class='kpi-label'>Immediate Freeze (Black / Dark Grey)</div><div class='kpi-num'>{f_freeze:,}</div><div class='kpi-desc'>AMLO Blacklist & Police Reports</div></div>", unsafe_allow_html=True)
        with k3:
            st.markdown(f"<div class='kpi-box' style='border-left: 3px solid #F59E0B;'><div class='kpi-label'>Proactive Flagged (Dark / Light Brown)</div><div class='kpi-num'>{f_watch:,}</div><div class='kpi-desc'>Behavioral anomalies requiring review</div></div>", unsafe_allow_html=True)
        with k4:
            st.markdown(f"<div class='kpi-box' style='border-left: 3px solid #3B82F6;'><div class='kpi-label'>Flagged Laundering Volume</div><div class='kpi-num'>THB {f_susp_vol/1_000_000:,.2f}M</div><div class='kpi-desc'>Across {len(dash_laundering_tx):,} confirmed transactions</div></div>", unsafe_allow_html=True)

    common_layout = dict(
        margin=dict(t=15, b=20, l=10, r=20),
        height=245,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=font_sel, color=chart_font_col, size=11),
        xaxis=dict(gridcolor=grid_col),
        yaxis=dict(gridcolor=grid_col)
    )

    r1_c1, r1_c2 = st.columns(2)
    with r1_c1:
        title_1 = "1. จำนวนบัญชีแยกตามระดับความเสี่ยง 5 ชั้น" if lang == "th" else "1. Account Count by BOT 5-Tier Classification"
        st.markdown(f"<div class='chart-title'>{title_1}</div>", unsafe_allow_html=True)
        
        tier_df = pd.DataFrame({
            "raw_tier": list(f_tier_counts.index),
            "count": list(f_tier_counts.values)
        })
        tier_df["label"] = tier_df["raw_tier"].apply(lambda x: translate_tier(x, lang))
        tier_df = tier_df.sort_values("count", ascending=True)
        
        c_map = {
            translate_tier("ม้าดำ (Blacklist ปปง.)", lang): "#1E293B",
            translate_tier("ม้าเทาเข้ม (มีผู้เสียหาย แจ้งความแล้ว)", lang): "#475569",
            translate_tier("ม้าเทาอ่อน (มีผู้เสียหาย ยังไม่แจ้งความ)", lang): "#94A3B8",
            translate_tier("ม้าน้ำตาลเข้ม (สงสัยระดับสูง)", lang): "#78350F",
            translate_tier("ม้าน้ำตาลอ่อน (เริ่มสงสัย พฤติกรรมแปลก)", lang): "#D97706",
            translate_tier("ปกติ (ความเสี่ยงต่ำ)", lang): "#10B981"
        }
        fig1 = px.bar(
            tier_df, x="count", y="label", orientation="h",
            color="label", color_discrete_map=c_map, text="count"
        )
        fig1.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig1.update_layout(**common_layout, showlegend=False, xaxis_title="", yaxis_title="")
        st.plotly_chart(fig1, use_container_width=True)

    with r1_c2:
        title_2 = "2. แนวโน้มมูลค่าธุรกรรมฟอกเงินรายสัปดาห์ (บาท)" if lang == "th" else "2. Weekly Confirmed Laundering Volume (THB)"
        st.markdown(f"<div class='chart-title'>{title_2}</div>", unsafe_allow_html=True)
        
        if not dash_laundering_tx.empty:
            wk_df = dash_laundering_tx.set_index("Timestamp").resample("W-SUN", label="left")["Amount Received"].sum().reset_index()
        else:
            wk_df = pd.DataFrame({"Timestamp": [], "Amount Received": []})
            
        fig2 = px.area(wk_df, x="Timestamp", y="Amount Received")
        fig2.update_traces(line_color="#EF4444", fillcolor="rgba(239, 68, 68, 0.2)")
        fig2.update_layout(**common_layout, xaxis_title="", yaxis_title="บาท (THB)" if lang == "th" else "THB")
        st.plotly_chart(fig2, use_container_width=True)

    r2_c1, r2_c2 = st.columns(2)
    with r2_c1:
        title_3 = "3. ช่วงเวลาที่เกิดธุรกรรมฟอกเงินใน 24 ชั่วโมง (ไฮไลต์ช่วงดึก 00:00-05:59 น.)" if lang == "th" else "3. 24-Hour Laundering Distribution (00:00-05:59 Danger Zone)"
        st.markdown(f"<div class='chart-title'>{title_3}</div>", unsafe_allow_html=True)
        
        hr_counts = dash_laundering_tx["hour"].value_counts().reindex(range(24), fill_value=0).reset_index()
        hr_counts.columns = ["hour", "count"]
        hr_counts["zone"] = np.where(hr_counts["hour"] < 6, "ช่วงวิกาล (00:00-05:59)" if lang == "th" else "Night Zone (00-05)", "ช่วงเวลาปกติ (06:00-23:59)" if lang == "th" else "Day/Evening (06-23)")
        
        fig3 = px.bar(
            hr_counts, x="hour", y="count", color="zone",
            color_discrete_map={
                "ช่วงวิกาล (00:00-05:59)": "#EF4444", "Night Zone (00-05)": "#EF4444",
                "ช่วงเวลาปกติ (06:00-23:59)": "#3B82F6", "Day/Evening (06-23)": "#3B82F6"
            }
        )
        fig3.update_layout(
            **common_layout,
            xaxis_title="ชั่วโมง (0 - 23 น.)" if lang == "th" else "Hour of Day (0-23)",
            yaxis_title="จำนวนรายการ" if lang == "th" else "Transactions",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title="")
        )
        st.plotly_chart(fig3, use_container_width=True)

    with r2_c2:
        title_4 = "4. จำนวนบัญชีที่มีความเสี่ยงสูงแยกตามอาชีพ" if lang == "th" else "4. High-Risk Mule Accounts by Occupation"
        st.markdown(f"<div class='chart-title'>{title_4}</div>", unsafe_allow_html=True)
        
        high_risk_sub = dash_pred[dash_pred["mule_tier"] != "ปกติ (ความเสี่ยงต่ำ)"].copy()
        if not high_risk_sub.empty and "person_occupation" in high_risk_sub.columns:
            high_risk_sub["occ_label"] = high_risk_sub["person_occupation"].fillna("unknown").apply(lambda x: translate_val(x, "occ", lang))
            occ_df = high_risk_sub["occ_label"].value_counts().reset_index()
            occ_df.columns = ["occ", "count"]
            occ_df = occ_df.sort_values("count", ascending=True)
        else:
            occ_df = pd.DataFrame({"occ": [], "count": []})
            
        fig4 = px.bar(
            occ_df, x="count", y="occ", orientation="h", text="count",
            color="occ",
            color_discrete_sequence=["#38BDF8", "#F43F5E", "#F59E0B", "#10B981", "#818CF8", "#A855F7", "#2DD4BF", "#FB923C"]
        )
        fig4.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig4.update_layout(**common_layout, showlegend=False, xaxis_title="", yaxis_title="")
        st.plotly_chart(fig4, use_container_width=True)

    r3_c1, r3_c2 = st.columns(2)
    with r3_c1:
        title_5 = "5. สัดส่วนธุรกรรมต้องสงสัยแยกตามช่องทางชำระเงิน" if lang == "th" else "5. Suspicious Transactions by Payment Channel"
        st.markdown(f"<div class='chart-title'>{title_5}</div>", unsafe_allow_html=True)
        
        if not dash_laundering_tx.empty:
            ch_df = dash_laundering_tx["Payment Format"].value_counts().reset_index()
            ch_df.columns = ["raw_chan", "count"]
            ch_df["chan_label"] = ch_df["raw_chan"].apply(lambda x: translate_val(x, "chan", lang))
        else:
            ch_df = pd.DataFrame({"chan_label": [], "count": []})
            
        fig5 = px.bar(ch_df, x="chan_label", y="count", text="count", color="chan_label")
        fig5.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig5.update_layout(**common_layout, showlegend=False, xaxis_title="", yaxis_title="")
        st.plotly_chart(fig5, use_container_width=True)

    with r3_c2:
        title_6 = "6. คะแนนความเสี่ยงเฉลี่ยแยกตามกลุ่มช่วงอายุ" if lang == "th" else "6. Average Risk Score by Customer Age Group"
        st.markdown(f"<div class='chart-title'>{title_6}</div>", unsafe_allow_html=True)
        
        ind_sub = dash_pred[dash_pred["person_age"].notna()].copy()
        if not ind_sub.empty:
            age_labels = ["เยาวชน (<=22 ปี)", "วัยทำงานต้น (23-35 ปี)", "วัยทำงาน (36-55 ปี)", "สูงอายุ (56+ ปี)"] if lang == "th" else ["Youth (<=22)", "Young Adult (23-35)", "Adult (36-55)", "Senior (56+)"]
            ind_sub["age_grp"] = pd.cut(ind_sub["person_age"], bins=[0, 22, 35, 55, 120], labels=age_labels)
            age_df = ind_sub.groupby("age_grp", observed=False)["risk_score"].mean().reset_index()
        else:
            age_df = pd.DataFrame({"age_grp": [], "risk_score": []})
            
        fig6 = px.bar(age_df, x="age_grp", y="risk_score", text="risk_score", color="risk_score", color_continuous_scale="Blues")
        fig6.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig6.update_layout(**common_layout, coloraxis_showscale=False, xaxis_title="", yaxis_title="คะแนนเฉลี่ย (0-100)" if lang == "th" else "Avg Score (0-100)")
        st.plotly_chart(fig6, use_container_width=True)

# =============================================================
# TAB 2: Case Triage Queue (With Review Status Filter!)
# =============================================================
with tab_queue:
    # แถวที่ 1: ตัวกรองหลัก (Primary Filters)
    q_col1, q_col2, q_col3, q_col4 = st.columns([2.5, 1.5, 1.5, 2.0])
    
    all_raw_tiers = list(pred_df["mule_tier"].unique())
    tier_disp_map = {translate_tier(t, lang): t for t in all_raw_tiers}
    default_tiers = [
        translate_tier("ม้าน้ำตาลเข้ม (สงสัยระดับสูง)", lang),
        translate_tier("ม้าเทาเข้ม (มีผู้เสียหาย แจ้งความแล้ว)", lang),
        translate_tier("ม้าเทาอ่อน (มีผู้เสียหาย ยังไม่แจ้งความ)", lang)
    ]
    
    with q_col1:
        q_tiers_disp = st.multiselect(
            "ระดับความเสี่ยง" if lang == "th" else "Risk Classification",
            options=list(tier_disp_map.keys()),
            default=default_tiers
        )
        q_raw_tiers = [tier_disp_map[d] for d in q_tiers_disp] if q_tiers_disp else all_raw_tiers

    with q_col2:
        sev_opts = ["ทั้งหมด", "สูงมาก (80 - 100)", "ปานกลาง (50 - 79)", "ต่ำ (0 - 49)"] if lang == "th" else ["All", "Critical (80 - 100)", "Medium (50 - 79)", "Low (0 - 49)"]
        q_sev = st.selectbox("ช่วงคะแนนความเสี่ยง" if lang == "th" else "Score Bracket", options=sev_opts, index=0)

    with q_col3:
        status_opts = ["ทั้งหมด", "รอตรวจสอบ", "ตรวจสอบแล้ว"] if lang == "th" else ["All", "Pending Review", "Reviewed"]
        q_status_filter = st.selectbox("สถานะงาน" if lang == "th" else "Workflow Status", options=status_opts, index=0)

    with q_col4:
        q_search = st.text_input("ค้นหาเลขที่บัญชี" if lang == "th" else "Search Account ID", placeholder="A009852")

    # แถวที่ 2: การจัดเรียงและจำนวนที่แสดง (เลือกเงื่อนไขหลักเพียงอย่างเดียวเพื่อไม่ให้เงื่อนไขขัดแย้งกัน)
    s_col1, s_col2, s_col3 = st.columns([2.5, 1.5, 3.0])
    
    with s_col1:
        sort_opts = [
            "คะแนนความเสี่ยง (จากมากไปน้อย)",
            "คะแนนความเสี่ยง (จากน้อยไปมาก)",
            "วันที่ทำรายการล่าสุด (ใหม่ไปเก่า)",
            "วันที่ทำรายการล่าสุด (เก่าไปใหม่)"
        ] if lang == "th" else [
            "Risk Score: Highest First",
            "Risk Score: Lowest First",
            "Last Active Date: Newest First",
            "Last Active Date: Oldest First"
        ]
        sel_sort = st.selectbox("จัดเรียงลำดับตาม" if lang == "th" else "Sort By", options=sort_opts, index=0)

    with s_col2:
        row_limit_opts = [50, 100, 200, 500, "ทั้งหมด (All)"] if lang == "th" else [50, 100, 200, 500, "All"]
        sel_row_limit = st.selectbox("จำนวนรายการที่แสดง" if lang == "th" else "Display Limit", options=row_limit_opts, index=0)

    with s_col3:
        st.write("")  # Alignment spacer

    # Map audit history
    audit_history_df = get_audit_history(limit=500)
    audit_map = {}
    if not audit_history_df.empty:
        for _, a_row in audit_history_df.iterrows():
            if a_row["account_id"] not in audit_map:
                audit_map[a_row["account_id"]] = f"{translate_action(a_row['action_code'], lang)} ({a_row['timestamp'][:10]})"

    q_df = pred_df[pred_df["mule_tier"].isin(q_raw_tiers)].copy()
    if "80 - 100" in q_sev:
        q_df = q_df[q_df["risk_score"] >= 80.0]
    elif "50 - 79" in q_sev:
        q_df = q_df[(q_df["risk_score"] >= 50.0) & (q_df["risk_score"] < 80.0)]
    elif "0 - 49" in q_sev:
        q_df = q_df[q_df["risk_score"] < 50.0]

    if q_status_filter in ["รอตรวจสอบ", "Pending Review"]:
        q_df = q_df[~q_df["account"].isin(audit_map.keys())]
    elif q_status_filter in ["ตรวจสอบแล้ว", "Reviewed"]:
        q_df = q_df[q_df["account"].isin(audit_map.keys())]
        
    if q_search.strip():
        q_df = q_df[q_df["account"].str.contains(q_search.strip(), case=False)]

    # จัดเรียงตามเงื่อนไขที่เลือกเพียงหนึ่งเดียว
    if "มากไปน้อย" in sel_sort or "Highest" in sel_sort:
        q_df = q_df.sort_values("risk_score", ascending=False)
    elif "น้อยไปมาก" in sel_sort or "Lowest" in sel_sort:
        q_df = q_df.sort_values("risk_score", ascending=True)
    elif "ใหม่ไปเก่า" in sel_sort or "Newest" in sel_sort:
        q_df = q_df.sort_values("last_active_date", ascending=False)
    elif "เก่าไปใหม่" in sel_sort or "Oldest" in sel_sort:
        q_df = q_df.sort_values("last_active_date", ascending=True)
    else:
        q_df = q_df.sort_values("risk_score", ascending=False)
    n_show = len(q_df) if ("All" in str(sel_row_limit) or "ทั้งหมด" in str(sel_row_limit)) else int(sel_row_limit)
    view_sub = q_df.head(n_show).copy()
    view_sub["disp_tier"] = view_sub["mule_tier"].apply(lambda x: translate_tier(x, lang))
    view_sub["risk_score"] = view_sub["risk_score"].round(1)
    view_sub["last_active_str"] = pd.to_datetime(view_sub["last_active_date"]).dt.strftime("%Y-%m-%d %H:%M").fillna("-")
    
    pending_txt = "รอตรวจสอบ" if lang == "th" else "Pending Review"
    view_sub["audit_status"] = view_sub["account"].apply(lambda acc: audit_map.get(acc, pending_txt))
    
    if lang == "th":
        st.caption(f"พบทั้งหมด **{len(q_df):,}** บัญชี (กำลังแสดง **{len(view_sub):,}** รายการตามเงื่อนไข)")
        cols_rename = {
            "account": "เลขที่บัญชี",
            "disp_tier": "ระดับความเสี่ยง",
            "risk_score": "คะแนนความเสี่ยง (0-100)",
            "last_active_str": "ทำรายการล่าสุด",
            "audit_status": "สถานะการตรวจสอบ"
        }
    else:
        st.caption(f"Found **{len(q_df):,}** accounts (Displaying **{len(view_sub):,}** accounts)")
        cols_rename = {
            "account": "Account ID",
            "disp_tier": "Risk Classification",
            "risk_score": "Risk Score (0-100)",
            "last_active_str": "Last Active Date",
            "audit_status": "Review Status"
        }
        
    st.dataframe(view_sub[list(cols_rename.keys())].rename(columns=cols_rename), use_container_width=True, hide_index=True)
    
    # Selection from ALL filtered accounts (or type directly!)
    all_filtered_accounts = q_df["account"].tolist()
    sel_c1, sel_c2 = st.columns([3, 1])
    with sel_c1:
        target_sel = st.selectbox(
            "เลือกบัญชีที่ต้องการตรวจสอบ (ค้นหาหรือเลือกจากทั้งหมดที่ผ่านตัวกรอง)" if lang == "th" else "Select Account for Investigation (Search or select from all filtered accounts)",
            options=all_filtered_accounts if all_filtered_accounts else [st.session_state["selected_account"]],
            help="สามารถพิมพ์รหัสบัญชีเพื่อค้นหาและเลือกได้ทันที" if lang == "th" else "Type account ID to search and select"
        )
    with sel_c2:
        st.write("")
        st.write("")
        if st.button("เปิดแฟ้มคดีนี้" if lang == "th" else "Investigate Case", type="primary", use_container_width=True):
            st.session_state["selected_account"] = target_sel
            st.success(f"เลือกบัญชี {target_sel} แล้ว เปิดแท็บ 'แฟ้มคดีและการสืบสวนเครือข่าย' เพื่อดูรายละเอียด" if lang == "th" else f"Loaded {target_sel} into Case Investigation")

# =============================================================
# TAB 3: Case Investigation & Network Analysis
# =============================================================
with tab_case:
    active_acc = st.session_state.get("selected_account", "A009852")
    
    p_row = pred_df[pred_df["account"] == active_acc]
    p_data = p_row.iloc[0].to_dict() if not p_row.empty else {"account": active_acc, "risk_score": 0.0, "mule_tier": "ปกติ (ความเสี่ยงต่ำ)"}
    
    f_row = feat_df[feat_df["account"] == active_acc]
    f_data = f_row.iloc[0].to_dict() if not f_row.empty else {}
    
    person_match = person_df[person_df["bank_account_number"] == active_acc]
    person_obj = person_match.iloc[0].to_dict() if not person_match.empty else None
    
    merchant_match = merchant_df[merchant_df["bank_account_number"] == active_acc]
    merchant_obj = merchant_match.iloc[0].to_dict() if not merchant_match.empty else None
    
    # Find Linked Entities / Nominee Ring
    owner_id, linked_accounts = find_linked_accounts(active_acc)
    pred_tier_lookup = pred_df.set_index("account")["mule_tier"].to_dict()
    
    narrative_res = generate_case_narrative(
        account_id=active_acc,
        pred_data=p_data,
        feat_data=f_data,
        tx_df=tx_df,
        person_data=person_obj,
        merchant_data=merchant_obj,
        lang=lang,
        pred_tier_lookup=pred_tier_lookup
    )
    
    disp_tier = narrative_res["tier"]
    current_risk = narrative_res["risk_score"]
    last_decision = get_account_last_action(active_acc)
    
    # Header Banner: Clearly display Human Decision override vs Raw AI Risk Score
    hc1, hc2, hc3 = st.columns([2.4, 2.2, 2.4])
    with hc1:
        st.markdown(f"#### {'บัญชีเป้าหมาย' if lang == 'th' else 'Target Account'}: `{active_acc}`")
        if last_decision:
            act_code = last_decision.get("action_code", "")
            if act_code == "DISMISS_NORMAL":
                disp_badge = "สถานะ: บัญชีปกติ / ยุติเรื่อง" if lang == "th" else "Status: Dismissed (Normal)"
                st.markdown(f"**<span style='color: #22C55E; font-size: 14px;'>{disp_badge}</span>**", unsafe_allow_html=True)
                st.caption(f"{'ระดับความเสี่ยงจากโมเดล' if lang == 'th' else 'Model Risk Tier'}: {disp_tier}")
            elif act_code == "FREEZE_AMLO_STR":
                disp_badge = "สถานะ: ระงับบัญชี 72 ชม. และส่ง ปปง." if lang == "th" else "Status: 72h Freeze & AMLO STR"
                st.markdown(f"**<span style='color: #EF4444; font-size: 14px;'>{disp_badge}</span>**", unsafe_allow_html=True)
                st.caption(f"{'ระดับความเสี่ยงจากโมเดล' if lang == 'th' else 'Model Risk Tier'}: {disp_tier}")
            elif act_code == "WATCHLIST_EDD":
                disp_badge = "สถานะ: เฝ้าระวังเข้มงวด / จำกัดวงเงิน" if lang == "th" else "Status: Watchlist (EDD)"
                st.markdown(f"**<span style='color: #F59E0B; font-size: 14px;'>{disp_badge}</span>**", unsafe_allow_html=True)
                st.caption(f"{'ระดับความเสี่ยงจากโมเดล' if lang == 'th' else 'Model Risk Tier'}: {disp_tier}")
            else:
                st.markdown(f"**{'ระดับความเสี่ยงจากโมเดล' if lang == 'th' else 'Model Risk Tier'}:** {disp_tier}")
        else:
            st.markdown(f"**{'ระดับความเสี่ยงจากโมเดล' if lang == 'th' else 'Model Risk Tier'}:** {disp_tier}")
            
    with hc2:
        st.markdown(f"**{'คะแนนความเสี่ยงจากโมเดล' if lang == 'th' else 'Model Risk Score'}:** `{current_risk:.1f} / 100`")
        st.progress(min(max(current_risk / 100.0, 0.0), 1.0))
        
    with hc3:
        if last_decision:
            officer_disp = last_decision.get('officer_name', officer_name)
            time_disp = last_decision['timestamp'][:16]
            st.success(f"**{'ผลการพิจารณาโดย' if lang == 'th' else 'Reviewed by'}:** {officer_disp}\n\n**{'คำสั่ง' if lang == 'th' else 'Order'}:** {translate_action(last_decision['action_code'], lang)} ({time_disp})")
        else:
            st.warning("สถานะ: รอการพิจารณาจากเจ้าหน้าที่" if lang == "th" else "Status: Pending Officer Review")

    st.markdown("---")

    inv_left, inv_right = st.columns([1.05, 1.95])
    
    with inv_left:
        # 1. KYC Profile + Linked Nominee Accounts
        st.markdown(f"**{'1. ข้อมูลผู้ถือบัญชีและบัญชีที่เกี่ยวข้อง' if lang == 'th' else '1. Customer Profile & Linked Entities'}**")
        with st.container(border=True):
            if person_obj:
                occ = translate_val(person_obj.get("person_occupation"), "occ", lang)
                mar = translate_val(person_obj.get("person_marital_status"), "marital", lang)
                edu = translate_val(person_obj.get("person_education"), "edu", lang)
                gen = translate_val(person_obj.get("person_gender"), "gender", lang)
                pid = person_obj.get("person_id", "-")
                bnk = person_obj.get("bank", "-")
                if lang == "th":
                    st.write(f"**รหัสประจำตัวบุคคล:** `{pid}` | **รหัสธนาคาร:** `{bnk}`")
                    st.write(f"**ประเภท:** บุคคลธรรมดา ({gen} อายุ {person_obj.get('person_age')} ปี)")
                    st.write(f"**อาชีพ:** {occ}")
                    st.write(f"**สถานภาพ:** {mar} | **การศึกษา:** {edu}")
                else:
                    st.write(f"**Person ID:** `{pid}` | **Bank ID:** `{bnk}`")
                    st.write(f"**Type:** Individual ({gen}, Age {person_obj.get('person_age')})")
                    st.write(f"**Occupation:** {occ}")
                    st.write(f"**Marital:** {mar} | **Education:** {edu}")
            elif merchant_obj:
                m_t = translate_val(merchant_obj.get("type"), "m_type", lang)
                ind = translate_val(merchant_obj.get("industry"), "ind", lang)
                cap = float(merchant_obj.get("registered_capital", 0))
                mid = merchant_obj.get("merchant_id", "-")
                bnk = merchant_obj.get("bank", "-")
                rep_id = merchant_obj.get("legal_representative_id", "-")
                rep_p = person_by_id_lookup.get(rep_id)
                if lang == "th":
                    st.write(f"**รหัสนิติบุคคล:** `{mid}` | **รหัสธนาคาร:** `{bnk}`")
                    st.write(f"**ประเภท:** นิติบุคคล ({m_t}) | **ธุรกิจ:** {ind}")
                    st.write(f"**ทุนจดทะเบียน:** ฿{cap:,.0f} บาท")
                    if rep_p:
                        rep_occ = translate_val(rep_p.get("person_occupation"), "occ", "th")
                        st.write(f"**ผู้แทนนิติบุคคล (กรรมการผู้มีอำนาจ):** `{rep_id}` (อายุ {rep_p.get('person_age', '-')} ปี | อาชีพ: {rep_occ})")
                    else:
                        st.write(f"**ผู้แทนนิติบุคคล (กรรมการผู้มีอำนาจ):** `{rep_id}`")
                else:
                    st.write(f"**Merchant ID:** `{mid}` | **Bank ID:** `{bnk}`")
                    st.write(f"**Type:** Corporate ({m_t}) | **Industry:** {ind}")
                    st.write(f"**Capital:** THB {cap:,.0f}")
                    if rep_p:
                        rep_occ = translate_val(rep_p.get("person_occupation"), "occ", "en")
                        st.write(f"**Legal Representative (Director):** `{rep_id}` (Age {rep_p.get('person_age', '-')} | {rep_occ})")
                    else:
                        st.write(f"**Legal Representative (Director):** `{rep_id}`")
            else:
                st.write("ไม่พบข้อมูลในระบบ" if lang == "th" else "External Account")
                
            if linked_accounts:
                if lang == "th":
                    st.markdown(f"**ตรวจพบบัญชีที่ถือโดยผู้แทนเดียวกัน ({owner_id}):** `{', '.join(linked_accounts)}`")
                else:
                    st.markdown(f"**Linked Accounts under Same Owner ({owner_id}):** `{', '.join(linked_accounts)}`")
            else:
                st.caption("ไม่พบบัญชีอื่นภายใต้ผู้ถือบัญชีเดียวกัน" if lang == "th" else "No other accounts registered under this owner")

        # 2. Behavioral Indicators
        st.markdown(f"**{'2. พฤติกรรมความผิดปกติที่ตรวจพบ' if lang == 'th' else '2. Behavioral Risk Indicators'}**")
        with st.container(border=True):
            rules = narrative_res.get("rules_detail", {})
            flow_info = narrative_res.get("flow_summary", {})
            t_in = flow_info.get("total_in_amt", 0.0)
            t_out = flow_info.get("total_out_amt", 0.0)

            if lang == "th":
                p_str = "ตรวจพบ" if rules.get("pass_through") else "ปกติ"
                v_str = "ตรวจพบ" if rules.get("rapid_outflow") else "ปกติ"
                s_str = "ตรวจพบ" if rules.get("structuring") else "ปกติ"
                rep_cnt = rules.get("repeat_amt", 0)
                rep_str = f"พบซ้ำ {rep_cnt} ครั้ง" if rep_cnt > 1 else "ไม่พบ"

                n_pct = rules.get("night_pct", 0.0)
                w_pct = rules.get("weekend_pct", 0.0)
                n_str = f"{n_pct:.1f}% (สูงผิดปกติ)" if n_pct > 15.0 else f"{n_pct:.1f}% (ปกติ)"
                w_str = f"{w_pct:.1f}%"

                is_drift = rules.get("is_drift", False)
                drift_str = "พุ่งสูงกะทันหัน" if is_drift else "สม่ำเสมอตามปกติ"

                if t_out > max(t_in * 1.8, 50_000):
                    flow_bal_str = f"โอนออกสูงกว่ารับเข้า {t_out / max(t_in, 1):.1f} เท่า"
                elif t_in > max(t_out * 1.8, 50_000):
                    flow_bal_str = f"รับเข้าสูงกว่าโอนออก {t_in / max(t_out, 1):.1f} เท่า"
                else:
                    flow_bal_str = "สมดุลตามปกติ"

                st.markdown("**พฤติกรรมการหมุนเวียนและส่งต่อเงิน:**")
                st.write(f"• การพักเงินแล้วโอนออกทันที: `{p_str}`")
                st.write(f"• การผันเงินออกเร็วภายใน 48 ชั่วโมง: `{v_str}`")
                st.write(f"• การแบ่งซอยยอดโอนเลี่ยงการตรวจสอบ: `{s_str}`")
                st.write(f"• การโอนเงินด้วยยอดเดิมซ้ำๆ: `{rep_str}`")
                st.write(f"• ความสมดุลของเงินรับเข้าและโอนออก: `{flow_bal_str}`")

                st.markdown("**ช่วงเวลาและจังหวะการทำรายการ:**")
                st.write(f"• ธุรกรรมช่วงวิกาล (00:00-05:59 น.): `{n_str}`")
                st.write(f"• ธุรกรรมช่วงวันหยุดเสาร์-อาทิตย์: `{w_str}`")

                st.markdown("**การเปลี่ยนแปลงของพฤติกรรมกะทันหัน:**")
                st.write(f"• ความถี่และยอดเงิน 30 วันล่าสุด: `{drift_str}`")
            else:
                p_str = "Detected" if rules.get("pass_through") else "Normal"
                v_str = "Detected" if rules.get("rapid_outflow") else "Normal"
                s_str = "Detected" if rules.get("structuring") else "Normal"
                rep_cnt = rules.get("repeat_amt", 0)
                rep_str = f"Repeated {rep_cnt} times" if rep_cnt > 1 else "None"

                n_pct = rules.get("night_pct", 0.0)
                w_pct = rules.get("weekend_pct", 0.0)
                n_str = f"{n_pct:.1f}% (Elevated)" if n_pct > 15.0 else f"{n_pct:.1f}% (Normal)"
                w_str = f"{w_pct:.1f}%"

                is_drift = rules.get("is_drift", False)
                drift_str = "Surged Abnormally" if is_drift else "Consistent Baseline"

                if t_out > max(t_in * 1.8, 50_000):
                    flow_bal_str = f"Outflow exceeds inflow by {t_out / max(t_in, 1):.1f}x"
                elif t_in > max(t_out * 1.8, 50_000):
                    flow_bal_str = f"Inflow exceeds outflow by {t_in / max(t_out, 1):.1f}x"
                else:
                    flow_bal_str = "Balanced"

                st.markdown("**Fund Flow & Transfer Patterns:**")
                st.write(f"• Rapid Pass-Through Transfer: `{p_str}`")
                st.write(f"• 48-Hour Rapid Outflow (>80%): `{v_str}`")
                st.write(f"• Structured Layering Transfers: `{s_str}`")
                st.write(f"• Repeated Identical Amounts: `{rep_str}`")
                st.write(f"• Inflow vs. Outflow Balance: `{flow_bal_str}`")

                st.markdown("**Timing & Activity Patterns:**")
                st.write(f"• Late-Night Activity (00:00-05:59): `{n_str}`")
                st.write(f"• Weekend Activity: `{w_str}`")

                st.markdown("**Sudden Activity Surge:**")
                st.write(f"• Recent 30-Day Velocity: `{drift_str}`")

        # 3. Cash Flow Summary
        st.markdown(f"**{'3. สรุปยอดเงินหมุนเวียน' if lang == 'th' else '3. Cash Flow Summary'}**")
        flow_info = narrative_res.get("flow_summary", {})
        with st.container(border=True):
            cf1, cf2 = st.columns(2)
            if lang == "th":
                with cf1:
                    st.metric("ยอดรับเข้ารวม", f"฿{flow_info.get('total_in_amt', 0):,.0f}")
                    st.metric("ผู้โอนเข้า", f"{flow_info.get('senders_count', 0)} บัญชี")
                with cf2:
                    st.metric("ยอดโอนออกรวม", f"฿{flow_info.get('total_out_amt', 0):,.0f}")
                    st.metric("ธุรกรรมฟอกเงิน", f"{flow_info.get('laundering_count', 0)} รายการ")
            else:
                with cf1:
                    st.metric("Total Incoming", f"THB {flow_info.get('total_in_amt', 0):,.0f}")
                    st.metric("Unique Senders", f"{flow_info.get('senders_count', 0)}")
                with cf2:
                    st.metric("Total Outgoing", f"THB {flow_info.get('total_out_amt', 0):,.0f}")
                    st.metric("Laundering Txns", f"{flow_info.get('laundering_count', 0)}")

    with inv_right:
        st.markdown(f"**{'แผนผังเครือข่ายเส้นทางการเงิน' if lang == 'th' else 'Transaction Ego-Network Topology'}**")
        nc1, nc2, nc3, nc4 = st.columns([1.45, 0.9, 1.35, 1.1])
        with nc1:
            layout_opts = ["กระจายรอบศูนย์", "ไหลซ้ายไปขวา"] if lang == "th" else ["Radial", "Left-to-Right Flow"]
            layout_sel = st.radio("รูปแบบผัง" if lang == "th" else "Layout", layout_opts, horizontal=True, index=0)
            layout_mode_val = "flow" if ("ซ้าย" in layout_sel or "Flow" in layout_sel) else "radial"
        with nc2:
            hops_val = st.radio("ระดับเชื่อมโยง" if lang == "th" else "Hops", [1, 2], horizontal=True, index=0)
        with nc3:
            edges_val = st.slider("จำนวนบัญชีที่แสดงบนผัง" if lang == "th" else "Max Counterparties", 10, 50, 20, 5)
        with nc4:
            st.write("")
            only_risky_val = st.checkbox("เฉพาะบัญชีเสี่ยงสูง" if lang == "th" else "High-Risk Only", value=False)

        network_html = build_ego_network_html(
            target_account=active_acc,
            tx_df=tx_df,
            pred_df=pred_df,
            person_df=person_df,
            merchant_df=merchant_df,
            hops=hops_val,
            max_edges=edges_val,
            theme=current_theme,
            lang=lang,
            height="435px",
            layout_mode=layout_mode_val,
            only_risky=only_risky_val
        )
        components.html(network_html, height=442, scrolling=False)
        if lang == "th":
            st.caption("สัญลักษณ์: โหนดสีทอง = เป้าหมาย | ขอบแดง/ส้ม = บัญชีม้าเสี่ยงสูง | สีเขียว = บัญชีปกติ || ลูกศรแดง = ฟอกเงินยืนยันแล้ว | ลูกศรฟ้า = โอนเข้าเป้าหมาย | ลูกศรส้ม = โอนออกจากเป้าหมาย")
        else:
            st.caption("Legend: Gold = Target | Red/Amber Border = High-Risk Mule | Green = Normal || Red Arrow = Confirmed Laundering | Blue Arrow = Incoming | Amber Arrow = Outgoing")

    st.markdown("---")

    # =============================================================
    # NEW: Top 5 Senders & Top 5 Downstream Beneficiaries + Chain Jump
    # =============================================================
    acc_tx_all = tx_df[(tx_df["From Account"] == active_acc) | (tx_df["To Account"] == active_acc)].copy()
    pred_tier_lookup = pred_df.set_index("account")["mule_tier"].to_dict()
    
    in_tx_sub = acc_tx_all[acc_tx_all["To Account"] == active_acc]
    out_tx_sub = acc_tx_all[acc_tx_all["From Account"] == active_acc]
    
    top_senders_col, top_rec_col = st.columns(2)
    with top_senders_col:
        st.markdown(f"**{'5 อันดับบัญชีที่โอนเงินเข้ามากที่สุด' if lang == 'th' else 'Top 5 Upstream Senders (Incoming Funds)'}**")
        if not in_tx_sub.empty:
            s_grp = in_tx_sub.groupby("From Account")["Amount Received"].agg(["count", "sum"]).reset_index().sort_values("sum", ascending=False).head(5)
            s_grp["bank"] = s_grp["From Account"].map(bank_lookup).fillna("-")
            s_grp["tier"] = s_grp["From Account"].apply(lambda a: translate_tier(pred_tier_lookup.get(a, "ปกติ (ความเสี่ยงต่ำ)"), lang))
            s_grp["sum"] = s_grp["sum"].round(2)
            if lang == "th":
                s_disp = s_grp[["bank", "From Account", "tier", "count", "sum"]].rename(columns={"bank": "ธนาคาร", "From Account": "เลขที่บัญชีผู้โอน", "tier": "ระดับความเสี่ยง", "count": "จำนวนครั้ง", "sum": "ยอดรวม (บาท)"})
            else:
                s_disp = s_grp[["bank", "From Account", "tier", "count", "sum"]].rename(columns={"bank": "Bank ID", "From Account": "Sender Account", "tier": "Risk Tier", "count": "Txns", "sum": "Total (THB)"})
            st.dataframe(s_disp, use_container_width=True, hide_index=True)
        else:
            st.caption("ไม่มีรายการรับเงินเข้า" if lang == "th" else "No incoming transactions")

    with top_rec_col:
        st.markdown(f"**{'5 อันดับบัญชีที่รับเงินโอนออกมากที่สุด' if lang == 'th' else 'Top 5 Downstream Beneficiaries (Outgoing Funds)'}**")
        if not out_tx_sub.empty:
            r_grp = out_tx_sub.groupby("To Account")["Amount Received"].agg(["count", "sum"]).reset_index().sort_values("sum", ascending=False).head(5)
            r_grp["bank"] = r_grp["To Account"].map(bank_lookup).fillna("-")
            r_grp["tier"] = r_grp["To Account"].apply(lambda a: translate_tier(pred_tier_lookup.get(a, "ปกติ (ความเสี่ยงต่ำ)"), lang))
            r_grp["sum"] = r_grp["sum"].round(2)
            if lang == "th":
                r_disp = r_grp[["bank", "To Account", "tier", "count", "sum"]].rename(columns={"bank": "ธนาคาร", "To Account": "เลขที่บัญชีผู้รับเงิน", "tier": "ระดับความเสี่ยง", "count": "จำนวนครั้ง", "sum": "ยอดรวม (บาท)"})
            else:
                r_disp = r_grp[["bank", "To Account", "tier", "count", "sum"]].rename(columns={"bank": "Bank ID", "To Account": "Beneficiary Account", "tier": "Risk Tier", "count": "Txns", "sum": "Total (THB)"})
            st.dataframe(r_disp, use_container_width=True, hide_index=True)
        else:
            st.caption("ไม่มีรายการโอนเงินออก" if lang == "th" else "No outgoing transactions")

    # 1-Click Chain Investigation Jump
    all_counterparties = sorted(list(set(acc_tx_all["From Account"]).union(set(acc_tx_all["To Account"])) - {active_acc}))
    if all_counterparties:
        jmp1, jmp2 = st.columns([3, 1])
        with jmp1:
            next_hop_acc = st.selectbox(
                "เลือกบัญชีที่ต้องการตามรอยเส้นเงินต่อ" if lang == "th" else "Chain Investigation: Select Bank Account to Trace",
                options=all_counterparties
            )
        with jmp2:
            st.write("")
            st.write("")
            if st.button("สลับไปสืบสวนบัญชีนี้" if lang == "th" else "Trace Selected Account", use_container_width=True):
                st.session_state["selected_account"] = next_hop_acc
                st.rerun()

    st.markdown("---")

    # =============================================================
    # NEW: Account Statement & Daily Inflow/Outflow Timeline
    # =============================================================
    st.markdown(f"**{'รายการเดินบัญชีและกราฟเวลาโอนเงินเข้า-ออกของบัญชีเป้าหมาย' if lang == 'th' else 'Account Statement & Inflow/Outflow Timeline'}**")
    stmt_c1, stmt_c2 = st.columns([1.15, 1.85])
    
    with stmt_c1:
        if not acc_tx_all.empty:
            acc_tx_all["direction_type"] = np.where(
                acc_tx_all["To Account"] == active_acc,
                "รับเงินเข้า" if lang == "th" else "Incoming",
                "โอนเงินออก" if lang == "th" else "Outgoing"
            )
            acc_tx_all["date_day"] = acc_tx_all["Timestamp"].dt.floor("D")
            daily_flow = acc_tx_all.groupby(["date_day", "direction_type"])["Amount Received"].sum().reset_index()
            
            fig_stmt = px.bar(
                daily_flow, x="date_day", y="Amount Received", color="direction_type", barmode="group",
                color_discrete_map={
                    "รับเงินเข้า": "#10B981", "Incoming": "#10B981",
                    "โอนเงินออก": "#EF4444", "Outgoing": "#EF4444"
                }
            )
            fig_stmt.update_layout(
                **common_layout,
                xaxis_title="",
                yaxis_title="บาท (THB)" if lang == "th" else "THB",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title="")
            )
            st.plotly_chart(fig_stmt, use_container_width=True)

    with stmt_c2:
        if not acc_tx_all.empty:
            stmt_view = acc_tx_all.sort_values("Timestamp", ascending=False).head(100).copy()
            stmt_view["counterparty"] = np.where(stmt_view["From Account"] == active_acc, stmt_view["To Account"], stmt_view["From Account"])
            stmt_view["bank_id"] = stmt_view["counterparty"].map(bank_lookup).fillna("-")
            stmt_view["chan_disp"] = stmt_view["Payment Format"].apply(lambda c: translate_val(c, "chan", lang))
            stmt_view["status_disp"] = np.where(
                stmt_view["Is Laundering"] == 1,
                "เส้นทางฟอกเงิน" if lang == "th" else "Confirmed Laundering",
                "ทั่วไป" if lang == "th" else "Standard"
            )
            stmt_view["Amount Received"] = stmt_view["Amount Received"].round(2)
            stmt_view["ts_str"] = stmt_view["Timestamp"].astype(str).str.slice(0, 19)
            
            if lang == "th":
                stmt_cols = {
                    "ts_str": "วัน-เวลา", "direction_type": "ทิศทาง", "bank_id": "ธนาคาร", "counterparty": "เลขที่บัญชี",
                    "Amount Received": "จำนวนเงิน (บาท)", "chan_disp": "ช่องทาง", "status_disp": "สถานะ"
                }
            else:
                stmt_cols = {
                    "ts_str": "Timestamp", "direction_type": "Direction", "bank_id": "Bank", "counterparty": "Account No.",
                    "Amount Received": "Amount (THB)", "chan_disp": "Channel", "status_disp": "Status"
                }
            st.dataframe(stmt_view[list(stmt_cols.keys())].rename(columns=stmt_cols), use_container_width=True, height=245, hide_index=True)

    st.markdown("---")

    # Assessment & Action Bar
    ca1, ca2 = st.columns([1.2, 1.2])
    with ca1:
        st.markdown(f"**{'สรุปผลการประเมินและมาตรการแนะนำ' if lang == 'th' else 'Compliance Assessment & Recommended Protocol'}**")
        with st.container(border=True):
            st.write(narrative_res["overall_assessment"])
            for rf in narrative_res.get("red_flags", []):
                st.write(f"• {rf}")
            recom = narrative_res.get("recommendation", {})
            st.markdown(f"**{'คำสั่งแนะนำ' if lang == 'th' else 'Recommended Action'}:** {recom.get('title')}")
            st.caption(f"{'อ้างอิง' if lang == 'th' else 'Authority'}: {recom.get('legal_basis')}")

    with ca2:
        st.markdown(f"**{'บันทึกคำสั่งของผู้ตรวจการ' if lang == 'th' else 'Compliance Officer Decision'}**")
        with st.form("decision_form"):
            notes_lbl = "ระบุเหตุผลประกอบการสั่งการ" if lang == "th" else "Decision Audit Notes"
            compliance_notes = st.text_area(notes_lbl, height=68)
            
            b1, b2, b3 = st.columns(3)
            with b1:
                sub_dismiss = st.form_submit_button("บัญชีปกติ (ยุติเรื่อง)" if lang == "th" else "Dismiss (Normal)", use_container_width=True)
            with b2:
                sub_edd = st.form_submit_button("เฝ้าระวัง / จำกัดวงเงิน" if lang == "th" else "Watchlist (EDD)", use_container_width=True)
            with b3:
                sub_freeze = st.form_submit_button("ระงับบัญชี 72 ชม." if lang == "th" else "Freeze 72h & STR", use_container_width=True)

            raw_t = p_data.get("mule_tier", "")
            _rules = narrative_res.get("rules_detail", {})
            _signals = [k for k in ("pass_through", "rapid_outflow", "structuring") if _rules.get(k)]
            if _rules.get("night_pct", 0.0) > 15.0:
                _signals.append("late_night")
            _ctx = {
                "total_in_amt": flow_info.get("total_in_amt", 0.0),
                "total_out_amt": flow_info.get("total_out_amt", 0.0),
                "laundering_count": flow_info.get("laundering_count", 0),
                "signals": _signals,
                "recommended_action": recommended_action(raw_t),
            }
            _chosen = "DISMISS_NORMAL" if sub_dismiss else "WATCHLIST_EDD" if sub_edd else "FREEZE_AMLO_STR" if sub_freeze else None
            if _chosen:
                log_decision(active_acc, _chosen, officer_name, raw_t, current_risk, compliance_notes, _ctx)
                st.rerun()

    st.markdown("---")

    # Integrated Copilot & Printable Official STR Form
    bot_c1, bot_c2 = st.columns(2)
    
    with bot_c1:
        st.markdown(f"**{'ถามและตีความแฟ้มคดี' if lang == 'th' else 'Case Interpretation Assistant'}**")

        # Build rich case context from actual transactions & counterparty risk
        pred_score_lookup = pred_df.set_index("account")["risk_score"].to_dict()
        high_risk_cps = []
        for cp_acc in all_counterparties:
            cp_raw_t = pred_tier_lookup.get(cp_acc, "ปกติ (ความเสี่ยงต่ำ)")
            cp_sc = float(pred_score_lookup.get(cp_acc, 0.0))
            if "ปกติ" not in cp_raw_t:
                high_risk_cps.append({
                    "account": cp_acc,
                    "bank": bank_lookup.get(cp_acc, "-"),
                    "tier": translate_tier(cp_raw_t, lang).split(" (")[0],
                    "score": cp_sc
                })
        high_risk_cps.sort(key=lambda x: x["score"], reverse=True)

        top_senders_list = []
        if not in_tx_sub.empty:
            s_top = in_tx_sub.groupby("From Account")["Amount Received"].agg(["count", "sum"]).reset_index().sort_values("sum", ascending=False).head(3)
            for _, sr in s_top.iterrows():
                ca = str(sr["From Account"])
                top_senders_list.append({
                    "account": ca,
                    "bank": bank_lookup.get(ca, "-"),
                    "tier": translate_tier(pred_tier_lookup.get(ca, "ปกติ (ความเสี่ยงต่ำ)"), lang).split(" (")[0],
                    "count": int(sr["count"]),
                    "sum": float(sr["sum"])
                })

        top_receivers_list = []
        if not out_tx_sub.empty:
            r_top = out_tx_sub.groupby("To Account")["Amount Received"].agg(["count", "sum"]).reset_index().sort_values("sum", ascending=False).head(3)
            for _, rr in r_top.iterrows():
                ca = str(rr["To Account"])
                top_receivers_list.append({
                    "account": ca,
                    "bank": bank_lookup.get(ca, "-"),
                    "tier": translate_tier(pred_tier_lookup.get(ca, "ปกติ (ความเสี่ยงต่ำ)"), lang).split(" (")[0],
                    "count": int(rr["count"]),
                    "sum": float(rr["sum"])
                })

        if not acc_tx_all.empty:
            top_chan_raw = acc_tx_all["Payment Format"].mode().iloc[0] if not acc_tx_all["Payment Format"].mode().empty else "N/A"
            top_chan_disp = translate_val(top_chan_raw, "chan", lang)
            night_tx_cnt = int((acc_tx_all["hour"] < 6).sum())
            night_tx_pct = float((night_tx_cnt / len(acc_tx_all)) * 100.0)
            act_days = max(int(acc_tx_all["Timestamp"].dt.date.nunique()), 1)
            cp_banks = {bank_lookup.get(c, "-") for c in all_counterparties if bank_lookup.get(c)}
            tx_stats_ctx = {
                "avg_in": float(in_tx_sub["Amount Received"].mean()) if not in_tx_sub.empty else 0.0,
                "max_in": float(in_tx_sub["Amount Received"].max()) if not in_tx_sub.empty else 0.0,
                "avg_out": float(out_tx_sub["Amount Received"].mean()) if not out_tx_sub.empty else 0.0,
                "max_out": float(out_tx_sub["Amount Received"].max()) if not out_tx_sub.empty else 0.0,
                "active_days": act_days,
                "tx_per_day": float(len(acc_tx_all) / act_days),
                "top_channel": top_chan_disp,
                "night_cnt": night_tx_cnt,
                "night_pct": night_tx_pct,
                "unique_banks_cnt": max(len(cp_banks), 1)
            }
        else:
            tx_stats_ctx = {}

        rep_person_ctx = person_by_id_lookup.get(merchant_obj.get("legal_representative_id")) if merchant_obj else None

        case_ctx = {
            "account_id": active_acc,
            "tier": disp_tier,
            "risk_score": current_risk,
            "demo_line": narrative_res.get("demo_findings", ["-"])[0],
            "person_obj": person_obj,
            "merchant_obj": merchant_obj,
            "rep_person_obj": rep_person_ctx,
            "flow_summary": flow_info,
            "tx_stats": tx_stats_ctx,
            "high_risk_cps": high_risk_cps,
            "top_senders_list": top_senders_list,
            "top_receivers_list": top_receivers_list,
            "red_flags": narrative_res.get("red_flags", []),
            "rules_detail": narrative_res.get("rules_detail", {}),
            "recommendation": narrative_res.get("recommendation", {}),
            "linked_entities": linked_accounts,
            "owner_id": owner_id
        }

        q_col_a, q_col_b, q_col_c = st.columns(3)
        if lang == "th":
            with q_col_a:
                q_btn_1 = st.button("สรุปสาเหตุความเสี่ยงของบัญชีนี้", use_container_width=True)
            with q_col_b:
                q_btn_2 = st.button("วิเคราะห์ความขัดแย้งข้อมูลอาชีพกับยอดโอน", use_container_width=True)
            with q_col_c:
                q_btn_3 = st.button("วิเคราะห์เส้นทางเงินและข้อกฎหมาย", use_container_width=True)
            custom_q = st.text_input("พิมพ์คำถามเกี่ยวกับการตีความบัญชีนี้:", placeholder="เช่น โอนไปบัญชีไหนมากที่สุด หรือ ควรใช้อำนาจตามมาตราใดระงับบัญชี?")
        else:
            with q_col_a:
                q_btn_1 = st.button("Explain Key Risk Drivers", use_container_width=True)
            with q_col_b:
                q_btn_2 = st.button("Analyze Profile vs Flow Discrepancy", use_container_width=True)
            with q_col_c:
                q_btn_3 = st.button("Trace Top Flows & Legal Mandate", use_container_width=True)
            custom_q = st.text_input("Ask about this case:", placeholder="e.g., Who are the top beneficiaries or what statute applies?")

        q_state_key = f"copilot_intent_{active_acc}"
        if q_btn_1:
            st.session_state[q_state_key] = ("risk_drivers", "สรุปสาเหตุความเสี่ยงของบัญชีนี้")
        elif q_btn_2:
            st.session_state[q_state_key] = ("profile_discrepancy", "วิเคราะห์ความขัดแย้งข้อมูลอาชีพกับยอดโอน")
        elif q_btn_3:
            st.session_state[q_state_key] = ("flow_and_law", "วิเคราะห์เส้นทางเงินและข้อกฎหมาย")
        elif custom_q.strip():
            st.session_state[q_state_key] = ("", custom_q.strip())

        saved_intent, saved_q = st.session_state.get(q_state_key, ("", ""))
        if saved_q:
            case_ctx["forced_intent"] = saved_intent
            with st.container(border=True):
                st.markdown(copilot.interpret_case(saved_q, case_ctx, lang=lang))

    with bot_c2:
        st.markdown(f"**{'แบบรายงาน ปปง. 1-03 ฉบับราชการพร้อมพิมพ์' if lang == 'th' else 'Official Printable AMLO STR Form 1-03'}**")
        btn_str_lbl = "สร้างเอกสารรายงาน ปปง. 1-03 ฉบับสมบูรณ์" if lang == "th" else "Generate Official Printable STR Form 1-03"
        if st.button(btn_str_lbl, type="primary", use_container_width=True):
            st.session_state[f"str_md_{active_acc}_{lang}"] = generate_aml_str_report(
                account_id=active_acc, narrative=narrative_res, tx_df=tx_df, investigator_name=officer_name, lang=lang, bank_lookup=bank_lookup
            )
            st.session_state[f"str_html_{active_acc}_{lang}"] = generate_aml_str_html(
                account_id=active_acc, narrative=narrative_res, tx_df=tx_df, investigator_name=officer_name, lang=lang, bank_lookup=bank_lookup
            )

        rep_md = st.session_state.get(f"str_md_{active_acc}_{lang}")
        rep_html = st.session_state.get(f"str_html_{active_acc}_{lang}")
        if rep_md and rep_html:
            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    label="ดาวน์โหลดฟอร์มราชการพร้อมพิมพ์ (.html / PDF)" if lang == "th" else "Download Printable Form (.html / PDF)",
                    data=rep_html,
                    file_name=f"AMLO_STR_Form103_{active_acc}.html",
                    mime="text/html",
                    use_container_width=True
                )
            with dl2:
                st.download_button(
                    label="ดาวน์โหลดไฟล์ข้อความ (.md)" if lang == "th" else "Download Markdown (.md)",
                    data=rep_md,
                    file_name=f"AMLO_STR_{active_acc}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            with st.expander("ดูตัวอย่างเอกสารราชการแบบ ปปง. 1-03" if lang == "th" else "Preview Official Printable Form", expanded=True):
                components.html(rep_html, height=420, scrolling=True)

# =============================================================
# TAB 4: Regulatory Knowledge Base
# =============================================================
with tab_kb:
    st.markdown(f"**{'คู่มือข้อกฎหมายและเกณฑ์การกำกับดูแล' if lang == 'th' else 'Regulatory & Statutory Reference Guide'}**")
    kb_search = st.text_input("ค้นหาข้อกำหนดกฎหมาย" if lang == "th" else "Search Regulations", placeholder="มาตรา 6, ม้าดำ, 72 ชั่วโมง" if lang == "th" else "Section 6, Black Mule, 72 hours")
    
    docs_to_show = copilot.query(kb_search, n_results=3, lang=lang) if kb_search.strip() else copilot.query("พรก 2566 ธปท ปปง", n_results=4, lang=lang)
    for d in docs_to_show:
        with st.container(border=True):
            st.markdown(f"**[{d['category']}] {d['title']}**")
            st.write(d["text"])

# =============================================================
# TAB 5: Audit Trail Log
# =============================================================
with tab_audit:
    st.markdown(f"**{'ประวัติการบันทึกคำสั่งของผู้ตรวจการ' if lang == 'th' else 'System-Wide Compliance Audit Trail'}**")

    chain_ok, chain_rows, chain_bad_id = verify_chain()
    if chain_rows == 0:
        st.caption("ยังไม่มีบันทึกในระบบ" if lang == "th" else "No records logged yet.")
    elif chain_ok:
        st.success(
            f"ตรวจสอบความถูกต้องของบันทึกแล้ว: {chain_rows} รายการ ไม่พบการแก้ไข" if lang == "th"
            else f"Integrity check passed: {chain_rows} records, no tampering detected."
        )
    else:
        st.error(
            f"พบความผิดปกติของบันทึกที่รายการลำดับ {chain_bad_id}" if lang == "th"
            else f"Integrity check FAILED at record ID {chain_bad_id}."
        )

    global_audit = get_audit_history(limit=5000)
    if not global_audit.empty:
        global_audit["ts"] = pd.to_datetime(global_audit["timestamp"])
        global_audit["date_only"] = global_audit["ts"].dt.date

        af1, af2, af3, af4, af5 = st.columns([1.6, 1.5, 1.5, 1.8, 1.4])
        with af1:
            d_min, d_max = global_audit["date_only"].min(), global_audit["date_only"].max()
            date_sel = st.date_input(
                "ช่วงวันที่" if lang == "th" else "Date Range",
                value=(d_min, d_max), min_value=d_min, max_value=d_max, key="aud_dates"
            )
        with af2:
            off_sel = st.multiselect(
                "เจ้าหน้าที่" if lang == "th" else "Officer",
                options=sorted(global_audit["reviewer_name"].unique()), key="aud_off"
            )
        with af3:
            act_disp_to_code = {translate_action(c, lang): c for c in ACTION_CODES}
            act_sel = st.multiselect(
                "คำสั่ง" if lang == "th" else "Action",
                options=list(act_disp_to_code.keys()), key="aud_act"
            )
        with af4:
            tier_opts = {translate_tier(t, lang): t for t in sorted(global_audit["tier_raw"].unique())}
            tier_sel = st.multiselect(
                "ระดับความเสี่ยง" if lang == "th" else "Risk Tier",
                options=list(tier_opts.keys()), key="aud_tier"
            )
        with af5:
            acc_sel = st.text_input("เลขที่บัญชี" if lang == "th" else "Account ID", key="aud_acc")

        f_aud = global_audit.copy()
        if isinstance(date_sel, (tuple, list)) and len(date_sel) == 2:
            f_aud = f_aud[(f_aud["date_only"] >= date_sel[0]) & (f_aud["date_only"] <= date_sel[1])]
        if off_sel:
            f_aud = f_aud[f_aud["reviewer_name"].isin(off_sel)]
        if act_sel:
            f_aud = f_aud[f_aud["action_code"].isin([act_disp_to_code[a] for a in act_sel])]
        if tier_sel:
            f_aud = f_aud[f_aud["tier_raw"].isin([tier_opts[t] for t in tier_sel])]
        if acc_sel.strip():
            f_aud = f_aud[f_aud["account_id"].str.contains(acc_sel.strip(), case=False)]

        f_aud["followed"] = f_aud["action_code"] == f_aud["recommended_action"]
        n_total = len(f_aud)
        n_freeze = int((f_aud["action_code"] == "FREEZE_AMLO_STR").sum())
        n_edd = int((f_aud["action_code"] == "WATCHLIST_EDD").sum())
        n_dismiss = int((f_aud["action_code"] == "DISMISS_NORMAL").sum())
        follow_rate = (f_aud["followed"].mean() * 100) if n_total else 0.0

        ak1, ak2, ak3, ak4 = st.columns(4)
        ak1.metric("คำสั่งทั้งหมด" if lang == "th" else "Total Decisions", f"{n_total:,}")
        ak2.metric("ระงับและรายงาน ปปง." if lang == "th" else "Freeze & STR", f"{n_freeze:,}")
        ak3.metric("เฝ้าระวัง / ยุติเรื่อง" if lang == "th" else "Watchlist / Dismissed", f"{n_edd:,} / {n_dismiss:,}")
        ak4.metric("ตรงกับคำแนะนำระบบ" if lang == "th" else "Aligned with System Advice", f"{follow_rate:.0f}%")

        if n_total:
            ac1, ac2 = st.columns(2)
            with ac1:
                st.markdown(f"<div class='chart-title'>{'จำนวนคำสั่งรายวันแยกตามประเภท' if lang == 'th' else 'Daily Decisions by Action'}</div>", unsafe_allow_html=True)
                daily = f_aud.groupby(["date_only", "action_code"]).size().reset_index(name="count")
                daily["action_lbl"] = daily["action_code"].apply(lambda c: translate_action(c, lang))
                fig_a1 = px.bar(
                    daily, x="date_only", y="count", color="action_lbl", barmode="stack",
                    color_discrete_map={
                        translate_action("DISMISS_NORMAL", lang): "#10B981",
                        translate_action("WATCHLIST_EDD", lang): "#F59E0B",
                        translate_action("FREEZE_AMLO_STR", lang): "#EF4444",
                    },
                )
                fig_a1.update_layout(**common_layout, xaxis_title="", yaxis_title="", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=""))
                st.plotly_chart(fig_a1, use_container_width=True)
            with ac2:
                st.markdown(f"<div class='chart-title'>{'คำสั่งของเจ้าหน้าที่เทียบกับคำแนะนำของระบบ' if lang == 'th' else 'Officer Decision vs System Recommendation'}</div>", unsafe_allow_html=True)
                cmp_df = pd.crosstab(f_aud["recommended_action"], f_aud["action_code"]).reset_index().melt(id_vars="recommended_action", var_name="action_code", value_name="count")
                cmp_df["rec_lbl"] = cmp_df["recommended_action"].apply(lambda c: translate_action(c, lang))
                cmp_df["act_lbl"] = cmp_df["action_code"].apply(lambda c: translate_action(c, lang))
                fig_a2 = px.bar(
                    cmp_df, x="rec_lbl", y="count", color="act_lbl", barmode="group",
                    color_discrete_map={
                        translate_action("DISMISS_NORMAL", lang): "#10B981",
                        translate_action("WATCHLIST_EDD", lang): "#F59E0B",
                        translate_action("FREEZE_AMLO_STR", lang): "#EF4444",
                    },
                )
                fig_a2.update_layout(
                    **common_layout,
                    xaxis_title="คำแนะนำของระบบ" if lang == "th" else "System recommendation",
                    yaxis_title="",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=""),
                )
                st.plotly_chart(fig_a2, use_container_width=True)

        view_aud = f_aud.copy()
        view_aud["action_disp"] = view_aud["action_code"].apply(lambda c: translate_action(c, lang))
        view_aud["rec_disp"] = view_aud["recommended_action"].apply(lambda c: translate_action(c, lang))
        view_aud["tier_disp"] = view_aud["tier_raw"].apply(lambda t: translate_tier(t, lang))
        view_aud["signals_disp"] = view_aud["signals"].apply(lambda s: translate_signals(s, lang))
        view_aud["followed_disp"] = view_aud["followed"].map(
            {True: "ตรง" if lang == "th" else "Yes", False: "ไม่ตรง" if lang == "th" else "No"}
        )
        if lang == "th":
            a_cols = {
                "id": "ลำดับ", "timestamp": "วัน-เวลา", "account_id": "เลขที่บัญชี", "reviewer_name": "ผู้ตรวจสอบ",
                "action_disp": "คำสั่ง", "rec_disp": "ระบบแนะนำ", "followed_disp": "ตรงกับระบบ",
                "tier_disp": "ระดับความเสี่ยง", "risk_score": "คะแนน", "total_in_amt": "รับเข้า (บาท)",
                "total_out_amt": "โอนออก (บาท)", "laundering_count": "ธุรกรรมฟอกเงิน",
                "signals_disp": "สัญญาณที่เห็น", "notes": "หมายเหตุ",
            }
        else:
            a_cols = {
                "id": "ID", "timestamp": "Timestamp", "account_id": "Account", "reviewer_name": "Officer",
                "action_disp": "Action", "rec_disp": "System advice", "followed_disp": "Aligned",
                "tier_disp": "Tier", "risk_score": "Score", "total_in_amt": "In (THB)",
                "total_out_amt": "Out (THB)", "laundering_count": "Laundering txns",
                "signals_disp": "Signals", "notes": "Notes",
            }
        st.dataframe(view_aud[list(a_cols.keys())].rename(columns=a_cols), use_container_width=True, hide_index=True)

        ex1, ex2, ex3 = st.columns([1.4, 2.2, 1.2])
        with ex1:
            export_df = view_aud[list(a_cols.keys()) + ["prev_hash", "record_hash"]].rename(columns=a_cols)
            st.download_button(
                label="ส่งออก CSV (ตามตัวกรอง)" if lang == "th" else "Export Filtered CSV",
                data=export_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="sherlock_audit_trail.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with ex2:
            case_pick = st.selectbox(
                "เปิดแฟ้มคดีของบัญชีในรายการ" if lang == "th" else "Open case file for account",
                options=list(dict.fromkeys(f_aud["account_id"].tolist())) or ["-"], key="aud_case_pick",
            )
        with ex3:
            st.write("")
            st.write("")
            if st.button("เปิดแฟ้มคดี" if lang == "th" else "Open Case", use_container_width=True, key="aud_open_case") and case_pick != "-":
                st.session_state["selected_account"] = case_pick
                st.rerun()
