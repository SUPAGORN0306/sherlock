"""
Financial Ego-Network Visualization Engine
Supports:
- Edge aggregation (combines multiple transfers between the same account pair)
- Directional edge coloring (Red = Confirmed Laundering, Sky Blue = Incoming, Amber = Outgoing)
- Two layout modes: Radial (anti-overlap physics) and Left-to-Right Hierarchical Flow (Senders -> Target -> Beneficiaries)
- High-risk-only filter & clean stroked typography with zero white iframe border
"""

import json
import numpy as np
import pandas as pd
from pyvis.network import Network
from src.dashboard.i18n import translate_tier, translate_val

TIER_COLORS = {
    "ม้าดำ (Blacklist ปปง.)": {"bg": "#18181B", "border": "#EF4444"},
    "ม้าเทาเข้ม (มีผู้เสียหาย แจ้งความแล้ว)": {"bg": "#334155", "border": "#F87171"},
    "ม้าเทาอ่อน (มีผู้เสียหาย ยังไม่แจ้งความ)": {"bg": "#475569", "border": "#FB923C"},
    "ม้าน้ำตาลเข้ม (สงสัยระดับสูง)": {"bg": "#451A03", "border": "#F59E0B"},
    "ม้าน้ำตาลอ่อน (เริ่มสงสัย พฤติกรรมแปลก)": {"bg": "#78350F", "border": "#FBBF24"},
    "ปกติ (ความเสี่ยงต่ำ)": {"bg": "#14532D", "border": "#22C55E"},
}

def format_thb_short(amt: float, lang: str = "th") -> str:
    if amt >= 1_000_000:
        return f"฿{amt/1_000_000:.1f}M"
    elif amt >= 1_000:
        return f"฿{amt/1_000:.0f}K"
    return f"฿{amt:.0f}"

def _aggregate_edges(df_edges: pd.DataFrame) -> pd.DataFrame:
    """Aggregate multiple transactions between the same (From Account, To Account) pair."""
    if df_edges.empty:
        return df_edges
    grouped = (
        df_edges.groupby(["From Account", "To Account"], as_index=False)
        .agg(
            total_amt=("Amount Received", "sum"),
            tx_count=("Amount Received", "count"),
            max_amt=("Amount Received", "max"),
            is_laundering=("Is Laundering", "max"),
            last_ts=("Timestamp", "max"),
            first_fmt=("Payment Format", "first"),
        )
    )
    return grouped

def build_ego_network_html(
    target_account: str,
    tx_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    person_df: pd.DataFrame = None,
    merchant_df: pd.DataFrame = None,
    hops: int = 1,
    max_edges: int = 25,
    theme: str = "dark",
    lang: str = "th",
    height: str = "460px",
    layout_mode: str = "radial",
    only_risky: bool = False,
) -> str:
    is_dark = (theme == "dark")
    bg_color = "#0B0F17" if is_dark else "#F8FAFC"
    card_border = "#263244" if is_dark else "#CBD5E1"
    font_color = "#F1F5F9" if is_dark else "#0F172A"
    sub_font_color = "#94A3B8" if is_dark else "#475569"

    pred_map = pred_df.set_index("account").to_dict(orient="index") if pred_df is not None else {}
    person_map = person_df.set_index("bank_account_number").to_dict(orient="index") if (person_df is not None and not person_df.empty) else {}
    person_id_map = (
        person_df.drop_duplicates("person_id").set_index("person_id").to_dict(orient="index")
        if (person_df is not None and not person_df.empty and "person_id" in person_df.columns)
        else {}
    )
    merchant_map = merchant_df.set_index("bank_account_number").to_dict(orient="index") if (merchant_df is not None and not merchant_df.empty) else {}

    hop1_tx = tx_df[(tx_df["From Account"] == target_account) | (tx_df["To Account"] == target_account)]

    if hop1_tx.empty:
        net = Network(height=height, width="100%", directed=True, bgcolor=bg_color, font_color=font_color)
        lbl = f"[เป้าหมาย] {target_account}" if lang == "th" else f"[TARGET] {target_account}"
        net.add_node(target_account, label=lbl, color="#D4AF37", size=28)
        return net.generate_html()

    # Aggregate 1-hop transactions by counterparty pair first
    hop1_agg = _aggregate_edges(hop1_tx)

    # Add counterparty risk score to prioritize risky + high-value links
    def _pair_priority(row):
        cp = row["To Account"] if row["From Account"] == target_account else row["From Account"]
        cp_info = pred_map.get(cp, {})
        cp_score = float(cp_info.get("risk_score", 0.0))
        cp_raw_t = cp_info.get("mule_tier", "ปกติ (ความเสี่ยงต่ำ)")
        is_risky = 0 if "ปกติ" in cp_raw_t else 1
        return (int(row["is_laundering"]), is_risky, cp_score, float(row["total_amt"]))

    hop1_agg["priority_tuple"] = hop1_agg.apply(_pair_priority, axis=1)
    hop1_agg = hop1_agg.sort_values(by="priority_tuple", ascending=False).drop(columns=["priority_tuple"])

    if only_risky:
        def _keep_risky(row):
            cp = row["To Account"] if row["From Account"] == target_account else row["From Account"]
            raw_t = pred_map.get(cp, {}).get("mule_tier", "ปกติ (ความเสี่ยงต่ำ)")
            return ("ปกติ" not in raw_t) or (int(row["is_laundering"]) == 1)
        filtered_hop1 = hop1_agg[hop1_agg.apply(_keep_risky, axis=1)]
        if not filtered_hop1.empty:
            hop1_agg = filtered_hop1

    sub_edges = hop1_agg.head(max_edges).copy()
    sub_edges["hop"] = 1

    if hops >= 2:
        hop1_cps = set(sub_edges["From Account"]).union(set(sub_edges["To Account"])) - {target_account}
        cp_tx = tx_df[
            ((tx_df["From Account"].isin(hop1_cps)) | (tx_df["To Account"].isin(hop1_cps)))
            & (tx_df["From Account"] != target_account)
            & (tx_df["To Account"] != target_account)
        ]
        if not cp_tx.empty:
            hop2_agg = _aggregate_edges(cp_tx)
            hop2_agg = hop2_agg.sort_values(by=["is_laundering", "total_amt"], ascending=False).head(max(max_edges // 2, 8))
            hop2_agg["hop"] = 2
            sub_edges = pd.concat([sub_edges, hop2_agg], ignore_index=True)

    net = Network(height=height, width="100%", directed=True, bgcolor=bg_color, font_color=font_color)

    if layout_mode == "flow":
        options = {
            "layout": {
                "hierarchical": {
                    "enabled": True,
                    "direction": "LR",
                    "sortMethod": "directed",
                    "levelSeparation": 250,
                    "nodeSpacing": 62,
                    "treeSpacing": 85,
                    "blockShifting": True,
                    "edgeMinimization": True,
                    "parentCentralization": True
                }
            },
            "physics": {"enabled": False},
            "edges": {
                "smooth": {"type": "cubicBezier", "forceDirection": "horizontal", "roundness": 0.45},
                "arrows": {"to": {"enabled": True, "scaleFactor": 0.65}}
            },
            "interaction": {
                "hover": True,
                "navigationButtons": True,
                "tooltipDelay": 50,
                "zoomView": True
            }
        }
    else:
        options = {
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -210,
                    "centralGravity": 0.012,
                    "springLength": 170,
                    "springConstant": 0.05,
                    "damping": 0.9,
                    "avoidOverlap": 0.92
                },
                "solver": "forceAtlas2Based",
                "stabilization": {"enabled": True, "iterations": 120, "updateInterval": 25}
            },
            "edges": {
                "smooth": {"type": "curvedCW", "roundness": 0.12},
                "arrows": {"to": {"enabled": True, "scaleFactor": 0.65}}
            },
            "interaction": {
                "hover": True,
                "navigationButtons": True,
                "tooltipDelay": 50,
                "zoomView": True
            }
        }
    net.set_options(json.dumps(options))

    # Determine hierarchical level for each node when using flow mode
    senders_1 = set(sub_edges[sub_edges["To Account"] == target_account]["From Account"])
    receivers_1 = set(sub_edges[sub_edges["From Account"] == target_account]["To Account"])

    def _get_level(acc: str) -> int:
        if acc == target_account:
            return 2
        if acc in senders_1 and acc not in receivers_1:
            return 1
        if acc in receivers_1:
            return 3
        return 4

    nodes_added = set()

    def get_node_attrs(acc: str):
        is_target = (acc == target_account)
        tier_info = pred_map.get(acc, {})
        raw_tier = tier_info.get("mule_tier", "ปกติ (ความเสี่ยงต่ำ)")
        disp_tier = translate_tier(raw_tier, lang)
        risk_score = float(tier_info.get("risk_score", 0.0))
        is_normal = ("ปกติ" in raw_tier)

        style = TIER_COLORS.get(raw_tier, {"bg": "#334155", "border": "#64748B"})

        p_info = person_map.get(acc)
        m_info = merchant_map.get(acc)
        bnk_code = (p_info or m_info or {}).get("bank", "-")

        if lang == "th":
            if p_info:
                occ = translate_val(p_info.get("person_occupation"), "occ", "th")
                pid = p_info.get("person_id", "-")
                demo_line = f"รหัสบุคคล: {pid} | บุคคลธรรมดา\nอายุ {p_info.get('person_age', '-')} ปี | อาชีพ: {occ}"
            elif m_info:
                ind = translate_val(m_info.get("industry"), "ind", "th")
                mid = m_info.get("merchant_id", "-")
                rep = m_info.get("legal_representative_id", "-")
                rep_p = person_id_map.get(rep)
                if rep_p:
                    rep_occ = translate_val(rep_p.get("person_occupation"), "occ", "th")
                    rep_str = f"{rep} (กรรมการ อายุ {rep_p.get('person_age', '-')} ปี, อาชีพ{rep_occ})"
                else:
                    rep_str = str(rep)
                demo_line = f"รหัสร้านค้า: {mid} | ผู้แทนนิติบุคคล: {rep_str}\nนิติบุคคล | ธุรกิจ: {ind} | ทุนจดทะเบียน: ฿{m_info.get('registered_capital', 0):,.0f}"
            else:
                demo_line = "บัญชีภายนอก"
            role_tag = f"[บัญชีเป้าหมาย | ธนาคาร {bnk_code}]" if is_target else f"[ธนาคาร {bnk_code}]"
            tooltip = (
                f"{role_tag} เลขบัญชี: {acc}\n"
                f"สถานะ: {disp_tier} (ความเสี่ยง {risk_score:.1f}/100)\n"
                f"{demo_line}"
            )
        else:
            if p_info:
                occ = translate_val(p_info.get("person_occupation"), "occ", "en")
                pid = p_info.get("person_id", "-")
                demo_line = f"Person ID: {pid} | Individual\nAge {p_info.get('person_age', '-')} | Occupation: {occ}"
            elif m_info:
                ind = translate_val(m_info.get("industry"), "ind", "en")
                mid = m_info.get("merchant_id", "-")
                rep = m_info.get("legal_representative_id", "-")
                rep_p = person_id_map.get(rep)
                if rep_p:
                    rep_occ = translate_val(rep_p.get("person_occupation"), "occ", "en")
                    rep_str = f"{rep} (Director, Age {rep_p.get('person_age', '-')}, {rep_occ})"
                else:
                    rep_str = str(rep)
                demo_line = f"Merchant ID: {mid} | Legal Rep: {rep_str}\nCorporate | {ind} | Capital: THB {m_info.get('registered_capital', 0):,.0f}"
            else:
                demo_line = "External Account"
            role_tag = f"[TARGET | BANK {bnk_code}]" if is_target else f"[BANK {bnk_code}]"
            tooltip = (
                f"{role_tag} Account: {acc}\n"
                f"Tier: {disp_tier} (Score {risk_score:.1f}/100)\n"
                f"{demo_line}"
            )

        short_tier = disp_tier.split(" (")[0]
        # Keep normal nodes compact (1 line) so labels don't clutter the graph
        if is_target:
            label = f"{acc}\n[{short_tier}]"
        elif not is_normal:
            label = f"{acc}\n({short_tier})"
        else:
            label = acc

        if is_target:
            size = 30
            color = {"background": "#EAB308", "border": "#FFFFFF", "highlight": {"background": "#FACC15", "border": "#FFFFFF"}}
            border_width = 3
            f_size = 12
            f_col = "#FFFFFF" if is_dark else "#0F172A"
        else:
            size = 20 if not is_normal else 13
            color = {"background": style["bg"], "border": style["border"], "highlight": {"background": style["border"], "border": "#FFFFFF"}}
            border_width = 2.5 if not is_normal else 1.5
            f_size = 10 if not is_normal else 9
            f_col = font_color if not is_normal else sub_font_color

        return {
            "label": label,
            "title": tooltip,
            "color": color,
            "size": size,
            "borderWidth": border_width,
            "level": _get_level(acc),
            "font": {
                "color": f_col,
                "size": f_size,
                "face": "IBM Plex Sans Thai, sans-serif",
                "strokeWidth": 3,
                "strokeColor": bg_color,
            }
        }

    for _, row in sub_edges.iterrows():
        u = str(row["From Account"])
        v = str(row["To Account"])
        amt = float(row["total_amt"])
        tx_cnt = int(row["tx_count"])
        is_laundering = int(row.get("is_laundering", 0)) == 1
        ts = str(row.get("last_ts", ""))[:16]
        fmt_raw = str(row.get("first_fmt", "N/A"))
        fmt = translate_val(fmt_raw, "chan", lang)
        hop_num = int(row.get("hop", 1))

        for acc in [u, v]:
            if acc not in nodes_added:
                attrs = get_node_attrs(acc)
                node_kwargs = dict(
                    label=attrs["label"],
                    title=attrs["title"],
                    color=attrs["color"],
                    size=attrs["size"],
                    borderWidth=attrs["borderWidth"],
                    font=attrs["font"]
                )
                if layout_mode == "flow":
                    node_kwargs["level"] = attrs["level"]
                net.add_node(acc, **node_kwargs)
                nodes_added.add(acc)

        # Color-code edges by role & laundering status
        if is_laundering:
            edge_color = "#EF4444"  # Red solid for confirmed laundering
            edge_width = 3.4
            dashes = False
            status_text = "ตรวจพบเส้นทางฟอกเงิน" if lang == "th" else "Confirmed Laundering Flow"
        elif u == target_account:
            edge_color = "#F59E0B"  # Amber for Outgoing from Target
            edge_width = float(np.clip(1.3 + np.log10(max(amt, 10)) * 0.45, 1.4, 3.0))
            dashes = False
            status_text = "โอนเงินออกจากเป้าหมาย" if lang == "th" else "Outgoing from Target"
        elif v == target_account:
            edge_color = "#38BDF8"  # Sky Blue for Incoming to Target
            edge_width = float(np.clip(1.3 + np.log10(max(amt, 10)) * 0.45, 1.4, 3.0))
            dashes = False
            status_text = "โอนเงินเข้าบัญชีเป้าหมาย" if lang == "th" else "Incoming to Target"
        else:
            edge_color = "#64748B"  # Slate dashed for 2-hop
            edge_width = 1.2
            dashes = True
            status_text = "ธุรกรรมเชื่อมโยงแถว 2" if lang == "th" else "2-Hop Linked Flow"

        if lang == "th":
            edge_title = (
                f"จาก: {u} -> ถึง: {v}\n"
                f"ยอดรวม: ฿{amt:,.2f} บาท ({tx_cnt} รายการ)\n"
                f"ช่องทางหลัก: {fmt}\n"
                f"รายการล่าสุด: {ts}\n"
                f"ประเภทเส้นทาง: {status_text}"
            )
        else:
            edge_title = (
                f"From: {u} -> To: {v}\n"
                f"Total Amount: THB {amt:,.2f} ({tx_cnt} txns)\n"
                f"Primary Channel: {fmt}\n"
                f"Latest Time: {ts}\n"
                f"Flow Type: {status_text}"
            )

        edge_lbl = f"{format_thb_short(amt, lang)} ({tx_cnt})" if tx_cnt > 1 else format_thb_short(amt, lang)

        net.add_edge(
            u, v,
            title=edge_title,
            label=edge_lbl if hop_num == 1 else "",
            color=edge_color,
            width=edge_width,
            dashes=dashes,
            font={"size": 8.5, "color": font_color, "strokeWidth": 3, "strokeColor": bg_color, "align": "middle"}
        )

    raw_html = net.generate_html()

    # Strip Pyvis's white .card border and inject clean styling
    custom_style = f"""
    <style>
        html, body {{
            margin: 0 !important;
            padding: 0 !important;
            background-color: {bg_color} !important;
            overflow: hidden !important;
        }}
        .card {{
            border: 1px solid {card_border} !important;
            border-radius: 8px !important;
            background-color: {bg_color} !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        .card-body {{
            padding: 0 !important;
            background-color: {bg_color} !important;
        }}
        #mynetwork {{
            border: none !important;
            background-color: {bg_color} !important;
        }}
        div.vis-network div.vis-navigation div.vis-button {{
            filter: grayscale(100%) brightness(0.85) contrast(1.2);
            opacity: 0.72;
        }}
        div.vis-network div.vis-navigation div.vis-button:hover {{
            opacity: 1.0;
            box-shadow: 0 0 6px rgba(56, 189, 248, 0.6) !important;
        }}
        div.vis-tooltip {{
            background-color: {'#1E293B' if is_dark else '#FFFFFF'} !important;
            color: {'#F8FAFC' if is_dark else '#0F172A'} !important;
            border: 1px solid {'#475569' if is_dark else '#CBD5E1'} !important;
            border-radius: 6px !important;
            padding: 8px 10px !important;
            font-family: 'IBM Plex Sans Thai', sans-serif !important;
            font-size: 12px !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35) !important;
            white-space: pre-line !important;
        }}
    </style>
    """
    raw_html = raw_html.replace("</head>", custom_style + "\n</head>")
    return raw_html
