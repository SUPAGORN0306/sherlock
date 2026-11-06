"""
Bilingual AI Case Interpretation Copilot & Official AMLO STR Report Drafter
100% Pure Thai in Thai mode and 100% Pure English in English mode.
Includes both Markdown and Print-Ready Official HTML Form (AMLO Form 1-03) with window.print() PDF support.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
from src.dashboard.i18n import translate_val

KNOWLEDGE_DOCS_TH = [
    {
        "id": "law_01",
        "title": "พ.ร.ก. 2566 มาตรา 6: การระงับธุรกรรมชั่วคราว 72 ชั่วโมง",
        "category": "กฎหมายหลัก",
        "text": """พ.ร.ก. มาตรการป้องกันและปราบปรามอาชญากรรมทางเทคโนโลยี พ.ศ. 2566 มาตรา 6:
เมื่อสถาบันการเงินพบเหตุอันควรสงสัยเอง หรือได้รับแจ้งจากผู้เสียหายว่ามีการทำธุรกรรมที่เกี่ยวข้องกับอาชญากรรมทางเทคโนโลยี ให้มีหน้าที่ระงับการทำธุรกรรมไว้ชั่วคราวไม่เกิน 72 ชั่วโมง และแจ้งศูนย์ปฏิบัติการ 1441 ทราบทันที"""
    },
    {
        "id": "law_02",
        "title": "พ.ร.ก. 2566 มาตรา 9-10: บทลงโทษผู้เปิดบัญชีม้าและผู้จัดหา",
        "category": "บทลงโทษ",
        "text": """มาตรา 9: ผู้เปิดหรือยินยอมให้ผู้อื่นใช้บัญชีเงินฝากหรือซิมการ์ด (บัญชีม้า) ต้องระวางโทษจำคุกไม่เกิน 3 ปี หรือปรับไม่เกิน 300,000 บาท หรือทั้งจำทั้งปรับ
มาตรา 10: ผู้เป็นธุระจัดหา โฆษณา หรือซื้อขายบัญชีม้า ต้องระวางโทษจำคุก 2 ถึง 5 ปี หรือปรับ 200,000 ถึง 500,000 บาท หรือทั้งจำทั้งปรับ"""
    },
    {
        "id": "bot_01",
        "title": "แนวนโยบาย ธปท.: การจัดระดับความเสี่ยงบัญชีม้า 5 ระดับ",
        "category": "เกณฑ์ ธปท.",
        "text": """การจัดระดับความเสี่ยงบัญชีม้า 5 ระดับ ของธนาคารแห่งประเทศไทย:
1. ม้าดำ: รายชื่อบัญชีดำ ปปง. ระงับทุกบัญชีทันที ห้ามเปิดบัญชีใหม่ทุกธนาคาร
2. ม้าเทาเข้ม: มีผู้เสียหายแจ้งความดำเนินคดีแล้ว ระงับช่องทางอิเล็กทรอนิกส์ทันที
3. ม้าเทาอ่อน: มีผู้เสียหายแจ้งธนาคารแต่ยังไม่มีใบแจ้งความ ระงับชั่วคราวและเรียกยืนยันตัวตนที่สาขา
4. ม้าน้ำตาลเข้ม: ธนาคารตรวจพบพฤติกรรมเสี่ยงสูงมากเอง ระงับชั่วคราว 72 ชั่วโมง และส่งรายงาน ปปง.
5. ม้าน้ำตาลอ่อน: เริ่มพบความผิดปกติ จำกัดวงเงินโอนต่อวันและบังคับสแกนใบหน้ายืนยันตัวตน"""
    },
    {
        "id": "amlo_01",
        "title": "พ.ร.บ. ป้องกันและปราบปรามการฟอกเงิน มาตรา 16: รายงานแบบ ปปง. 1-03",
        "category": "รายงาน ปปง.",
        "text": """สถาบันการเงินต้องส่งรายงานธุรกรรมที่มีเหตุอันควรสงสัย (แบบ ปปง. 1-03) ต่อสำนักงาน ปปง. ภายใน 7 วันทำการนับแต่วันที่ตรวจพบเหตุสงสัย โดยระบุข้อมูลผู้ถือบัญชี พฤติการณ์การโอนเงิน รหัสธนาคารและเลขที่บัญชีที่รับ-โอน รวมถึงมาตรการระงับบัญชีที่ได้ดำเนินการไปแล้ว"""
    }
]

KNOWLEDGE_DOCS_EN = [
    {
        "id": "law_01_en",
        "title": "Decree B.E. 2566 Section 6: 72-Hour Proactive Transaction Freeze",
        "category": "Primary Statute",
        "text": """Anti-Technology Crime Decree B.E. 2566 Section 6:
When a financial institution proactively detects suspicious activity or receives a victim complaint regarding technology crime transactions, it must temporarily freeze outgoing transactions for up to 72 hours and notify the AOC 1441 War Room immediately."""
    },
    {
        "id": "law_02_en",
        "title": "Decree B.E. 2566 Sections 9-10: Criminal Penalties for Mule Accounts & Brokers",
        "category": "Criminal Penalties",
        "text": """Section 9: Any person who opens or allows another person to use their bank account or SIM card as a mule account faces up to 3 years imprisonment, a fine up to THB 300,000, or both.
Section 10: Any broker who procures, advertises, or traffics mule accounts faces 2 to 5 years imprisonment, a fine of THB 200,000 to 500,000, or both."""
    },
    {
        "id": "bot_01_en",
        "title": "Bank of Thailand Policy: 5-Tier Mule Account Classification Framework",
        "category": "BOT Guidelines",
        "text": """Bank of Thailand 5-Tier Mule Risk Taxonomy:
1. Black Mule: Official AMLO Blacklist. Immediate freeze across all accounts and ban on new account openings.
2. Dark Grey Mule: Confirmed police report filed. Immediate suspension of all electronic banking channels.
3. Light Grey Mule: Victim notified bank pending formal police report. Temporary suspension and mandatory in-branch KYC.
4. Dark Brown Mule: High-confidence proactive bank detection. Enforce 72-hour temporary freeze and file AMLO STR Form 1-03.
5. Light Brown Mule: Early behavioral anomaly detected. Cap daily transfer limits and enforce biometric facial verification."""
    },
    {
        "id": "amlo_01_en",
        "title": "Anti-Money Laundering Act Section 16: AMLO STR Form 1-03 Filing",
        "category": "AMLO Reporting",
        "text": """Financial institutions must submit a Suspicious Transaction Report (AMLO Form 1-03) to the Anti-Money Laundering Office within 7 business days of detecting suspicious activity, detailing customer KYC profile, transaction anomalies, bank and account numbers along the money trail, and freeze actions executed."""
    }
]


class CopilotRAG:
    """Lightweight, zero-memory regulatory knowledge base & reasoning engine."""
    def __init__(self):
        pass

    def query(self, query_text: str, n_results: int = 2, lang: str = "th") -> List[Dict[str, Any]]:
        docs = KNOWLEDGE_DOCS_EN if lang == "en" else KNOWLEDGE_DOCS_TH
        if not query_text or not query_text.strip():
            return docs[:n_results]

        keywords = [k.strip().lower() for k in query_text.split() if len(k.strip()) > 1]
        if not keywords:
            return docs[:n_results]

        scored = []
        for d in docs:
            content = f"{d['title']} {d['category']} {d['text']}".lower()
            score = sum(3 for k in keywords if k in d['title'].lower()) + sum(1 for k in keywords if k in content)
            scored.append((score, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:n_results]]

    def interpret_case(self, question: str, case_context: Dict[str, Any], lang: str = "th") -> str:
        acc = case_context.get("account_id", "N/A")
        tier = case_context.get("tier", "-")
        risk_score = float(case_context.get("risk_score", 0.0))
        flow = case_context.get("flow_summary", {})
        rules = case_context.get("rules_detail", {})
        linked_info = case_context.get("linked_entities", [])
        owner_id = case_context.get("owner_id", "-")
        person_obj = case_context.get("person_obj")
        merchant_obj = case_context.get("merchant_obj")
        rep_person_obj = case_context.get("rep_person_obj")
        tx_stats = case_context.get("tx_stats", {})
        high_risk_cps = case_context.get("high_risk_cps", [])
        top_senders = case_context.get("top_senders_list", [])
        top_receivers = case_context.get("top_receivers_list", [])
        recom = case_context.get("recommendation", {})

        q_lower = question.lower().strip()
        intent = case_context.get("forced_intent", "")
        if not intent:
            if any(k in q_lower for k in ["อาชีพ", "อายุ", "ขัดแย้ง", "ผู้ถือบัญชี", "ผู้แทน", "นอมินี", "รายได้", "ประวัติ", "ลูกค้า", "นิติบุคคล", "กรรมการ", "profile", "occupation", "discrepancy", "kyc", "age", "rep"]):
                intent = "profile_discrepancy"
            elif any(k in q_lower for k in ["เส้นทาง", "โอนไป", "รับจาก", "แถว 2", "ธนาคาร", "ใคร", "trace", "counterparty", "beneficiary", "sender", "route"]):
                intent = "flow_and_law"
            elif any(k in q_lower for k in ["กฎหมาย", "มาตรา", "ระงับ", "อายัด", "ปปง", "พรก", "72", "โทษ", "จำคุก", "law", "statute", "section", "freeze", "str", "amlo"]):
                intent = "legal_authority"
            else:
                intent = "risk_drivers"

        in_amt = float(flow.get("total_in_amt", 0.0))
        out_amt = float(flow.get("total_out_amt", 0.0))
        in_cnt = int(flow.get("in_tx_count", 0))
        out_cnt = int(flow.get("out_tx_count", 0))
        senders_cnt = int(flow.get("senders_count", 0))
        receivers_cnt = int(flow.get("receivers_count", 0))
        laund_cnt = int(flow.get("laundering_count", 0))
        laund_amt = float(flow.get("laundering_amt", 0.0))
        total_turnover = in_amt + out_amt
        total_tx = in_cnt + out_cnt
        total_cp = senders_cnt + receivers_cnt

        avg_in = float(tx_stats.get("avg_in", in_amt / max(in_cnt, 1)))
        max_in = float(tx_stats.get("max_in", 0.0))
        avg_out = float(tx_stats.get("avg_out", out_amt / max(out_cnt, 1)))
        max_out = float(tx_stats.get("max_out", 0.0))
        active_days = int(tx_stats.get("active_days", 1))
        tx_per_day = float(tx_stats.get("tx_per_day", total_tx / max(active_days, 1)))
        top_channel = tx_stats.get("top_channel", "-")
        night_cnt = int(tx_stats.get("night_cnt", 0))
        night_pct = float(tx_stats.get("night_pct", rules.get("night_pct", 0.0)))
        unique_banks_cnt = int(tx_stats.get("unique_banks_cnt", 1))

        # ---------------------------------------------------------
        # ENGLISH MODE (100% Pure English)
        # ---------------------------------------------------------
        if lang == "en":
            if intent == "profile_discrepancy":
                ans = f"**KYC Profile vs. Financial Activity Discrepancy Analysis — Account `{acc}`**\n\n"
                if person_obj:
                    occ_raw = str(person_obj.get("person_occupation", "unknown"))
                    occ_en = translate_val(occ_raw, "occ", "en")
                    age = int(person_obj.get("person_age", 0))
                    edu_en = translate_val(person_obj.get("person_education"), "edu", "en")
                    mar_en = translate_val(person_obj.get("person_marital_status"), "marital", "en")
                    pid = person_obj.get("person_id", "-")
                    bnk = person_obj.get("bank", "-")
                    ans += (
                        f"**1. Declared Customer KYC Profile:**\n"
                        f"- **Person ID:** `{pid}` | **Bank ID:** `{bnk}`\n"
                        f"- **Demographics:** Age {age} | Declared Occupation: **{occ_en}** | Education: {edu_en} | Marital Status: {mar_en}\n\n"
                        f"**2. Actual Transaction Volume vs. Declared Profile:**\n"
                        f"- **Total Turnover:** THB {total_turnover:,.2f} across {total_tx} transactions ({in_cnt} incoming / {out_cnt} outgoing)\n"
                        f"- **Single Transfer Peak:** Max Outgoing THB {max_out:,.2f} (Avg THB {avg_out:,.2f}) | Max Incoming THB {max_in:,.2f}\n"
                        f"- **Account Dispersion:** Transacted with **{total_cp} unique accounts** ({senders_cnt} senders, {receivers_cnt} beneficiaries) across {unique_banks_cnt} banks\n\n"
                        f"**3. Specific Discrepancy Assessment:**\n"
                    )
                    is_high_volume = (out_amt >= 50_000 or avg_out >= 3_000 or total_turnover >= 150_000 or len(high_risk_cps) >= 5)
                    ratio_str = f"exceeds incoming by {out_amt / max(in_amt, 1):.1f}x" if out_amt > in_amt else f"represents {out_amt / max(in_amt, 1) * 100:.1f}% of incoming funds"

                    if occ_raw in ["unemployed", "student", "retired"] or age <= 22 or age >= 60:
                        if is_high_volume or len(high_risk_cps) > 0:
                            ans += (
                                f"- **High Conflict Detected:** The account holder is registered as **{occ_en} (Age {age})**, a segment without regular commercial cash-flow, "
                                f"yet executed {out_cnt} outgoing transfers totaling THB {out_amt:,.2f} to {receivers_cnt} different accounts ({ratio_str}).\n"
                                f"- **Mule Typology Match:** High account dispersion combined with an inactive-income occupation strongly indicates a **rented or nominee mule account**.\n"
                            )
                        else:
                            ans += (
                                f"- **Consistent with Standard Living Expenses:** The account holder is registered as **{occ_en} (Age {age})**. "
                                f"Although {out_cnt} outgoing transfers were executed, the average amount is only **THB {avg_out:,.2f}** per transfer (Total THB {out_amt:,.2f}), "
                                f"which is consistent with everyday retail subsistence spending.\n"
                                f"- **Conclusion:** Transaction turnover aligns with individual economic status; does not match nominee mule typologies.\n"
                            )
                    else:
                        if is_high_volume or len(high_risk_cps) > 0:
                            ans += (
                                f"- **Behavioral Deviation:** While the holder declares **{occ_en} (Age {age})**, the transaction network ({total_cp} accounts across {unique_banks_cnt} banks) "
                                f"and linkage to {len(high_risk_cps)} high-risk accounts deviate from standard personal banking behavior.\n"
                            )
                        else:
                            ans += (
                                f"- **Behavior Consistent with Profile:** Average transaction size (THB {avg_out:,.2f}) and total volume (THB {out_amt:,.2f}) "
                                f"align with standard personal expenditure for {occ_en}.\n"
                            )
                elif merchant_obj:
                    mid = merchant_obj.get("merchant_id", "-")
                    bnk = merchant_obj.get("bank", "-")
                    ind_en = translate_val(merchant_obj.get("industry"), "ind", "en")
                    m_type_en = translate_val(merchant_obj.get("type"), "m_type", "en")
                    cap = float(merchant_obj.get("registered_capital", 0.0))
                    rep_id = merchant_obj.get("legal_representative_id", "-")
                    rep_line = f"`{rep_id}`"
                    if rep_person_obj:
                        r_occ = translate_val(rep_person_obj.get("person_occupation"), "occ", "en")
                        r_age = rep_person_obj.get("person_age", "-")
                        r_edu = translate_val(rep_person_obj.get("person_education"), "edu", "en")
                        rep_line = f"`{rep_id}` (Age {r_age}, Occupation: **{r_occ}**, Education: {r_edu})"
                    ans += (
                        f"**1. Corporate KYC & Legal Representative Profile:**\n"
                        f"- **Merchant ID:** `{mid}` | **Bank ID:** `{bnk}` | **Type:** {m_type_en} ({ind_en})\n"
                        f"- **Registered Capital:** THB {cap:,.0f}\n"
                        f"- **Legal Representative (Authorized Director):** {rep_line}\n\n"
                        f"**2. Corporate Activity vs. Representative Profile:**\n"
                        f"- **Total Turnover:** THB {total_turnover:,.2f} ({total_tx} txns across {total_cp} accounts)\n"
                        f"- **Assessment:** Corporate account is linked to {len(high_risk_cps)} high-risk mule accounts; verify whether the legal representative ({rep_line}) is acting as a nominee director for a shell company.\n"
                    )
                other_linked = [x for x in linked_info if x != acc]
                if other_linked:
                    ans += f"\n- **Shared Ownership Ring:** Holder `{owner_id}` also controls {len(other_linked)} other account(s): `{', '.join(other_linked)}`."
                else:
                    ans += f"\n- **Shared Ownership Ring:** No other accounts registered under holder `{owner_id}` in system."
                return ans

            elif intent in ["flow_and_law", "legal_authority"]:
                ans = f"**Fund Flow Tracing & Statutory Mandate — Account `{acc}` ({tier})**\n\n"
                if top_senders:
                    ans += "**1. Top Upstream Senders (Incoming Funds):**\n"
                    for s in top_senders[:3]:
                        ans += f"- Account `{s['account']}` (Bank `{s['bank']}` | {s['tier']}): {s['count']} txns, Total **THB {s['sum']:,.2f}**\n"
                if top_receivers:
                    ans += "\n**2. Top Downstream Beneficiaries (Layer-2 Outgoing):**\n"
                    for r in top_receivers[:3]:
                        ans += f"- Account `{r['account']}` (Bank `{r['bank']}` | {r['tier']}): {r['count']} txns, Total **THB {r['sum']:,.2f}**\n"
                ans += (
                    f"\n**3. Applicable Statutory Authority & Protocol:**\n"
                    f"- **Mandated Action:** {recom.get('title', '-')}\n"
                    f"- **Legal Basis:** {recom.get('legal_basis', '-')}\n"
                )
                for step in recom.get("protocol", []):
                    ans += f"  - {step}\n"
                return ans

            else:
                ans = (
                    f"**Key Risk Drivers Breakdown — Account `{acc}` ({tier} | Score: {risk_score:.1f}/100)**\n\n"
                    f"**1. Cash Flow Imbalance & Velocity:**\n"
                    f"- **Incoming:** THB {in_amt:,.2f} ({in_cnt} txns from {senders_cnt} senders | Avg THB {avg_in:,.2f}, Max THB {max_in:,.2f})\n"
                    f"- **Outgoing:** THB {out_amt:,.2f} ({out_cnt} txns to {receivers_cnt} beneficiaries | Avg THB {avg_out:,.2f}, Max THB {max_out:,.2f})\n"
                )
                if out_amt > in_amt * 1.5 and out_cnt > 0:
                    ans += f"- **Flow Pattern:** Outgoing volume is **{out_amt / max(in_amt, 1):.1f}x higher** than incoming, dispersing funds across {receivers_cnt} downstream accounts.\n"
                elif in_amt > out_amt * 1.5 and in_cnt > 0:
                    ans += f"- **Flow Pattern:** Incoming volume is **{in_amt / max(out_amt, 1):.1f}x higher** than outgoing, aggregating funds from {senders_cnt} upstream senders.\n"
                ans += f"\n**2. Network Contagion & Illicit Links:**\n"
                if high_risk_cps:
                    ex_str = ", ".join([f"`{c['account']}` (Bank `{c['bank']}`, {c['tier']})" for c in high_risk_cps[:5]])
                    ans += f"- **High-Risk Linked Accounts:** Directly transacted with **{len(high_risk_cps)} high-risk mule account(s)** (including {ex_str}).\n"
                if laund_cnt > 0:
                    ans += f"- **Confirmed Laundering Flows:** {laund_cnt} confirmed transactions totaling **THB {laund_amt:,.2f}**.\n"
                else:
                    ans += f"- **Network Risk:** Even where individual transactions are pending police adjudication, direct transfers with blacklisted/high-risk accounts drive the {risk_score:.1f}/100 risk score.\n"
                ans += (
                    f"\n**3. Behavioral & Temporal Indicators:**\n"
                    f"- **Activity Frequency:** {total_tx} txns over {active_days} active days ({tx_per_day:.1f} txns/day) | Primary Channel: **{top_channel}**\n"
                    f"- **Late-Night Transfers (00:00–05:59):** {night_cnt} txns ({night_pct:.1f}%)\n"
                )
                return ans

        # ---------------------------------------------------------
        # THAI MODE (100% Pure Thai - No Leftover English Words)
        # ---------------------------------------------------------
        if intent == "profile_discrepancy":
            ans = f"**การวิเคราะห์ความขัดแย้งระหว่างข้อมูลผู้ถือบัญชีกับธุรกรรมจริง — บัญชี `{acc}` ({tier})**\n\n"
            if person_obj:
                occ_raw = str(person_obj.get("person_occupation", "unknown"))
                occ_th = translate_val(occ_raw, "occ", "th")
                age = int(person_obj.get("person_age", 0))
                edu_th = translate_val(person_obj.get("person_education"), "edu", "th")
                mar_th = translate_val(person_obj.get("person_marital_status"), "marital", "th")
                gen_th = translate_val(person_obj.get("person_gender"), "gender", "th")
                pid = person_obj.get("person_id", "-")
                bnk = person_obj.get("bank", "-")

                ans += (
                    f"**1. ข้อมูลผู้ถือบัญชีตามทะเบียนลูกค้า:**\n"
                    f"- **รหัสประจำตัวบุคคล:** `{pid}` | **รหัสธนาคาร:** `{bnk}`\n"
                    f"- **ข้อมูลส่วนบุคคล:** เพศ{gen_th} อายุ **{age} ปี** | อาชีพที่แจ้ง: **{occ_th}** | การศึกษา: {edu_th} | สถานภาพ: {mar_th}\n\n"
                    f"**2. ปริมาณธุรกรรมจริงเทียบกับสถานะทางเศรษฐกิจ:**\n"
                    f"- **ยอดเงินหมุนเวียนรวมรับ-จ่าย:** ฿{total_turnover:,.2f} บาท จากทั้งหมด {total_tx} รายการ (รับเข้า {in_cnt} ครั้ง / โอนออก {out_cnt} ครั้ง)\n"
                    f"- **ขนาดธุรกรรมต่อครั้ง:** โอนออกเฉลี่ยครั้งละ ฿{avg_out:,.2f} บาท (สูงสุดครั้งละ **฿{max_out:,.2f} บาท**) | รับเข้าเฉลี่ยครั้งละ ฿{avg_in:,.2f} บาท (สูงสุด ฿{max_in:,.2f} บาท)\n"
                    f"- **การกระจายตัวของบัญชีที่รับ-โอน:** ทำธุรกรรมเชื่อมโยงกับบัญชีอื่นถึง **{total_cp} บัญชี** (ผู้โอนเข้า {senders_cnt} บัญชี และผู้รับโอนออก {receivers_cnt} บัญชี) ข้าม **{unique_banks_cnt} ธนาคาร**\n\n"
                    f"**3. ข้อสังเกตความขัดแย้งที่ตรวจพบ:**\n"
                )
                is_high_volume = (out_amt >= 50_000 or avg_out >= 3_000 or total_turnover >= 150_000 or len(high_risk_cps) >= 5)
                ratio_str = f"สูงกว่ายอดรับเข้า {out_amt / max(in_amt, 1):.1f} เท่า" if out_amt > in_amt else f"คิดเป็น {out_amt / max(in_amt, 1) * 100:.1f}% ของยอดรับเข้า"

                if occ_raw in ["unemployed", "student", "retired"] or age <= 22 or age >= 60:
                    if is_high_volume or len(high_risk_cps) > 0:
                        ans += (
                            f"- **พบความขัดแย้งระดับสูง:** ผู้ถือบัญชีแจ้งสถานะอาชีพ **`{occ_th}` (อายุ {age} ปี)** ซึ่งเป็นกลุ่มที่ไม่มีรายได้ประจำจากการประกอบธุรกิจ "
                            f"แต่กลับมีการโอนเงินออกถึง **{out_cnt} รายการ รวม ฿{out_amt:,.2f} บาท ไปยัง {receivers_cnt} บัญชีปลายทาง** ({ratio_str})\n"
                            f"- **ข้อบ่งชี้บัญชีม้ารับจ้างเปิด:** การที่ผู้มีสถานะ `{occ_th}` มีการโอนเชื่อมโยงถึง {total_cp} บัญชี และพัวพันกับบัญชีม้าความเสี่ยงสูงอีก {len(high_risk_cps)} บัญชี "
                            f"ไม่สอดคล้องกับการใช้บัญชีเพื่อการยังชีพปกติ เข้าข่ายลักษณะ **บัญชีม้ารับจ้างเปิดหรือถูกสวมสิทธิ์** ตามเกณฑ์ ธปท.\n"
                        )
                    else:
                        ans += (
                            f"- **สอดคล้องกับการดำรงชีพปกติ (ไม่พบความขัดแย้งรุนแรง):** ผู้ถือบัญชีแจ้งสถานะอาชีพ **`{occ_th}` (อายุ {age} ปี)** "
                            f"แม้จะมีการโอนเงินออก {out_cnt} รายการ แต่มียอดโอนเฉลี่ยเพียงครั้งละ **฿{avg_out:,.2f} บาท** (รวมทั้งสิ้น ฿{out_amt:,.2f} บาท) "
                            f"ซึ่งสอดคล้องกับการใช้จ่ายในชีวิตประจำวันทั่วไปตามสถานะทางเศรษฐกิจ\n"
                            f"- **ข้อสรุป:** พฤติกรรมการหมุนเวียนของเงินอยู่ในเกณฑ์ปกติสำหรับการยังชีพ ไม่เข้าข่ายลักษณะบัญชีม้ารับจ้างเปิด\n"
                        )
                else:
                    if is_high_volume or len(high_risk_cps) > 0:
                        ans += (
                            f"- **พบความผิดปกติเทียบกับกลุ่มอาชีพ `{occ_th}`:** แม้ผู้ถือบัญชีจะแจ้งอาชีพ `{occ_th}` (อายุ {age} ปี) แต่รูปแบบการทำธุรกรรมที่เชื่อมโยงถึง **{total_cp} บัญชี** "
                            f"(โอนออกไปยัง {receivers_cnt} บัญชี รวม ฿{out_amt:,.2f} บาท) และมีเส้นทางเงินพัวพันกับบัญชีความเสี่ยงสูงในเครือข่ายถึง **{len(high_risk_cps)} บัญชี** "
                            f"เบี่ยงเบนไปจากพฤติกรรมบัญชีเงินเดือนหรือบัญชีใช้จ่ายส่วนบุคคลตามปกติอย่างมีนัยสำคัญ\n"
                        )
                    else:
                        ans += (
                            f"- **พฤติกรรมสอดคล้องกับกลุ่มอาชีพ `{occ_th}`:** ขนาดธุรกรรมเฉลี่ยครั้งละ ฿{avg_out:,.2f} บาท "
                            f"และยอดโอนรวม ฿{out_amt:,.2f} บาท สอดคล้องกับพฤติกรรมการใช้จ่ายส่วนบุคคลตามปกติ ไม่พบสัญญาณความผิดปกติเชิงโครงสร้าง\n"
                        )
            elif merchant_obj:
                mid = merchant_obj.get("merchant_id", "-")
                bnk = merchant_obj.get("bank", "-")
                ind_th = translate_val(merchant_obj.get("industry"), "ind", "th")
                m_type_th = translate_val(merchant_obj.get("type"), "m_type", "th")
                cap = float(merchant_obj.get("registered_capital", 0.0))
                rep_id = merchant_obj.get("legal_representative_id", "-")
                if rep_person_obj:
                    r_occ_raw = str(rep_person_obj.get("person_occupation", "unknown"))
                    r_occ_th = translate_val(r_occ_raw, "occ", "th")
                    r_age = int(rep_person_obj.get("person_age", 0))
                    r_edu_th = translate_val(rep_person_obj.get("person_education"), "edu", "th")
                    rep_detail = f"รหัสบุคคล `{rep_id}` (อายุ **{r_age} ปี** | อาชีพที่แจ้ง: **{r_occ_th}** | การศึกษา: **{r_edu_th}**)"
                else:
                    r_occ_raw = "unknown"
                    r_age = 35
                    rep_detail = f"รหัสบุคคล `{rep_id}`"

                ans += (
                    f"**1. ข้อมูลทะเบียนนิติบุคคลและผู้แทนตามกฎหมาย:**\n"
                    f"- **รหัสนิติบุคคล:** `{mid}` | **รหัสธนาคาร:** `{bnk}` | **ประเภท:** {m_type_th} (กลุ่มธุรกิจ: **{ind_th}**)\n"
                    f"- **ทุนจดทะเบียน:** ฿{cap:,.0f} บาท\n"
                    f"- **ผู้แทนนิติบุคคล (กรรมการผู้มีอำนาจลงนาม):** {rep_detail}\n\n"
                    f"**2. ปริมาณธุรกรรมและข้อสังเกตความขัดแย้ง:**\n"
                    f"- **ยอดเงินหมุนเวียนรวม:** ฿{total_turnover:,.2f} บาท ({total_tx} รายการ เชื่อมโยงกับ {total_cp} บัญชี)\n"
                )
                if r_occ_raw in ["unemployed", "student", "retired"] or r_age <= 24:
                    ans += (
                        f"- **ความขัดแย้งของข้อมูลกรรมการผู้มีอำนาจ:** นิติบุคคลแจ้งทุนจดทะเบียนสูงถึง **฿{cap:,.0f} บาท** แต่ผู้แทนนิติบุคคล (`{rep_id}`) "
                        f"กลับมีข้อมูลส่วนบุคคลเป็นเพียงผู้มีอายุ {r_age} ปี อาชีพ `{translate_val(r_occ_raw, 'occ', 'th')}` ซึ่งเข้าข่าย **การใช้ตัวแทนอำพราง (นอมินี) จดทะเบียนบริษัทม้าบังหน้า**\n"
                    )
                else:
                    ans += (
                        f"- **ความเสี่ยงด้านเส้นทางเงินของนิติบุคคล:** บัญชีนิติบุคคลกลุ่ม `{ind_th}` มีการโอนเงินเชื่อมโยงกับบัญชีม้าความเสี่ยงสูงในเครือข่ายถึง **{len(high_risk_cps)} บัญชี** "
                        f"ต้องตรวจสอบเอกสารการจัดตั้งบริษัทและแหล่งที่มาของรายได้จริงของกรรมการผู้มีอำนาจ (`{rep_id}`)\n"
                    )
            else:
                ans += f"- ไม่พบข้อมูลทะเบียนประวัติลูกค้าภายในระบบ (บัญชีภายนอก)\n"

            other_linked = [x for x in linked_info if x != acc]
            if other_linked:
                ans += f"\n**4. เครือข่ายบัญชีภายใต้ผู้ถือครองเดียวกัน (`{owner_id}`):**\n- ตรวจพบบัญชีอื่นที่เปิดโดยบุคคลหรือผู้แทนรายเดียวกันอีก **{len(other_linked)} บัญชี** ได้แก่ `{', '.join(other_linked)}` (ควรตรวจสอบความเชื่อมโยงพร้อมกันทั้งกลุ่ม)"
            else:
                ans += f"\n**4. การถือครองบัญชีอื่นในระบบ:**\n- ไม่พบบัญชีอื่นภายใต้รหัสผู้ถือบัญชี `{owner_id}` ในระบบ"
            return ans

        elif intent in ["flow_and_law", "legal_authority"]:
            retrieved = self.query(question, n_results=2, lang="th")
            ans = f"**การวิเคราะห์เส้นทางเงินและฐานอำนาจกฎหมาย — บัญชี `{acc}` ({tier})**\n\n"
            if top_senders:
                ans += "**1. บัญชีต้นทางที่โอนเงินเข้าสูงสุด:**\n"
                for s in top_senders[:3]:
                    ans += f"- บัญชี `{s['account']}` (ธนาคาร `{s['bank']}` | {s['tier']}): โอนเข้า {s['count']} ครั้ง รวม **฿{s['sum']:,.2f} บาท**\n"
            else:
                ans += "**1. บัญชีต้นทางที่โอนเงินเข้า:** ไม่มีรายการรับเงินเข้า\n"

            if top_receivers:
                ans += "\n**2. บัญชีปลายทางที่รับโอนเงินต่อสูงสุด (บัญชีแถว 2):**\n"
                for r in top_receivers[:3]:
                    ans += f"- บัญชี `{r['account']}` (ธนาคาร `{r['bank']}` | {r['tier']}): รับโอนออก {r['count']} ครั้ง รวม **฿{r['sum']:,.2f} บาท**\n"
            else:
                ans += "\n**2. บัญชีปลายทางที่รับโอนเงินต่อ:** ไม่มีรายการโอนเงินออก\n"

            ans += (
                f"\n**3. แนวทางสั่งการและฐานอำนาจตามกฎหมายสำหรับบัญชีระดับ `{tier}`:**\n"
                f"- **มาตรการที่แนะนำ:** {recom.get('title', '-')}\n"
                f"- **ฐานอำนาจกฎหมาย:** {recom.get('legal_basis', '-')}\n"
            )
            for step in recom.get("protocol", []):
                ans += f"  - {step}\n"

            if retrieved:
                ans += f"\n**ข้อกฎหมายอ้างอิงเพิ่มเติม ({retrieved[0]['title']}):**\n{retrieved[0]['text']}"
            return ans

        else:
            ans = (
                f"**วิเคราะห์สาเหตุความเสี่ยงเชิงลึกของบัญชี `{acc}` ({tier} | คะแนนความเสี่ยง {risk_score:.1f}/100)**\n\n"
                f"**1. พฤติกรรมการเคลื่อนไหวของกระแสเงิน:**\n"
                f"- **ฝั่งรับเงินเข้า:** รวม **฿{in_amt:,.2f} บาท** ({in_cnt} รายการ จาก {senders_cnt} บัญชีต้นทาง | เฉลี่ยครั้งละ ฿{avg_in:,.2f} บาท, สูงสุด ฿{max_in:,.2f} บาท)\n"
                f"- **ฝั่งโอนเงินออก:** รวม **฿{out_amt:,.2f} บาท** ({out_cnt} รายการ ไปยัง {receivers_cnt} บัญชีปลายทาง | เฉลี่ยครั้งละ ฿{avg_out:,.2f} บาท, สูงสุด ฿{max_out:,.2f} บาท)\n"
            )
            if out_amt > in_amt * 1.5 and out_cnt > 0:
                ans += (
                    f"- **สาเหตุความผิดปกติของกระแสเงิน:** ยอดโอนออกสูงกว่ายอดรับเข้าถึง **{out_amt / max(in_amt, 1):.1f} เท่า** "
                    f"และมีความถี่การโอนออกสูงถึง {out_cnt} ครั้ง กระจายไปยังบัญชีปลายทาง {receivers_cnt} บัญชี เข้าข่ายพฤติกรรม **บัญชีพักและกระจายเงินออกหลายทอด**\n"
                )
            elif in_amt > out_amt * 1.5 and in_cnt > 0:
                ans += (
                    f"- **สาเหตุความผิดปกติของกระแสเงิน:** ยอดรับเงินเข้าสูงกว่ายอดโอนออก **{in_amt / max(out_amt, 1):.1f} เท่า** "
                    f"โดยรับโอนสะสมจากบัญชีต้นทางถึง {senders_cnt} บัญชี เข้าข่ายพฤติกรรม **บัญชีรวบรวมเงินจากหลายต้นทาง**\n"
                )
            elif in_amt > 0 and out_amt > 0:
                ans += (
                    f"- **สาเหตุความผิดปกติของกระแสเงิน:** สัดส่วนยอดโอนออกคิดเป็น **{(out_amt / max(in_amt, 1)) * 100:.1f}%** ของยอดเงินที่รับเข้า "
                    f"แสดงพฤติกรรมรับเงินแล้วผันออกต่อทันทีโดยไม่คงเหลือเงินติดบัญชี (**บัญชีทางผ่าน**)\n"
                )

            ans += "\n**2. ความเสี่ยงจากการเชื่อมโยงเครือข่ายบัญชีม้า:**\n"
            if high_risk_cps:
                ex_list = [f"`{c['account']}` (ธนาคาร `{c['bank']}`, {c['tier']})" for c in high_risk_cps[:4]]
                ans += (
                    f"- **ตรวจพบการรับ-โอนกับบัญชีความเสี่ยงสูงโดยตรง:** บัญชีนี้มีธุรกรรมเชื่อมโยงโดยตรงกับบัญชีม้าความเสี่ยงสูงในเครือข่ายจำนวน **{len(high_risk_cps)} บัญชี** "
                    f"ได้แก่ {', '.join(ex_list)}{' และบัญชีอื่นในเครือข่าย' if len(high_risk_cps) > 4 else ''}\n"
                )
            if laund_cnt > 0:
                ans += f"- **ธุรกรรมฟอกเงินที่ยืนยันแล้ว:** พบธุรกรรมในเส้นทางฟอกเงินโดยตรงจำนวน **{laund_cnt} รายการ** มูลค่ารวม **฿{laund_amt:,.2f} บาท**\n"
            else:
                ans += (
                    f"- **เหตุผลที่คะแนนความเสี่ยงอยู่ที่ {risk_score:.1f}/100:** แม้ธุรกรรมรายชิ้นของบัญชีนี้จะยังไม่อยู่ในรายการแจ้งความยืนยันฟอกเงินโดยตรง "
                    f"แต่การมีเส้นทางเงินเชื่อมโยงกับกลุ่มบัญชีม้าดำ/ม้าเทาหลายบัญชี และความไม่สมดุลของกระแสเงิน ส่งผลให้ระบบจัดชั้นไว้ที่ **`{tier}`**\n"
                )

            p_flag = "ตรวจพบ" if rules.get("pass_through") else "ไม่พบ"
            v_flag = "ตรวจพบ" if rules.get("rapid_outflow") else "ไม่พบ"
            s_flag = "ตรวจพบ" if rules.get("structuring") else "ไม่พบ"
            ans += (
                f"\n**3. ตัวชี้วัดพฤติกรรมและช่วงเวลาทำธุรกรรม:**\n"
                f"- **ความถี่และช่องทางหลัก:** ทำรายการรวม {total_tx} ครั้ง ในช่วง {active_days} วันที่มีความเคลื่อนไหว (เฉลี่ย **{tx_per_day:.1f} รายการ/วัน**) | ช่องทางหลัก: **{top_channel}**\n"
                f"- **ธุรกรรมช่วงวิกาล (00:00–05:59 น.):** จำนวน **{night_cnt} รายการ** (คิดเป็น **{night_pct:.1f}%** ของธุรกรรมทั้งหมด)\n"
                f"- **สัญญาณพฤติกรรมเฉพาะ:** การพักเงินแล้วส่งต่อทันที (`{p_flag}`) | การผันเงินออกเร็วภายใน 48 ชม. (`{v_flag}`) | การแบ่งซอยยอดโอน (`{s_flag}`)"
            )
            return ans


def generate_aml_str_report(
    account_id: str,
    narrative: Dict[str, Any],
    tx_df: Optional[pd.DataFrame] = None,
    investigator_name: str = "เจ้าหน้าที่กำกับการปฏิบัติตามกฎเกณฑ์",
    lang: str = "th",
    bank_lookup: Optional[Dict[str, str]] = None,
) -> str:
    now = datetime.now()
    report_no = f"STR-{now.strftime('%Y%m%d')}-{account_id}"
    date_str = now.strftime("%d/%m/%Y %H:%M:%S")
    bank_map = bank_lookup or {}

    tier = narrative.get("tier", "-")
    risk_score = narrative.get("risk_score", 0.0)
    overall_assessment = narrative.get("overall_assessment", "")
    demo = narrative.get("demo_findings", ["-"])[0]
    red_flags = narrative.get("red_flags", [])
    flow = narrative.get("flow_summary", {})
    recom = narrative.get("recommendation", {})
    rules = narrative.get("rules_detail", {})

    tx_sample_rows = []
    if tx_df is not None and not tx_df.empty:
        acc_tx = tx_df[(tx_df["From Account"] == account_id) | (tx_df["To Account"] == account_id)]
        acc_tx_sorted = acc_tx.sort_values(by=["Is Laundering", "Amount Received"], ascending=False).head(12)
        for _, r in acc_tx_sorted.iterrows():
            other_acc = str(r["To Account"]) if str(r["From Account"]) == account_id else str(r["From Account"])
            other_bnk = bank_map.get(other_acc, "-")
            if lang == "th":
                direction = "โอนออก" if str(r["From Account"]) == account_id else "รับเข้า"
                chan = translate_val(r.get("Payment Format"), "chan", "th")
                is_laund = "ยืนยันเส้นทางฟอกเงิน" if int(r.get("Is Laundering", 0)) == 1 else "ธุรกรรมทั่วไป"
                tx_sample_rows.append(
                    f"| {str(r.get('Timestamp', ''))[:19]} | {direction} | {other_bnk} | {other_acc} | ฿{float(r['Amount Received']):,.2f} บาท | {chan} | {is_laund} |"
                )
            else:
                direction = "Outgoing" if str(r["From Account"]) == account_id else "Incoming"
                chan = translate_val(r.get("Payment Format"), "chan", "en")
                is_laund = "Confirmed Laundering" if int(r.get("Is Laundering", 0)) == 1 else "Standard"
                tx_sample_rows.append(
                    f"| {str(r.get('Timestamp', ''))[:19]} | {direction} | {other_bnk} | {other_acc} | THB {float(r['Amount Received']):,.2f} | {chan} | {is_laund} |"
                )

    tx_table_str = "\n".join(tx_sample_rows) if tx_sample_rows else "| - | - | - | - | - | - | - |"
    red_flags_str = "\n".join([f"- {rf}" for rf in red_flags]) if red_flags else ("- ไม่พบความเสี่ยงวิกฤตเพิ่มเติม" if lang == "th" else "- No additional critical flags")

    if lang == "en":
        return f"""# SUSPICIOUS TRANSACTION REPORT (AMLO FORM 1-03)
**Report Reference:** {report_no}  
**Date & Time:** {date_str}  
**Reporting Institution:** Commercial Bank AOC 1441 Member  
**Compliance Officer:** {investigator_name}  

---

### PART 1: SUBJECT & ACCOUNT PROFILE
- **Account Number:** `{account_id}`
- **Customer KYC Profile:** {demo}
- **BOT Risk Classification:** {tier}
- **Unified Risk Score:** {risk_score:.1f} / 100

---

### PART 2: SUSPICIOUS CIRCUMSTANCES & TRANSACTION ANALYSIS
1. **Executive Assessment:** {overall_assessment}
2. **Detected Risk Indicators:**
{red_flags_str}
3. **Behavioral Breakdown:**
   - Rapid Pass-Through Activity: {'Detected' if rules.get('pass_through') else 'Not Detected'}
   - 48-Hour Rapid Outflow (>80%): {'Detected' if rules.get('rapid_outflow') else 'Not Detected'}
   - Transaction Structuring: {'Detected' if rules.get('structuring') else 'Not Detected'}
   - Late-Night Activity Ratio (00:00-05:59): {rules.get('night_pct', 0.0):.1f}%
4. **Financial Flow Summary:**
   - Total Incoming: THB {flow.get('total_in_amt', 0):,.2f} ({flow.get('in_tx_count', 0)} transactions from {flow.get('senders_count', 0)} accounts)
   - Total Outgoing: THB {flow.get('total_out_amt', 0):,.2f} ({flow.get('out_tx_count', 0)} transactions to {flow.get('receivers_count', 0)} accounts)
   - Confirmed Illicit Laundering Links: {flow.get('laundering_count', 0)} transactions (THB {flow.get('laundering_amt', 0):,.2f})

5. **Critical Transaction Timeline (Top 12 Records):**

| Timestamp | Direction | Bank ID | Account No. | Amount | Channel | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{tx_table_str}

---

### PART 3: INSTITUTIONAL ACTIONS & STATUTORY AUTHORITY
- **Action Enforced:** {recom.get('title', '-')}
- **Legal Authority:** {recom.get('legal_basis', '-')}
- **Operational Protocol Executed:**
""" + "\n".join([f"  {s}" for s in recom.get("protocol", [])]) + f"""

**Signed:** ................................................................... ({investigator_name})
"""

    return f"""# แบบรายงานธุรกรรมที่มีเหตุอันควรสงสัย (แบบ ปปง. 1-03)
**เลขที่เอกสารรายงาน:** {report_no}  
**วันที่และเวลาที่จัดทำ:** {date_str}  
**สถาบันการเงินผู้รายงาน:** ธนาคารพาณิชย์สมาชิกศูนย์ปฏิบัติการ 1441  
**เจ้าหน้าที่ผู้รับผิดชอบ:** {investigator_name}  

---

### ส่วนที่ 1: ข้อมูลผู้ถูกรายงานและบัญชีเงินฝาก
- **เลขที่บัญชีเงินฝาก:** `{account_id}`
- **ข้อมูลเจ้าของบัญชี:** {demo}
- **การจัดชั้นความเสี่ยงตามเกณฑ์ ธปท.:** {tier}
- **คะแนนความเสี่ยงประเมินรวม:** {risk_score:.1f} / 100

---

### ส่วนที่ 2: พฤติการณ์และข้อเท็จจริงอันควรสงสัยโดยละเอียด
1. **บทสรุปพฤติการณ์:** {overall_assessment}
2. **สิ่งบ่งชี้ความเสี่ยงที่ตรวจพบ:**
{red_flags_str}
3. **รายละเอียดตัวชี้วัดพฤติกรรมผิดปกติ:**
   - การพักเงินแล้วโอนออกทันที: {'ตรวจพบความผิดปกติ' if rules.get('pass_through') else 'ปกติ'}
   - การผันเงินออกเกิน 80% ภายใน 48 ชั่วโมง: {'ตรวจพบความผิดปกติ' if rules.get('rapid_outflow') else 'ปกติ'}
   - การแบ่งซอยยอดโอนเลี่ยงเกณฑ์ตรวจสอบ: {'ตรวจพบความผิดปกติ' if rules.get('structuring') else 'ปกติ'}
   - สัดส่วนการทำธุรกรรมช่วงวิกาล (00:00 - 05:59 น.): {rules.get('night_pct', 0.0):.1f}%
4. **สถิติกระแสการเงินหมุนเวียนในรอบการตรวจสอบ:**
   - ยอดเงินรับเข้ารวมทั้งสิ้น: ฿{flow.get('total_in_amt', 0):,.2f} บาท ({flow.get('in_tx_count', 0)} รายการ จาก {flow.get('senders_count', 0)} บัญชีต้นทาง)
   - ยอดเงินโอนออกรวมทั้งสิ้น: ฿{flow.get('total_out_amt', 0):,.2f} บาท ({flow.get('out_tx_count', 0)} รายการ ไปยัง {flow.get('receivers_count', 0)} บัญชีปลายทาง)
   - ธุรกรรมที่เชื่อมโยงกับเส้นทางฟอกเงินในคดี: {flow.get('laundering_count', 0)} รายการ (มูลค่ารวม ฿{flow.get('laundering_amt', 0):,.2f} บาท)

5. **ตารางรายการธุรกรรมสำคัญที่ตรวจพบ (12 รายการสูงสุด):**

| วัน-เวลาที่ทำรายการ | ทิศทาง | ธนาคาร | เลขที่บัญชี | จำนวนเงิน | ช่องทางธุรกรรม | สถานะการตรวจสอบ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{tx_table_str}

---

### ส่วนที่ 3: การดำเนินการของสถาบันการเงินและข้อกฎหมายอ้างอิง
- **มาตรการที่สั่งการ:** {recom.get('title', '-')}
- **ฐานอำนาจตามกฎหมาย:** {recom.get('legal_basis', '-')}
- **ขั้นตอนการปฏิบัติงานที่ได้ดำเนินการแล้ว:**
""" + "\n".join([f"  {s}" for s in recom.get("protocol", [])]) + f"""

**ลงชื่อ:** ................................................................... ({investigator_name})
"""


def generate_aml_str_html(
    account_id: str,
    narrative: Dict[str, Any],
    tx_df: Optional[pd.DataFrame] = None,
    investigator_name: str = "เจ้าหน้าที่กำกับการปฏิบัติตามกฎเกณฑ์",
    lang: str = "th",
    bank_lookup: Optional[Dict[str, str]] = None,
) -> str:
    """
    Generates a formal, print-ready Official HTML document for AMLO Form 1-03 (100% Thai or 100% English).
    """
    now = datetime.now()
    report_no = f"STR-{now.strftime('%Y%m%d')}-{account_id}"
    date_str = now.strftime("%d/%m/%Y %H:%M:%S")
    bank_map = bank_lookup or {}

    tier = narrative.get("tier", "-")
    risk_score = narrative.get("risk_score", 0.0)
    overall_assessment = narrative.get("overall_assessment", "")
    demo = narrative.get("demo_findings", ["-"])[0]
    red_flags = narrative.get("red_flags", [])
    flow = narrative.get("flow_summary", {})
    recom = narrative.get("recommendation", {})

    tx_rows_html = ""
    if tx_df is not None and not tx_df.empty:
        acc_tx = tx_df[(tx_df["From Account"] == account_id) | (tx_df["To Account"] == account_id)]
        acc_tx_sorted = acc_tx.sort_values(by=["Is Laundering", "Amount Received"], ascending=False).head(12)
        for _, r in acc_tx_sorted.iterrows():
            direction = ("โอนออก" if lang == "th" else "Outgoing") if str(r["From Account"]) == account_id else ("รับเข้า" if lang == "th" else "Incoming")
            other_acc = str(r["To Account"]) if str(r["From Account"]) == account_id else str(r["From Account"])
            other_bnk = bank_map.get(other_acc, "-")
            chan = translate_val(r.get("Payment Format"), "chan", lang)
            is_laund = int(r.get("Is Laundering", 0)) == 1
            status_lbl = ("ยืนยันเส้นทางฟอกเงิน" if lang == "th" else "Confirmed Laundering") if is_laund else ("ทั่วไป" if lang == "th" else "Standard")
            row_style = "color: #B91C1C; font-weight: 600;" if is_laund else ""
            amt_str = f"฿{float(r['Amount Received']):,.2f}" if lang == "th" else f"THB {float(r['Amount Received']):,.2f}"
            tx_rows_html += f"""
            <tr style="{row_style}">
                <td>{str(r.get('Timestamp', ''))[:19]}</td>
                <td>{direction}</td>
                <td>{other_bnk}</td>
                <td>{other_acc}</td>
                <td style="text-align: right;">{amt_str}</td>
                <td>{chan}</td>
                <td>{status_lbl}</td>
            </tr>
            """

    flags_html = "".join([f"<li>{rf}</li>" for rf in red_flags]) if red_flags else "<li>-</li>"
    proto_html = "".join([f"<li>{s}</li>" for s in recom.get("protocol", [])])

    if lang == "en":
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{report_no}</title>
    <link href="https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        body {{ font-family: 'Sarabun', sans-serif; font-size: 13px; color: #111827; background: #F3F4F6; margin: 0; padding: 20px; }}
        .page {{ max-width: 820px; margin: 0 auto; background: #FFFFFF; padding: 32px 40px; border: 1px solid #D1D5DB; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
        .header-box {{ border-bottom: 2px solid #111827; padding-bottom: 12px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: flex-start; }}
        .doc-title {{ font-size: 18px; font-weight: 700; margin: 0; }}
        .doc-sub {{ font-size: 13px; color: #4B5563; margin-top: 2px; }}
        .meta-box {{ text-align: right; font-size: 12px; line-height: 1.5; }}
        .sec-title {{ background: #1F2937; color: #FFFFFF; padding: 5px 10px; font-weight: 600; font-size: 13.5px; margin-top: 18px; margin-bottom: 10px; }}
        table.info-grid {{ width: 100%; border-collapse: collapse; margin-bottom: 10px; }}
        table.info-grid td {{ padding: 5px 8px; border: 1px solid #E5E7EB; }}
        table.tx-table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 8px; }}
        table.tx-table th, table.tx-table td {{ border: 1px solid #D1D5DB; padding: 5px 8px; text-align: left; }}
        table.tx-table th {{ background: #F3F4F6; font-weight: 600; }}
        .sign-area {{ margin-top: 32px; text-align: right; padding-right: 20px; }}
        .print-btn {{ background: #1D4ED8; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-family: 'Sarabun', sans-serif; font-size: 13px; margin-bottom: 12px; }}
        @media print {{ body {{ background: #FFFFFF; padding: 0; }} .page {{ box-shadow: none; border: none; padding: 10px; }} .no-print {{ display: none; }} }}
    </style>
</head>
<body>
    <div style="max-width: 820px; margin: 0 auto;" class="no-print">
        <button class="print-btn" onclick="window.print()">Print Official Report / Save as PDF</button>
    </div>
    <div class="page">
        <div class="header-box">
            <div>
                <div class="doc-title">SUSPICIOUS TRANSACTION REPORT (AMLO FORM 1-03)</div>
                <div class="doc-sub">Anti-Money Laundering Office (AMLO)</div>
            </div>
            <div class="meta-box">
                <div><b>Report No:</b> {report_no}</div>
                <div><b>Date:</b> {date_str}</div>
                <div><b>Officer:</b> {investigator_name}</div>
            </div>
        </div>

        <div class="sec-title">PART 1: SUBJECT & ACCOUNT PROFILE</div>
        <table class="info-grid">
            <tr>
                <td width="25%"><b>Account Number:</b></td>
                <td width="25%">{account_id}</td>
                <td width="25%"><b>BOT Risk Tier:</b></td>
                <td width="25%"><b>{tier}</b></td>
            </tr>
            <tr>
                <td><b>Customer KYC Profile:</b></td>
                <td colspan="3">{demo}</td>
            </tr>
            <tr>
                <td><b>Unified Risk Score:</b></td>
                <td>{risk_score:.1f} / 100</td>
                <td><b>Confirmed Laundering:</b></td>
                <td>{flow.get('laundering_count', 0)} txns (THB {flow.get('laundering_amt', 0):,.2f})</td>
            </tr>
        </table>

        <div class="sec-title">PART 2: SUSPICIOUS CIRCUMSTANCES & TRANSACTION ANALYSIS</div>
        <p style="margin: 6px 0;"><b>Executive Assessment:</b> {overall_assessment}</p>
        <p style="margin: 6px 0;"><b>Detected Risk Indicators:</b></p>
        <ul style="margin: 4px 0 10px 20px;">{flags_html}</ul>

        <table class="info-grid">
            <tr>
                <td><b>Total Incoming:</b> THB {flow.get('total_in_amt', 0):,.2f} ({flow.get('in_tx_count', 0)} txns / {flow.get('senders_count', 0)} accounts)</td>
                <td><b>Total Outgoing:</b> THB {flow.get('total_out_amt', 0):,.2f} ({flow.get('out_tx_count', 0)} txns / {flow.get('receivers_count', 0)} accounts)</td>
            </tr>
        </table>

        <div><b>Key Transaction Records (Top 12):</b></div>
        <table class="tx-table">
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Direction</th>
                    <th>Bank ID</th>
                    <th>Account No.</th>
                    <th style="text-align: right;">Amount (THB)</th>
                    <th>Channel</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {tx_rows_html}
            </tbody>
        </table>

        <div class="sec-title">PART 3: INSTITUTIONAL ACTIONS & STATUTORY AUTHORITY</div>
        <p style="margin: 4px 0;"><b>Enforced Action:</b> {recom.get('title', '-')}</p>
        <p style="margin: 4px 0;"><b>Legal Authority:</b> {recom.get('legal_basis', '-')}</p>
        <ul style="margin: 4px 0 10px 20px;">{proto_html}</ul>

        <div class="sign-area">
            <div>Signed ................................................................... Reporting Officer</div>
            <div style="margin-top: 4px;">({investigator_name})</div>
            <div style="font-size: 11.5px; color: #4B5563;">Financial Crime Compliance & AML Investigation Division</div>
        </div>
    </div>
</body>
</html>"""

    return f"""<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <title>{report_no}</title>
    <link href="https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        body {{ font-family: 'Sarabun', sans-serif; font-size: 13px; color: #111827; background: #F3F4F6; margin: 0; padding: 20px; }}
        .page {{ max-width: 820px; margin: 0 auto; background: #FFFFFF; padding: 32px 40px; border: 1px solid #D1D5DB; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
        .header-box {{ border-bottom: 2px solid #111827; padding-bottom: 12px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: flex-start; }}
        .doc-title {{ font-size: 18px; font-weight: 700; margin: 0; }}
        .doc-sub {{ font-size: 13px; color: #4B5563; margin-top: 2px; }}
        .meta-box {{ text-align: right; font-size: 12px; line-height: 1.5; }}
        .sec-title {{ background: #1F2937; color: #FFFFFF; padding: 5px 10px; font-weight: 600; font-size: 13.5px; margin-top: 18px; margin-bottom: 10px; }}
        table.info-grid {{ width: 100%; border-collapse: collapse; margin-bottom: 10px; }}
        table.info-grid td {{ padding: 5px 8px; border: 1px solid #E5E7EB; }}
        table.tx-table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 8px; }}
        table.tx-table th, table.tx-table td {{ border: 1px solid #D1D5DB; padding: 5px 8px; text-align: left; }}
        table.tx-table th {{ background: #F3F4F6; font-weight: 600; }}
        .sign-area {{ margin-top: 32px; text-align: right; padding-right: 20px; }}
        .print-btn {{ background: #1D4ED8; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-family: 'Sarabun', sans-serif; font-size: 13px; margin-bottom: 12px; }}
        @media print {{ body {{ background: #FFFFFF; padding: 0; }} .page {{ box-shadow: none; border: none; padding: 10px; }} .no-print {{ display: none; }} }}
    </style>
</head>
<body>
    <div style="max-width: 820px; margin: 0 auto;" class="no-print">
        <button class="print-btn" onclick="window.print()">พิมพ์เอกสารรายงาน / บันทึกเป็นไฟล์ PDF</button>
    </div>
    <div class="page">
        <div class="header-box">
            <div>
                <div class="doc-title">แบบรายงานธุรกรรมที่มีเหตุอันควรสงสัย (แบบ ปปง. 1-03)</div>
                <div class="doc-sub">สำนักงานป้องกันและปราบปรามการฟอกเงิน (ปปง.)</div>
            </div>
            <div class="meta-box">
                <div><b>เลขที่รายงาน:</b> {report_no}</div>
                <div><b>วันที่จัดทำ:</b> {date_str}</div>
                <div><b>ผู้จัดทำ:</b> {investigator_name}</div>
            </div>
        </div>

        <div class="sec-title">ส่วนที่ 1: ข้อมูลผู้ถูกรายงานและบัญชีเงินฝาก</div>
        <table class="info-grid">
            <tr>
                <td width="25%"><b>เลขที่บัญชี:</b></td>
                <td width="25%">{account_id}</td>
                <td width="25%"><b>ระดับความเสี่ยง ธปท.:</b></td>
                <td width="25%"><b>{tier}</b></td>
            </tr>
            <tr>
                <td><b>ข้อมูลผู้ถือบัญชี:</b></td>
                <td colspan="3">{demo}</td>
            </tr>
            <tr>
                <td><b>คะแนนความเสี่ยงรวม:</b></td>
                <td>{risk_score:.1f} / 100</td>
                <td><b>ธุรกรรมฟอกเงินที่ยืนยัน:</b></td>
                <td>{flow.get('laundering_count', 0)} รายการ (฿{flow.get('laundering_amt', 0):,.2f})</td>
            </tr>
        </table>

        <div class="sec-title">ส่วนที่ 2: พฤติการณ์และข้อเท็จจริงอันควรสงสัย</div>
        <p style="margin: 6px 0;"><b>บทสรุปพฤติการณ์:</b> {overall_assessment}</p>
        <p style="margin: 6px 0;"><b>สิ่งบ่งชี้ความเสี่ยงที่ตรวจพบ:</b></p>
        <ul style="margin: 4px 0 10px 20px;">{flags_html}</ul>

        <table class="info-grid">
            <tr>
                <td><b>ยอดเงินรับเข้ารวม:</b> ฿{flow.get('total_in_amt', 0):,.2f} บาท ({flow.get('in_tx_count', 0)} รายการ / {flow.get('senders_count', 0)} บัญชี)</td>
                <td><b>ยอดเงินโอนออกรวม:</b> ฿{flow.get('total_out_amt', 0):,.2f} บาท ({flow.get('out_tx_count', 0)} รายการ / {flow.get('receivers_count', 0)} บัญชี)</td>
            </tr>
        </table>

        <div><b>รายการธุรกรรมสำคัญที่ตรวจพบ (12 รายการสูงสุด):</b></div>
        <table class="tx-table">
            <thead>
                <tr>
                    <th>วัน-เวลา</th>
                    <th>ทิศทาง</th>
                    <th>ธนาคาร</th>
                    <th>เลขที่บัญชี</th>
                    <th style="text-align: right;">จำนวนเงิน (บาท)</th>
                    <th>ช่องทาง</th>
                    <th>สถานะ</th>
                </tr>
            </thead>
            <tbody>
                {tx_rows_html}
            </tbody>
        </table>

        <div class="sec-title">ส่วนที่ 3: มาตรการระงับบัญชีและฐานอำนาจตามกฎหมาย</div>
        <p style="margin: 4px 0;"><b>คำสั่งมาตรการ:</b> {recom.get('title', '-')}</p>
        <p style="margin: 4px 0;"><b>ฐานอำนาจกฎหมาย:</b> {recom.get('legal_basis', '-')}</p>
        <ul style="margin: 4px 0 10px 20px;">{proto_html}</ul>

        <div class="sign-area">
            <div>ลงชื่อ ................................................................... ผู้รายงาน</div>
            <div style="margin-top: 4px;">({investigator_name})</div>
            <div style="font-size: 11.5px; color: #4B5563;">ฝ่ายกำกับการปฏิบัติตามกฎเกณฑ์และป้องกันอาชญากรรมทางการเงิน</div>
        </div>
    </div>
</body>
</html>"""
