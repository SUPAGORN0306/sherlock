"""
Concise, Bilingual Case Narrative & Compliance Recommendation Generator
Free of internal rule codes (no S1/S3/S6) and ML jargon.
"""

import pandas as pd
from typing import Dict, Any, Optional
from src.dashboard.i18n import translate_tier, translate_val

def generate_case_narrative(
    account_id: str,
    pred_data: Dict[str, Any],
    feat_data: Optional[Dict[str, Any]] = None,
    tx_df: Optional[pd.DataFrame] = None,
    person_data: Optional[Dict[str, Any]] = None,
    merchant_data: Optional[Dict[str, Any]] = None,
    lang: str = "th"
) -> Dict[str, Any]:
    raw_tier = pred_data.get("mule_tier", "ปกติ (ความเสี่ยงต่ำ)")
    disp_tier = translate_tier(raw_tier, lang)
    risk_score = float(pred_data.get("risk_score", 0.0))
    
    red_flags = []
    
    # 1. Concise Overall Assessment
    if lang == "th":
        if risk_score >= 80.0:
            overall_assessment = (
                f"คะแนนความเสี่ยง {risk_score:.1f}/100 (ระดับสูงมาก): พบรูปแบบการรับและโอนเงินออกอย่างรวดเร็ว "
                f"เชื่อมโยงกับกลุ่มบัญชีที่มีประวัติทุจริตชัดเจน"
            )
        elif risk_score >= 50.0:
            overall_assessment = (
                f"คะแนนความเสี่ยง {risk_score:.1f}/100 (ระดับปานกลาง-สูง): พฤติกรรมการเคลื่อนไหวของเงินผิดปกติจากบัญชีทั่วไป ต้องตรวจสอบเพิ่มเติม"
            )
        else:
            overall_assessment = (
                f"คะแนนความเสี่ยง {risk_score:.1f}/100 (ระดับต่ำ): พฤติกรรมธุรกรรมปกติ ไม่พบความเชื่อมโยงที่น่าสงสัย"
            )
    else:
        if risk_score >= 80.0:
            overall_assessment = (
                f"Risk Score {risk_score:.1f}/100 (Critical): Exhibits rapid pass-through fund movements "
                f"directly linked to known illicit mule networks."
            )
        elif risk_score >= 50.0:
            overall_assessment = (
                f"Risk Score {risk_score:.1f}/100 (Elevated): Transaction velocity deviates from standard customer baselines."
            )
        else:
            overall_assessment = (
                f"Risk Score {risk_score:.1f}/100 (Low): Normal transaction activity with no suspicious network links."
            )

    # 2. Concise Demographic Assessment
    demo_findings = []
    if person_data:
        age = person_data.get("person_age", 0)
        occ_raw = str(person_data.get("person_occupation", "unknown"))
        occ = translate_val(occ_raw, "occ", lang)
        edu = translate_val(person_data.get("person_education"), "edu", lang)
        mar = translate_val(person_data.get("person_marital_status"), "marital", lang)
        
        if lang == "th":
            demo_str = f"บุคคลธรรมดา | อายุ {age} ปี | อาชีพ: {occ} | การศึกษา: {edu} | สถานภาพ: {mar}"
            if (age <= 22 or age >= 60 or occ_raw in ["retired", "student", "unemployed"]) and risk_score >= 20:
                red_flags.append(f"ข้อมูลผู้ถือบัญชี ({occ} อายุ {age} ปี) ไม่สอดคล้องกับปริมาณธุรกรรมและจำนวนคู่สัญญาที่สูงผิดปกติ")
        else:
            demo_str = f"Individual | Age: {age} | Occupation: {occ} | Education: {edu} | Marital: {mar}"
            if (age <= 22 or age >= 60 or occ_raw in ["retired", "student", "unemployed"]) and risk_score >= 20:
                red_flags.append(f"Customer profile ({occ}, age {age}) conflicts with elevated transaction volume and counterparty dispersion")
        demo_findings.append(demo_str)
    elif merchant_data:
        m_type = translate_val(merchant_data.get("type"), "m_type", lang)
        capital = float(merchant_data.get("registered_capital", 0))
        ind = translate_val(merchant_data.get("industry"), "ind", lang)
        rep_id = str(merchant_data.get("legal_representative_id", "-"))
        
        if lang == "th":
            demo_str = f"นิติบุคคล ({m_type}) | ธุรกิจ: {ind} | ทุนจดทะเบียน: ฿{capital:,.0f} บาท | ผู้แทนนิติบุคคล: {rep_id}"
            if risk_score >= 20:
                red_flags.append(f"นิติบุคคลกลุ่ม {ind} มีเส้นทางการเงินเชื่อมโยงกับกลุ่มบัญชีเสี่ยงสูง อาจเป็นนิติบุคคลบังหน้า")
        else:
            demo_str = f"Corporate ({m_type}) | Industry: {ind} | Capital: THB {capital:,.0f} | Legal Rep: {rep_id}"
            if risk_score >= 20:
                red_flags.append(f"Corporate entity ({ind}) linked to high-risk mule flows; potential shell company")
        demo_findings.append(demo_str)
    else:
        demo_findings.append("ไม่พบข้อมูลทะเบียนลูกค้า" if lang == "th" else "No internal KYC profile found")

    # 3. Behavioral Indicators (No S1/S3/S6 codes!)
    rules_detail = {}
    if feat_data:
        s1_hit = feat_data.get("s1_hit", 0)
        s3_hit = feat_data.get("s3_hit", 0)
        s6_hit = feat_data.get("s6_hit", 0)
        s7_hit = feat_data.get("s7_same_amount_cnt", 0)
        night_ratio = feat_data.get("night_txn_ratio", 0.0)
        drift_txn = feat_data.get("drift_txn_cnt", 0.0)
        
        rules_detail = {
            "pass_through": s1_hit > 0,
            "rapid_outflow": s3_hit > 0,
            "structuring": s6_hit > 0,
            "repeat_amt": int(s7_hit),
            "night_pct": night_ratio * 100,
            "drift_ratio": drift_txn
        }
        
        if lang == "th":
            if s1_hit > 0:
                red_flags.append("มีการพักเงินแล้วโอนออกไปยังบัญชีอื่นทันทีอย่างต่อเนื่อง")
            if s3_hit > 0:
                red_flags.append("เงินที่รับเข้าถูกผันออกจากบัญชีมากกว่า 80% ภายใน 48 ชั่วโมง")
            if s6_hit > 0:
                red_flags.append("มีพฤติกรรมแบ่งซอยยอดโอนใกล้เคียงกันเพื่อเลี่ยงเกณฑ์ตรวจสอบ")
            if night_ratio > 0.15:
                red_flags.append(f"ทำธุรกรรมช่วงวิกาล (00:00-05:59 น.) สูงถึง {night_ratio*100:.1f}%")
        else:
            if s1_hit > 0:
                red_flags.append("Continuous rapid pass-through transfers immediately after receiving funds")
            if s3_hit > 0:
                red_flags.append("Over 80% of incoming funds drained within 48 hours")
            if s6_hit > 0:
                red_flags.append("Transaction structuring detected to evade reporting thresholds")
            if night_ratio > 0.15:
                red_flags.append(f"High late-night transaction ratio (00:00-05:59) at {night_ratio*100:.1f}%")

    # 4. Transaction Flow Summary
    flow_summary = {}
    if tx_df is not None and not tx_df.empty:
        acc_tx = tx_df[(tx_df["From Account"] == account_id) | (tx_df["To Account"] == account_id)]
        in_flow = acc_tx[acc_tx["To Account"] == account_id]
        out_flow = acc_tx[acc_tx["From Account"] == account_id]
        laundering_tx = acc_tx[acc_tx["Is Laundering"] == 1]
        
        flow_summary = {
            "total_in_amt": float(in_flow["Amount Received"].sum()),
            "total_out_amt": float(out_flow["Amount Received"].sum()),
            "in_tx_count": int(len(in_flow)),
            "out_tx_count": int(len(out_flow)),
            "senders_count": int(in_flow["From Account"].nunique()),
            "receivers_count": int(out_flow["To Account"].nunique()),
            "laundering_count": int(len(laundering_tx)),
            "laundering_amt": float(laundering_tx["Amount Received"].sum())
        }
        
        if len(laundering_tx) > 0:
            if lang == "th":
                red_flags.append(f"พบธุรกรรมเชื่อมโยงเส้นทางฟอกเงินโดยตรง {len(laundering_tx)} รายการ (รวม ฿{laundering_tx['Amount Received'].sum():,.2f} บาท)")
            else:
                red_flags.append(f"Confirmed illicit laundering links in {len(laundering_tx)} transactions (Total THB {laundering_tx['Amount Received'].sum():,.2f})")
                
        t_in = flow_summary["total_in_amt"]
        t_out = flow_summary["total_out_amt"]
        if t_out > max(t_in * 1.8, 50_000) and flow_summary["receivers_count"] >= 10:
            if lang == "th":
                red_flags.append(f"ยอดโอนออก (฿{t_out:,.2f}) สูงกว่ายอดรับเข้า {t_out/max(t_in, 1):.1f} เท่า โดยกระจายไปยัง {flow_summary['receivers_count']} บัญชีปลายทาง")
            else:
                red_flags.append(f"Outgoing volume (THB {t_out:,.2f}) exceeds incoming by {t_out/max(t_in, 1):.1f}x across {flow_summary['receivers_count']} beneficiary accounts")
        elif t_in > max(t_out * 1.8, 50_000) and flow_summary["senders_count"] >= 10:
            if lang == "th":
                red_flags.append(f"ยอดรับเข้า (฿{t_in:,.2f}) สูงกว่ายอดโอนออก {t_in/max(t_out, 1):.1f} เท่า โดยรับโอนสะสมจาก {flow_summary['senders_count']} บัญชีต้นทาง")
            else:
                red_flags.append(f"Incoming volume (THB {t_in:,.2f}) exceeds outgoing by {t_in/max(t_out, 1):.1f}x from {flow_summary['senders_count']} sender accounts")

    # 5. Concise Regulatory Recommendations
    if lang == "th":
        if "ม้าดำ" in raw_tier:
            recommendation = {
                "title": "ระงับทุกบัญชีและทุกช่องทางทันที",
                "legal_basis": "บัญชีรายชื่อเฝ้าระวัง ปปง. (HR-03) และคำสั่งศูนย์ AOC",
                "protocol": [
                    "1. ระงับการทำธุรกรรมทุกช่องทางและห้ามเปิดบัญชีใหม่",
                    "2. ส่งรายงาน ปปง. 1-03 และประสานพนักงานสอบสวน"
                ]
            }
        elif "ม้าเทาเข้ม" in raw_tier:
            recommendation = {
                "title": "ระงับช่องทางอิเล็กทรอนิกส์ทันที",
                "legal_basis": "พ.ร.ก. ป้องกันอาชญากรรมทางเทคโนโลยี พ.ศ. 2566 มาตรา 6",
                "protocol": [
                    "1. ระงับช่องทางอิเล็กทรอนิกส์ชั่วคราว (มีผู้เสียหายแจ้งความแล้ว)",
                    "2. จัดส่งประวัติเดินบัญชีให้พนักงานสอบสวน"
                ]
            }
        elif "ม้าเทาอ่อน" in raw_tier:
            recommendation = {
                "title": "ระงับชั่วคราวและเรียกยืนยันตัวตน ณ สาขา",
                "legal_basis": "แนวนโยบายการจัดการบัญชีม้า ธนาคารแห่งประเทศไทย",
                "protocol": [
                    "1. ระงับการโอนออกทางอิเล็กทรอนิกส์ชั่วคราว",
                    "2. ให้ลูกค้านำเอกสารยืนยันแหล่งที่มาของเงินที่สาขา"
                ]
            }
        elif "ม้าน้ำตาลเข้ม" in raw_tier:
            recommendation = {
                "title": "ระงับธุรกรรมชั่วคราว 72 ชั่วโมง และส่งรายงาน ปปง.",
                "legal_basis": "พ.ร.ก. 2566 มาตรา 6 (สถาบันการเงินตรวจพบเหตุสงสัยเอง)",
                "protocol": [
                    "1. สั่งระงับช่องทางอิเล็กทรอนิกส์ชั่วคราว 72 ชั่วโมง",
                    "2. แจ้งศูนย์ AOC 1441 และยื่นแบบรายงาน ปปง. 1-03"
                ]
            }
        elif "ม้าน้ำตาลอ่อน" in raw_tier:
            recommendation = {
                "title": "เฝ้าระวังเข้มงวด จำกัดวงเงิน และบังคับสแกนใบหน้า",
                "legal_basis": "แนวปฏิบัติการบริหารความเสี่ยง ธนาคารแห่งประเทศไทย",
                "protocol": [
                    "1. ปรับลดวงเงินโอนต่อวันไม่เกิน 50,000 บาท",
                    "2. บังคับสแกนใบหน้ายืนยันตัวตนทุกครั้งที่โอนออก"
                ]
            }
        else:
            recommendation = {
                "title": "ให้บริการตามปกติ",
                "legal_basis": "มาตรการดูแลลูกค้าตามเกณฑ์มาตรฐาน",
                "protocol": ["1. ไม่พบความเสี่ยงที่มีนัยสำคัญ อนุญาตให้ทำธุรกรรมตามปกติ"]
            }
    else:
        if "ม้าดำ" in raw_tier:
            recommendation = {
                "title": "Immediate Total Account Freeze",
                "legal_basis": "AMLO Blacklist (HR-03) & AOC Mandate",
                "protocol": ["1. Freeze all electronic channels and block new account opening", "2. Submit AMLO STR Form 1-03 immediately"]
            }
        elif "ม้าเทาเข้ม" in raw_tier:
            recommendation = {
                "title": "Immediate Electronic Channel Suspension",
                "legal_basis": "Anti-Technology Crime Decree B.E. 2566 Section 6",
                "protocol": ["1. Suspend electronic channels (Police report filed)", "2. Forward account statements to investigating officers"]
            }
        elif "ม้าเทาอ่อน" in raw_tier:
            recommendation = {
                "title": "Temporary Suspension & Enhanced In-Branch KYC",
                "legal_basis": "Bank of Thailand Mule Risk Guidelines",
                "protocol": ["1. Suspend outgoing electronic transfers", "2. Require in-person identity and source-of-funds verification"]
            }
        elif "ม้าน้ำตาลเข้ม" in raw_tier:
            recommendation = {
                "title": "72-Hour Proactive Freeze & STR Filing",
                "legal_basis": "Decree B.E. 2566 Section 6 (Proactive Institution Detection)",
                "protocol": ["1. Enforce 72-hour temporary electronic freeze", "2. Notify AOC 1441 and file AMLO STR Form 1-03"]
            }
        elif "ม้าน้ำตาลอ่อน" in raw_tier:
            recommendation = {
                "title": "Enhanced Due Diligence & Biometric Limit Control",
                "legal_basis": "Bank of Thailand Risk Control Framework",
                "protocol": ["1. Cap daily transfer limit to THB 50,000", "2. Enforce facial recognition for all outgoing transfers"]
            }
        else:
            recommendation = {
                "title": "Standard Account Operation",
                "legal_basis": "Standard KYC Monitoring",
                "protocol": ["1. No material anomalies detected; maintain normal service"]
            }

    return {
        "account_id": account_id,
        "raw_tier": raw_tier,
        "tier": disp_tier,
        "risk_score": risk_score,
        "overall_assessment": overall_assessment,
        "demo_findings": demo_findings,
        "rules_detail": rules_detail,
        "red_flags": red_flags,
        "flow_summary": flow_summary,
        "recommendation": recommendation
    }
