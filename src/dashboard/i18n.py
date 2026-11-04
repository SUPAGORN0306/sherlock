"""
Bilingual (Thai / English) Translation & Mapping Module for Sherlock
Ensures 100% Thai in TH mode and 100% English in EN mode with concise, professional banking terms.
"""

TIER_MAP_TH = {
    "ม้าดำ (Blacklist ปปง.)": "ม้าดำ (บัญชีดำ ปปง.)",
    "ม้าเทาเข้ม (มีผู้เสียหาย แจ้งความแล้ว)": "ม้าเทาเข้ม (แจ้งความแล้ว)",
    "ม้าเทาอ่อน (มีผู้เสียหาย ยังไม่แจ้งความ)": "ม้าเทาอ่อน (รอแจ้งความ)",
    "ม้าน้ำตาลเข้ม (สงสัยระดับสูง)": "ม้าน้ำตาลเข้ม (ความเสี่ยงสูง)",
    "ม้าน้ำตาลอ่อน (เริ่มสงสัย พฤติกรรมแปลก)": "ม้าน้ำตาลอ่อน (เฝ้าระวัง)",
    "ปกติ (ความเสี่ยงต่ำ)": "บัญชีปกติ"
}

TIER_MAP_EN = {
    "ม้าดำ (Blacklist ปปง.)": "Black Mule (AMLO Blacklist)",
    "ม้าเทาเข้ม (มีผู้เสียหาย แจ้งความแล้ว)": "Dark Grey (Police Report)",
    "ม้าเทาอ่อน (มีผู้เสียหาย ยังไม่แจ้งความ)": "Light Grey (Unfiled Claim)",
    "ม้าน้ำตาลเข้ม (สงสัยระดับสูง)": "Dark Brown (High Risk)",
    "ม้าน้ำตาลอ่อน (เริ่มสงสัย พฤติกรรมแปลก)": "Light Brown (Watchlist)",
    "ปกติ (ความเสี่ยงต่ำ)": "Normal Account"
}

OCCUPATION_MAP = {
    "th": {
        "student": "นักเรียน / นักศึกษา",
        "retired": "เกษียณอายุ",
        "unemployed": "ว่างงาน",
        "freelancer": "อาชีพอิสระ",
        "white-collar worker": "พนักงานบริษัท",
        "blue-collar worker": "พนักงานทั่วไป",
        "business owner": "เจ้าของกิจการ",
        "unknown": "ไม่ระบุ"
    },
    "en": {
        "student": "Student",
        "retired": "Retired",
        "unemployed": "Unemployed",
        "freelancer": "Freelancer",
        "white-collar worker": "Office Worker",
        "blue-collar worker": "Manual Worker",
        "business owner": "Business Owner",
        "unknown": "Unknown"
    }
}

MARITAL_MAP = {
    "th": {
        "single": "โสด",
        "married": "สมรส",
        "divorced": "หย่าร้าง",
        "widowed": "หม้าย",
        "unknown": "ไม่ระบุ"
    },
    "en": {
        "single": "Single",
        "married": "Married",
        "divorced": "Divorced",
        "widowed": "Widowed",
        "unknown": "Unknown"
    }
}

EDUCATION_MAP = {
    "th": {
        "below high school": "ต่ำกว่ามัธยมศึกษา",
        "high school": "มัธยมศึกษา",
        "associate degree": "อนุปริญญา / ปวส.",
        "bachelor's degree": "ปริญญาตรี",
        "graduate degree or higher": "ปริญญาโทหรือสูงกว่า",
        "unknown": "ไม่ระบุ"
    },
    "en": {
        "below high school": "Below High School",
        "high school": "High School",
        "associate degree": "Associate Degree",
        "bachelor's degree": "Bachelor's Degree",
        "graduate degree or higher": "Graduate Degree+",
        "unknown": "Unknown"
    }
}

GENDER_MAP = {
    "th": {"male": "ชาย", "female": "หญิง", "unknown": "ไม่ระบุ"},
    "en": {"male": "Male", "female": "Female", "unknown": "Unknown"}
}

MERCHANT_TYPE_MAP = {
    "th": {"Small": "ขนาดเล็ก", "Medium": "ขนาดกลาง", "Large": "ขนาดใหญ่"},
    "en": {"Small": "Small Enterprise", "Medium": "Medium Enterprise", "Large": "Large Enterprise"}
}

INDUSTRY_MAP = {
    "th": {
        "Travel & Hospitality": "ท่องเที่ยวและโรงแรม",
        "Retail": "ค้าปลีก",
        "Telecommunications": "โทรคมนาคม",
        "Food & Beverage": "อาหารและเครื่องดื่ม",
        "Utilities": "สาธารณูปโภค",
        "Healthcare": "การแพทย์และสุขภาพ",
        "E-commerce": "พาณิชย์อิเล็กทรอนิกส์",
        "Manufacturing": "อุตสาหกรรมการผลิต",
        "Entertainment": "บันเทิงและสื่อ",
        "Transportation": "ขนส่งและโลจิสติกส์",
        "Financial Services": "บริการทางการเงิน",
        "Education": "การศึกษา"
    },
    "en": {
        "Travel & Hospitality": "Travel & Hospitality",
        "Retail": "Retail",
        "Telecommunications": "Telecommunications",
        "Food & Beverage": "Food & Beverage",
        "Utilities": "Utilities",
        "Healthcare": "Healthcare",
        "E-commerce": "E-commerce",
        "Manufacturing": "Manufacturing",
        "Entertainment": "Entertainment",
        "Transportation": "Transportation",
        "Financial Services": "Financial Services",
        "Education": "Education"
    }
}

CHANNEL_MAP = {
    "th": {
        "Mobile": "แอปพลิเคชันมือถือ",
        "Cash": "เงินสด",
        "Card": "บัตรอิเล็กทรอนิกส์",
        "Transfer": "โอนเงินระหว่างบัญชี"
    },
    "en": {
        "Mobile": "Mobile Banking",
        "Cash": "Cash",
        "Card": "Card Payment",
        "Transfer": "Bank Transfer"
    }
}

def translate_tier(raw_tier: str, lang: str = "th") -> str:
    if lang == "en":
        return TIER_MAP_EN.get(raw_tier, raw_tier)
    return TIER_MAP_TH.get(raw_tier, raw_tier)

def translate_val(val: str, category: str, lang: str = "th") -> str:
    val_str = str(val) if val is not None else "unknown"
    maps = {
        "occ": OCCUPATION_MAP,
        "marital": MARITAL_MAP,
        "edu": EDUCATION_MAP,
        "gender": GENDER_MAP,
        "m_type": MERCHANT_TYPE_MAP,
        "ind": INDUSTRY_MAP,
        "chan": CHANNEL_MAP
    }
    target_map = maps.get(category, {}).get(lang, {})
    return target_map.get(val_str, val_str)


ACTION_MAP = {
    "th": {
        "DISMISS_NORMAL": "บัญชีปกติ / ยุติเรื่อง",
        "WATCHLIST_EDD": "เฝ้าระวังเข้มงวด / จำกัดวงเงิน",
        "FREEZE_AMLO_STR": "ระงับธุรกรรม 72 ชม. และส่ง ปปง.",
    },
    "en": {
        "DISMISS_NORMAL": "Dismissed (Normal)",
        "WATCHLIST_EDD": "Watchlist (EDD / Limit Control)",
        "FREEZE_AMLO_STR": "72h Freeze & AMLO STR",
    },
}

SIGNAL_MAP = {
    "th": {
        "pass_through": "พักเงินแล้วส่งต่อทันที",
        "rapid_outflow": "ผันเงินออกเร็วใน 48 ชม.",
        "structuring": "แบ่งซอยยอดโอน",
        "late_night": "โอนช่วงวิกาลสูง",
    },
    "en": {
        "pass_through": "Rapid pass-through",
        "rapid_outflow": "48h rapid outflow",
        "structuring": "Structuring",
        "late_night": "High late-night activity",
    },
}


def translate_action(code: str, lang: str = "th") -> str:
    return ACTION_MAP.get(lang, ACTION_MAP["th"]).get(str(code), str(code))


def translate_signals(codes: str, lang: str = "th") -> str:
    if not codes:
        return "-"
    m = SIGNAL_MAP.get(lang, SIGNAL_MAP["th"])
    return ", ".join(m.get(c, c) for c in str(codes).split(",") if c)
