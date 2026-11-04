"""
Append-only, hash-chained audit trail for compliance decisions.

- Stores raw codes only (action code, raw tier); display text is translated at render time.
- Records the context the officer saw (flows, signals, system recommendation) for later review.
- SQLite triggers block UPDATE and DELETE; each row carries a SHA-256 hash linked to the previous row.
"""

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "audit_trail_v2.db"

ACTION_CODES = ["DISMISS_NORMAL", "WATCHLIST_EDD", "FREEZE_AMLO_STR"]
GENESIS_HASH = "0" * 64

HASH_FIELDS = [
    "timestamp", "account_id", "action_code", "reviewer_name", "tier_raw", "risk_score",
    "notes", "total_in_amt", "total_out_amt", "laundering_count", "signals", "recommended_action",
]


def recommended_action(raw_tier: str) -> str:
    """Action implied by the BOT tier, used to compare officer decisions with the system suggestion."""
    if raw_tier and raw_tier.startswith("ปกติ"):
        return "DISMISS_NORMAL"
    if raw_tier and raw_tier.startswith("ม้าน้ำตาลอ่อน"):
        return "WATCHLIST_EDD"
    return "FREEZE_AMLO_STR"


def _connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def init_db(db_path: Path = DB_PATH):
    with _connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                account_id TEXT NOT NULL,
                action_code TEXT NOT NULL,
                reviewer_name TEXT NOT NULL,
                tier_raw TEXT NOT NULL,
                risk_score REAL,
                notes TEXT,
                total_in_amt REAL,
                total_out_amt REAL,
                laundering_count INTEGER,
                signals TEXT,
                recommended_action TEXT,
                prev_hash TEXT NOT NULL,
                record_hash TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit_log
            BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit_log
            BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END
        """)
        conn.commit()


def _compute_hash(prev_hash: str, record: Dict[str, Any]) -> str:
    payload = {k: record.get(k) for k in HASH_FIELDS}
    raw = prev_hash + json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def log_decision(
    account_id: str,
    action_code: str,
    reviewer_name: str,
    tier_raw: str,
    risk_score: float,
    notes: str = "",
    context: Optional[Dict[str, Any]] = None,
    db_path: Path = DB_PATH,
) -> bool:
    if action_code not in ACTION_CODES:
        raise ValueError(f"Unknown action code: {action_code}")
    init_db(db_path)
    context = context or {}
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "account_id": account_id,
        "action_code": action_code,
        "reviewer_name": reviewer_name,
        "tier_raw": tier_raw,
        "risk_score": round(float(risk_score), 2),
        "notes": notes or "",
        "total_in_amt": round(float(context.get("total_in_amt", 0.0)), 2),
        "total_out_amt": round(float(context.get("total_out_amt", 0.0)), 2),
        "laundering_count": int(context.get("laundering_count", 0)),
        "signals": ",".join(context.get("signals", [])),
        "recommended_action": context.get("recommended_action") or recommended_action(tier_raw),
    }
    with _connect(db_path) as conn:
        row = conn.execute("SELECT record_hash FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()
        prev_hash = row[0] if row else GENESIS_HASH
        record["prev_hash"] = prev_hash
        record["record_hash"] = _compute_hash(prev_hash, record)
        cols = list(record.keys())
        conn.execute(
            f"INSERT INTO audit_log ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
            [record[c] for c in cols],
        )
        conn.commit()
    return True


def get_audit_history(
    account_id: Optional[str] = None,
    limit: int = 500,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    init_db(db_path)
    with _connect(db_path) as conn:
        if account_id:
            return pd.read_sql_query(
                "SELECT * FROM audit_log WHERE account_id = ? ORDER BY id DESC LIMIT ?",
                conn, params=(account_id, limit),
            )
        return pd.read_sql_query("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", conn, params=(limit,))


def get_account_last_action(account_id: str, db_path: Path = DB_PATH):
    df = get_audit_history(account_id=account_id, limit=1, db_path=db_path)
    if df.empty:
        return None
    r = df.iloc[0]
    return {
        "action_code": r["action_code"],
        "reviewer_name": r["reviewer_name"],
        "notes": r["notes"],
        "timestamp": r["timestamp"],
    }


def verify_chain(db_path: Path = DB_PATH):
    """Recompute every hash in order. Returns (is_valid, rows_checked, first_bad_id)."""
    init_db(db_path)
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM audit_log ORDER BY id ASC").fetchall()
    prev = GENESIS_HASH
    for r in rows:
        rec = dict(r)
        if rec["prev_hash"] != prev or _compute_hash(prev, rec) != rec["record_hash"]:
            return False, len(rows), rec["id"]
        prev = rec["record_hash"]
    return True, len(rows), None
