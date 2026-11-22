
import json
import sqlite3

from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.app.risk_engine import assess_risk


ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "verification_cases.sqlite3"

router = APIRouter(
    prefix="/api/verification-cases",
    tags=["Verification Cases"],
)


BankingAction = Literal[
    "balance_inquiry",
    "transaction_status",
    "password_reset",
    "mobile_number_change",
    "transfer_limit_increase",
    "high_value_transfer",
]

AuthenticationStatus = Literal[
    "verified",
    "unverified",
    "failed",
]

CaseStatus = Literal[
    "PENDING_CUSTOMER_RESPONSE",
    "CONFIRMED_PENDING_VERIFICATION",
    "CONFIRMED_PENDING_HUMAN_REVIEW",
    "DISPUTED",
    "ESCALATED_FOR_INVESTIGATION",
    "REVIEW_CLOSED_NO_BANKING_ACTION",
]


class CreateCaseRequest(BaseModel):
    banking_action: BankingAction
    authentication_status: AuthenticationStatus
    unusual_activity: bool = False


class CustomerResponse(BaseModel):
    response: Literal["confirm", "deny"]


class OfficerReview(BaseModel):
    decision: Literal[
        "escalate",
        "close_no_action",
    ]

    note: str = Field(
        min_length=8,
        max_length=500,
    )


def utc_now():
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def connect_db():
    db = sqlite3.connect(
        DB_PATH,
        timeout=10,
    )

    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")

    return db


def initialize_database():
    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with closing(connect_db()) as db:
        with db:
            # Existing table: unchanged schema
            db.execute("""
                CREATE TABLE IF NOT EXISTS
                verification_cases (
                    case_id TEXT PRIMARY KEY,
                    banking_action TEXT NOT NULL,
                    authentication_status TEXT NOT NULL,
                    unusual_activity INTEGER NOT NULL,
                    risk_score INTEGER NOT NULL,
                    risk_level TEXT NOT NULL,
                    policy_action TEXT NOT NULL,
                    reasons_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # Existing table: unchanged schema
            db.execute("""
                CREATE TABLE IF NOT EXISTS
                verification_events (
                    event_id INTEGER PRIMARY KEY
                        AUTOINCREMENT,
                    case_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (case_id)
                    REFERENCES verification_cases(case_id)
                )
            """)

            # New table for simulated officer reviews
            db.execute("""
                CREATE TABLE IF NOT EXISTS
                case_reviews (
                    review_id INTEGER PRIMARY KEY
                        AUTOINCREMENT,
                    case_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    note TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (case_id)
                    REFERENCES verification_cases(case_id)
                )
            """)

            db.execute("""
                CREATE INDEX IF NOT EXISTS
                idx_cases_updated
                ON verification_cases(updated_at)
            """)

            db.execute("""
                CREATE INDEX IF NOT EXISTS
                idx_cases_status
                ON verification_cases(status)
            """)


def get_case_record(case_id):
    initialize_database()

    with closing(connect_db()) as db:
        row = db.execute(
            """
            SELECT *
            FROM verification_cases
            WHERE case_id = ?
            """,
            (case_id,),
        ).fetchone()

        if row is None:
            raise HTTPException(
                status_code=404,
                detail="Verification case not found",
            )

        events = db.execute(
            """
            SELECT event_id, event_type, timestamp
            FROM verification_events
            WHERE case_id = ?
            ORDER BY event_id ASC
            """,
            (case_id,),
        ).fetchall()

        reviews = db.execute(
            """
            SELECT review_id, decision, note, timestamp
            FROM case_reviews
            WHERE case_id = ?
            ORDER BY review_id ASC
            """,
            (case_id,),
        ).fetchall()

    return {
        "case_id": row["case_id"],
        "banking_context": {
            "banking_action": row["banking_action"],
            "authentication_status": (
                row["authentication_status"]
            ),
            "unusual_activity": bool(
                row["unusual_activity"]
            ),
        },
        "risk_assessment": {
            "risk_score": row["risk_score"],
            "risk_level": row["risk_level"],
            "action": row["policy_action"],
            "reasons": json.loads(
                row["reasons_json"]
            ),
            "audio_policy": "MONITOR_ONLY",
        },
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "audit_events": [
            {
                "event_id": event["event_id"],
                "event_type": event["event_type"],
                "timestamp": event["timestamp"],
            }
            for event in events
        ],
        "officer_reviews": [
            {
                "review_id": review["review_id"],
                "decision": review["decision"],
                "note": review["note"],
                "timestamp": review["timestamp"],
            }
            for review in reviews
        ],
        "mode": "SIMULATION",
    }


# =========================================
# CREATE VERIFICATION CASE
# =========================================

@router.post("", status_code=201)
def create_verification_case(
    request: CreateCaseRequest,
):
    risk = assess_risk(
        banking_action=request.banking_action,
        authentication_status=(
            request.authentication_status
        ),
        unusual_activity=request.unusual_activity,
    )

    if risk["action"] == "STANDARD_BANKING_FLOW":
        raise HTTPException(
            status_code=400,
            detail=(
                "This scenario does not require "
                "an additional verification case."
            ),
        )

    initialize_database()

    case_id = uuid4().hex
    timestamp = utc_now()

    with closing(connect_db()) as db:
        with db:
            db.execute(
                """
                INSERT INTO verification_cases (
                    case_id,
                    banking_action,
                    authentication_status,
                    unusual_activity,
                    risk_score,
                    risk_level,
                    policy_action,
                    reasons_json,
                    status,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    case_id,
                    request.banking_action,
                    request.authentication_status,
                    int(request.unusual_activity),
                    risk["risk_score"],
                    risk["risk_level"],
                    risk["action"],
                    json.dumps(risk["reasons"]),
                    "PENDING_CUSTOMER_RESPONSE",
                    timestamp,
                    timestamp,
                ),
            )

            db.execute(
                """
                INSERT INTO verification_events (
                    case_id, event_type, timestamp
                )
                VALUES (?, ?, ?)
                """,
                (
                    case_id,
                    "CASE_CREATED",
                    timestamp,
                ),
            )

    return get_case_record(case_id)


# =========================================
# LIST AND FILTER CASES — NEW
# =========================================

@router.get("")
def list_verification_cases(
    status: CaseStatus | None = None,
    search: str = Query(
        default="",
        max_length=100,
    ),
    limit: int = Query(
        default=10,
        ge=1,
        le=100,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
):
    initialize_database()

    filters = []
    parameters = []

    if status:
        filters.append("status = ?")
        parameters.append(status)

    search = search.strip()

    if search:
        filters.append(
            "("
            "instr(lower(case_id), lower(?)) > 0 "
            "OR "
            "instr(lower(banking_action), lower(?)) > 0"
            ")"
        )
        parameters.extend([search, search])

    where_clause = ""

    if filters:
        where_clause = (
            " WHERE " + " AND ".join(filters)
        )

    with closing(connect_db()) as db:
        total = db.execute(
            "SELECT COUNT(*) FROM verification_cases"
            + where_clause,
            parameters,
        ).fetchone()[0]

        rows = db.execute(
            """
            SELECT
                case_id,
                banking_action,
                authentication_status,
                risk_score,
                risk_level,
                status,
                created_at,
                updated_at
            FROM verification_cases
            """
            + where_clause
            + """
            ORDER BY updated_at DESC, rowid DESC
            LIMIT ? OFFSET ?
            """,
            [*parameters, limit, offset],
        ).fetchall()

        counts_rows = db.execute(
            """
            SELECT status, COUNT(*) AS total
            FROM verification_cases
            GROUP BY status
            """
        ).fetchall()

    status_counts = {
        row["status"]: row["total"]
        for row in counts_rows
    }

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "status_counts": status_counts,
        "items": [
            dict(row)
            for row in rows
        ],
        "mode": "SIMULATION",
    }


# =========================================
# GET SINGLE CASE
# =========================================

@router.get("/{case_id}")
def get_verification_case(case_id: str):
    return get_case_record(case_id)


# =========================================
# RECORD CUSTOMER RESPONSE
# =========================================

@router.post("/{case_id}/respond")
def respond_to_verification_case(
    case_id: str,
    request: CustomerResponse,
):
    initialize_database()

    timestamp = utc_now()

    with closing(connect_db()) as db:
        with db:
            # Serializes competing writes.
            db.execute("BEGIN IMMEDIATE")

            case = db.execute(
                """
                SELECT status, policy_action
                FROM verification_cases
                WHERE case_id = ?
                """,
                (case_id,),
            ).fetchone()

            if case is None:
                raise HTTPException(
                    status_code=404,
                    detail="Verification case not found",
                )

            if (
                case["status"]
                != "PENDING_CUSTOMER_RESPONSE"
            ):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "This case has already received "
                        "a customer response."
                    ),
                )

            if request.response == "deny":
                new_status = "DISPUTED"
                event_type = "CUSTOMER_DENIED"

            elif (
                case["policy_action"]
                == "HUMAN_REVIEW_REQUIRED"
            ):
                new_status = (
                    "CONFIRMED_PENDING_HUMAN_REVIEW"
                )
                event_type = "CUSTOMER_CONFIRMED"

            else:
                new_status = (
                    "CONFIRMED_PENDING_VERIFICATION"
                )
                event_type = "CUSTOMER_CONFIRMED"

            db.execute(
                """
                UPDATE verification_cases
                SET status = ?, updated_at = ?
                WHERE case_id = ?
                """,
                (
                    new_status,
                    timestamp,
                    case_id,
                ),
            )

            db.execute(
                """
                INSERT INTO verification_events (
                    case_id, event_type, timestamp
                )
                VALUES (?, ?, ?)
                """,
                (
                    case_id,
                    event_type,
                    timestamp,
                ),
            )

    return get_case_record(case_id)


# =========================================
# SIMULATED OFFICER REVIEW — NEW
# =========================================

@router.post("/{case_id}/review")
def review_verification_case(
    case_id: str,
    request: OfficerReview,
):
    initialize_database()

    note = request.note.strip()

    if len(note) < 8:
        raise HTTPException(
            status_code=400,
            detail=(
                "Review note must contain "
                "at least 8 characters."
            ),
        )

    timestamp = utc_now()

    with closing(connect_db()) as db:
        with db:
            db.execute("BEGIN IMMEDIATE")

            case = db.execute(
                """
                SELECT status
                FROM verification_cases
                WHERE case_id = ?
                """,
                (case_id,),
            ).fetchone()

            if case is None:
                raise HTTPException(
                    status_code=404,
                    detail="Verification case not found",
                )

            allowed = {
                "DISPUTED",
                "CONFIRMED_PENDING_HUMAN_REVIEW",
                "ESCALATED_FOR_INVESTIGATION",
            }

            if case["status"] not in allowed:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "This case is not eligible "
                        "for simulated officer review."
                    ),
                )

            if request.decision == "escalate":
                if (
                    case["status"]
                    == "ESCALATED_FOR_INVESTIGATION"
                ):
                    raise HTTPException(
                        status_code=409,
                        detail="Case is already escalated.",
                    )

                new_status = (
                    "ESCALATED_FOR_INVESTIGATION"
                )
                event_type = "CASE_ESCALATED"

            else:
                new_status = (
                    "REVIEW_CLOSED_NO_BANKING_ACTION"
                )
                event_type = "REVIEW_CLOSED_NO_ACTION"

            db.execute(
                """
                UPDATE verification_cases
                SET status = ?, updated_at = ?
                WHERE case_id = ?
                """,
                (
                    new_status,
                    timestamp,
                    case_id,
                ),
            )

            db.execute(
                """
                INSERT INTO case_reviews (
                    case_id,
                    decision,
                    note,
                    timestamp
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    case_id,
                    request.decision,
                    note,
                    timestamp,
                ),
            )

            db.execute(
                """
                INSERT INTO verification_events (
                    case_id,
                    event_type,
                    timestamp
                )
                VALUES (?, ?, ?)
                """,
                (
                    case_id,
                    event_type,
                    timestamp,
                ),
            )

    return get_case_record(case_id)
