"""SQLite persistence for jobs and uploaded profiles."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles (
    profile_id TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    role       TEXT NOT NULL,
    builtin    INTEGER NOT NULL DEFAULT 0,
    color_space TEXT NOT NULL,
    device_class TEXT NOT NULL,
    pcs        TEXT NOT NULL,
    size       INTEGER NOT NULL,
    sha256     TEXT NOT NULL,
    filename   TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id          TEXT PRIMARY KEY,
    image_sha256    TEXT NOT NULL,
    width           INTEGER NOT NULL,
    height          INTEGER NOT NULL,
    image_mode      TEXT NOT NULL,
    source_profile  TEXT NOT NULL,
    press_profile   TEXT NOT NULL,
    display_profile TEXT NOT NULL,
    source_intent   TEXT NOT NULL,
    proof_intent    TEXT NOT NULL,
    threshold       REAL NOT NULL,
    use_embedded    INTEGER NOT NULL DEFAULT 0,
    embedded_sha256 TEXT,
    parameters_json TEXT NOT NULL,
    profile_digests_json TEXT NOT NULL,
    de_mean         REAL,
    de_p95          REAL,
    de_max          REAL,
    over_ratio      REAL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def upsert_profile(conn: sqlite3.Connection, record) -> None:
    conn.execute(
        """
        INSERT INTO profiles (profile_id, name, role, builtin, color_space,
                              device_class, pcs, size, sha256, filename)
        VALUES (:profile_id, :name, :role, :builtin, :color_space,
                :device_class, :pcs, :size, :sha256, :filename)
        ON CONFLICT(profile_id) DO UPDATE SET
            name=excluded.name, role=excluded.role,
            color_space=excluded.color_space, device_class=excluded.device_class,
            pcs=excluded.pcs, size=excluded.size, sha256=excluded.sha256
        """,
        {
            "profile_id": record.profile_id,
            "name": record.name,
            "role": record.role,
            "builtin": 1 if record.builtin else 0,
            "color_space": record.color_space,
            "device_class": record.device_class,
            "pcs": record.pcs,
            "size": record.size,
            "sha256": record.sha256,
            "filename": record.filename,
        },
    )
    conn.commit()


def insert_job(conn: sqlite3.Connection, job: dict) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO jobs
        (job_id, image_sha256, width, height, image_mode, source_profile,
         press_profile, display_profile, source_intent, proof_intent,
         threshold, use_embedded, embedded_sha256, parameters_json,
         profile_digests_json, de_mean, de_p95, de_max, over_ratio)
        VALUES (:job_id, :image_sha256, :width, :height, :image_mode,
                :source_profile, :press_profile, :display_profile,
                :source_intent, :proof_intent, :threshold, :use_embedded,
                :embedded_sha256, :parameters_json, :profile_digests_json,
                :de_mean, :de_p95, :de_max, :over_ratio)
        """,
        {
            "job_id": job["job_id"],
            "image_sha256": job["image_sha256"],
            "width": job["width"],
            "height": job["height"],
            "image_mode": job["image_mode"],
            "source_profile": job["source_profile"],
            "press_profile": job["press_profile"],
            "display_profile": job["display_profile"],
            "source_intent": job["source_intent"],
            "proof_intent": job["proof_intent"],
            "threshold": job["threshold"],
            "use_embedded": 1 if job.get("use_embedded") else 0,
            "embedded_sha256": job.get("embedded_sha256"),
            "parameters_json": json.dumps(job["parameters"], sort_keys=True, ensure_ascii=False),
            "profile_digests_json": json.dumps(job["profile_digests"], sort_keys=True, ensure_ascii=False),
            "de_mean": job["stats"]["mean"],
            "de_p95": job["stats"]["p95"],
            "de_max": job["stats"]["max"],
            "over_ratio": job["stats"]["over_ratio"],
        },
    )
    conn.commit()


def get_job(conn: sqlite3.Connection, job_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()


def recent_jobs(conn: sqlite3.Connection, limit: int = 20) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM jobs ORDER BY created_at DESC, job_id DESC LIMIT ?", (limit,)
    ).fetchall()
