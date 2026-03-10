"""SQLAlchemy engine factory.

Detects DATABASE_URL environment variable. Falls back to the local SQLite
database included in the repository when no DATABASE_URL is set.
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

_DEFAULT_SQLITE_PATH = Path(__file__).parent.parent / "nba_sql.db"


def get_engine() -> Engine:
    """Return a SQLAlchemy engine for either PostgreSQL or SQLite."""
    database_url = os.environ.get("DATABASE_URL")

    if database_url:
        # Heroku-style postgres:// → postgresql:// compatibility fix
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)
        return create_engine(database_url, pool_pre_ping=True)

    # Local SQLite fallback
    sqlite_url = f"sqlite:///{_DEFAULT_SQLITE_PATH}"
    return create_engine(sqlite_url, connect_args={"check_same_thread": False})


def check_connection(engine: Engine) -> bool:
    """Return True if the database is reachable."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
