from __future__ import annotations
import os
from pathlib import Path
import pandas as pd


class Store:
    """
    Dual-backend store: parquet (local dev) or PostgreSQL (production).

    Parquet  → cfg.storage.dir / <name>.parquet
    SQL      → one table per dataset name in the Postgres database pointed
               to by the env var named in cfg.storage.sql_url_env.

    The SQL tables are created (or replaced) on every pipeline run so the
    dashboard always reads the freshest data without manual migrations.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.backend = cfg.storage.backend

        if self.backend == "parquet":
            self.dir = Path(cfg.storage.dir)
            self.dir.mkdir(parents=True, exist_ok=True)
            self._engine = None

        elif self.backend == "sql":
            from sqlalchemy import create_engine
            url = os.environ.get(cfg.storage.sql_url_env)
            if not url:
                raise RuntimeError(
                    f"Set the {cfg.storage.sql_url_env} environment variable "
                    "to a SQLAlchemy database URL, e.g. "
                    "postgresql://user:pass@host:5432/dbname"
                )
            self._engine = create_engine(url, pool_pre_ping=True)

        else:
            raise ValueError(f"Unknown storage backend: {self.backend!r}")

    # ------------------------------------------------------------------
    def save(self, name: str, data: pd.DataFrame) -> None:
        if self.backend == "parquet":
            data.to_parquet(self.dir / f"{name}.parquet")

        else:  # sql
            # index (DatetimeIndex) is stored as a plain column called "date"
            # so PostgreSQL doesn't choke on unnamed indices.
            df = data.copy()
            df.index.name = "date"
            df.to_sql(
                name,
                con=self._engine,
                if_exists="replace",   # full refresh on every pipeline run
                index=True,
                index_label="date",
                method="multi",        # faster multi-row INSERT
                chunksize=2000,
            )
            print(f"[store] saved {len(df)} rows → table '{name}'")

    # ------------------------------------------------------------------
    def load(self, name: str) -> pd.DataFrame:
        if self.backend == "parquet":
            return pd.read_parquet(self.dir / f"{name}.parquet")

        else:  # sql
            df = pd.read_sql(
                f'SELECT * FROM "{name}" ORDER BY date',
                con=self._engine,
                index_col="date",
                parse_dates=["date"],
            )
            return df
