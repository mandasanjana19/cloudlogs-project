"""Tiny helpers to write small tables as files that the dashboard can read safely.

Atomic write = write to a temp file, then rename. A reader never sees a half-written file.
"""
import json
import os
from pathlib import Path

import pandas as pd


def write_parquet(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def read_parquet(path):
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default if default is not None else {}


class RollingTable:
    """Keeps the newest `max_rows` rows in memory and mirrors them to one parquet file."""

    def __init__(self, path, max_rows):
        self.path, self.max_rows, self.df = Path(path), max_rows, pd.DataFrame()

    def append(self, new):
        if new is None or len(new) == 0:
            return
        self.df = new.copy() if self.df.empty else pd.concat([self.df, new], ignore_index=True)
        self.df = self.df.tail(self.max_rows).reset_index(drop=True)
        write_parquet(self.df, self.path)

    def replace(self, new):
        self.df = new.copy()
        write_parquet(self.df, self.path)