from __future__ import annotations
from pathlib import Path
import yaml


class Cfg:
    """
    Attribute-access config node that ALSO behaves like a dict.
    This means every caller works regardless of whether it uses
    cfg.key  or  cfg["key"]  or  cfg.items() / cfg.get(k, default).
    """

    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, Cfg(v) if isinstance(v, dict) else v)

    # --- dict-like interface so callers can do cfg.fred_series.items() etc. ---
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return hasattr(self, key)

    def items(self):
        return vars(self).items()

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __repr__(self):
        return f"Cfg({vars(self)})"


def load_config(path=None) -> Cfg:
    path = path or Path(__file__).parents[1] / "config.yaml"
    with open(path) as f:
        return Cfg(yaml.safe_load(f))
