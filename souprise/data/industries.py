"""Industry profiles: configuration-driven vertical adaptation.

An industry profile is a JSON file under souprise/industries/<name>/profile.json
describing entities, record types and field generators. One generic,
seeded generator turns any profile into records in the same
"Field: value" format the verified and compute paths understand. The
core stays one codebase; industries are data, not branches.

License: Apache-2.0
Copyright 2026 Michael Kupermann
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from souprise.data.generators.business import BusinessEntry

DEFAULT_BASE_DIR = Path(__file__).resolve().parents[1] / "industries"

_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

_REQUIRED_KEYS = ("name", "display", "entities", "record_types")


class ProfileError(ValueError):
    """Invalid or unknown industry profile."""


def list_profiles(base_dir: Optional[str] = None) -> List[str]:
    base = Path(base_dir) if base_dir else DEFAULT_BASE_DIR
    if not base.is_dir():
        return []
    return sorted(p.parent.name for p in base.glob("*/profile.json"))


def load_profile(name: str, base_dir: Optional[str] = None) -> Dict:
    base = Path(base_dir) if base_dir else DEFAULT_BASE_DIR
    path = base / name / "profile.json"
    if not path.is_file():
        known = ", ".join(list_profiles(base_dir)) or "none installed"
        raise ProfileError(f"unknown industry '{name}' (available: {known})")
    profile = json.loads(path.read_text(encoding="utf-8"))
    for key in _REQUIRED_KEYS:
        if key not in profile:
            raise ProfileError(f"profile '{name}' lacks required key '{key}'")
    weights = [rt.get("weight", 1.0) for rt in profile["record_types"]]
    if any(w <= 0 for w in weights):
        raise ProfileError(f"profile '{name}': record type weights must be > 0")
    return profile


def _entity_pool(profile: Dict) -> Dict[str, List[str]]:
    pools = {}
    for ename, spec in profile["entities"].items():
        prefix = spec.get("prefix", f"{ename}_")
        count = int(spec.get("count", 100))
        pools[ename] = [f"{prefix}{i:04d}" for i in range(count)]
    return pools


def _field_value(rng, spec: Dict, pools: Dict[str, List[str]]) -> str:
    kind = spec["kind"]
    if kind == "choice":
        return str(rng.choice(spec["values"]))
    if kind == "entity":
        return str(rng.choice(pools[spec["ref"]]))
    if kind == "money":
        return f"${rng.uniform(spec.get('min', 0), spec.get('max', 10000)):,.2f}"
    if kind == "int":
        return str(rng.randint(spec.get("min", 0), spec.get("max", 10000) + 1))
    if kind == "percent":
        return f"{rng.uniform(spec.get('min', 0), spec.get('max', 100)):.1f}%"
    if kind == "date":
        year = int(rng.choice(spec.get("years", [2024, 2025])))
        return f"{rng.randint(1, 29)} {rng.choice(_MONTHS)} {year}"
    raise ProfileError(f"unknown field kind '{kind}'")


def generate_industry_data(profile: Dict, n: int = 10000,
                           seed: int = 42) -> List[BusinessEntry]:
    """Generate seeded records for an industry profile.

    Every record is one entity's record of one type; titles are
    "<type title pattern>" with {entity} substituted, so the verified
    path's entity handling works unchanged.
    """
    rng = np.random.RandomState(seed)
    pools = _entity_pool(profile)
    types = profile["record_types"]
    weights = np.array([rt.get("weight", 1.0) for rt in types], dtype=float)
    weights = weights / weights.sum()

    entries: List[BusinessEntry] = []
    for _ in range(n):
        rt = types[int(rng.choice(len(types), p=weights))]
        entity = str(rng.choice(pools[rt["entity"]]))
        title = rt.get("title", "{type} {entity}").format(
            type=rt["type"], entity=entity)
        lines = [f"{f['name']}: {_field_value(rng, f, pools)}"
                 for f in rt["fields"]]
        entries.append(BusinessEntry(
            title=title,
            content="\n".join(lines),
            tags=list(rt.get("tags", [])) + [profile["name"]],
        ))
    return entries
