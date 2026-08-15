"""Record-level access policies, enforced before search.

A policy defines which records a role may see (field-value conditions)
and which fields are hidden from it. The visibility mask is applied to
the hypervector index BEFORE distance computation, so similarity scores
over forbidden records never exist and cannot leak. Field masks redact
values from answers, record dumps, aggregation and generative context.

Policies are in-process enforcement objects; user authentication is the
REST API's job (issue #29).

License: Apache-2.0
Copyright 2026 Michael Kupermann
"""

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List

import numpy as np

DENIAL_TEXT = ("Your role's access policy does not permit this answer. "
               "Nothing was retrieved outside your permissions.")


@dataclass(frozen=True)
class AccessPolicy:
    """Visibility rules for one role.

    Args:
        name: Role name, recorded in audit events.
        visible_where: Field-value conditions; a record is visible iff
            for every (field, allowed values) pair its value is in the
            allowed set. Empty dict means all records are visible.
        hidden_fields: Field names whose values this role never sees.
    """
    name: str = "unrestricted"
    visible_where: Dict[str, FrozenSet[str]] = field(default_factory=dict)
    hidden_fields: FrozenSet[str] = frozenset()

    @property
    def is_unrestricted(self) -> bool:
        return not self.visible_where and not self.hidden_fields


UNRESTRICTED = AccessPolicy()


def load_policy(path: str) -> AccessPolicy:
    """Load a policy from a JSON file.

    Format: {"name": "eu_sales",
             "visible_where": {"Region": ["EU"]},
             "hidden_fields": ["Margin"]}
    """
    import json
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return AccessPolicy(
        name=raw.get("name", "unnamed"),
        visible_where={k: frozenset(v)
                       for k, v in raw.get("visible_where", {}).items()},
        hidden_fields=frozenset(raw.get("hidden_fields", [])),
    )


def _fields_of(text: str) -> Dict[str, str]:
    return dict(line.split(": ", 1) for line in text.splitlines() if ": " in line)


def visible_mask(entries: List[Dict[str, Any]], policy: AccessPolicy) -> np.ndarray:
    """Boolean visibility mask over entries for a policy."""
    if not policy.visible_where:
        return np.ones(len(entries), dtype=bool)
    mask = np.zeros(len(entries), dtype=bool)
    for i, entry in enumerate(entries):
        fields = _fields_of(entry.get("text", ""))
        mask[i] = all(
            fields.get(f, "").lower() in {v.lower() for v in allowed}
            for f, allowed in policy.visible_where.items()
        )
    return mask


def redact_text(text: str, policy: AccessPolicy) -> str:
    """Remove hidden fields' lines from a record text."""
    if not policy.hidden_fields:
        return text
    hidden = {f.lower() for f in policy.hidden_fields}
    kept = [line for line in text.splitlines()
            if not (": " in line and line.split(": ", 1)[0].lower() in hidden)]
    redacted = len(text.splitlines()) - len(kept)
    if redacted:
        kept.append(f"({redacted} field(s) redacted by policy)")
    return "\n".join(kept)


def filter_entries(entries: List[Dict[str, Any]],
                   policy: AccessPolicy) -> List[Dict[str, Any]]:
    """Visible entries with hidden fields redacted (for aggregation and
    deterministic scans)."""
    if policy.is_unrestricted:
        return entries
    mask = visible_mask(entries, policy)
    out = []
    for entry, ok in zip(entries, mask):
        if not ok:
            continue
        if policy.hidden_fields:
            entry = dict(entry, text=redact_text(entry.get("text", ""), policy))
        out.append(entry)
    return out
