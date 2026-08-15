"""Physically isolated tenants: one directory, index and audit log each.

A tenant is a directory under a common base. Its index file, audit log
and policy files live only there; nothing is shared between tenants.
Cross-tenant leakage is structurally impossible because a query only
ever opens one tenant's files — there is no shared index to filter.

License: Apache-2.0
Copyright 2026 Michael Kupermann
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

DEFAULT_BASE_DIR = "tenants"

# Slug only: forbids path separators, dots and anything that could
# escape the base directory.
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class TenantError(ValueError):
    """Invalid tenant name or unknown tenant."""


@dataclass(frozen=True)
class Tenant:
    """One tenant's private file layout."""
    name: str
    root: Path

    @property
    def index_path(self) -> str:
        return str(self.root / "index.db")

    @property
    def audit_path(self) -> str:
        return str(self.root / "audit.db")

    @property
    def policies_dir(self) -> Path:
        return self.root / "policies"

    def policy_path(self, name: str) -> str:
        """Resolve a policy by bare name inside this tenant's directory."""
        if not _NAME_RE.match(name):
            raise TenantError(f"invalid policy name '{name}'")
        return str(self.policies_dir / f"{name}.json")


class TenantManager:
    """Create and resolve tenants under a base directory."""

    def __init__(self, base_dir: str = DEFAULT_BASE_DIR):
        self.base = Path(base_dir)

    def _validate(self, name: str) -> None:
        if not _NAME_RE.match(name):
            raise TenantError(
                f"invalid tenant name '{name}': lowercase letters, digits, "
                f"'-' and '_' only, max 64 chars")

    def create(self, name: str) -> Tenant:
        self._validate(name)
        root = self.base / name
        (root / "policies").mkdir(parents=True, exist_ok=True)
        return Tenant(name=name, root=root)

    def get(self, name: str) -> Tenant:
        self._validate(name)
        root = self.base / name
        if not root.is_dir():
            raise TenantError(
                f"unknown tenant '{name}' under {self.base} — create it "
                f"with 'souprise tenant create {name}'")
        return Tenant(name=name, root=root)

    def list(self) -> List[str]:
        if not self.base.is_dir():
            return []
        return sorted(p.name for p in self.base.iterdir()
                      if p.is_dir() and _NAME_RE.match(p.name))
