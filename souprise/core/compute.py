"""Deterministic aggregation over index entries. Code owns every number.

Aggregate questions (sum, count, average, min, max with simple filters)
are parsed rule-based and computed with Decimal arithmetic over ALL
entries in the index, not a top-k sample. The language model plays no
part in any figure.

License: Apache-2.0
Copyright 2026 Michael Kupermann
"""

import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, List, Optional

_AGG_PATTERNS = [
    (r"\b(average|mean|durchschnitt)\b", "avg"),
    (r"\b(how many|count|number of|wie viele|anzahl)\b", "count"),
    (r"\b(total|sum|summe|combined|insgesamt)\b", "sum"),
    (r"\b(highest|largest|maximum|max|höchste)\b", "max"),
    (r"\b(lowest|smallest|minimum|min|niedrigste)\b", "min"),
]

_FIELD_PATTERNS = [
    (r"annual revenue|revenue|umsatz", "Annual Revenue"),
    (r"amount|betrag|invoices? (worth|value)|owed", "Amount"),
    (r"stock|lagerbestand|units", "Stock"),
    (r"price|preis", "Price"),
    (r"quantity|menge", "Quantity"),
    (r"allocated", "Allocated"),
    (r"spent", "Spent"),
    (r"remaining", "Remaining"),
]

_STATUS_WORDS = ["overdue", "paid", "open", "cancelled", "confirmed",
                 "shipped", "delivered", "in production"]
_REGION_WORDS = ["north", "south", "east", "west", "us", "eu", "apac", "global"]

_ENTITY_RE = re.compile(r"\b[A-Za-z]+_[A-Za-z0-9]+\b")
_NUM_RE = re.compile(r"-?[\d,]+(?:\.\d+)?")


@dataclass
class ComputeResult:
    """A deterministic aggregate with full provenance."""
    text: str
    operation: str = ""
    field_name: str = ""
    value: str = ""
    record_count: int = 0
    filters: Dict[str, str] = field(default_factory=dict)
    computed: bool = True


def parse_aggregate(question: str):
    """(operation, field, filters) or None if not an aggregate question."""
    q = question.lower()
    operation = next((op for pat, op in _AGG_PATTERNS if re.search(pat, q)), None)
    if operation is None:
        return None
    field_name = next((f for pat, f in _FIELD_PATTERNS if re.search(pat, q)), None)
    if field_name is None and operation != "count":
        return None

    filters: Dict[str, str] = {}
    for status in _STATUS_WORDS:
        if re.search(rf"\b{status}\b", q):
            filters["Status"] = status
            break
    for region in _REGION_WORDS:
        if re.search(rf"\b{region}\b(?! \w*_)", q) and f"{region} " not in ("us ",):
            filters["Region"] = region.upper() if len(region) <= 4 else region.capitalize()
            break
    entity = _ENTITY_RE.search(question)
    if entity:
        filters["_entity"] = entity.group(0).lower()
    return operation, field_name, filters


def _fields_of(text: str) -> Dict[str, str]:
    return dict(line.split(": ", 1) for line in text.splitlines() if ": " in line)


def _to_decimal(raw: str) -> Optional[Decimal]:
    match = _NUM_RE.search(raw.replace("$", ""))
    if not match:
        return None
    try:
        return Decimal(match.group(0).replace(",", ""))
    except ArithmeticError:
        return None


def _matches(entry_text: str, fields: Dict[str, str], filters: Dict[str, str]) -> bool:
    for key, wanted in filters.items():
        if key == "_entity":
            if wanted not in entry_text.lower():
                return False
        else:
            value = fields.get(key, "")
            if value.lower() != wanted.lower():
                return False
    return True


def compute_aggregate(question: str, entries: List[Dict[str, Any]]) -> Optional[ComputeResult]:
    """Compute an aggregate over ALL index entries, or None.

    Args:
        question: The user's question.
        entries: Every index entry ({'id', 'text', ...}).
    """
    parsed = parse_aggregate(question)
    if parsed is None:
        return None
    operation, field_name, filters = parsed

    values: List[Decimal] = []
    matched = 0
    for entry in entries:
        fields = _fields_of(entry.get("text", ""))
        if not _matches(entry.get("text", ""), fields, filters):
            continue
        if field_name is None:
            matched += 1
            continue
        raw = fields.get(field_name)
        if raw is None:
            continue
        value = _to_decimal(raw)
        if value is not None:
            values.append(value)
            matched += 1

    filter_text = ", ".join(
        f"{'entity' if k == '_entity' else k}={v}" for k, v in filters.items()
    ) or "no filter"

    if field_name is None:  # plain count
        result_value = str(matched)
        summary = f"Count of records ({filter_text}): {matched}"
    else:
        if not values:
            return ComputeResult(
                text=(f"No records with a {field_name} value match "
                      f"({filter_text}); nothing to compute."),
                operation=operation, field_name=field_name,
                record_count=0, filters=filters,
            )
        if operation == "sum":
            result = sum(values)
        elif operation == "avg":
            result = (sum(values) / len(values)).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP)
        elif operation == "max":
            result = max(values)
        elif operation == "min":
            result = min(values)
        else:  # count with a field
            result = Decimal(len(values))
        result_value = f"{result:,}"
        summary = (f"{operation} of {field_name} over {len(values)} records "
                   f"({filter_text}): {result_value}")

    return ComputeResult(
        text=f"{summary}\nComputed deterministically from the index; "
             f"no language model involved.",
        operation=operation, field_name=field_name or "",
        value=result_value, record_count=matched, filters=filters,
    )
