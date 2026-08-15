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
    (r"\b(are there any|do we have any|gibt es)\b", "count"),
    (r"\b(total|sum|summe|combined|insgesamt|owe)\b", "sum"),
    (r"\b(highest|largest|biggest|maximum|max|höchste|größte)\b", "max"),
    (r"\b(lowest|smallest|minimum|min|niedrigste|kleinste)\b", "min"),
]

_TREND_WORDS = {
    "negative": "declining", "declining": "declining", "down": "declining",
    "falling": "declining", "rising": "rising", "positive": "rising",
    "up": "rising", "stable": "stable",
}

_THRESHOLD_RE = re.compile(
    r"(greater than|more than|over|above|exceeding|mehr als|über|"
    r"less than|under|below|fewer than|weniger als|unter)\s+\$?([\d,]+(?:\.\d+)?)")
_BELOW_WORDS = ("less than", "under", "below", "fewer than",
                "weniger als", "unter")

_FIELD_PATTERNS = [
    (r"annual revenue|revenue|umsatz", "Annual Revenue"),
    (r"\b(amount|betrag|owed?|owes)\b|invoices? (worth|value)", "Amount"),
    (r"value of .*orders?|orders? .*value|order value", "Total"),
    (r"open tickets|tickets", "Open Tickets"),
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
    for word, trend in _TREND_WORDS.items():
        if re.search(rf"\b{word}\b.*\btrend\b|\btrend\b.*\b{word}\b", q):
            filters["Trend"] = trend
            break
    threshold = _THRESHOLD_RE.search(q)
    if threshold:
        direction = "<" if threshold.group(1) in _BELOW_WORDS else ">"
        filters["_threshold"] = f"{direction}{threshold.group(2).replace(',', '')}"
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
        if key == "_threshold":
            continue  # applied to the target field's value, not here
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

    entity = filters.get("_entity")
    if entity and not any(entity in e.get("text", "").lower() for e in entries):
        # The entity does not exist in the (visible) index. A count of
        # zero here would be misleading; fall through to the verified
        # path, which refuses on unknown entities.
        return None

    threshold = filters.get("_threshold")
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
        if value is None:
            continue
        if threshold:
            bound = Decimal(threshold[1:])
            if threshold[0] == ">" and not value > bound:
                continue
            if threshold[0] == "<" and not value < bound:
                continue
        values.append(value)
        matched += 1

    filter_text = ", ".join(
        f"{'entity' if k == '_entity' else k}={v}" for k, v in filters.items()
    ) or "no filter"

    if field_name is None:  # plain count
        result_value = str(matched)
        summary = f"Count of records ({filter_text}): {matched}"
    else:
        if not values and operation != "count":
            return ComputeResult(
                text=(f"No records with a {field_name} value match "
                      f"({filter_text}); nothing to compute."),
                operation=operation, field_name=field_name,
                record_count=0, filters=filters,
            )
        if not values:  # a count of zero is a valid, exact answer
            return ComputeResult(
                text=(f"count of {field_name} over 0 records "
                      f"({filter_text}): 0\nComputed deterministically from "
                      f"the index; no language model involved."),
                operation=operation, field_name=field_name, value="0",
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
