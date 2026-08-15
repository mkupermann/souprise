"""Verified answers: values are copied from records, never generated.

The verified answer mode removes the language model from the factual
path. A rule-based detector maps the question to a record field, the
value is copied verbatim from the retrieved record, ambiguity yields all
candidates instead of a guess, and weak retrieval yields a refusal.
By construction, a verified answer cannot contain a fabricated figure.

License: Apache-2.0
Copyright 2026 Michael Kupermann
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from souprise.core.pipeline import RetrievalResult

REFUSAL_TEXT = ("No matching records found for this question. "
                "I don't answer without a source.")

# Question keywords -> record field. First match wins; order matters
# (e.g. "annual revenue" before "amount"-ish words).
_FIELD_PATTERNS = [
    (r"annual revenue|revenue|umsatz", "Annual Revenue"),
    (r"amount|betrag|owe|charge|bill", "Amount"),
    (r"status|bearbeitungsstand|paid or overdue|overdue or paid", "Status"),
    (r"stock|lagerbestand|units .* in stock|inventory", "Stock"),
    (r"price|preis", "Price"),
    (r"margin|marge", "Margin"),
    (r"quantity|menge", "Quantity"),
    (r"total", "Total"),
    (r"allocated|budget allocated", "Allocated"),
    (r"spent|ausgegeben", "Spent"),
    (r"remaining|verbleibend", "Remaining"),
    (r"utilization|auslastung", "Utilization"),
    (r"segment", "Segment"),
    (r"region", "Region"),
    (r"due date|due|fällig", "Due Date"),
    (r"contact|ansprechpartner", "Contact"),
    (r"open tickets|tickets", "Open Tickets"),
]

DEFAULT_TEMPLATE = "{summary}\nSource: {source}"


@dataclass
class VerifiedAnswer:
    """A deterministic answer whose values are copied from records."""
    text: str
    verified: bool = True
    refused: bool = False
    ambiguous: bool = False
    field_name: Optional[str] = None
    values: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)


def detect_field(question: str) -> Optional[str]:
    """Map a question to a record field name, or None."""
    q = question.lower()
    for pattern, field_name in _FIELD_PATTERNS:
        if re.search(pattern, q):
            return field_name
    return None


def _fields_of(result: RetrievalResult) -> Dict[str, str]:
    return dict(
        line.split(": ", 1)
        for line in result.content.splitlines() if ": " in line
    )


def _entity_of(title: str) -> str:
    """Entity key of a record title (strips time qualifiers)."""
    for token in title.split():
        if "_" in token:
            return token
    return title


_ENTITY_TOKEN_RE = re.compile(r"\b[A-Za-z]+_[A-Za-z0-9]+\b")


def entities_in(question: str) -> List[str]:
    """Entity identifiers mentioned in the question (Customer_0042 style)."""
    return [t.lower() for t in _ENTITY_TOKEN_RE.findall(question)]


def answer_verified(
    question: str,
    results: List[RetrievalResult],
    min_score: float = 0.52,
    template: str = DEFAULT_TEMPLATE,
) -> VerifiedAnswer:
    """Build a deterministic answer from retrieval results.

    Args:
        question: The user's question.
        results: Retrieval results, best first.
        min_score: Below this top score the answer is a refusal.
        template: Answer template with {summary} and {source}.
    """
    if not results or results[0].score < min_score:
        return VerifiedAnswer(text=REFUSAL_TEXT, refused=True)

    top = results[0]

    # Entity verification: if the question names a specific entity, the
    # answering record must actually carry it. Similarity scores alone
    # would happily return the closest OTHER entity; that is a wrong
    # answer, so it becomes a refusal instead.
    asked = entities_in(question)
    if asked:
        matching = [r for r in results
                    if any(e in r.title.lower() or e in r.content.lower()
                           for e in asked)]
        if not matching:
            return VerifiedAnswer(text=REFUSAL_TEXT, refused=True)
        results = matching
        top = results[0]

    field_name = detect_field(question)

    if field_name is None:
        # No specific field detected: return the whole matching record,
        # verbatim. Still verified; nothing is generated.
        summary = f"Closest matching record:\n{top.content}"
        return VerifiedAnswer(
            text=template.format(summary=summary, source=top.title),
            sources=[top.title],
        )

    # Candidates: records of the same entity that carry the field
    entity = _entity_of(top.title)
    candidates = []
    for result in results:
        if _entity_of(result.title) != entity:
            continue
        value = _fields_of(result).get(field_name)
        if value is not None:
            candidates.append((result.title, value))

    if not candidates:
        summary = (f"The matching record does not carry a field "
                   f"'{field_name}':\n{top.content}")
        return VerifiedAnswer(
            text=template.format(summary=summary, source=top.title),
            field_name=field_name, sources=[top.title],
        )

    distinct_values = {value for _, value in candidates}
    if len(distinct_values) > 1:
        # Conflicting records for the same entity: list all, guess none.
        lines = "\n".join(f"- {title}: {field_name} = {value}"
                          for title, value in candidates)
        summary = (f"Multiple records for {entity} carry different values "
                   f"for {field_name}; all of them:\n{lines}")
        return VerifiedAnswer(
            text=template.format(
                summary=summary,
                source=", ".join(t for t, _ in candidates)),
            ambiguous=True, field_name=field_name,
            values=sorted(distinct_values),
            sources=[t for t, _ in candidates],
        )

    title, value = candidates[0]
    summary = f"{field_name} for {title}: {value}"
    return VerifiedAnswer(
        text=template.format(summary=summary, source=title),
        field_name=field_name, values=[value], sources=[title],
    )
