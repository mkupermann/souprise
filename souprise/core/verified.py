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
    (r"trend|trending|sales trend", "Trend"),
    (r"30.day sales|recent sales|monthly sales", "30-Day Sales"),
]

# Questions that ask for a record overview rather than a single field.
_RECORD_INTENT_RE = re.compile(
    r"\b(profile|details|overview|pull up|tell me about|summari[sz]e|"
    r"how is .+ (performing|doing)|budget status|akte|übersicht)\b")

DEFAULT_TEMPLATE = "{summary}\nSource: {source}"


@dataclass
class VerifiedAnswer:
    """A deterministic answer whose values are copied from records."""
    text: str
    verified: bool = True
    refused: bool = False
    ambiguous: bool = False
    # Which route produced the answer: "field", "ambiguous", "record",
    # "record_intent", or "refusal". Used for coverage measurement.
    path: str = "field"
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
        if _ENTITY_TOKEN_RE.fullmatch(token):
            return token
    return title


# Underscore ids (Customer_0042) and hyphenated ids containing a digit
# (INV-2026-0815); plain hyphenated words like "re-run" do not match.
_ENTITY_TOKEN_RE = re.compile(
    r"\b[A-Za-z]+_[A-Za-z0-9]+\b|\b[A-Za-z]+(?:-[A-Za-z0-9]*\d[A-Za-z0-9]*)+\b")

# Company-shaped natural names: capitalized words ending in a legal-form
# suffix, or "Name & Name" pairs.
_COMPANY_NAME_RE = re.compile(
    r"\b(?:[A-ZÄÖÜ][\w&.\-]*\s+)+?(?:GmbH|AG|SE|KG|Corp\.?|Corporation|"
    r"Inc\.?|Ltd\.?|LLC|Co\.)\b|"
    r"\b[A-ZÄÖÜ][a-zäöüß]+\s+&\s+[A-ZÄÖÜ][a-zäöüß]+\b")

_VOCAB_FIELDS = ("Customer", "Product", "Department", "Contact")


def known_entities(entries: List[Dict[str, str]]) -> set:
    """Entity vocabulary of an index: field values and title id-tokens."""
    vocabulary = set()
    for entry in entries:
        text = entry.get("text", "")
        for line in text.splitlines():
            if ": " in line:
                key, value = line.split(": ", 1)
                if key in _VOCAB_FIELDS and value.strip():
                    vocabulary.add(value.strip().lower())
        for token in _ENTITY_TOKEN_RE.findall(text):
            vocabulary.add(token.lower())
    return vocabulary


def entities_in(question: str, vocabulary: Optional[set] = None) -> List[str]:
    """Entity mentions in the question.

    Underscore identifiers always count. With a vocabulary, natural
    names known to the index count too; company-shaped names NOT in the
    vocabulary are returned as well, so the caller refuses instead of
    answering about the closest lookalike.
    """
    found = [t.lower() for t in _ENTITY_TOKEN_RE.findall(question)]
    if vocabulary:
        q = question.lower()
        for name in vocabulary:
            if len(name) >= 4 and name in q:
                found.append(name)
    for match in _COMPANY_NAME_RE.finditer(question):
        found.append(match.group(0).strip().lower())
    return list(dict.fromkeys(found))


def answer_verified(
    question: str,
    results: List[RetrievalResult],
    min_score: float = 0.52,
    template: str = DEFAULT_TEMPLATE,
    all_entries: Optional[List[Dict[str, str]]] = None,
    vocabulary: Optional[set] = None,
) -> VerifiedAnswer:
    """Build a deterministic answer from retrieval results.

    Args:
        question: The user's question.
        results: Retrieval results, best first.
        min_score: Below this top score the answer is a refusal.
        template: Answer template with {summary} and {source}.
        all_entries: Full index contents; when the question names an
            entity that missed the top-k, a deterministic scan over all
            entries finds its records (or proves it absent).
    """
    if not results or results[0].score < min_score:
        return VerifiedAnswer(text=REFUSAL_TEXT, refused=True, path="refusal")

    top = results[0]

    # Entity verification: if the question names a specific entity, the
    # answering record must actually carry it. Similarity scores alone
    # would happily return the closest OTHER entity; that is a wrong
    # answer, so it becomes a refusal instead.
    if vocabulary is None and all_entries:
        vocabulary = known_entities(all_entries)
    asked = entities_in(question, vocabulary)
    if asked:
        matching = [r for r in results
                    if any(e in r.title.lower() or e in r.content.lower()
                           for e in asked)]
        if not matching and all_entries:
            # Deterministic scan: top-k similarity can miss a rare
            # entity that the index does contain.
            matching = [
                RetrievalResult(title=str(e.get("id", "")), content=e["text"],
                                score=1.0)
                for e in all_entries
                if any(a in e.get("text", "").lower()
                       or a in str(e.get("id", "")).lower() for a in asked)
            ][:5]
        if not matching:
            return VerifiedAnswer(text=REFUSAL_TEXT, refused=True, path="refusal")
        results = matching
        top = results[0]

    # Record-overview intent: the right answer IS the verbatim record.
    if _RECORD_INTENT_RE.search(question.lower()):
        shown = results[:3]
        summary = "\n\n".join(f"--- {r.title} ---\n{r.content}" for r in shown)
        return VerifiedAnswer(
            text=template.format(summary=summary,
                                 source=", ".join(r.title for r in shown)),
            sources=[r.title for r in shown], path="record_intent",
        )

    field_name = detect_field(question)

    if field_name is None:
        # No specific field detected: return the whole matching record,
        # verbatim. Still verified; nothing is generated.
        summary = f"Closest matching record:\n{top.content}"
        return VerifiedAnswer(
            text=template.format(summary=summary, source=top.title),
            sources=[top.title], path="record",
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
            ambiguous=True, field_name=field_name, path="ambiguous",
            values=sorted(distinct_values),
            sources=[t for t, _ in candidates],
        )

    title, value = candidates[0]
    summary = f"{field_name} for {title}: {value}"
    return VerifiedAnswer(
        text=template.format(summary=summary, source=title),
        field_name=field_name, values=[value], sources=[title],
    )
