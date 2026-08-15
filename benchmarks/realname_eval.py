"""BENCH-7 Q2: natural-name entity verification per benchmarks/PROTOCOL.md.

Builds a corpus whose entities are natural company names, then measures
value accuracy on known names and refusal rate on unknown names.

Usage:
    PYTHONPATH=. python3 benchmarks/realname_eval.py

License: Apache-2.0
Copyright 2026 Michael Kupermann
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from souprise import RAGConfig, SoupriseRAG  # noqa: E402

KNOWN = [
    "ACME GmbH", "Meyer & Söhne", "Nordwind Logistik AG", "Baumann KG",
    "Delta Systems Inc.", "Hafen Handel GmbH", "Krüger & Partner",
    "Orion Tech Ltd.", "Schmidt Maschinenbau AG", "Weser Chemie SE",
    "Falkenberg GmbH", "TransAlp Spedition AG", "Bergmann & Vogel",
    "Lindner Elektro KG", "Pacific Trade Corp.", "Rhein Metall Handel GmbH",
    "Steiner & Brandt", "Nova Print Ltd.", "Küstenwerk AG", "Adler Logistik GmbH",
    "Brandt Automation SE", "Fischer & Weber", "Global Parts Inc.",
    "Hansa Marine GmbH", "Isartal Software AG", "Jäger Antriebe KG",
    "Kraft Anlagenbau GmbH", "Lorenz & Söhne", "Møller Nordic Corp.",
    "Neumann Textil AG",
]

UNKNOWN = [
    "Zorblatt Industries Inc.", "Phantom Consulting GmbH", "Vexcorp Ltd.",
    "Nimbus & Fry", "Quantex Systems AG", "Umbra Trading KG",
    "Wexler & Stone", "Novagene Pharma SE", "Kestrel Logistics Corp.",
    "Bergfeld & Sohn", "Xandria Metals GmbH", "Polarwind Shipping AG",
    "Trentor Machines Ltd.", "Grimm & Falk", "Solvex Chemie GmbH",
    "Aurora Deck Inc.", "Marlow & Reed", "Cinderhall Group SE",
    "Vantor Electric KG", "Obsidian Freight Corp.",
]


def main():
    rng = random.Random(9)
    entries = []
    expected = {}
    for i, name in enumerate(KNOWN):
        amount = f"${rng.randint(1_000, 90_000)},{rng.randint(100, 999)//10:02d}0.00"
        amount = f"${rng.randint(1_000, 90_000):,}.00"
        status = rng.choice(["overdue", "paid", "open"])
        entries.append({
            "id": f"Invoice {name} 2026-{i:02d}",
            "text": (f"Invoice {name} 2026-{i:02d}\nCustomer: {name}\n"
                     f"Amount: {amount}\nStatus: {status}\nRegion: EU"),
        })
        expected[name] = (amount, status)

    rag = SoupriseRAG(RAGConfig(retriever="simple", answer_mode="verified",
                                retrieval_k=5))
    rag.index_from_entries(entries)

    correct = 0
    for name in KNOWN:
        field = "amount" if rng.random() < 0.5 else "status"
        result = rag.query(f"What is the {field} of the invoice for {name}?")
        want = expected[name][0 if field == "amount" else 1]
        correct += (not result.refused
                    and want.replace(",", "") in result.answer.replace(",", ""))
    known_accuracy = correct / len(KNOWN)
    print(f"known-name value accuracy: {known_accuracy:.3f} ({correct}/{len(KNOWN)})")

    refusals = 0
    for name in UNKNOWN:
        result = rag.query(f"What is the amount of the invoice for {name}?")
        refusals += result.refused
    refusal_rate = refusals / len(UNKNOWN)
    print(f"unknown-name refusal rate: {refusal_rate:.3f} ({refusals}/{len(UNKNOWN)})")

    results = {
        "known_accuracy": round(known_accuracy, 4),
        "unknown_refusal_rate": round(refusal_rate, 4),
        "bars": {"known": "= 1.000", "unknown": "= 1.000"},
        "verdicts": {
            "known": "pass" if known_accuracy == 1.0 else "fail",
            "unknown": "pass" if refusal_rate == 1.0 else "fail",
        },
    }
    out = Path("benchmarks/results")
    out.mkdir(parents=True, exist_ok=True)
    (out / "realname_eval.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results["verdicts"], indent=2))


if __name__ == "__main__":
    main()
