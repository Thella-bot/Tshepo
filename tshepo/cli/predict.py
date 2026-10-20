"""Classify messages with the rule engine or a trained model.

    python -m tshepo.cli.predict "Your PIN is required to..." --sender +26658123456 --model hybrid
    python -m tshepo.cli.predict --file data/real_labelled.jsonl --model tfidf --limit 20

--model: rules | tfidf | hybrid | transformer
"""
import argparse
import json
from typing import Callable, Optional

from ..config import LABELS, model_path
from ..evaluation import load_jsonl
from ..models.gate import load_gate, load_transformer_gate
from ..parsing.parser import classify
from ..resilience import CircuitBreaker, fallback_gate

GateFn = Callable[[str, Optional[str]], str]


def build_model(name: str, path: Optional[str] = None) -> GateFn:
    """Return a gate callable wrapped in a circuit breaker that falls back to
    the zero-dependency rule gate on repeated model failures."""
    if name == "rules":
        base: GateFn = lambda body, sender: classify(body, sender)["label"]
    elif name == "transformer":
        base = load_transformer_gate(path or "models/transformer")
    else:
        base = load_gate(path or model_path(f"models/{name}.joblib"))

    breaker = CircuitBreaker(failure_threshold=3, cooldown_seconds=15.0)
    return fallback_gate(base, breaker=breaker)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("body", nargs="?", help="a single message; omit to read --file")
    ap.add_argument("--file", default=None, help="JSONL file with body/sender/label per line")
    ap.add_argument("--sender", default=None)
    ap.add_argument("--model", default="rules", choices=["rules", "tfidf", "hybrid", "transformer"])
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--limit", type=int, default=10)
    a = ap.parse_args()

    predict = build_model(a.model, a.model_path)

    if a.body:
        res = classify(a.body, a.sender)
        print(json.dumps({"model": a.model, "label": predict(a.body, a.sender),
                          "rule_label": res["label"], "flags": res["flags"],
                          "parsed": res["parsed"].__dict__}, indent=2, ensure_ascii=False))
        return

    if not a.file:
        ap.error("give a message body or --file")

    rows = load_jsonl(a.file)[:a.limit]
    correct = 0
    for r in rows:
        label = predict(r.get("body", ""), r.get("sender"))
        if r.get("label"):
            correct += int(label == r["label"])
        print(f"{label:20} | {r.get('sender',''):14} | {r.get('body','')[:90]}")
    if rows and all(r.get("label") for r in rows):
        print(f"\nagreement with labels: {correct}/{len(rows)}")
    print("labels:", LABELS)


if __name__ == "__main__":
    main()
