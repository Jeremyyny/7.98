"""Strict answer extraction; never reward a number found inside a derivation."""
from __future__ import annotations

import re
from fractions import Fraction
from functools import lru_cache


def unbox(text: str) -> str | None:
    text = text.strip()
    if not text.startswith(r"\boxed{"):
        return None
    depth = 1
    for i in range(7, len(text)):
        if text[i] == "{" and (i == 0 or text[i - 1] != "\\"):
            depth += 1
        elif text[i] == "}" and (i == 0 or text[i - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return text[7:i].strip() if not text[i + 1:].strip() else None
    return None


def extract_final(text: str) -> str | None:
    """Require one explicit final declaration on the final nonblank line."""
    lines = [s.strip() for s in (text or "").splitlines() if s.strip()]
    if not lines or sum(s.startswith("FINAL_ANSWER:") for s in lines) != 1:
        return None
    if not lines[-1].startswith("FINAL_ANSWER:"):
        return None
    answer = unbox(lines[-1].removeprefix("FINAL_ANSWER:").strip())
    return answer if answer and len(answer) <= 1024 else None


def as_draft(text: str) -> str:
    """Keep the full derivation, removing only its terminal declaration."""
    return "\n".join(line for line in text.splitlines()
                     if not line.strip().startswith("FINAL_ANSWER:")).strip()


def _number(text: str) -> Fraction | None:
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+-]?\d+)?", text):
        try:
            return Fraction(text)
        except (ValueError, ZeroDivisionError):
            return None
    return None


@lru_cache(maxsize=8192)
def _parse(text: str):
    from math_verify import LatexExtractionConfig, parse
    return parse(r"\boxed{" + text + "}", extraction_config=[LatexExtractionConfig()],
                 fallback_mode="no_fallback", extraction_mode="first_match")


def valid_gold(gold: str) -> bool:
    if not gold or gold.strip().lower() in {"proof", "unknown", "none", "n/a"}:
        return False
    return _number(gold.strip()) is not None or bool(_parse(gold.strip()))


def equivalent(prediction: str | None, gold: str) -> bool:
    if prediction is None:
        return False
    p, g = prediction.strip(), str(gold).strip()
    if not p or len(p) > 1024:
        return False
    pn, gn = _number(p), _number(g)
    if pn is not None and gn is not None:
        return pn == gn
    # Missing dependencies are setup errors, not silently wrong answers.
    from math_verify import verify
    gold_parsed, pred_parsed = _parse(g), _parse(p)
    # Do not let decimal rounding accept a near-miss integer/fraction wrapped
    # in LaTeX (the plain-number path above is exact as well).
    from sympy import Float, Rational
    if len(gold_parsed) == len(pred_parsed) == 1 and all(isinstance(x, (Rational, Float)) for x in (gold_parsed[0], pred_parsed[0])):
        return Rational(str(gold_parsed[0])) == Rational(str(pred_parsed[0]))
    return bool(verify(gold_parsed, pred_parsed))


def correct(text: str, gold: str) -> bool:
    return equivalent(extract_final(text), gold)
