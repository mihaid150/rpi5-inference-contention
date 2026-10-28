import re


def quality(case, text, stop):
    lowered = text.lower()
    missing = [s for s in case["required"] if s.lower() not in lowered]
    forbidden = [s for s in case.get("forbidden", []) if s.lower() in lowered]
    if (
        case["id"] == "eval-negation"
        and "gateway" in lowered
        and ("not" not in lowered)
    ):
        forbidden.append("gateway without explicit negation")
    unsupported = case.get("unsupported", case["id"] == "eval-unsupported")
    strict = not missing and (not forbidden) and (stop == "eos")
    safety = strict
    if unsupported:
        citation = re.escape(case.get("source", "NETWORK:1"))
        safety = stop == "eos" and bool(
            re.fullmatch(
                "unknown(?:\\s*\\[(?:source_id:)?" + citation + "\\])?\\.?",
                text.strip(),
                re.I,
            )
        )
        strict = safety and lowered.strip().strip(".") == "unknown"
    return dict(
        strict_passed=strict,
        factual_passed=safety,
        missing=missing,
        forbidden=forbidden,
        eos=stop == "eos",
        unsupported=unsupported,
    )
