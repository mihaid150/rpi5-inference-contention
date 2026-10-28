import re

ARMS = [
    dict(name="raw-mb2", microbatch=2, repack=False, mask=[]),
    dict(name="raw-mb8", microbatch=8, repack=False, mask=[]),
    dict(name="raw-mb32", microbatch=32, repack=False, mask=[]),
    dict(name="raw-mb64", microbatch=64, repack=False, mask=[]),
    dict(name="repack-mb2", microbatch=2, repack=True, mask=[]),
    dict(name="repack-mb64", microbatch=64, repack=True, mask=[]),
]


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


def paired_gate(baseline, candidate):
    base = {r["fixture"]: r for r in baseline}
    cand = {r["fixture"]: r for r in candidate}
    if (
        len(base) != len(baseline)
        or len(cand) != len(candidate)
        or (not base)
        or (base.keys() != cand.keys())
    ):
        return dict(passed=False, reason="incomplete or duplicate paired fixtures")
    facts = [name for name, r in cand.items() if not r["quality"]["factual_passed"]]
    regressions = [
        name
        for name, r in cand.items()
        if base[name]["quality"]["strict_passed"]
        and (not r["quality"]["strict_passed"])
    ]
    return dict(
        passed=not facts and (not regressions),
        factual_failures=facts,
        strict_regressions=regressions,
        strict_failures=[
            name for name, r in cand.items() if not r["quality"]["strict_passed"]
        ],
        scope="development paired checks, no independent semantic adjudication",
    )


def select(baseline, candidates):
    eligible = []
    gates = {}
    for arm, rows in candidates.items():
        gate = paired_gate(baseline, rows)
        gates[arm] = gate
        if gate["passed"]:
            eligible.append((sum((r["prefill_compute_s"] for r in rows)), arm))
    return (min(eligible)[1] if eligible else None, gates)
