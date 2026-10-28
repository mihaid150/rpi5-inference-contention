import re
import statistics

SERVER_ARMS = [
    dict(name="server-64-f32", microbatch=64, cache="f32", flash="off"),
    dict(name="server-256-f32", microbatch=256, cache="f32", flash="off"),
    dict(name="server-512-f32", microbatch=512, cache="f32", flash="off"),
    dict(name="server-512-default", microbatch=512, cache="f16", flash="auto"),
]


def parse_answer(text):
    citations = re.findall("\\[([^\\[\\]]+):(\\d+)\\]", text)
    return (re.sub("\\[[^\\[\\]]+:\\d+\\]", "", text).strip(), citations)


def score(case, text, stop, evaluator):
    answer, citations = parse_answer(text)
    metrics = evaluator.compute_turn_score(case["document_id"], 1, answer)
    correct = (
        stop == "eos"
        and max((evaluator.compute_exact(g, answer) for g in case["gold_answers"])) == 1
    )
    allowed = set(map(str, case.get("citation_chunks", [0])))
    cited = any(
        (source == case["source"] and chunk in allowed for source, chunk in citations)
    )
    if case.get("unsupported"):
        cited = not citations and answer.lower().strip() == "unknown"
    return dict(
        **metrics,
        answer=answer,
        citations=citations,
        eos=stop == "eos",
        exact_any_reference=correct,
        source_cited=cited,
        automatic_supported_correct=correct and cited,
        human_semantic_review="pending",
        scope="automatic lexical and citation checks; not semantic adjudication"
    )


def select_standard(rows, case_ids, check):
    candidates = []
    gates = {}
    for arm in SERVER_ARMS:
        rs = [r for r in rows if r["arm"] == arm["name"]]
        complete = len(rs) == 3 * len(case_ids) and all(
            (sum((r["fixture"] == c for r in rs)) == 3 for c in case_ids)
        )
        passed = complete and all((check(r)["strict_passed"] for r in rs))
        gates[arm["name"]] = dict(complete=complete, passed=passed)
        if passed:
            candidates.append(
                (statistics.median((r["total_request_s"] for r in rs)), arm["name"])
            )
    return (min(candidates)[1] if candidates else None, gates)


def paired_regressions(baseline, candidate):
    base = {r["fixture"]: r for r in baseline}
    cand = {r["fixture"]: r for r in candidate}
    if (
        len(base) != len(baseline)
        or len(cand) != len(candidate)
        or base.keys() != cand.keys()
        or (not base)
    ):
        return dict(passed=False, reason="missing or duplicate evaluation cases")
    lost = [
        key
        for key, row in base.items()
        if row["assessment"]["automatic_supported_correct"]
        and (not cand[key]["assessment"]["automatic_supported_correct"])
    ]
    return dict(
        passed=not lost,
        regressions=lost,
        scope="paired automatic correctness/citation regression only; independent human review pending",
    )


def application_verdict(episodes, admitted):
    by = {}
    for row in episodes:
        by.setdefault(row["policy"], []).append(row)
    base = by.get("standard-concurrent", [])
    if len(base) != 3:
        return dict(status="incomplete", benefit=False)
    result = {}
    for name, rows in by.items():
        if name == "standard-concurrent":
            continue
        eligible = len(rows) == 3 and all(
            (
                r["current_passed"] and r["fresh_passed"] and r["retrieval_passed"]
                for r in rows
            )
        )
        bm = {
            k: statistics.median((r[k] for r in base))
            for k in ("current_answer_s", "publication_s", "fresh_answer_s")
        }
        cm = (
            {k: statistics.median((r[k] for r in rows)) for k in bm}
            if len(rows) == 3
            else {}
        )
        base_valid = all(
            (
                r["current_passed"] and r["fresh_passed"] and r["retrieval_passed"]
                for r in base
            )
        )
        nondominated = (
            bool(cm)
            and all((cm[k] <= bm[k] for k in bm))
            and any((cm[k] < bm[k] for k in bm))
        )
        quality_gate = name.startswith("standard-") or admitted.get(name, False)
        result[name] = dict(
            automatic_joint_benefit=eligible
            and base_valid
            and quality_gate
            and nondominated,
            quality_gate=quality_gate,
            episode_checks=eligible,
            baseline_episode_checks=base_valid,
            medians=cm,
            baseline_medians=bm,
        )
    return dict(
        status="completed",
        comparisons=result,
        scope="three descriptive episodes per policy, no confidence/noninferiority/SLA claim",
        publication_novelty_established=False,
        human_review_complete=False,
    )
