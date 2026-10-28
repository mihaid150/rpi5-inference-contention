import re
import statistics


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
