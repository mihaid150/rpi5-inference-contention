import statistics


def estimate(arrival, phases):
    submission = max(arrival, phases["publication_s"]) + phases["retrieval_s"]
    slot = phases["current_complete_s"]
    end = max(submission, slot) + phases["fresh_service_s"]
    bottleneck = (
        "current-slot"
        if slot >= submission
        else (
            "arrival/retrieval"
            if arrival >= phases["publication_s"]
            else "publication/retrieval"
        )
    )
    return dict(
        current_complete_s=slot,
        publication_s=phases["publication_s"],
        fresh_submission_s=submission,
        fresh_complete_s=end,
        fresh_latency_from_actual_arrival_s=end - arrival,
        bottleneck=bottleneck,
    )


def fit(rows):
    values = {}
    for policy in ("shared", "partition-2-2"):
        rs = [
            r
            for r in rows
            if r["cell"]["policy"] == policy and r["cell"]["update"] == "incremental"
        ]
        if len(rs) != 6:
            raise ValueError(
                "Expected six E2 incremental observations per fixed policy"
            )
        phases = {
            k: statistics.median((r["metrics"][k] for r in rs))
            for k in ("current_complete_s", "publication_s", "retrieval_s")
        }
        phases["fresh_service_s"] = statistics.median(
            (
                (
                    r["timeline"]["fresh"]["server_timings"]["prompt_ms"]
                    + r["timeline"]["fresh"]["server_timings"]["predicted_ms"]
                )
                / 1000
                for r in rs
            )
        )
        values[policy] = phases
    return values
