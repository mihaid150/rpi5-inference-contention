def allocation(policy, cpus):
    cpus = sorted(cpus)
    if len(cpus) != 4:
        raise ValueError("Exactly four allowed physical CPU IDs required")
    if policy == "partition-2-2":
        return (dict(cpus=cpus[:2], prefill_threads=2, decode_threads=2), cpus[2:], 2)
    if policy == "partition-3-1":
        return (dict(cpus=cpus[:3], prefill_threads=3, decode_threads=2), cpus[3:], 1)
    if policy in ("shared", "inference-only", "update-only"):
        return (dict(cpus=cpus, prefill_threads=4, decode_threads=2), cpus, 2)
    raise ValueError(policy)


def diagnosis_rows(plan):
    rows = [r for r in plan["rows"] if r["stage"] == "diagnosis"]
    if len(rows) != 27 or len({r["id"] for r in rows}) != 27:
        raise ValueError("Incorrect E1 schedule")
    for block in range(1, 4):
        rs = [r for r in rows if r["block"] == block]
        if len(rs) != 9 or sum((r["policy"] == "update-only" for r in rs)) != 1:
            raise ValueError("Incorrect block")
        keys = {(r["fixture"], r["policy"]) for r in rs if r["fixture"] is not None}
        fixtures = {k[0] for k in keys}
        if len(fixtures) != 2 or keys != {
            (f, a)
            for f in fixtures
            for a in ("shared", "inference-only", "partition-2-2", "partition-3-1")
        }:
            raise ValueError("Incomplete matched policy matrix")
    return rows


def regime_rows(plan):
    rows = [r for r in plan["rows"] if r["stage"] == "regimes"]
    if len(rows) != 36 or len({r["id"] for r in rows}) != 36:
        raise ValueError("Incorrect E2 schedule")
    expected = {
        (mode, offset, policy)
        for mode in ("incremental", "full-rebuild")
        for offset in (0, 90)
        for policy in ("shared", "partition-2-2", "deferred")
    }
    for block in range(1, 4):
        rs = [r for r in rows if r["block"] == block]
        if (
            len({r["fixture"] for r in rs}) != 1
            or {(r["update"], r["fresh_arrival_s"], r["policy"]) for r in rs}
            != expected
        ):
            raise ValueError("Incomplete E2 matched block")
    return rows


def remaining_prefix(cells, previous_rows, expected_ids):
    done = [row["cell"]["id"] for row in previous_rows]
    if done != [cell["id"] for cell in cells[: len(done)]] or done != expected_ids:
        raise ValueError("Resume must be an exact prefix of original schedule")
    return cells[len(done) :]


def prediction_rows(plan):
    rows = [r for r in plan["rows"] if r["stage"] == "prediction-check"]
    if len(rows) != 12 or len({r["id"] for r in rows}) != 12:
        raise ValueError("Incorrect E3 schedule")
    expected = {
        (arrival, policy)
        for arrival in (30, 60)
        for policy in ("shared", "partition-2-2")
    }
    for block in range(1, 4):
        rs = [r for r in rows if r["block"] == block]
        if (
            len({r["fixture"] for r in rs}) != 1
            or {(r["fresh_arrival_s"], r["policy"]) for r in rs} != expected
        ):
            raise ValueError("Incomplete E3 block")
        if any((r["update"] != "incremental" for r in rs)):
            raise ValueError("E3 incremental only")
    if {r["fixture"] for r in rows} & {r["fixture"] for r in regime_rows(plan)}:
        raise ValueError("E3 timing documents overlap E2")
    return rows
