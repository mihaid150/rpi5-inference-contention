import numpy as np


def log_probs(logits):
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim != 1 or not x.size or (not np.isfinite(x).all()):
        raise ValueError("Invalid logits")
    x = x - x.max()
    return x - np.log(np.exp(x).sum())


def metrics(reference, candidate, target):
    p = log_probs(reference)
    q = log_probs(candidate)
    if p.shape != q.shape or not 0 <= target < len(p):
        raise ValueError("Vocabulary/target mismatch")
    kl = float(np.sum(np.exp(p) * (p - q)))
    if not np.isfinite(kl) or kl < -1e-10:
        raise ValueError("Invalid KL")
    return dict(
        kl_reference_to_candidate=max(0.0, kl),
        reference_logp=float(p[target]),
        candidate_logp=float(q[target]),
        top1_agreement=bool(p.argmax() == q.argmax()),
    )


def teacher_forced(step, prompt, answer):
    if not prompt or not answer:
        raise ValueError("Empty prompt/answer")
    for pos, t in enumerate(prompt):
        logits = step(pos, t)
    for j, target in enumerate(answer):
        yield (j, target, np.asarray(logits).copy())
        if j + 1 < len(answer):
            logits = step(len(prompt) + j, target)
