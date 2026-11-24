"""Part D: retrieval metrics. `ranked` = list of abstract ids, best first; `relevant` = set/list of relevant ids."""


def hit_at_k(ranked, relevant, k):
    return float(len(set(ranked[:k]) & set(relevant)) >= 1)


def recall_at_k(ranked, relevant, k):
    """Share of relevant abstracts found, out of as many as could fit in k: min(k, |R|)."""
    rel = set(relevant)
    return len(set(ranked[:k]) & rel) / min(k, len(rel))


def mrr_at_10(ranked, relevant):
    rel = set(relevant)
    for rank, d in enumerate(ranked[:10], start=1):
        if d in rel:
            return 1.0 / rank
    return 0.0


def evaluate_rankings(rankings, relevant_lists, ks=(1, 5, 10)):
    """Average over all questions."""
    n = len(rankings)
    out = {}
    for k in ks:
        out[f"Hit@{k}"] = sum(hit_at_k(r, rel, k) for r, rel in zip(rankings, relevant_lists)) / n
    for k in ks:
        out[f"Recall@{k}"] = sum(recall_at_k(r, rel, k) for r, rel in zip(rankings, relevant_lists)) / n
    out["MRR@10"] = sum(mrr_at_10(r, rel) for r, rel in zip(rankings, relevant_lists)) / n
    return out
