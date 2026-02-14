from collections import defaultdict

import torch


@torch.no_grad()
def inbatch_mrr(dot_products: torch.Tensor) -> list[float]:
    """
    dot_products: [B, B] where dot_products[i, j] is score of source i vs target j,
    and the positive for source i is at j == i.

    Returns: list of MRR scores for this batch.
    """
    B = dot_products.size(0)
    sorted_idx = dot_products.argsort(dim=1, descending=True)  # [B, B]
    target_indices = torch.arange(B, device=dot_products.device)  # [B]

    positions = (sorted_idx == target_indices.unsqueeze(1)).nonzero(as_tuple=False)[:, 1]  # [B]
    ranks = positions.to(torch.float32) + 1.0  # [B]

    rr = 1.0 / ranks
    return rr.cpu().float().tolist()


@torch.no_grad()
def inbatch_hitrate_at_k(queries: torch.Tensor, candidates: torch.Tensor, ks=(10, 100, 1000)) -> dict[int, list[float]]:
    """
    queries:    [B, D]
    candidates: [B, D]
    Assumes the positive for query i is candidate i.

    Returns: dict {k: hitrate_at_k} where hitrate_at_k is a boolean tensor of shape [B].
    """
    scores = queries @ candidates.T  # [B, B]
    B = scores.size(0)

    max_k = max(ks)
    assert max_k <= B, "max_k cannot be larger than batch size"

    topk_idx = scores.topk(k=max_k, dim=1).indices  # [B, max_k]
    target_indices = torch.arange(B, device=scores.device).unsqueeze(1)  # [B, 1]

    hitrates = defaultdict(list)
    for k in ks:
        hits = (topk_idx[:, :k] == target_indices).any(dim=1).float()  # [B]
        hitrates[k].extend(hits.cpu().tolist())
    return hitrates
