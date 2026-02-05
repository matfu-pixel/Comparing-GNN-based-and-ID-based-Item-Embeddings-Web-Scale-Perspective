import torch


@torch.no_grad()
def inbatch_mrr(dot_products: torch.Tensor) -> torch.Tensor:
    """
    dot_products: [B, B] where dot_products[i, j] is score of source i vs target j,
    and the positive for source i is at j == i.

    Returns: scalar MRR for this batch (torch.Tensor).
    """
    B = dot_products.size(0)
    sorted_idx = dot_products.argsort(dim=1, descending=True)  # [B, B]
    target_indices = torch.arange(B, device=dot_products.device)  # [B]

    positions = (sorted_idx == target_indices.unsqueeze(1)).nonzero(as_tuple=False)[:, 1]  # [B]
    ranks = positions.to(torch.float32) + 1.0  # [B]

    rr = 1.0 / ranks
    mrr = float(rr.mean())
    return mrr


@torch.no_grad()
def inbatch_recall_at_k(queries: torch.Tensor, candidates: torch.Tensor, ks=(10, 100, 1000)) -> dict:
    """
    queries:    [B, D]
    candidates: [B, D]
    Assumes the positive for query i is candidate i.

    Returns: dict {k: recall_at_k} where recall_at_k is a scalar tensor.
    """
    scores = queries @ candidates.T  # [B, B]
    B = scores.size(0)
    max_k = min(max(ks), B)  # cannot be larger than batch size

    topk_idx = scores.topk(k=max_k, dim=1).indices  # [B, max_k]
    target_indices = torch.arange(B, device=scores.device).unsqueeze(1)  # [B, 1]

    recalls = {}
    for k in ks:
        k_eff = min(k, max_k)  # clamp if k > B
        hits = (topk_idx[:, :k_eff] == target_indices).any(dim=1).float()  # [B]
        recalls[k] = float(hits.mean())
    return recalls
