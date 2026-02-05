import torch
import torch.nn as nn

from gnn_utils.metrics import inbatch_mrr, inbatch_recall_at_k
from gnn_utils.utils import make_pairs


class TwhinLoss(nn.Module):
    def __init__(self, reg_weight=0.0):
        super().__init__()
        self.reg_weight = reg_weight
        self.scale = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, source_embeddings, target_embeddings, calculate_mrr):
        batch_size = source_embeddings.size(0)
        num_negatives = batch_size * batch_size - batch_size
        num_positives = batch_size
        neg_weight = float(num_positives) / num_negatives
        dot_products = torch.matmul(source_embeddings, target_embeddings.t())

        logits = torch.cat(
            [dot_products.diag(), dot_products.flatten()[1:].view(batch_size - 1, batch_size + 1)[:, :-1].flatten()]
        )
        labels = torch.zeros_like(logits)
        labels[:batch_size] = 1.0
        weights = torch.full_like(logits, neg_weight)
        weights[:batch_size] = 1.0

        loss = (
            nn.functional.binary_cross_entropy_with_logits(
                logits * torch.exp(self.scale) + self.bias, labels, weights, reduction="sum"
            )
            / 2
            / batch_size
        )
        reg = (source_embeddings**2).mean() + (target_embeddings**2).mean()

        if not calculate_mrr:
            return (
                loss,
                self.reg_weight * reg,
            )
        return (loss, self.reg_weight * reg, inbatch_mrr(dot_products))


class InBatchSampledSoftmax(nn.Module):
    MIN_TEMPERATURE = 0.01
    MAX_TEMPERATURE = 100.0

    def __init__(self):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(0.0))

    @property
    def temperature(self):
        return torch.clip(torch.exp(self.tau), min=self.MIN_TEMPERATURE, max=self.MAX_TEMPERATURE)

    def forward(self, queries, candidates, recall_at=None):
        logits = self.temperature * queries @ candidates.T
        if recall_at is None:
            return -torch.log_softmax(logits, dim=1).diag().mean()
        return -torch.log_softmax(logits, dim=1).diag().mean(), inbatch_recall_at_k(queries, candidates, recall_at)


class CalibratedPairwiseLogistic(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets, lengths):
        pairs = make_pairs(lengths)
        target0, target1 = targets[pairs]
        weight = target0 > target1
        weight_sum = weight.sum()
        if weight_sum > 0:
            context0, context1 = logits[pairs]
            loss = -torch.nn.functional.log_softmax(
                torch.stack([torch.nn.functional.logsigmoid(context0), torch.nn.functional.logsigmoid(context1)]), dim=0
            )[0]
            return ((loss * weight).sum() / weight_sum).to(logits.dtype)
        else:
            return torch.zeros((), dtype=logits.dtype, device=logits.device)
