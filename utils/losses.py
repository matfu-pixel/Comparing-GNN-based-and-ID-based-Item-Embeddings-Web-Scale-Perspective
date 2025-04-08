import torch
import torch.nn as nn


class TwhinLoss(nn.Module):
    def __init__(self, reg_weight=0.):
        super().__init__()
        self.reg_weight = reg_weight
        self.scale = nn.Parameter(torch.zeros(1)) 
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, source_embeddings, target_embeddings):
        batch_size = source_embeddings.size(0)
        num_negatives = batch_size * batch_size - batch_size
        num_positives = batch_size
        neg_weight = float(num_positives) / num_negatives
        dot_products = torch.matmul(source_embeddings, target_embeddings.t())

        logits = torch.cat([dot_products.diag(), dot_products.flatten()[1:].view(batch_size - 1, batch_size + 1)[:,:-1].flatten()])
        labels = torch.zeros_like(logits)
        labels[:batch_size] = 1.
        weights = torch.full_like(logits, neg_weight)
        weights[:batch_size] = 1.

        loss = nn.functional.binary_cross_entropy_with_logits(logits * torch.exp(self.scale) + self.bias, labels, weights, reduction="sum") / 2 / batch_size
        reg = (source_embeddings ** 2).mean() + (target_embeddings ** 2).mean()

        return loss, self.reg_weight * reg,


class InBatchSampledSoftmax(nn.Module):
    MIN_TEMPERATURE = 0.01
    MAX_TEMPERATURE = 100.

    def __init__(self):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(0.))

    @property
    def temperature(self):
        return torch.clip(torch.exp(self.tau), min=self.MIN_TEMPERATURE, max=self.MAX_TEMPERATURE)

    def forward(self, queries, candidates):
        logits = self.temperature * queries @ candidates.T
        return -torch.log_softmax(logits, dim=1).diag().mean()
    

def make_groups(lengths: torch.Tensor) -> torch.Tensor:
    if lengths.dim() != 1 or lengths.is_floating_point():
        raise ValueError("Invalid lengths tensor")
    if lengths.size(0) == 0:
        return lengths.clone()
    offset = lengths.cumsum(dim=0)
    addition = torch.zeros(int(offset[-1]) + 1, dtype=offset.dtype, device=offset.device)
    addition = addition.scatter_add_(dim=0, index=offset, src=torch.ones_like(offset))
    return addition[:-1].cumsum(dim=0)


def make_pairs(lengths: torch.Tensor) -> torch.Tensor:
    repeat = lengths[make_groups(lengths)]
    first = make_groups(repeat)
    addition = torch.ones(first.size(0) + 1, dtype=first.dtype, device=first.device)
    addition.scatter_add_(dim=0, index=lengths.square().cumsum(dim=0), src=lengths)
    addition.scatter_add_(dim=0, index=repeat.cumsum(dim=0), src=-repeat)
    addition[0].add_(-1)
    second = addition[:-1].cumsum(dim=0)
    return torch.stack([first, second])


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
                torch.stack([
                    torch.nn.functional.logsigmoid(context0),
                    torch.nn.functional.logsigmoid(context1)
                ]),
                dim=0
            )[0]
            return ((loss * weight).sum() / weight_sum).to(logits.dtype)
        else:
            return torch.zeros((), dtype=logits.dtype, device=logits.device)
