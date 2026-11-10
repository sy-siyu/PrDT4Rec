import torch
from .objective import per_trajectory_categorical_nll

def per_trajectory_mean_action_nll(logits, actions, mask):
    """Divide each trajectory's summed NLL by its own valid executed count.

    The numerator and denominator use exactly the same mask and real-action
    predicate. Accepted complete episodes always have a positive count; an
    invalid empty trajectory produces nonfinite loss and fails the unchanged
    gradient/epoch finite checks, without a silent denominator clamp.
    """
    if mask.dtype != torch.bool:
        raise ValueError("mean-action NLL requires the boolean valid-action mask")
    summed = per_trajectory_categorical_nll(logits, actions, mask)
    counts = (mask & (actions >= 2)).sum(dim=1).to(dtype=summed.dtype)
    return summed / counts

