from __future__ import annotations
import torch
from torch.nn import functional as F

def global_softmax_weights(scores: torch.Tensor, beta: float) -> torch.Tensor:
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("scores must be a non-empty vector")
    if beta < 0:
        raise ValueError("beta must be non-negative")
    if beta == 0:
        return torch.full_like(scores, 1.0 / scores.numel())
    scaled = beta * (scores - scores.max())
    log_normalizer = torch.logsumexp(scaled, dim=0)
    return torch.exp(scaled - log_normalizer)


def per_trajectory_categorical_nll(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    token_multipliers: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits.shape[:2] != targets.shape or targets.shape != mask.shape:
        raise ValueError("logits, target and mask shapes are inconsistent")
    # Mask invalid target IDs before CE: inf * 0 would otherwise produce NaN.
    valid = mask & (targets >= 2)
    safe_targets = targets.masked_fill(~valid, -100)
    token_nll = F.cross_entropy(logits.transpose(1, 2), safe_targets, reduction="none", ignore_index=-100)
    if token_multipliers is not None:
        if token_multipliers.shape != targets.shape or token_multipliers.requires_grad:
            raise ValueError("action multipliers must be detached and aligned to targets")
        token_nll = token_nll * token_multipliers.masked_fill(~valid, 0.)
    return token_nll.sum(dim=1)


def bradley_terry_loss(preferred_scores: torch.Tensor, rejected_scores: torch.Tensor) -> torch.Tensor:
    return F.softplus(-(preferred_scores - rejected_scores)).mean()


def bradley_terry_target_loss(
    first_scores: torch.Tensor,
    second_scores: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Bradley--Terry cross-entropy with explicit fractional pair targets.

    A target of one prefers the first endpoint, zero prefers the second, and
    one half represents an exact tie. Integer targets are rejected so callers
    cannot accidentally erase the tie semantics through an implicit cast.
    """

    if first_scores.shape != second_scores.shape or first_scores.shape != targets.shape:
        raise ValueError("scores and Bradley--Terry targets must have the same shape")
    if not torch.is_floating_point(targets):
        raise TypeError("Bradley--Terry targets must be floating point")
    if not torch.isfinite(targets).all() or bool(((targets < 0) | (targets > 1)).any()):
        raise ValueError("Bradley--Terry targets must be finite and lie in [0, 1]")
    typed_targets = targets.to(device=first_scores.device, dtype=first_scores.dtype)
    return F.binary_cross_entropy_with_logits(first_scores - second_scores, typed_targets)

