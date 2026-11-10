from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
from .models import ConvFormerBlock

@dataclass(frozen=True)
class ContinuousSequenceOutput:
    mean: torch.Tensor
    fixed_log_std: torch.Tensor

class ContinuousSequencePolicy(nn.Module):
    """Unconditioned causal Decision ConvFormer with a bounded Gaussian mean."""

    def __init__(self, *, mode: str, observation_dim: int=91, action_dim: int=27, max_length: int=30, hidden_dim: int=64, layers: int=2, heads: int=4, dropout: float=0.0, conv_window: int=4, return_scale: float=100.0, fixed_std: float=0.1) -> None:
        super().__init__()
        if mode != 'prdt':
            raise ValueError('only unconditioned PrDT is included')
        if hidden_dim % heads or max_length <= 0:
            raise ValueError('invalid hidden/head/length geometry')
        if fixed_std <= 0 or not math.isfinite(fixed_std):
            raise ValueError('fixed_std must be finite and positive')
        if return_scale <= 0 or not math.isfinite(return_scale):
            raise ValueError('return_scale must be finite and positive')
        self.mode = mode
        self.max_length = max_length
        self.action_dim = action_dim
        self.return_scale = float(return_scale)
        self.state_projection = nn.Linear(observation_dim, hidden_dim)
        self.action_projection = nn.Linear(action_dim, hidden_dim)
        self.time_embedding = nn.Embedding(max_length, hidden_dim)
        self.return_projection = None
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList([ConvFormerBlock(hidden_dim, conv_window, dropout) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.register_buffer('_fixed_log_std', torch.full((action_dim,), math.log(fixed_std)))

    def tokenize(self, observations: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor, returns_to_go: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        if observations.ndim != 3 or actions.ndim != 3:
            raise ValueError('observations/actions must have [batch,time,feature] shape')
        batch, length, _ = observations.shape
        if actions.shape[:2] != (batch, length) or mask.shape != (batch, length):
            raise ValueError('sequence shapes are inconsistent')
        if mask.dtype != torch.bool or not mask.any(dim=1).all():
            raise ValueError('every sequence needs a boolean valid mask')
        if length > self.max_length:
            raise ValueError('sequence exceeds configured maximum')
        positions = torch.arange(length, device=observations.device).unsqueeze(0).expand(batch, -1)
        time = self.time_embedding(positions)
        state_tokens = self.state_projection(observations) + time
        action_tokens = self.action_projection(actions) + time
        if returns_to_go is not None:
            raise ValueError('PrDT is unconditioned and rejects desired returns')
        tokens = torch.stack((state_tokens, action_tokens), dim=2).flatten(1, 2)
        token_mask = torch.stack((mask, mask), dim=2).flatten(1, 2)
        return (self.input_norm(tokens) * token_mask.unsqueeze(-1), token_mask)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor, returns_to_go: torch.Tensor | None=None) -> ContinuousSequenceOutput:
        hidden, token_mask = self.tokenize(observations, actions, mask, returns_to_go)
        for block in self.blocks:
            hidden = block(hidden, token_mask)
        hidden = self.final_norm(hidden) * token_mask.unsqueeze(-1)
        state_hidden = hidden[:, 0::2]
        mean = torch.tanh(self.action_head(state_hidden))
        return ContinuousSequenceOutput(mean, self._fixed_log_std.expand_as(mean))

def per_trajectory_continuous_nll(output: ContinuousSequenceOutput, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Fixed-variance Gaussian NLL, mean over valid timesteps per trajectory.

    Action dimensions are summed because they form one 27D action event. The
    timestep reduction is a mean, preventing trajectory length from silently
    changing a trajectory's imitation weight.
    """
    if output.mean.shape != targets.shape or mask.shape != targets.shape[:2]:
        raise ValueError('continuous target/mask shapes are inconsistent')
    variance = torch.exp(2 * output.fixed_log_std)
    per_dimension = 0.5 * ((targets - output.mean).square() / variance + 2 * output.fixed_log_std + math.log(2 * math.pi))
    per_step = per_dimension.sum(dim=-1) * mask.to(per_dimension.dtype)
    return per_step.sum(dim=1) / mask.sum(dim=1).clamp_min(1).to(per_step.dtype)

class ContinuousTrajectoryScorer(nn.Module):
    """Completed-trajectory scorer for later Bradley--Terry preference fitting."""

    def __init__(self, observation_dim: int=91, action_dim: int=27, max_length: int=30, hidden_dim: int=64, layers: int=2, heads: int=4, dropout: float=0.0) -> None:
        super().__init__()
        self.max_length = max_length
        self.state_projection = nn.Linear(observation_dim, hidden_dim)
        self.action_projection = nn.Linear(action_dim, hidden_dim)
        self.reward_projection = nn.Linear(1, hidden_dim)
        self.time_embedding = nn.Embedding(max_length, hidden_dim)
        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=heads, dim_feedforward=4 * hidden_dim, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, length = mask.shape
        if length > self.max_length or observations.shape[:2] != (batch, length) or actions.shape[:2] != (batch, length):
            raise ValueError('scorer sequence shapes are inconsistent')
        positions = torch.arange(length, device=mask.device).unsqueeze(0).expand(batch, -1)
        hidden = (self.state_projection(observations) + self.action_projection(actions) + self.reward_projection(rewards.unsqueeze(-1)) + self.time_embedding(positions)) * mask.unsqueeze(-1)
        hidden = self.encoder(hidden, src_key_padding_mask=~mask)
        hidden = self.norm(hidden) * mask.unsqueeze(-1)
        pooled = hidden.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        return self.head(pooled).squeeze(-1)
