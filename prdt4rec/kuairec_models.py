from __future__ import annotations
import math
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

class TrajectoryScorer(nn.Module):
    """Full-trajectory attention scorer used only before policy fitting.

    The scorer may inspect current feedback because it scores completed logged
    trajectories. The policy below receives only previous feedback.
    """

    def __init__(self, n_users: int, n_items: int, max_length: int, hidden_dim: int=512, layers: int=3, heads: int=4, dropout: float=0.1) -> None:
        super().__init__()
        if max_length <= 0:
            raise ValueError('max_length must be positive')
        self.max_length = int(max_length)
        self.item_embedding = nn.Embedding(n_items, hidden_dim, padding_idx=0)
        self.feedback_projection = nn.Linear(1, hidden_dim)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)
        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=heads, dim_feedforward=4 * hidden_dim, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.score_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, users: torch.Tensor, actions: torch.Tensor, feedback: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, length = actions.shape
        if length > self.max_length:
            raise ValueError(f'sequence length {length} exceeds max_length {self.max_length}')
        positions = torch.arange(length, device=actions.device).unsqueeze(0).expand(batch_size, -1)
        hidden = self.item_embedding(actions) + self.feedback_projection(feedback.unsqueeze(-1)) + self.position_embedding(positions)
        hidden = hidden * mask.unsqueeze(-1)
        hidden = self.encoder(hidden, src_key_padding_mask=~mask)
        hidden = self.final_norm(hidden) * mask.unsqueeze(-1)
        pooled = hidden.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        return self.score_head(pooled).squeeze(-1)

class CausalModalityConv(nn.Module):
    """Official DC-style causal depthwise filtering for two interleaved modalities.

    Official DC interleaves (RTG, observation, action) and takes convolution
    outputs at 0::3, 1::3 and 2::3 using three distinct filters. PrDT4Rec has
    no RTG token, so this layer interleaves (observation, action) and takes
    outputs at 0::2 and 1::2. The filters still run over the flattened causal
    token stream, matching the official implementation's semantics.
    """

    def __init__(self, hidden_dim: int, window_size: int) -> None:
        super().__init__()
        self.window_size = int(window_size)
        self.observation_filter = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=window_size, groups=hidden_dim)
        self.action_filter = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=window_size, groups=hidden_dim)

    def forward(self, tokens: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        masked = tokens * token_mask.unsqueeze(-1)
        if tokens.device.type == 'cpu':
            windows = F.pad(masked, (0, 0, self.window_size - 1, 0)).unfold(1, self.window_size, 1)
            observation = (windows[:, 0::2] * self.observation_filter.weight[:, 0]).sum(-1)
            action = (windows[:, 1::2] * self.action_filter.weight[:, 0]).sum(-1)
            observation = observation + self.observation_filter.bias
            action = action + self.action_filter.bias
            return torch.stack((observation, action), dim=2).flatten(1, 2) * token_mask.unsqueeze(-1)
        sequence = F.pad(masked, (0, 0, self.window_size - 1, 0)).transpose(1, 2)
        observation = self.observation_filter(sequence)[:, :, 0::2]
        action = self.action_filter(sequence)[:, :, 1::2]
        output = torch.stack((observation, action), dim=3).flatten(2, 3).transpose(1, 2)
        return output * token_mask.unsqueeze(-1)

class ConvFormerBlock(nn.Module):

    def __init__(self, hidden_dim: int, window_size: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.filter = CausalModalityConv(hidden_dim, window_size)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, 4 * hidden_dim), nn.GELU(), nn.Linear(4 * hidden_dim, hidden_dim), nn.Dropout(dropout))

    def forward(self, hidden: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.filter(self.norm1(hidden), token_mask)
        hidden = (hidden + self.mlp(self.norm2(hidden))) * token_mask.unsqueeze(-1)
        return hidden

def add_initial_context_to_first_state(state_embeddings: torch.Tensor, item_embedding: nn.Module, initial_context_actions: torch.Tensor | None) -> torch.Tensor:
    """Add an unrewarded reset item to state0 without creating a token.

    The storage/adapter boundary guarantees values are either zero (no reset
    context) or real item IDs in ``[2, n_items)``. ``torch.where`` preserves
    state rows exactly when the optional context is zero.
    """
    if initial_context_actions is None:
        return state_embeddings
    batch_size = state_embeddings.shape[0]
    if initial_context_actions.shape != (batch_size,):
        raise ValueError('initial_context_actions must have shape [batch]')
    if initial_context_actions.dtype not in (torch.int32, torch.int64):
        raise TypeError('initial_context_actions must contain integer item IDs')
    if initial_context_actions.device != state_embeddings.device:
        raise ValueError('initial_context_actions must be on the model input device')
    context = item_embedding(initial_context_actions).unsqueeze(1)
    positions = torch.arange(state_embeddings.shape[1], device=state_embeddings.device)
    use_context = (initial_context_actions != 0).view(batch_size, 1, 1) & (positions == 0).view(1, -1, 1)
    return torch.where(use_context, state_embeddings + context, state_embeddings)

class RecommendationPolicy(nn.Module):
    """RTG-free categorical policy with causal ConvFormer blocks.

    An optional unrewarded reset item enters only observation state0 through
    the same item embedding used by chosen-action history.
    """

    def __init__(self, n_users: int, n_items: int, max_length: int, hidden_dim: int, layers: int, heads: int, dropout: float, conv_window: int, backbone: str) -> None:
        super().__init__()
        if backbone != 'convformer':
            raise ValueError(f'unknown backbone: {backbone}')
        if max_length <= 0:
            raise ValueError('max_length must be positive')
        self.n_items = n_items
        self.max_length = int(max_length)
        self.backbone_name = backbone
        self.user_embedding = nn.Embedding(n_users, hidden_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(n_items, hidden_dim, padding_idx=0)
        self.previous_feedback_projection = nn.Linear(1, hidden_dim, bias=False)
        self.time_embedding = nn.Embedding(max_length, hidden_dim)
        self.modality_embedding = nn.Embedding(2, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList([ConvFormerBlock(hidden_dim, conv_window, dropout) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.action_head = nn.Linear(hidden_dim, n_items)

    def tokenize(self, users: torch.Tensor, actions: torch.Tensor, feedback: torch.Tensor, mask: torch.Tensor, initial_context_actions: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, length = actions.shape
        if length > self.max_length:
            raise ValueError(f'sequence length {length} exceeds max_length {self.max_length}')
        positions = torch.arange(length, device=actions.device).unsqueeze(0).expand(batch_size, -1)
        previous_feedback = torch.zeros_like(feedback)
        previous_feedback[:, 1:] = feedback[:, :-1]
        time = self.time_embedding(positions)
        observation = self.user_embedding(users).unsqueeze(1) + self.previous_feedback_projection(previous_feedback.unsqueeze(-1)) + time + self.modality_embedding.weight[0]
        observation = add_initial_context_to_first_state(observation, self.item_embedding, initial_context_actions)
        action = self.item_embedding(actions) + time + self.modality_embedding.weight[1]
        tokens = torch.stack((observation, action), dim=2).flatten(1, 2)
        token_mask = torch.stack((mask, mask), dim=2).flatten(1, 2)
        return (self.input_norm(tokens) * token_mask.unsqueeze(-1), token_mask)

    def forward(self, users: torch.Tensor, actions: torch.Tensor, feedback: torch.Tensor, mask: torch.Tensor, initial_context_actions: torch.Tensor | None=None) -> torch.Tensor:
        hidden, token_mask = self.tokenize(users, actions, feedback, mask, initial_context_actions)
        for block in self.blocks:
            hidden = block(hidden, token_mask)
        hidden = self.final_norm(hidden) * token_mask.unsqueeze(-1)
        observation_states = hidden[:, 0::2]
        logits = self.action_head(observation_states)
        unavailable = torch.arange(self.n_items, device=logits.device) < 2
        return logits.masked_fill(unavailable, -torch.inf)
