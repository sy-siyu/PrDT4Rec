import torch
from torch import nn
from .kuairec_models import RecommendationPolicy, TrajectoryScorer
N_ITEMS=3329
ITEM_FEATURE_WIDTH=41
HORIZON=40

def projected_item_lookup(table: torch.Tensor, hidden_dim: int) -> nn.Sequential:
    if table.shape != (N_ITEMS, ITEM_FEATURE_WIDTH):
        raise ValueError("item feature table has incompatible shape")
    return nn.Sequential(
        nn.Embedding.from_pretrained(table, freeze=True, padding_idx=0),
        nn.Linear(ITEM_FEATURE_WIDTH, hidden_dim, bias=False),
    )


class ZeroUserEmbedding(nn.Module):
    """Return an identity-free state channel for any batch of user indices."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)

    def forward(self, users: torch.Tensor) -> torch.Tensor:
        return torch.zeros((*users.shape, self.hidden_dim), device=users.device, dtype=torch.float32)


class KuairecTrajectoryScorer(TrajectoryScorer):
    """Completed-trajectory Transformer scorer with frozen released 41D items."""

    def __init__(self, item_features: torch.Tensor, **kwargs) -> None:
        super().__init__(1, N_ITEMS, HORIZON, **kwargs)
        hidden = self.position_embedding.embedding_dim
        self.item_embedding = projected_item_lookup(item_features, hidden)


class KuairecPrDTPolicy(RecommendationPolicy):
    """RTG-free categorical Decision ConvFormer with no user-ID shortcut."""

    def __init__(self, item_features: torch.Tensor, **kwargs) -> None:
        super().__init__(1, N_ITEMS, HORIZON, backbone="convformer", **kwargs)
        hidden = self.action_head.in_features
        self.user_embedding = ZeroUserEmbedding(hidden)
        self.item_embedding = projected_item_lookup(item_features, hidden)

