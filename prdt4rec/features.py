import torch
from torch import nn
from .models import RecommendationPolicy, TrajectoryScorer

def _lookup_arrays(corpus, representation):
    users = torch.as_tensor(representation["user_vectors"], dtype=torch.float32).clone()
    vectors = torch.as_tensor(representation["item_vectors"], dtype=torch.float32).clone()
    if (users.ndim != 2 or vectors.ndim != 2 or users.shape[1] != 40
            or vectors.shape[1] != 40 or users.shape[0] < corpus.n_users
            or vectors.shape[0] != corpus.n_items - 2
            or not torch.isfinite(users).all() or not torch.isfinite(vectors).all()
            or torch.count_nonzero(users[0])):
        raise ValueError("shared lookups must be finite aligned 40D TRAIN user/item vectors with zero PAD user")
    items = torch.zeros((corpus.n_items, vectors.shape[1]), dtype=torch.float32)
    items[2:] = vectors
    return users, items


def _projected_lookup(values, hidden):
    # Identical indexing, freezing and bias-free projection to SharedLookupDT.
    return nn.Sequential(nn.Embedding.from_pretrained(values, freeze=True, padding_idx=0),
                         nn.Linear(values.shape[1], hidden, bias=False))


class SharedLookupDC(RecommendationPolicy):
    """Original causal DC forward with the exact fixed DT lookup features."""

    def __init__(self, corpus, representation, **kwargs):
        super().__init__(corpus.n_users, corpus.n_items, corpus.max_length,
                         backbone="convformer", **kwargs)
        users, items = _lookup_arrays(corpus, representation)
        hidden = self.action_head.in_features
        self.user_embedding = _projected_lookup(users, hidden)
        self.item_embedding = _projected_lookup(items, hidden)


class SharedLookupScorer(TrajectoryScorer):
    """Completed-trajectory attention scorer; its user input stays unused."""

    def __init__(self, corpus, representation, **kwargs):
        super().__init__(corpus.n_users, corpus.n_items, corpus.max_length, **kwargs)
        _, items = _lookup_arrays(corpus, representation)
        self.item_embedding = _projected_lookup(items, self.position_embedding.embedding_dim)


def make_dc(corpus, representation, config, device):
    p = config.get("policy", {})
    return SharedLookupDC(corpus, representation,
        hidden_dim=int(p.get("hidden_dim", 256)), layers=int(p.get("layers", 4)),
        heads=int(p.get("heads", 4)), dropout=float(p.get("dropout", .1)),
        conv_window=int(p.get("conv_window", 16))).to(device)


def _scorer_config(config):
    s = config.get("scorer", {})
    return {"hidden_dim": int(s.get("hidden_dim", 128)), "layers": int(s.get("layers", 3)),
            "heads": int(s.get("heads", 4)), "dropout": float(s.get("dropout", .1)),
            "epochs": int(s.get("epochs", 10)), "batch_size": int(s.get("batch_size", 128)),
            "learning_rate": float(s.get("learning_rate", s.get("lr", 3e-4))),
            "lambda_score": float(s.get("lambda_score", .001))}


def make_scorer(corpus, representation, config, device):
    s = _scorer_config(config)
    return SharedLookupScorer(corpus, representation,
        **{key: s[key] for key in ("hidden_dim", "layers", "heads", "dropout")}).to(device)

