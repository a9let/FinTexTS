"""Independent group MLPs and learnable price/text fusion."""

import torch
from torch import nn


TEXT_GROUPS = ("macro", "sector", "targetCompany", "relatedCompany", "filing")
EMBEDDING_SIZE = 384


def mlp_structure(hidden_size, pred_len, num_layers):
    """Serializable layer description, shared by every independent group MLP."""
    if num_layers < 1 or hidden_size < 1 or pred_len < 1:
        raise ValueError("num_layers, hidden_size and pred_len must be positive")
    sizes = [EMBEDDING_SIZE] + [hidden_size] * (num_layers - 1) + [pred_len]
    layers = []
    for index, (in_features, out_features) in enumerate(zip(sizes, sizes[1:])):
        layers.append({"type": "Linear", "in_features": in_features, "out_features": out_features})
        if index < num_layers - 1:
            layers.append({"type": "ReLU"})
    return layers


class TextModel(nn.Module):
    def __init__(self, hidden_size, pred_len, num_layers, used_col_prefixes=TEXT_GROUPS,
                 initial_text_weight=0.1, fusion_mode="blend"):
        super().__init__()
        if not 0 < initial_text_weight < 1:
            raise ValueError("initial_text_weight must satisfy 0 < text_weight < 1")
        if fusion_mode not in ("blend", "residual"):
            raise ValueError("fusion_mode must be blend or residual")
        self.used_col_prefixes = tuple(used_col_prefixes)
        if not self.used_col_prefixes or len(set(self.used_col_prefixes)) != len(self.used_col_prefixes):
            raise ValueError("Text groups must be nonempty and unique")
        if not set(self.used_col_prefixes).issubset(TEXT_GROUPS):
            raise ValueError(f"Unknown text group; expected groups from {TEXT_GROUPS}")
        self.fusion_mode = fusion_mode
        self.pred_len = pred_len
        structure = mlp_structure(hidden_size, pred_len, num_layers)
        self.group_mlps = nn.ModuleDict()
        for group in self.used_col_prefixes:
            layers = []
            for layer in structure:
                if layer["type"] == "Linear":
                    layers.append(nn.Linear(layer["in_features"], layer["out_features"]))
                else:
                    layers.append(nn.ReLU())
            self.group_mlps[group] = nn.Sequential(*layers)
        self.G = nn.Parameter(torch.zeros(len(self.used_col_prefixes)))
        self.alpha_logit = nn.Parameter(torch.logit(torch.tensor(initial_text_weight, dtype=torch.float32)))

    def get_alpha(self):
        return torch.sigmoid(self.alpha_logit)

    def get_group_weights(self):
        return torch.softmax(self.G, dim=0)

    def fusion_state(self):
        return {
            "text_alpha": self.get_alpha().detach().cpu().item(),
            "group_weights": dict(zip(self.used_col_prefixes, self.get_group_weights().detach().cpu().tolist())),
        }

    def forward(self, price_output, last_price, embeddings):
        """Fuse [B,H,C] prices with last-day embeddings [B,G,384]."""
        if embeddings.ndim != 3 or embeddings.shape[1:] != (len(self.used_col_prefixes), EMBEDDING_SIZE):
            raise ValueError("Expected last-day text embeddings [batch, groups, 384]")
        if (price_output.ndim != 3 or price_output.shape[:2] != (embeddings.shape[0], self.pred_len)
                or last_price.shape != (embeddings.shape[0], price_output.shape[-1])):
            raise ValueError("Expected price_output [batch, pred_len, channels] and last_price [batch, channels]")
        present = embeddings.abs().sum(dim=-1) >= 1e-6
        outputs = torch.stack([
            self.group_mlps[group](embeddings[:, index, :])
            for index, group in enumerate(self.used_col_prefixes)
        ], dim=1)
        # Mask AFTER each MLP: its biases must never contribute for absent text.
        masked = outputs * present.unsqueeze(-1).to(outputs.dtype)
        delta = (masked * self.get_group_weights().view(1, -1, 1)).sum(dim=1)
        delta = delta.unsqueeze(-1).expand(-1, -1, price_output.shape[-1])
        alpha = self.get_alpha()
        if self.fusion_mode == "residual":
            return price_output + alpha * delta
        text_prediction = last_price.unsqueeze(1) + delta
        text_prediction = text_prediction * present.any(dim=1).view(-1, 1, 1).to(delta.dtype)
        return (1 - alpha) * price_output + alpha * text_prediction
