"""Independent group MLPs and learnable price/text fusion."""

import math

import torch
from torch import nn


TEXT_GROUPS = ("macro", "sector", "targetCompany", "relatedCompany", "filing")
EMBEDDING_SIZE = 384


def validate_initial_group_weights(values, num_groups):
    """Validate ordered CLI/model weights without consuming any random numbers."""
    if values is None:
        return [1.0 / num_groups] * num_groups
    if len(values) != num_groups:
        raise ValueError(f"initial_group_weights requires exactly {num_groups} values")
    try:
        weights = [float(value) for value in values]
    except (TypeError, ValueError) as error:
        raise ValueError("initial_group_weights must be numeric") from error
    if any(not math.isfinite(value) or value <= 0 for value in weights):
        raise ValueError("initial_group_weights must be finite and strictly positive")
    # Bound individual values before summing, including very large finite inputs.
    if any(value > 1 + 1e-6 for value in weights) or not math.isclose(
        math.fsum(weights), 1.0, rel_tol=0.0, abs_tol=1e-6
    ):
        raise ValueError("initial_group_weights must sum to 1 (absolute tolerance 1e-6)")
    return weights


def validate_fixed_uniform_weights(values, num_groups):
    weights = validate_initial_group_weights(values, num_groups)
    if any(not math.isclose(value, 1 / num_groups, rel_tol=0, abs_tol=1e-8) for value in weights):
        raise ValueError("fixed_uniform rejects nonuniform initial_group_weights")
    return weights


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
                 initial_text_weight=0.1, fusion_mode="blend", initial_group_weights=None):
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
        initial_weights = validate_initial_group_weights(initial_group_weights, len(self.used_col_prefixes))
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
        # Heads retain their original RNG initialization order. Only G changes.
        with torch.no_grad():
            self.G.copy_(torch.tensor(initial_weights, dtype=torch.float64).log())

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


class FixedUniformTextModel(nn.Module):
    """Five independent linear heads with fixed averaging and learnable alpha."""

    def __init__(self, hidden_size, pred_len, num_layers, used_col_prefixes=TEXT_GROUPS,
                 initial_text_weight=0.1, fusion_mode="blend", initial_group_weights=None):
        super().__init__()
        if tuple(used_col_prefixes) != TEXT_GROUPS:
            raise ValueError(f"fixed_uniform requires all five groups in order: {TEXT_GROUPS}")
        if num_layers != 1:
            raise ValueError("fixed_uniform requires text_num_layers=1 (linear heads only)")
        if pred_len < 1:
            raise ValueError("pred_len must be positive")
        if not 0 < initial_text_weight < 1:
            raise ValueError("initial_text_weight must satisfy 0 < text_weight < 1")
        if fusion_mode not in ("blend", "residual"):
            raise ValueError("fusion_mode must be blend or residual")
        validate_fixed_uniform_weights(initial_group_weights, len(TEXT_GROUPS))
        self.used_col_prefixes = TEXT_GROUPS
        self.pred_len = pred_len
        self.fusion_mode = fusion_mode
        # Same head names and random initialization order as the one-layer TextModel.
        self.group_mlps = nn.ModuleDict({
            group: nn.Sequential(nn.Linear(EMBEDDING_SIZE, pred_len)) for group in TEXT_GROUPS
        })
        self.alpha_logit = nn.Parameter(torch.logit(torch.tensor(initial_text_weight, dtype=torch.float32)))

    def get_alpha(self):
        return torch.sigmoid(self.alpha_logit)

    def fixed_group_coefficients(self):
        return {group: 1 / len(self.used_col_prefixes) for group in self.used_col_prefixes}

    def fusion_state(self):
        return {
            "text_alpha": self.get_alpha().detach().cpu().item(),
            "group_weight_mode": "fixed_uniform",
            "fixed_group_coefficients": self.fixed_group_coefficients(),
        }

    def forward(self, price_output, last_price, embeddings):
        if embeddings.ndim != 3 or embeddings.shape[1:] != (len(TEXT_GROUPS), EMBEDDING_SIZE):
            raise ValueError("Expected last-day text embeddings [batch, 5, 384]")
        if (price_output.ndim != 3 or price_output.shape[:2] != (embeddings.shape[0], self.pred_len)
                or last_price.shape != (embeddings.shape[0], price_output.shape[-1])):
            raise ValueError("Expected price_output [batch, pred_len, channels] and last_price [batch, channels]")
        present = embeddings.abs().sum(dim=-1) >= 1e-6
        outputs = torch.stack([
            self.group_mlps[group](embeddings[:, index])
            for index, group in enumerate(self.used_col_prefixes)
        ], dim=1)
        delta = (outputs * present.unsqueeze(-1).to(outputs.dtype)).sum(dim=1) / len(TEXT_GROUPS)
        delta = delta.unsqueeze(-1).expand(-1, -1, price_output.shape[-1])
        alpha = self.get_alpha()
        if self.fusion_mode == "residual":
            return price_output + alpha * delta
        text_prediction = (last_price.unsqueeze(1) + delta) * present.any(dim=1).view(-1, 1, 1).to(delta.dtype)
        return (1 - alpha) * price_output + alpha * text_prediction


def build_text_model(config, group_weight_mode="learnable"):
    if group_weight_mode == "fixed_uniform":
        return FixedUniformTextModel(**config)
    if group_weight_mode == "learnable":
        return TextModel(**config)
    raise ValueError(f"Unknown group_weight_mode: {group_weight_mode}")
