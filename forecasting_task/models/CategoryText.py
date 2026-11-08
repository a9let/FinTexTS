"""Same-date category scoring; five original group heads and price-only gate."""

import torch
from torch import nn

from forecasting_task.models.GroupText import ContextNoveltyTextModel, TEXT_GROUPS


def scorer_config(mode):
    return {"mode": mode, "embedding_size": 384, "price_context_size": 8, "hidden_size": 16,
            "shared_groups": list(TEXT_GROUPS[:4]), "value_projection": False, "category_position_feature": False,
            "v_initialization": "zeros", "b_initialization": "zeros",
            "parameters": 6304 if mode == "attention" else 0}


class CategoryScorer(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.W_e = nn.Linear(384, 16, bias=False)
            self.W_c = nn.Linear(8, 16, bias=False)
        self.b = nn.Parameter(torch.zeros(16))
        self.v = nn.Parameter(torch.zeros(16))

    def forward(self, values, context):
        hidden = torch.tanh(self.W_e(values)+self.W_c(context)[:, None, None, :]+self.b)
        return (hidden*self.v).sum(-1)


class CategoryTextModel(ContextNoveltyTextModel):
    def __init__(self, category_pool, category_scorer_config=None, **kwargs):
        if category_pool not in ("uniform", "attention") or kwargs.get("group_gate") != "context_price_only":
            raise ValueError("Category pooling requires uniform/attention and context_price_only")
        if category_scorer_config is not None and category_scorer_config != scorer_config(category_pool):
            raise ValueError("Conflicting saved category scorer architecture")
        super().__init__(**kwargs)  # Preserve original FiLM/head/gate initialization order.
        self.category_pool = category_pool
        self.category_scorer = CategoryScorer() if category_pool == "attention" else None

    def fusion_state(self):
        return {**super().fusion_state(), "category_pool": self.category_pool}

    def forward(self, price_output, last_price, embeddings, *, price_window, category_embeddings,
                category_present, filing_present, return_details=False):
        values, present = category_embeddings, category_present
        if values.ndim != 4 or values.shape[1:] != (4, 5, 384) or present.shape != values.shape[:-1]:
            raise ValueError("Expected category values [batch,4,5,384] and explicit masks")
        if not torch.isfinite(values).all() or not torch.all((present == 0) | (present == 1)):
            raise ValueError("Invalid category values/masks")
        present = present.bool()
        if (values.masked_select(~present[..., None].expand_as(values)) != 0).any():
            raise ValueError("Absent category vectors must be zero")
        if filing_present.shape != (values.shape[0],) or not torch.all((filing_present == 0) | (filing_present == 1)):
            raise ValueError("Invalid filing mask")
        context = torch.cat((price_window[:, -1]-price_window[:, -2], price_window[:, -1]-price_window[:, -6]), -1)
        scores = self.category_scorer(values, context) if self.category_scorer is not None else values.new_zeros(values.shape[:-1])
        scores = scores.masked_fill(~present, float("-inf"))
        scores = torch.where(present.any(-1, keepdim=True), scores, torch.zeros_like(scores))
        beta = scores.softmax(-1)*present.to(scores.dtype)
        pooled = (beta[..., None]*values).sum(-2)
        grouped = torch.cat((pooled, embeddings[:, 4:5]), 1)
        group_present = torch.cat((present.any(-1), filing_present.bool()[:, None]), -1)
        zeros = values.new_zeros((values.shape[0], 5))
        output, detail = super().forward(price_output, last_price, grouped, price_window=price_window,
                                         novelty=zeros, pair_available=zeros, group_present=group_present, return_details=True)
        if return_details:
            heads = torch.stack([self.group_mlps[g](values[:, i]) for i, g in enumerate(TEXT_GROUPS[:4])], 1)
            contributions = self.fixed_alpha*detail["weights"][:, :4, None, None]*beta[..., None]*heads
            return output, {**detail, "category_beta": beta, "category_present": present,
                            "category_heads": heads, "category_contributions": contributions,
                            "category_pooled": pooled, "category_context": context}
        return output
