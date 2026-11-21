"""Independent group encoders for the memory ablation."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from forecasting_task.models.GroupText import TEXT_GROUPS


FAMILIES = ("memto_group", "titans_linear_group", "xtime_group")


def masked_softmax(values, mask, dim=-1):
    weights = values.masked_fill(~mask, torch.finfo(values.dtype).min).softmax(dim) * mask
    return weights / weights.sum(dim, keepdim=True).clamp_min(torch.finfo(values.dtype).tiny)


def last_present(values, present):
    index = (present * torch.arange(1, present.shape[1]+1, device=present.device)).amax(1)
    return values[torch.arange(len(values), device=values.device), (index-1).clamp_min(0)]


class PrototypeMemory(nn.Module):
    def __init__(self, count):
        super().__init__()
        self.projection = nn.Linear(384, 32)
        self.qkv = nn.Linear(32, 96)
        self.output = nn.Linear(32, 32)
        self.norm = nn.LayerNorm(32)
        self.decoder = nn.Linear(64, 384)
        self.old_gate = nn.Linear(32, 32, bias=False)
        self.new_gate = nn.Linear(32, 32, bias=False)
        self.register_buffer("bank", torch.randn(count, 32) / math.sqrt(32))
        self.register_buffer("center", torch.zeros(4))
        self.register_buffer("scale", torch.ones(4))

    def encode(self, x, present):
        z = self.projection(x)
        q, k, v = self.qkv(z).chunk(3, -1)
        causal = torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device).tril()
        weights = masked_softmax(q @ k.transpose(-1, -2) / math.sqrt(32), causal & present[:, None, :])
        return self.norm(z + self.output(weights @ v)) * present[..., None]

    def write(self, queries, present):
        weights = masked_softmax(self.bank @ queries.transpose(-1, -2), present[:, None, :])
        update = weights @ queries
        gate = torch.sigmoid(self.old_gate(self.bank) + self.new_gate(update))
        candidate = (1-gate)*self.bank + gate*update
        return torch.where(present.any(1)[:, None, None], candidate, self.bank), weights

    def read(self, queries, bank):
        weights = (queries @ bank.transpose(-1, -2)).softmax(-1)
        return weights @ bank, weights

    def reconstruction(self, x, present):
        queries = self.encode(x, present)
        candidate, _ = self.write(queries, present)
        read, weights = self.read(queries, candidate)
        error = (self.decoder(torch.cat((queries, read), -1)) - x).square().mean(-1)
        entropy = -(weights * weights.clamp_min(1e-12).log()).sum(-1)
        return (error + .01*entropy)[present].mean(), candidate

    @torch.no_grad()
    def commit(self, candidates, present):
        # All windows read the same old snapshot. Publish the mean of nonempty
        # candidate banks only after the backward pass for the whole batch.
        valid = present.any(1)
        if valid.any():
            self.bank.copy_(candidates[valid].mean(0))

    def features(self, x, present):
        queries = self.encode(x, present)
        read, _ = self.read(queries, self.bank)
        lsd = (queries[..., None, :] - self.bank).square().sum(-1).amin(-1)
        isd = (self.decoder(torch.cat((queries, read), -1)) - x).square().sum(-1)
        anomaly = masked_softmax(lsd, present) * isd
        scalars = torch.stack((last_present(lsd, present), last_present(isd, present),
                               last_present(anomaly, present), anomaly.amax(1)), -1)
        vectors = last_present(torch.cat((queries, read), -1), present)
        return vectors, scalars, torch.stack((lsd*present, isd*present, anomaly), -1)


class LinearMemory(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.keys = nn.Linear(384, hidden)
        self.values = nn.Linear(384, hidden)
        self.queries = nn.Linear(384, hidden)
        self.coefficients = nn.Linear(384, 3)
        nn.init.zeros_(self.coefficients.weight)
        with torch.no_grad():
            self.coefficients.bias.copy_(torch.tensor([.05, .5, .1]).logit())
        self.initial = nn.Parameter(torch.zeros(hidden, hidden))

    @staticmethod
    def gradient(memory, key, value):
        error = (memory @ key[..., None]).squeeze(-1) - value
        return error[..., :, None] * key[..., None, :]

    def forward(self, x, present):
        keys = F.normalize(self.keys(x), dim=-1, eps=1e-8)
        values = self.values(x)
        queries = F.normalize(self.queries(x), dim=-1, eps=1e-8)
        coeff = self.coefficients(x).sigmoid()
        memory = self.initial.expand(len(x), -1, -1)
        surprise = torch.zeros_like(memory)
        norms = []
        for i in range(x.shape[1]):
            alpha, eta, theta = (v[:, None, None] for v in coeff[:, i].unbind(-1))
            gradient = self.gradient(memory, keys[:, i], values[:, i])
            active = present[:, i, None, None]
            surprise = torch.where(active, eta*surprise-theta*gradient, surprise)
            memory = torch.where(active, (1-alpha)*memory+surprise, memory)
            norms.append(gradient.flatten(1).norm(dim=-1))
        final = (memory @ last_present(queries, present)[..., None]).squeeze(-1)
        return final, torch.cat((torch.stack(norms, 1)[..., None], coeff), -1)*present[..., None]


class ExpertRouter(nn.Module):
    def __init__(self, pred_len, normal_heads):
        super().__init__()
        normal = nn.ModuleDict({group: normal_heads[group][0] for group in TEXT_GROUPS})
        self.experts = nn.ModuleList([normal] + [
            nn.ModuleDict({g: nn.Linear(384, pred_len) for g in TEXT_GROUPS}) for _ in range(2)])
        self.router = nn.Sequential(nn.Linear(3*pred_len*4, 16), nn.ReLU(), nn.Linear(16, 3))
        self.register_buffer("normal_only", torch.tensor(True))

    def outputs(self, text, present, only_normal=False):
        pooled = (text * present[..., None]).sum(1) / present.sum(1).clamp_min(1)[..., None]
        experts = self.experts[:1] if only_normal else self.experts
        return torch.stack([torch.stack([expert[g](pooled[:, i]) for i,g in enumerate(TEXT_GROUPS)],1)
                            for expert in experts],1)

    @staticmethod
    def top_two(logits):
        values, indices = logits.topk(2, dim=-1)
        return torch.zeros_like(logits).scatter(-1, indices, values.softmax(-1))
