"""Original PyTorch interview implementations; optional dependency.

CPU-friendly teaching code, without fused kernels, GQA or cache storage.
Masks below always mean True = allowed. RoPE uses adjacent dimension pairs.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, dimension, heads, dropout=0.0):
        super().__init__()
        if heads <= 0 or dimension <= 0 or dimension % heads:
            raise ValueError("dimension must be a positive multiple of heads")
        self.heads, self.head_dim, self.dropout = heads, dimension // heads, dropout
        self.qkv = nn.Linear(dimension, 3 * dimension)
        self.out = nn.Linear(dimension, dimension)

    def forward(self, x, allowed=None, causal=False):
        batch, length, dimension = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = q.float() @ k.float().transpose(-1, -2) / math.sqrt(self.head_dim)
        if causal:
            causal_allowed = torch.arange(length, device=x.device)[:, None] >= torch.arange(length, device=x.device)[None, :]
            allowed = causal_allowed if allowed is None else allowed & causal_allowed
        if allowed is not None:
            if allowed.dtype != torch.bool:
                raise ValueError("allowed must be boolean")
            scores = scores.masked_fill(~allowed, float('-inf'))
            if torch.isneginf(scores).all(dim=-1).any():
                raise ValueError("fully masked query row")
        probabilities = F.dropout(scores.softmax(dim=-1).to(v.dtype), self.dropout, self.training)
        values = (probabilities @ v).transpose(1, 2).reshape(batch, length, dimension)
        return self.out(values)


def rotary_adjacent(x, positions, base=10000.0):
    # x [..., T, D]; positions [T]; this is one explicit pairing convention.
    if x.ndim < 2 or x.shape[-1] == 0 or x.shape[-1] % 2 or base <= 0:
        raise ValueError("[...,T,D] with nonempty even D and positive base required")
    if positions.ndim != 1 or positions.numel() != x.shape[-2]:
        raise ValueError("positions must have exactly one value for each sequence position")
    frequency = base ** (-torch.arange(0, x.shape[-1], 2, device=x.device, dtype=torch.float32) / x.shape[-1])
    angle = positions.to(device=x.device, dtype=torch.float32)[:, None] * frequency[None, :]
    c, s = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even*c-odd*s, even*s+odd*c), dim=-1).flatten(-2)


class LoRALinear(nn.Module):
    def __init__(self, input_features, output_features, rank=4, alpha=8.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = nn.Linear(input_features, output_features)
        self.base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, input_features))
        self.b = nn.Parameter(torch.zeros(output_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(x, self.a), self.b)

    def merged_weight(self):
        return self.base.weight + self.scale * (self.b @ self.a)


def info_nce(queries, keys, temperature=0.1, symmetric=False):
    if temperature <= 0 or queries.shape != keys.shape or queries.ndim != 2 or queries.shape[0] == 0 or queries.shape[1] == 0:
        raise ValueError("paired [N,D] matrices and positive temperature required")
    queries, keys = F.normalize(queries.float(), dim=-1), F.normalize(keys.float(), dim=-1)
    scores = queries @ keys.T / temperature
    labels = torch.arange(len(queries), device=queries.device)
    loss = F.cross_entropy(scores, labels)
    return (loss + F.cross_entropy(scores.T, labels)) / 2 if symmetric else loss


def response_logprobs(logits, token_ids, response_mask):
    # logits at t predict token at t+1; response_mask is aligned to token_ids.
    if logits.shape[:2] != token_ids.shape or token_ids.shape != response_mask.shape:
        raise ValueError("sequence shapes differ")
    logprobs = logits[:, :-1].float().log_softmax(dim=-1)
    targets = token_ids[:, 1:]
    mask = response_mask[:, 1:].bool()
    safe_targets = targets.masked_fill(~mask, 0)
    values = logprobs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    return values.masked_fill(~mask, 0).sum(dim=-1)


def dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta=0.1):
    if beta <= 0:
        raise ValueError("beta must be positive")
    margin = ((policy_chosen - policy_rejected)
              - (reference_chosen.detach() - reference_rejected.detach()))
    return -F.logsigmoid(beta * margin).mean()
