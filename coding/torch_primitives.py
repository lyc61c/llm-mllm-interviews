"""Original PyTorch interview implementations; optional dependency.

CPU-friendly teaching code, without fused kernels or cache storage.
Masks below always mean True = allowed. RoPE uses adjacent dimension pairs.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, input_features, hidden_features, output_features):
        super().__init__()
        if min(input_features, hidden_features, output_features) <= 0:
            raise ValueError("feature dimensions must be positive")
        self.first = nn.Linear(input_features, hidden_features)
        self.second = nn.Linear(hidden_features, output_features)

    def forward(self, x):
        # Linear transforms the last dimension, preserving leading dimensions.
        return self.second(F.gelu(self.first(x)))


class ImageMLP(nn.Module):
    def __init__(self, image_shape, hidden_features, classes):
        super().__init__()
        self.image_shape = tuple(image_shape)
        if len(self.image_shape) != 3 or any(size <= 0 for size in self.image_shape):
            raise ValueError("image_shape must be positive (channels, height, width)")
        self.mlp = MLP(math.prod(self.image_shape), hidden_features, classes)

    def forward(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != self.image_shape:
            raise ValueError("images must be [batch, channels, height, width] with the configured shape")
        # Fixed-resolution image classifier; output logits, not probabilities.
        return self.mlp(images.flatten(start_dim=1))


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


def vae_reparameterize(mean, logvar, noise=None):
    """Draw diagonal Gaussian latent samples; optional noise enables testing."""
    if mean.shape != logvar.shape or mean.ndim < 2 or not mean.numel():
        raise ValueError("nonempty matching [batch, latent...] shapes required")
    if not mean.is_floating_point() or not logvar.is_floating_point() or mean.device != logvar.device:
        raise ValueError("floating tensors on one device required")
    dtype = torch.float64 if mean.dtype == torch.float64 or logvar.dtype == torch.float64 else torch.float32
    mean, logvar = mean.to(dtype), logvar.to(dtype)
    if noise is None:
        noise = torch.randn_like(mean)
    elif noise.shape != mean.shape or noise.device != mean.device or not noise.is_floating_point():
        raise ValueError("noise must match latent shape and device")
    return mean + (0.5 * logvar).exp() * noise.to(dtype)


def vae_loss(reconstruction, target, mean, logvar, likelihood="bernoulli", beta=1.0):
    """Negative ELBO: sum event dimensions per example, then mean batch.

    Bernoulli reconstruction is logits; unit_gaussian reconstruction is mean.
    Returns (total, reconstruction_nll, posterior_kl) as differentiable scalars.
    """
    if reconstruction.shape != target.shape or reconstruction.ndim < 2 or not reconstruction.numel():
        raise ValueError("nonempty matching [batch, observation...] shapes required")
    if mean.shape != logvar.shape or mean.ndim < 2 or not mean.numel() or len(mean) != len(target):
        raise ValueError("matching [batch, latent...] shapes required")
    tensors = (reconstruction, target, mean, logvar)
    if any(not value.is_floating_point() or value.device != reconstruction.device for value in tensors):
        raise ValueError("floating tensors on one device required")
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    dtype = torch.float64 if any(value.dtype == torch.float64 for value in tensors) else torch.float32
    reconstruction, target, mean, logvar = (value.to(dtype) for value in tensors)
    if likelihood == "bernoulli":
        if ((target < 0) | (target > 1)).any():
            raise ValueError("Bernoulli targets must lie in [0,1]")
        element_nll = F.binary_cross_entropy_with_logits(reconstruction, target, reduction="none")
    elif likelihood == "unit_gaussian":
        element_nll = 0.5 * ((reconstruction - target).square() + math.log(2 * math.pi))
    else:
        raise ValueError("unknown observation likelihood")
    reconstruction_nll = element_nll.flatten(1).sum(1).mean()
    posterior_kl = 0.5 * (mean.square() + logvar.exp() - 1 - logvar).flatten(1).sum(1).mean()
    return reconstruction_nll + beta * posterior_kl, reconstruction_nll, posterior_kl


def grouped_query_attention(q, k, v, allowed=None, causal=False, query_offset=0):
    """Projected GQA tensors: Q [B,Hq,Tq,D], K/V [B,Hkv,Tk,D].

    Consecutive equal-size query groups share a KV head. Boolean masks use
    True=allowed and must broadcast to [B,Hq,Tq,Tk]. No dropout or projections;
    explicit KV repetition is pedagogical, not a fused production kernel.
    """
    tensors = (q, k, v)
    if any(value.ndim != 4 or min(value.shape) <= 0 for value in tensors):
        raise ValueError("q/k/v must be nonempty [B,H,T,D]")
    if any(not value.is_floating_point() for value in tensors):
        raise TypeError("q/k/v must be floating point")
    if any(value.device != q.device or value.dtype != q.dtype for value in tensors):
        raise ValueError("q/k/v must share device and dtype")
    b, hq, tq, d = q.shape
    bk, hkv, tk, dk = k.shape
    if v.shape != k.shape or b != bk or d != dk or hq % hkv:
        raise ValueError("matching batch/features and Hq divisible by Hkv required")
    if isinstance(query_offset, bool) or not isinstance(query_offset, int) or query_offset < 0:
        raise ValueError("query_offset must be a nonnegative integer")
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("q/k/v must be finite")
    compute_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q_work = q.to(compute_dtype)
    k_work = k.to(compute_dtype).repeat_interleave(hq // hkv, dim=1)
    v_work = v.to(compute_dtype).repeat_interleave(hq // hkv, dim=1)
    scores = (q_work @ k_work.transpose(-1, -2)) / math.sqrt(d)
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("attention scores overflowed")
    if allowed is not None:
        if allowed.dtype != torch.bool or allowed.device != q.device:
            raise ValueError("allowed must be boolean on the same device")
        try:
            allowed = torch.broadcast_to(allowed, scores.shape)
        except RuntimeError as error:
            raise ValueError("allowed cannot broadcast to attention scores") from error
    if causal:
        causal_allowed = torch.arange(tk, device=q.device)[None, :] <= (
            query_offset + torch.arange(tq, device=q.device)[:, None])
        allowed = causal_allowed if allowed is None else allowed & causal_allowed
    if allowed is not None:
        if bool((~allowed).all(dim=-1).any()):
            raise ValueError("fully masked query row")
        scores = scores.masked_fill(~allowed, -math.inf)
    return (scores.softmax(dim=-1) @ v_work).to(v.dtype)


@torch.no_grad()
def variation_keep_indices(previous, current, visual_mask, keep_visual):
    """Single-sample V2Drop variation/selection core; int64 original-order indices.

    Hidden [N,D], bool visual mask [N]. Text/special tokens always survive;
    FP32 L2 scores, stable earlier-index ties. No position/mask/KV integration.
    """
    if previous.ndim != 2 or current.shape != previous.shape or current.shape[1] <= 0:
        raise ValueError("hidden states must have the same [N,D] shape, D > 0")
    if not previous.is_floating_point() or not current.is_floating_point():
        raise TypeError("hidden states must be floating point")
    if visual_mask.dtype != torch.bool or visual_mask.shape != (current.shape[0],):
        raise ValueError("visual_mask must be bool [N]")
    if previous.device != current.device or visual_mask.device != current.device:
        raise ValueError("all tensors must share a device")
    if isinstance(keep_visual, bool) or not isinstance(keep_visual, int):
        raise TypeError("keep_visual must be an integer")
    visual = torch.where(visual_mask)[0]
    if not 0 <= keep_visual <= visual.numel():
        raise ValueError("visual budget is out of range")
    delta = current[visual].float() - previous[visual].float()
    scores = torch.linalg.vector_norm(delta, ord=2, dim=-1)
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("visual variation scores must be finite")
    ranking = torch.argsort(scores, descending=True, stable=True)
    selected_visual = visual[ranking[:keep_visual]]
    other = torch.where(~visual_mask)[0]
    return torch.sort(torch.cat((other, selected_visual))).values


def moe_top_k_router(logits, k=2, renormalize=True):
    """[T,E] logits -> ([T,k] int64 expert ids, [T,k] floating gate weights).

    Stable lower-id ties; FP32 gate arithmetic unless input is float64. Hard
    selected ids are nondifferentiable, while selected weights keep gradients.
    No experts, capacity/drop policy, all-to-all or load-balancing loss.
    """
    if logits.ndim != 2 or min(logits.shape) <= 0:
        raise ValueError("logits must be nonempty [T,E]")
    if not logits.is_floating_point():
        raise TypeError("logits must be floating point")
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= logits.shape[1]:
        raise ValueError("k must be an integer in [1,E]")
    if not isinstance(renormalize, bool):
        raise TypeError("renormalize must be boolean")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("router logits must be finite")
    work = logits if logits.dtype == torch.float64 else logits.float()
    indices = torch.argsort(work, dim=-1, descending=True, stable=True)[:, :k]
    if renormalize:
        weights = work.gather(1, indices).softmax(-1)
    else:
        weights = work.softmax(-1).gather(1, indices)
    return indices, weights


def layer_norm_last_dim(x, gamma=None, beta=None, epsilon=1e-5):
    """Handwritten last-D LayerNorm; population variance and epsilon inside sqrt.

    Optional gamma/beta [D] are differentiable; use FP32 moments unless input
    is float64, and return x.dtype. No running/batch statistics are used.
    """
    if x.ndim < 1 or min(x.shape) <= 0 or not x.is_floating_point():
        raise ValueError("x must be nonempty floating [...,D]")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("x must be finite")
    for parameter in (gamma, beta):
        if parameter is not None and (parameter.shape != (x.shape[-1],) or
                not parameter.is_floating_point() or parameter.device != x.device or
                not bool(torch.isfinite(parameter).all())):
            raise ValueError("gamma/beta must be finite floating [D] on x.device")
    compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    work = x.to(compute_dtype)
    mean = work.mean(dim=-1, keepdim=True)
    variance = (work - mean).square().mean(dim=-1, keepdim=True)
    if not bool(torch.isfinite(variance).all()):
        raise ValueError("normalization moments overflowed")
    result = (work - mean) * torch.rsqrt(variance + epsilon)
    if gamma is not None:
        result = result * gamma.to(compute_dtype)
    if beta is not None:
        result = result + beta.to(compute_dtype)
    return result.to(x.dtype)
