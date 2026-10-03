"""Small, original, standard-library implementations for interview practice.

These are correctness references, not GPU kernels or a production model.
Matrix convention: a list of row vectors; masks use True = allowed.
"""
import math
import random
import heapq
from collections import OrderedDict


def softmax(logits):
    if not logits or any(math.isnan(x) or x == math.inf for x in logits):
        raise ValueError("nonempty logits with no NaN or +inf required")
    maximum = max(logits)
    if maximum == -math.inf:
        raise ValueError("all logits are masked")
    weights = [math.exp(x - maximum) for x in logits]
    total = sum(weights)
    return [x / total for x in weights]


def cross_entropy(logits, target):
    if not 0 <= target < len(logits):
        raise ValueError("invalid target")
    # Validate the same numerical domain as softmax; compute log-sum-exp
    # directly so a very unlikely target does not underflow to log(0).
    softmax(logits)
    maximum = max(logits)
    return maximum + math.log(sum(math.exp(x - maximum) for x in logits)) - logits[target]


def _matrix(x, name):
    if not x or not x[0] or any(len(row) != len(x[0]) for row in x):
        raise ValueError(f"{name} must be nonempty and rectangular")


def attention(q, k, v, allowed=None):
    for name, value in (("q", q), ("k", k), ("v", v)):
        _matrix(value, name)
    if len(k) != len(v) or len(q[0]) != len(k[0]):
        raise ValueError("incompatible shapes")
    if allowed is not None and (len(allowed) != len(q) or any(len(r) != len(k) for r in allowed)):
        raise ValueError("mask shape must be [query_length, key_length]")
    output = []
    for i, query in enumerate(q):
        scores = [sum(a * b for a, b in zip(query, key)) / math.sqrt(len(query))
                  if allowed is None or allowed[i][j] else -math.inf
                  for j, key in enumerate(k)]
        probabilities = softmax(scores)
        output.append([sum(probabilities[j] * v[j][d] for j in range(len(v))) for d in range(len(v[0]))])
    return output


def causal_mask(query_length, key_length, query_offset=0):
    # Absolute query positions are query_offset + i. With a past KV cache,
    # query_offset is the number of cached tokens, not zero.
    return [[j <= query_offset + i for j in range(key_length)] for i in range(query_length)]


def rope(vector, position, base=10000.0):
    if not vector or len(vector) % 2 or base <= 0:
        raise ValueError("positive base and nonempty even dimension required")
    result = []
    for i in range(len(vector) // 2):
        angle = position * base ** (-2.0 * i / len(vector))
        c, s = math.cos(angle), math.sin(angle)
        a, b = vector[2 * i:2 * i + 2]
        result.extend([a * c - b * s, a * s + b * c])
    return result


def info_nce(queries, keys, temperature=0.1, normalize=True):
    _matrix(queries, "queries")
    _matrix(keys, "keys")
    if temperature <= 0 or len(queries) != len(keys) or len(queries[0]) != len(keys[0]):
        raise ValueError("paired matrices and positive temperature required")
    def unit(row):
        norm = math.sqrt(sum(x * x for x in row))
        if norm == 0:
            raise ValueError("cannot normalize a zero vector")
        return [x / norm for x in row]
    q = [unit(row) for row in queries] if normalize else queries
    k = [unit(row) for row in keys] if normalize else keys
    return sum(cross_entropy([sum(a * b for a, b in zip(row, key)) / temperature for key in k], i)
               for i, row in enumerate(q)) / len(q)


def sample_logits(logits, temperature=1.0, top_k=None, top_p=1.0, rng=None):
    if temperature < 0 or not 0 < top_p <= 1 or (top_k is not None and top_k < 1):
        raise ValueError("invalid sampling parameters")
    softmax(logits)
    if temperature == 0:
        return max(range(len(logits)), key=logits.__getitem__)
    ordered = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)
    if top_k is not None:
        ordered = ordered[:top_k]
    # Convention: temperature -> top-k -> renormalize -> top-p -> renormalize.
    pivot = logits[ordered[0]]
    probs = softmax([(logits[i] - pivot) / temperature for i in ordered])
    kept, weights, mass = [], [], 0.0
    for i, p in zip(ordered, probs):
        kept.append(i)
        weights.append(p)
        mass += p
        if mass >= top_p:
            break
    return (rng or random).choices(kept, weights=weights, k=1)[0]


def lora_forward(x, weight, a, b, alpha=1.0):
    # weight[out,in], a[rank,in], b[out,rank]; no bias.
    for name, value in (("weight", weight), ("a", a), ("b", b)):
        _matrix(value, name)
    if len(weight[0]) != len(x) or len(a[0]) != len(x) or len(b) != len(weight) or len(b[0]) != len(a):
        raise ValueError("incompatible LoRA shapes")
    hidden = [sum(u * t for u, t in zip(row, x)) for row in a]
    return [sum(u * t for u, t in zip(row, x)) + alpha / len(a) * sum(u * t for u, t in zip(br, hidden))
            for row, br in zip(weight, b)]


def dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta=0.1):
    if beta <= 0:
        raise ValueError("beta must be positive")
    z = beta * ((policy_chosen - reference_chosen) - (policy_rejected - reference_rejected))
    return max(-z, 0.0) + math.log1p(math.exp(-abs(z)))


def grpo_advantages(rewards, epsilon=1e-8):
    if not rewards or epsilon <= 0:
        raise ValueError("nonempty rewards and positive epsilon required")
    mean = sum(rewards) / len(rewards)
    # Population std (correction=0); this choice must be explicit.
    std = math.sqrt(sum((x - mean) ** 2 for x in rewards) / len(rewards))
    return [(x - mean) / (std + epsilon) for x in rewards]


def kth_largest(values, k):
    if not 1 <= k <= len(values):
        raise ValueError("k out of range")
    heap = []
    for x in values:
        if len(heap) < k:
            heapq.heappush(heap, x)
        elif x > heap[0]:
            heapq.heapreplace(heap, x)
    return heap[0]


def number_of_islands(grid):
    if not grid:
        return 0
    columns = len(grid[0])
    if any(len(row) != columns for row in grid):
        raise ValueError("ragged grid")
    seen, count = set(), 0
    for r in range(len(grid)):
        for c in range(columns):
            if grid[r][c] != 1 or (r, c) in seen:
                continue
            count += 1
            stack = [(r, c)]
            seen.add((r, c))
            while stack:
                i, j = stack.pop()
                for u, v in ((i-1, j), (i+1, j), (i, j-1), (i, j+1)):
                    if 0 <= u < len(grid) and 0 <= v < columns and grid[u][v] == 1 and (u, v) not in seen:
                        seen.add((u, v))
                        stack.append((u, v))
    return count


def edit_distance(left, right):
    # Rolling rows, O(len(left)*len(right)) time and O(min(m,n)) space.
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(previous[j] + 1, current[j-1] + 1, previous[j-1] + (a != b)))
        previous = current
    return previous[-1]


class LRUCache:
    def __init__(self, capacity):
        if capacity < 0:
            raise ValueError("capacity must be nonnegative")
        self.capacity = capacity
        self.data = OrderedDict()

    def get(self, key, default=-1):
        if key not in self.data:
            return default
        self.data.move_to_end(key)
        return self.data[key]

    def put(self, key, value):
        if self.capacity == 0:
            return
        self.data[key] = value
        self.data.move_to_end(key)
        if len(self.data) > self.capacity:
            self.data.popitem(last=False)


def kv_cache_bytes(batch, sequence, layers, kv_heads, head_dim, bytes_per_element=2):
    if any(x < 0 for x in (batch, sequence, layers, kv_heads, head_dim, bytes_per_element)):
        raise ValueError("dimensions must be nonnegative")
    return 2 * batch * sequence * layers * kv_heads * head_dim * bytes_per_element
