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


def two_sum(values, target):
    # Lookup before insertion: one array element cannot be used twice.
    seen = {}
    for index, value in enumerate(values):
        complement = target - value
        if complement in seen:
            return [seen[complement], index]
        seen.setdefault(value, index)
    raise ValueError("no pair of distinct indices has the target sum")


def longest_common_subsequence(left, right):
    # Return the length; subsequence characters need not be contiguous.
    # Rolling rows: O(m*n) time, O(min(m,n)) auxiliary space.
    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for a in left:
        current = [0]
        for j, b in enumerate(right, 1):
            current.append(previous[j-1] + 1 if a == b else max(previous[j], current[j-1]))
        previous = current
    return previous[-1]


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


class BucketBatchSampler:
    """Single-process, no-replacement length buckets yielding index lists.

    Pass to DataLoader(batch_sampler=...), not to its sampler argument.
    Buckets contain batch_size * bucket_multiplier adjacent sorted lengths.
    """
    def __init__(self, lengths, batch_size, bucket_multiplier=4,
                 shuffle=True, drop_last=False, seed=0):
        for name, value in (("batch_size", batch_size),
                            ("bucket_multiplier", bucket_multiplier)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(name + " must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self.lengths = tuple(lengths)
        if any(isinstance(n, bool) or not isinstance(n, (int, float))
               or n < 0 or (isinstance(n, float) and not math.isfinite(n)) for n in self.lengths):
            raise ValueError("lengths must be finite nonnegative numbers")
        self.batch_size = batch_size
        self.bucket_size = batch_size * bucket_multiplier
        self.shuffle, self.drop_last, self.seed = shuffle, drop_last, seed
        self.epoch = 0
        self.sorted_indices = tuple(sorted(range(len(self.lengths)),
                                           key=self.lengths.__getitem__))

    def set_epoch(self, epoch):
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self.epoch = epoch

    def __len__(self):
        n = len(self.lengths)
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        # Fresh local RNG: repeated iteration of one epoch is reproducible.
        rng = random.Random(self.seed + self.epoch)
        batches = []
        for offset in range(0, len(self.sorted_indices), self.bucket_size):
            bucket = list(self.sorted_indices[offset:offset + self.bucket_size])
            if self.shuffle:
                rng.shuffle(bucket)
            for start in range(0, len(bucket), self.batch_size):
                batch = bucket[start:start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches


def integer_sqrt(value):
    """Return floor(sqrt(value)) with integer binary search."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("a nonnegative integer is required")
    if value < 2:
        return value
    low, high = 1, value // 2 + 1
    while low <= high:
        middle = (low + high) // 2
        # Division avoids relying on fixed-width multiplication not overflowing.
        if middle <= value // middle:
            low = middle + 1
        else:
            high = middle - 1
    return high


def longest_palindromic_substring(text):
    """Center expansion; ties return the occurrence with the earliest start."""
    start, length = 0, 0
    for center in range(len(text)):
        for left, right in ((center, center), (center, center + 1)):
            while left >= 0 and right < len(text) and text[left] == text[right]:
                candidate = right - left + 1
                if candidate > length or (candidate == length and left < start):
                    start, length = left, candidate
                left -= 1
                right += 1
    return text[start:start + length]


def unique_permutations(values):
    """Return all distinct permutations of sortable values, without mutation.

    Empty input has one permutation: the empty list.
    """
    ordered = sorted(values)
    used = [False] * len(ordered)
    path, result = [], []

    def visit():
        if len(path) == len(ordered):
            result.append(path.copy())
            return
        for index, value in enumerate(ordered):
            # Equal unused siblings produce the same branch at this depth.
            if used[index] or (index > 0 and value == ordered[index - 1] and not used[index - 1]):
                continue
            used[index] = True
            path.append(value)
            visit()
            path.pop()
            used[index] = False

    visit()
    return result


class ListNode:
    def __init__(self, value, next=None):
        self.value, self.next = value, next


def reverse_linked_list(head):
    """Reverse acyclic nodes in place; reject a cycle before changing links."""
    slow = fast = head
    while fast is not None and fast.next is not None:
        slow, fast = slow.next, fast.next.next
        if slow is fast:
            raise ValueError("cyclic linked list")
    previous, current = None, head
    while current is not None:
        following = current.next
        current.next = previous
        previous, current = current, following
    return previous


def max_stock_profit(prices):
    """At most one buy followed by a later sell; declining input returns 0."""
    lowest, best = None, 0
    for price in prices:
        if lowest is not None:
            best = max(best, price - lowest)
        lowest = price if lowest is None else min(lowest, price)
    return best
