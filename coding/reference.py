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


def _gqa_shape4(value, name):
    """Validate nonempty, rectangular [B,H,T,D] numeric nested sequences."""
    if not value or not value[0] or not value[0][0] or not value[0][0][0]:
        raise ValueError(f"{name} must be nonempty [B,H,T,D]")
    shape = (len(value), len(value[0]), len(value[0][0]), len(value[0][0][0]))
    for batch in value:
        if len(batch) != shape[1]:
            raise ValueError(f"{name} has ragged heads")
        for head in batch:
            if len(head) != shape[2] or any(len(row) != shape[3] for row in head):
                raise ValueError(f"{name} has ragged tokens/features")
            if any(not math.isfinite(item) for row in head for item in row):
                raise ValueError(f"{name} must be finite")
    return shape


def grouped_query_attention(q, k, v, allowed=None, causal=False, query_offset=0):
    """GQA [B,Hq,Tq,D] x [B,Hkv,Tk,D], consecutive equal-size head groups.

    Returns [B,Hq,Tq,D]. Mask is bool [Tq,Tk] or exact [B,Hq,Tq,Tk],
    True=allowed; unlike the torch version, arbitrary broadcasting is not used.
    No projections, dropout, RoPE or cache allocation are included.
    """
    b, hq, tq, d = _gqa_shape4(q, "q")
    bk, hkv, tk, dk = _gqa_shape4(k, "k")
    if _gqa_shape4(v, "v") != (bk, hkv, tk, dk) or b != bk or d != dk or hq % hkv:
        raise ValueError("GQA requires matching batch/features and Hq divisible by Hkv")
    if isinstance(query_offset, bool) or not isinstance(query_offset, int) or query_offset < 0:
        raise ValueError("query_offset must be a nonnegative integer")
    per_head = False
    if allowed is not None:
        shared = (len(allowed) == tq and all(len(row) == tk and
                  all(isinstance(item, bool) for item in row) for row in allowed))
        if not shared:
            try:
                per_head = (len(allowed) == b and all(len(batch) == hq and
                            all(len(head) == tq and all(len(row) == tk and
                                all(isinstance(item, bool) for item in row) for row in head)
                                for head in batch) for batch in allowed))
            except (TypeError, IndexError):
                per_head = False
            if not per_head:
                raise ValueError("allowed must be bool [Tq,Tk] or [B,Hq,Tq,Tk]")
    group_size = hq // hkv
    result = []
    for batch in range(b):
        heads = []
        for head in range(hq):
            mask = allowed[batch][head] if per_head else allowed
            if causal:
                mask = [[j <= query_offset + i and (mask is None or mask[i][j])
                         for j in range(tk)] for i in range(tq)]
            kv_head = head // group_size
            heads.append(attention(q[batch][head], k[batch][kv_head], v[batch][kv_head], mask))
        result.append(heads)
    return result


def max_stock_profit_two_transactions(prices):
    """At most two nonoverlapping buy/sell pairs; state updates use the old day.

    Finite signed numeric prices are accepted. Only one position is held.
    There is no transaction fee or cooldown in this arithmetic reference.
    Empty input returns 0 and the iterable is consumed once.
    """
    hold1 = hold2 = -math.inf
    cash1 = cash2 = 0
    for price in prices:
        if not math.isfinite(price):
            raise ValueError("prices must be finite")
        old_hold1, old_cash1, old_hold2, old_cash2 = hold1, cash1, hold2, cash2
        hold1 = max(old_hold1, -price)
        cash1 = max(old_cash1, old_hold1 + price)
        hold2 = max(old_hold2, old_cash1 - price)
        cash2 = max(old_cash2, old_hold2 + price)
    return cash2


def variation_keep_indices(previous, current, visual_mask, keep_visual):
    """Single-sample V2Drop score/index core, returning original-order indices.

    Hidden states are numeric [N,D] lists; all nonvisual tokens survive. Scores
    use Python floating-point precision, not the torch version's FP32 cast.
    This is not a model, position/mask gather or per-layer KV implementation.
    """
    if len(previous) != len(current) or len(visual_mask) != len(current):
        raise ValueError("hidden states and mask must have the same N")
    if any(not isinstance(value, bool) for value in visual_mask):
        raise ValueError("visual_mask must contain booleans")
    if isinstance(keep_visual, bool) or not isinstance(keep_visual, int):
        raise TypeError("keep_visual must be an integer")
    if current:
        _matrix(previous, "previous")
        _matrix(current, "current")
        if len(previous[0]) != len(current[0]):
            raise ValueError("hidden states must have the same D")
    visual = [index for index, flag in enumerate(visual_mask) if flag]
    if not 0 <= keep_visual <= len(visual):
        raise ValueError("visual budget is out of range")
    scores = {}
    for index in visual:
        delta = [float(after) - float(before) for before, after in zip(previous[index], current[index])]
        score = math.hypot(*delta)
        if not math.isfinite(score):
            raise ValueError("visual variation scores must be finite")
        scores[index] = score
    ranking = sorted(visual, key=lambda index: (-scores[index], index))
    keep = set(ranking[:keep_visual])
    return [index for index, flag in enumerate(visual_mask) if not flag or index in keep]


def float_sqrt(value, abs_tol=0.0, rel_tol=1e-12, max_iterations=128):
    """Approximate sqrt of a finite nonnegative int/float, without sqrt/pow.

    Scale by powers of two, then bisect a bounded mantissa. Stop when the
    half-width is <= abs_tol + rel_tol*estimate or floating-point stagnates.
    Requested tolerances below floating-point resolution cannot be promised.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("value must be an int or float")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError("value must be representable as a finite float") from error
    if not math.isfinite(value) or value < 0:
        raise ValueError("value must be finite and nonnegative")
    if (not math.isfinite(abs_tol) or not math.isfinite(rel_tol) or
            abs_tol < 0 or rel_tol < 0 or abs_tol + rel_tol == 0):
        raise ValueError("finite nonnegative tolerances, at least one positive, required")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")
    if value == 0:
        return 0.0
    mantissa, exponent = math.frexp(value)
    if exponent % 2:
        mantissa *= 2.0
        exponent -= 1
    scale = exponent // 2
    low, high = 0.5, 2.0
    for _ in range(max_iterations):
        middle = (low + high) / 2.0
        estimate = math.ldexp(middle, scale)
        half_width = math.ldexp((high - low) / 2.0, scale)
        if (half_width <= abs_tol + rel_tol * estimate or middle == low or
                middle == high or middle * middle == mantissa):
            return estimate
        if middle * middle < mantissa:
            low = middle
        else:
            high = middle
    raise RuntimeError("square-root bisection did not converge within max_iterations")


def longest_increasing_matrix_path(matrix):
    """Longest strictly increasing four-neighbor path via topological BFS.

    Four-neighbor moves only, without diagonals or boundary wrapping.
    Values must be finite; each step must be strictly increasing.
    Empty rectangular input returns 0. No recursive stack is used.
    """
    from collections import deque
    if not matrix:
        return 0
    columns = len(matrix[0])
    if any(len(row) != columns for row in matrix):
        raise ValueError("matrix must be rectangular")
    if columns == 0:
        return 0
    if any(not math.isfinite(value) for row in matrix for value in row):
        raise ValueError("matrix values must be finite")
    rows = len(matrix)
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    indegree = [[0] * columns for _ in range(rows)]
    frontier = deque()
    for row in range(rows):
        for column in range(columns):
            indegree[row][column] = sum(
                0 <= row + dr < rows and 0 <= column + dc < columns and
                matrix[row + dr][column + dc] < matrix[row][column]
                for dr, dc in directions)
            if indegree[row][column] == 0:
                frontier.append((row, column))
    length = 0
    while frontier:
        length += 1
        for _ in range(len(frontier)):
            row, column = frontier.popleft()
            for dr, dc in directions:
                nr, nc = row + dr, column + dc
                if 0 <= nr < rows and 0 <= nc < columns and matrix[nr][nc] > matrix[row][column]:
                    indegree[nr][nc] -= 1
                    if indegree[nr][nc] == 0:
                        frontier.append((nr, nc))
    return length


def moe_top_k_router(logits, k=2, renormalize=True):
    """Top-k score/index core for [T,E] finite logits, no expert execution.

    Return (expert_indices, weights), both [T,k]. Ties choose lower expert id.
    With renormalize=True, selected weights sum to 1; otherwise retain the
    full softmax gate values. This omits capacity, dispatch and auxiliary loss.
    """
    _matrix(logits, "logits")
    experts = len(logits[0])
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= experts:
        raise ValueError("k must be an integer in [1,E]")
    if not isinstance(renormalize, bool):
        raise TypeError("renormalize must be boolean")
    if any(not math.isfinite(value) for row in logits for value in row):
        raise ValueError("router logits must be finite")
    indices, weights = [], []
    for row in logits:
        chosen = sorted(range(experts), key=lambda expert: (-row[expert], expert))[:k]
        if renormalize:
            gate = softmax([row[expert] for expert in chosen])
        else:
            probabilities = softmax(row)
            gate = [probabilities[expert] for expert in chosen]
        indices.append(chosen)
        weights.append(gate)
    return indices, weights


def layer_norm_last_dim(x, gamma=None, beta=None, epsilon=1e-5):
    """LayerNorm on last D of a nonempty rectangular nested numeric sequence.

    Population (biased) variance, epsilon inside sqrt, optional [D] affine.
    Supports vectors and arbitrary leading dimensions; returns fresh lists.
    """
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    def shape(value):
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError("x must have nonempty rectangular dimensions")
        if all(not isinstance(item, (list, tuple)) for item in value):
            if any(not math.isfinite(item) for item in value):
                raise ValueError("x must be finite")
            return (len(value),)
        shapes = [shape(item) for item in value]
        if any(child != shapes[0] for child in shapes):
            raise ValueError("x must be rectangular")
        return (len(value),) + shapes[0]
    dimensions = shape(x)
    features = dimensions[-1]
    gamma = [1.] * features if gamma is None else gamma
    beta = [0.] * features if beta is None else beta
    if (len(gamma) != features or len(beta) != features or
            any(not math.isfinite(value) for value in gamma) or
            any(not math.isfinite(value) for value in beta)):
        raise ValueError("gamma/beta must be finite [D]")
    def normalize(value, depth):
        if depth > 1:
            return [normalize(child, depth - 1) for child in value]
        mean = math.fsum(float(item) / features for item in value)
        try:
            variance = math.fsum((float(item) - mean) ** 2 for item in value) / features
        except OverflowError as error:
            raise ValueError("normalization moments overflowed") from error
        if not math.isfinite(variance):
            raise ValueError("normalization moments must be finite")
        denominator = math.sqrt(variance + epsilon)
        return [((float(item) - mean) / denominator) * scale + offset
                for item, scale, offset in zip(value, gamma, beta)]
    return normalize(x, len(dimensions))
