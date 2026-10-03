import math
import random
import unittest
from reference import *


class ReferenceTests(unittest.TestCase):
    def test_softmax_translation_and_masks(self):
        expected = softmax([1, 2, 3])
        for actual, target in zip(softmax([1001, 1002, 1003]), expected):
            self.assertAlmostEqual(actual, target)
        self.assertEqual(softmax([-math.inf, 0]), [0.0, 1.0])
        with self.assertRaises(ValueError):
            softmax([-math.inf, -math.inf])

    def test_cross_entropy_extreme_logits(self):
        self.assertAlmostEqual(cross_entropy([1000, -1000], 1), 2000)
        self.assertAlmostEqual(cross_entropy([0, 0], 0), math.log(2))

    def test_causal_attention_cannot_see_future(self):
        q = k = [[1, 0], [0, 1], [1, 1]]
        v = [[1, 2], [3, 4], [5, 6]]
        full = attention(q, k, v, causal_mask(3, 3))
        changed = attention(q, k, [[1, 2], [3, 4], [100, 100]], causal_mask(3, 3))
        self.assertEqual(full[:2], changed[:2])
        self.assertEqual(full[0], v[0])
        step = attention([q[2]], k, v, causal_mask(1, 3, query_offset=2))
        self.assertEqual(step[0], full[2])

    def test_rope_norm_and_relative_dot(self):
        a, b = [1, 2, 3, 4], [3, -2, 2, 1]
        dot = lambda x, y: sum(u*v for u, v in zip(x, y))
        self.assertAlmostEqual(dot(rope(a, 50), rope(a, 50)), dot(a, a))
        self.assertAlmostEqual(dot(rope(a, 4), rope(b, 9)), dot(a, rope(b, 5)))

    def test_infonce_positive_pairing(self):
        q = [[1, 0], [0, 1]]
        good = info_nce(q, q, temperature=0.2)
        bad = info_nce(q, q[::-1], temperature=0.2)
        self.assertLess(good, bad)
        self.assertAlmostEqual(info_nce(q, q, temperature=1), math.log1p(math.exp(-1)))

    def test_sampling_crossing_token_and_seed(self):
        # P=[0.6,0.3,0.1]; p=0.7 must keep the token that crosses 0.7.
        logits = [math.log(0.6), math.log(0.3), math.log(0.1)]
        rng = random.Random(1)
        draws = {sample_logits(logits, top_p=0.7, rng=rng) for _ in range(100)}
        self.assertEqual(draws, {0, 1})
        self.assertEqual(sample_logits(logits, temperature=0), 0)
        self.assertEqual(sample_logits(logits, top_k=1), 0)
        self.assertEqual(sample_logits([1.0, 2.0], temperature=1e-320), 1)

    def test_lora_matches_merged_weight(self):
        x, w, a, b = [2, 3], [[1, 2], [3, 4]], [[2, 1]], [[3], [4]]
        merged = [[w[i][j] + 2*b[i][0]*a[0][j] for j in range(2)] for i in range(2)]
        self.assertEqual(lora_forward(x, w, a, b, alpha=2), [sum(u*v for u, v in zip(row, x)) for row in merged])

    def test_dpo_sign_and_extremes(self):
        baseline = dpo_loss(-2, -2, -2, -2)
        self.assertAlmostEqual(baseline, math.log(2))
        self.assertLess(dpo_loss(-1, -3, -2, -2), baseline)
        self.assertGreater(dpo_loss(-3, -1, -2, -2), baseline)
        self.assertTrue(math.isfinite(dpo_loss(-1e6, 0, 0, 0, beta=1)))

    def test_grpo_zero_variance_and_scale(self):
        self.assertEqual(grpo_advantages([3, 3, 3]), [0, 0, 0])
        values = grpo_advantages([1, 2, 3])
        self.assertAlmostEqual(sum(values), 0)
        self.assertAlmostEqual(sum(x*x for x in values)/3, 1, places=6)

    def test_kth_largest_against_sorted_random(self):
        rng = random.Random(9)
        for _ in range(200):
            values = [rng.randrange(-10, 11) for _ in range(rng.randrange(1, 30))]
            k = rng.randrange(1, len(values)+1)
            self.assertEqual(kth_largest(values, k), sorted(values, reverse=True)[k-1])

    def test_two_sum_against_exhaustive_pairs(self):
        self.assertEqual(two_sum([2, 7, 11, 15], 9), [0, 1])
        self.assertEqual(two_sum([3, 3], 6), [0, 1])
        self.assertEqual(two_sum([-3, 0, 3], 0), [0, 2])
        rng = random.Random(17)
        for _ in range(200):
            values = [rng.randrange(-5, 6) for _ in range(rng.randrange(9))]
            target = rng.randrange(-10, 11)
            pairs = [(i, j) for i in range(len(values)) for j in range(i+1, len(values))
                     if values[i] + values[j] == target]
            if pairs:
                i, j = two_sum(values, target)
                self.assertIn((i, j), pairs)
            else:
                with self.assertRaises(ValueError):
                    two_sum(values, target)
        with self.assertRaises(ValueError):
            two_sum([3], 6)

    def test_lcs_against_exhaustive_subsequences(self):
        from itertools import combinations
        def subsequences(text):
            return {''.join(text[i] for i in positions)
                    for length in range(len(text)+1)
                    for positions in combinations(range(len(text)), length)}
        for a, b, expected in (('abcde', 'ace', 3), ('abc', 'def', 0),
                               ('', 'abc', 0), ('aaaa', 'aa', 2), ('abc', 'abc', 3)):
            self.assertEqual(longest_common_subsequence(a, b), expected)
        rng = random.Random(18)
        for _ in range(120):
            a = ''.join(rng.choice('abc') for _ in range(rng.randrange(7)))
            b = ''.join(rng.choice('abc') for _ in range(rng.randrange(7)))
            expected = max(map(len, subsequences(a) & subsequences(b)))
            self.assertEqual(longest_common_subsequence(a, b), expected)
            self.assertEqual(longest_common_subsequence(b, a), expected)

    def test_islands_against_union_find_random(self):
        rng = random.Random(8)
        for _ in range(50):
            grid = [[rng.randrange(2) for _ in range(5)] for _ in range(4)]
            parent = {(r,c):(r,c) for r in range(4) for c in range(5) if grid[r][c]}
            def find(x):
                while parent[x] != x:
                    x = parent[x]
                return x
            for r, c in parent:
                for neighbor in ((r+1,c), (r,c+1)):
                    if neighbor in parent:
                        parent[find(neighbor)] = find((r,c))
            self.assertEqual(number_of_islands(grid), len({find(x) for x in parent}))
        self.assertEqual(number_of_islands([]), 0)

    def test_edit_distance_against_recursive_small(self):
        from functools import lru_cache
        @lru_cache(None)
        def brute(a, b):
            if not a or not b:
                return max(len(a), len(b))
            return min(brute(a[:-1], b)+1, brute(a, b[:-1])+1, brute(a[:-1], b[:-1])+(a[-1] != b[-1]))
        for a in ("", "a", "ab", "ba", "abc"):
            for b in ("", "b", "ab", "bb", "abc"):
                self.assertEqual(edit_distance(a,b), brute(a,b))
        self.assertEqual(edit_distance("horse", "ros"), 3)

    def test_lru_eviction_update_zero_capacity(self):
        cache = LRUCache(2)
        cache.put(1, 1); cache.put(2, 2)
        self.assertEqual(cache.get(1), 1)
        cache.put(3, 3)
        self.assertEqual(cache.get(2), -1)
        cache.put(1, 4); cache.put(4, 4)
        self.assertEqual(cache.get(3), -1)
        self.assertEqual(cache.get(1), 4)
        zero = LRUCache(0); zero.put(1, 1)
        self.assertEqual(zero.get(1), -1)

    def test_kv_cache_units(self):
        self.assertEqual(kv_cache_bytes(1,4096,32,8,128,2), 536870912)

    def test_bucket_sampler_coverage_and_length(self):
        rng = random.Random(23)
        for count in range(42):
            lengths = [rng.randrange(101) for _ in range(count)]
            for batch_size in (1, 3, 8):
                for drop_last in (False, True):
                    sampler = BucketBatchSampler(lengths, batch_size, drop_last=drop_last, seed=9)
                    batches = list(sampler)
                    indices = [i for batch in batches for i in batch]
                    self.assertEqual(len(batches), len(sampler))
                    self.assertEqual(len(indices), len(set(indices)))
                    self.assertTrue(set(indices) <= set(range(count)))
                    self.assertTrue(all(0 < len(batch) <= batch_size for batch in batches))
                    if drop_last:
                        self.assertEqual(len(indices), count // batch_size * batch_size)
                        self.assertTrue(all(len(batch) == batch_size for batch in batches))
                    else:
                        self.assertEqual(sorted(indices), list(range(count)))

    def test_bucket_sampler_epoch_reproducibility(self):
        a = BucketBatchSampler(range(60), 3, seed=8)
        b = BucketBatchSampler(range(60), 3, seed=8)
        original = list(a)
        self.assertEqual(original, list(a))
        self.assertEqual(original, list(b))
        a.set_epoch(1); b.set_epoch(1)
        self.assertEqual(list(a), list(b))
        self.assertNotEqual(original, list(a))
        self.assertNotEqual(original, list(BucketBatchSampler(range(60), 3, seed=18)))

    def test_bucket_sampler_empty_ties_and_order(self):
        self.assertEqual(list(BucketBatchSampler([], 5)), [])
        self.assertEqual(list(BucketBatchSampler([8, 2, 2, 10, 0], 2, shuffle=False)),
                         [[4, 1], [2, 0], [3]])
        self.assertEqual(list(BucketBatchSampler([7], 2, drop_last=True)), [])
        lengths = [2, 1]
        sampler = BucketBatchSampler(lengths, 2, shuffle=False)
        lengths[:] = []
        self.assertEqual(list(sampler), [[1, 0]])

    def test_bucket_sampler_padding_budget(self):
        lengths = [1] * 64 + [100] * 64
        sampler = BucketBatchSampler(lengths, 8, bucket_multiplier=4, seed=13)
        padded = sum(max(lengths[i] for i in batch) * len(batch) for batch in sampler)
        shuffled = list(range(len(lengths)))
        random.Random(13).shuffle(shuffled)
        random_padded = sum(max(lengths[i] for i in shuffled[start:start+8]) * 8
                            for start in range(0, len(lengths), 8))
        self.assertEqual(padded, sum(lengths))
        self.assertLess(padded, random_padded)

    def test_bucket_sampler_invalid_parameters(self):
        for args in (([1], 0), ([1], True), ([-1], 2), ([float('nan')], 2),
                     ([float('inf')], 2), (['3'], 2)):
            with self.assertRaises(ValueError):
                BucketBatchSampler(*args)
        for kwargs in ({'bucket_multiplier': 0}, {'seed': 1.5}):
            with self.assertRaises(ValueError):
                BucketBatchSampler([1], 2, **kwargs)
        for epoch in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                BucketBatchSampler([1], 2).set_epoch(epoch)

    def test_integer_sqrt_exact_boundaries_and_large_values(self):
        values = list(range(500))
        rng = random.Random(24)
        for bits in (8, 64, 256, 1024):
            for _ in range(20):
                root = rng.getrandbits(bits)
                values.extend((root * root, root * root + 1, max(0, root * root - 1)))
        for value in values:
            actual = integer_sqrt(value)
            self.assertEqual(actual, math.isqrt(value))
            self.assertLessEqual(actual * actual, value)
            self.assertGreater((actual + 1) * (actual + 1), value)
        for value in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                integer_sqrt(value)

    def test_longest_palindrome_against_exhaustive_substrings(self):
        def brute(text):
            best = ''
            for i in range(len(text)):
                for j in range(i + 1, len(text) + 1):
                    candidate = text[i:j]
                    if candidate == candidate[::-1] and len(candidate) > len(best):
                        best = candidate
            return best
        self.assertEqual(longest_palindromic_substring('babad'), 'bab')
        self.assertEqual(longest_palindromic_substring('cbbd'), 'bb')
        self.assertEqual(longest_palindromic_substring(''), '')
        self.assertEqual(longest_palindromic_substring('aaaaa'), 'aaaaa')
        rng = random.Random(25)
        for _ in range(250):
            text = ''.join(rng.choice('abc') for _ in range(rng.randrange(12)))
            self.assertEqual(longest_palindromic_substring(text), brute(text))

    def test_unique_permutations_against_itertools(self):
        from itertools import permutations
        from collections import Counter
        self.assertEqual(unique_permutations([]), [[]])
        self.assertEqual(unique_permutations([1, 1, 2]), [[1, 1, 2], [1, 2, 1], [2, 1, 1]])
        rng = random.Random(26)
        for _ in range(100):
            values = [rng.randrange(-2, 3) for _ in range(rng.randrange(7))]
            original = values.copy()
            actual = unique_permutations(values)
            tuples = [tuple(row) for row in actual]
            self.assertEqual(set(tuples), set(permutations(values)))
            self.assertEqual(len(tuples), len(set(tuples)))
            self.assertEqual(values, original)
            self.assertTrue(all(Counter(row) == Counter(values) for row in actual))
            if len(actual) > 1 and values:
                previous = actual[1].copy()
                actual[0][0] = 'changed'
                self.assertEqual(actual[1], previous)
        with self.assertRaises(TypeError):
            unique_permutations([1, 'a'])

    def test_reverse_linked_list_identity_termination_and_cycles(self):
        for count in range(20):
            nodes = [ListNode(index % 3) for index in range(count)]
            for before, after in zip(nodes, nodes[1:]):
                before.next = after
            head = reverse_linked_list(nodes[0] if nodes else None)
            traversed, seen = [], set()
            while head is not None:
                self.assertNotIn(id(head), seen)
                seen.add(id(head))
                traversed.append(head)
                head = head.next
            self.assertEqual(traversed, nodes[::-1])
            if nodes:
                self.assertIsNone(nodes[0].next)
                original_head = reverse_linked_list(nodes[-1])
                self.assertIs(original_head, nodes[0])
                for before, after in zip(nodes, nodes[1:]):
                    self.assertIs(before.next, after)
        nodes = [ListNode(i) for i in range(4)]
        for before, after in zip(nodes, nodes[1:]):
            before.next = after
        nodes[-1].next = nodes[1]
        original_next = [node.next for node in nodes]
        with self.assertRaises(ValueError):
            reverse_linked_list(nodes[0])
        self.assertEqual([node.next for node in nodes], original_next)
        single = ListNode(3)
        single.next = single
        with self.assertRaises(ValueError):
            reverse_linked_list(single)
        self.assertIs(single.next, single)

    def test_single_transaction_stock_profit_against_all_pairs(self):
        for prices, expected in (([], 0), ([4], 0), ([7, 6, 4, 3, 1], 0),
                                 ([7, 1, 5, 3, 6, 4], 5), ([1, 2, 3, 4], 3)):
            self.assertEqual(max_stock_profit(prices), expected)
        rng = random.Random(27)
        for _ in range(250):
            prices = [rng.randrange(30) for _ in range(rng.randrange(20))]
            expected = max([0] + [prices[j] - prices[i] for i in range(len(prices))
                                  for j in range(i + 1, len(prices))])
            original = prices.copy()
            self.assertEqual(max_stock_profit(prices), expected)
            self.assertEqual(max_stock_profit(iter(prices)), expected)
            self.assertEqual(prices, original)

    def test_infonce_normalization_temperature_and_paired_diagonal(self):
        q = [[1., 0.], [0., 1.]]
        k = [[8., 0.], [0., 0.25]]
        for temperature in (0.2, 1., 3.):
            expected = math.log1p(math.exp(-1. / temperature))
            self.assertAlmostEqual(info_nce(q, k, temperature), expected)
            # Positive rescaling per feature vector cannot change cosine logits.
            self.assertAlmostEqual(info_nce([[7., 0.], [0., 11.]], k, temperature), expected)
        self.assertEqual(info_nce([[1., 2.]], [[3., 4.]]), 0.)
        for temperature in (0., -1.):
            with self.assertRaises(ValueError):
                info_nce(q, k, temperature)
        with self.assertRaises(ValueError):
            info_nce([[0., 0.]], [[1., 0.]])

    def test_infonce_logsumexp_extremes_and_raw_dot_option(self):
        q = [[1., 0.], [0., 1.]]
        k = [[1000., 0.], [0., 1000.]]
        # The wrong positive is 1000 below the negative: no exp(1000) required.
        self.assertEqual(info_nce(q, k[::-1], 1., normalize=False), 1000.)
        self.assertEqual(info_nce(q, k[::-1], 0.5, normalize=False), 2000.)
        self.assertEqual(info_nce(q, k, 1., normalize=False), 0.)
        self.assertAlmostEqual(info_nce(q, k, 1.), math.log1p(math.exp(-1.)))


    def test_gqa_consecutive_groups_and_mha_mqa_endpoints(self):
        q = [[[[0., 0.]] for _ in range(4)]]
        k = [[[[0., 0.]], [[0., 0.]]]]
        v = [[[[3., 4.]], [[8., 9.]]]]
        # With a single key, the output selects the KV group independently of softmax.
        self.assertEqual(grouped_query_attention(q, k, v), [[[[3., 4.]], [[3., 4.]], [[8., 9.]], [[8., 9.]]]])
        q3 = [[[[0.]] for _ in range(3)]]
        self.assertEqual(grouped_query_attention(q3, [[[[0.]]]], [[[[7.]]]]), [[[[7.]], [[7.]], [[7.]]]])
        values = [[[[1.]], [[2.]], [[3.]]]]
        self.assertEqual(grouped_query_attention(q3, q3, values), values)

    def test_gqa_against_independent_per_head_formula(self):
        rng = random.Random(31)
        def tensor(b, h, t, d):
            return [[[[rng.uniform(-2, 2) for _ in range(d)] for _ in range(t)] for _ in range(h)] for _ in range(b)]
        for hq, hkv in ((6, 2), (4, 1), (3, 3)):
            q, k, v = tensor(2, hq, 2, 3), tensor(2, hkv, 4, 3), tensor(2, hkv, 4, 3)
            result = grouped_query_attention(q, k, v)
            for b in range(2):
                for h in range(hq):
                    kv = h // (hq // hkv)
                    for t in range(2):
                        logits = [math.fsum(a * z for a, z in zip(q[b][h][t], row)) / math.sqrt(3) for row in k[b][kv]]
                        weights = [math.exp(z - max(logits)) for z in logits]
                        weights = [z / math.fsum(weights) for z in weights]
                        expected = [math.fsum(w * row[d] for w, row in zip(weights, v[b][kv])) for d in range(3)]
                        for actual, target in zip(result[b][h][t], expected):
                            self.assertAlmostEqual(actual, target)

    def test_gqa_mask_offset_and_input_contract(self):
        q = [[[[1.], [1.]], [[1.], [1.]]]]
        k = [[[[0.], [1.], [2.]]]]
        v = [[[[3.], [4.], [100.]]]]
        mask = [[True, False, False], [False, True, False]]
        self.assertEqual(grouped_query_attention(q, k, v, mask), [[[[3.], [4.]], [[3.], [4.]]]])
        masks = [[mask, mask[::-1]]]
        self.assertEqual(grouped_query_attention(q, k, v, masks), [[[[3.], [4.]], [[4.], [3.]]]])
        causal = grouped_query_attention(q, k, v, causal=True, query_offset=1)
        shifted = grouped_query_attention(q, k, [[[[3.], [4.], [10000.]]]], causal=True, query_offset=1)
        self.assertEqual(causal[0][0][0], shifted[0][0][0])
        self.assertNotEqual(causal[0][0][1], shifted[0][0][1])
        for bad in ([[False] * 3] * 2, [[1, 0, 0]] * 2, [[True]]):
            with self.assertRaises(ValueError):
                grouped_query_attention(q, k, v, bad)
        with self.assertRaises(ValueError):
            grouped_query_attention(q + q, k, v)
        with self.assertRaises(ValueError):
            grouped_query_attention(q, k * 2, v * 2)

    def test_two_transaction_stock_profit_against_all_ordered_pairs(self):
        def brute(prices):
            n = len(prices)
            trades = [(i, j, prices[j] - prices[i]) for i in range(n) for j in range(i + 1, n)]
            return max([0] + [profit for _, _, profit in trades] +
                       [p1 + p2 for _, sell, p1 in trades for buy, _, p2 in trades if sell < buy])
        for prices, expected in (([], 0), ([8], 0), ([7, 6, 4, 3, 1], 0),
                                 ([3, 3, 5, 0, 0, 3, 1, 4], 6), ([1, 2, 3, 4, 5], 4),
                                 ([-3, -1, -4, 2], 8)):
            self.assertEqual(max_stock_profit_two_transactions(prices), expected)
        rng = random.Random(32)
        for _ in range(300):
            prices = [rng.randrange(-5, 20) for _ in range(rng.randrange(10))]
            original = prices.copy()
            self.assertEqual(max_stock_profit_two_transactions(prices), brute(prices))
            self.assertEqual(max_stock_profit_two_transactions(iter(prices)), brute(prices))
            self.assertEqual(prices, original)
        with self.assertRaises(ValueError):
            max_stock_profit_two_transactions([1, math.nan])

    def test_variation_keep_indices_budget_stability_and_independent_scores(self):
        previous = [[0., 0.]] * 6
        current = [[99., 99.], [3., 4.], [0., 8.], [5., 0.], [4., 3.], [-9., -9.]]
        mask = [False, True, True, True, True, False]
        self.assertEqual(variation_keep_indices(previous, current, mask, 2), [0, 1, 2, 5])
        self.assertEqual(variation_keep_indices(previous, current, mask, 0), [0, 5])
        self.assertEqual(variation_keep_indices(previous, current, mask, 4), list(range(6)))
        self.assertEqual(variation_keep_indices([], [], [], 0), [])
        self.assertEqual(variation_keep_indices([[0.]], [[math.nan]], [False], 0), [0])
        rng = random.Random(33)
        for _ in range(100):
            n = rng.randrange(1, 15)
            before = [[rng.randrange(-5, 6) for _ in range(3)] for _ in range(n)]
            after = [[rng.randrange(-5, 6) for _ in range(3)] for _ in range(n)]
            visual = [rng.choice((True, False)) for _ in range(n)]
            budget = rng.randrange(sum(visual) + 1)
            ranking = sorted((i for i in range(n) if visual[i]),
                             key=lambda i: (-sum((a-b)**2 for a,b in zip(after[i], before[i])), i))
            selected = set(ranking[:budget])
            expected = [i for i in range(n) if not visual[i] or i in selected]
            self.assertEqual(variation_keep_indices(before, after, visual, budget), expected)

    def test_variation_multistage_original_positions_and_bad_inputs(self):
        original_ids = [10, 20, 30, 40, 50, 60]
        before = [[0.]] * 6
        after = [[0.], [2.], [8.], [3.], [9.], [0.]]
        mask = [False, True, True, True, True, False]
        first = variation_keep_indices(before, after, mask, 3)
        ids = [original_ids[i] for i in first]
        next_before = [after[i] for i in first]
        next_after = [[a[0] + change] for a, change in zip(next_before, [0, 5, 7, 1, 0])]
        second = variation_keep_indices(next_before, next_after, [mask[i] for i in first], 1)
        self.assertEqual([ids[i] for i in second], [10, 40, 60])
        for bad_budget in (-1, 5):
            with self.assertRaises(ValueError):
                variation_keep_indices(before, after, mask, bad_budget)
        with self.assertRaises(TypeError):
            variation_keep_indices(before, after, mask, True)
        with self.assertRaises(ValueError):
            variation_keep_indices([[0.]], [[math.inf]], [True], 0)
        with self.assertRaises(ValueError):
            variation_keep_indices(before, after, [1] * 6, 2)
        with self.assertRaises(ValueError):
            variation_keep_indices([[0., 1.]], [[2.]], [True], 1)

    def test_float_sqrt_against_math_extreme_scales_and_random_values(self):
        import sys
        cases = [0., 1., 2., 4., 0.01, 1e-300, 1e300, sys.float_info.max,
                 sys.float_info.min, math.ulp(0.0), 3 * math.ulp(0.0)]
        rng = random.Random(34)
        cases += [math.ldexp(rng.uniform(0.5, 1), rng.randrange(-1073, 1024)) for _ in range(400)]
        for value in cases:
            result = float_sqrt(value)
            expected = math.sqrt(value)
            self.assertTrue(math.isfinite(result))
            self.assertTrue(math.isclose(result, expected, rel_tol=1.1e-12, abs_tol=0.), (value, result, expected))
        self.assertLessEqual(abs(float_sqrt(2, abs_tol=1e-4, rel_tol=0) - math.sqrt(2)), 1e-4)
        self.assertLessEqual(abs(float_sqrt(2, rel_tol=1e-25) - math.sqrt(2)), math.ulp(math.sqrt(2)))

    def test_float_sqrt_validation_and_iteration_limit(self):
        for value in (-1., math.nan, math.inf, 10**1000):
            with self.assertRaises(ValueError):
                float_sqrt(value)
        for value in (True, '2', 2j):
            with self.assertRaises(TypeError):
                float_sqrt(value)
        for kwargs in ({'rel_tol': -1}, {'rel_tol': math.nan}, {'abs_tol': math.inf},
                       {'abs_tol': 0, 'rel_tol': 0}, {'max_iterations': 0}):
            with self.assertRaises(ValueError):
                float_sqrt(2., **kwargs)
        with self.assertRaises(RuntimeError):
            float_sqrt(2, max_iterations=1)

    def test_longest_matrix_path_against_exhaustive_small_dfs(self):
        def brute(matrix):
            if not matrix or not matrix[0]:
                return 0
            rows, columns = len(matrix), len(matrix[0])
            def visit(row, col):
                candidates = [1]
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < rows and 0 <= nc < columns and matrix[nr][nc] > matrix[row][col]:
                        candidates.append(1 + visit(nr, nc))
                return max(candidates)
            return max(visit(r, c) for r in range(rows) for c in range(columns))
        self.assertEqual(longest_increasing_matrix_path([]), 0)
        self.assertEqual(longest_increasing_matrix_path([[], []]), 0)
        self.assertEqual(longest_increasing_matrix_path([[1, 1], [1, 1]]), 1)
        self.assertEqual(longest_increasing_matrix_path([[9, 9, 4], [6, 6, 8], [2, 1, 1]]), 4)
        self.assertEqual(longest_increasing_matrix_path([list(range(2000))]), 2000)
        rng = random.Random(35)
        for _ in range(250):
            matrix = [[rng.randrange(-3, 5) for _ in range(rng.randrange(1, 4))]]
            columns = len(matrix[0])
            matrix += [[rng.randrange(-3, 5) for _ in range(columns)] for _ in range(rng.randrange(3))]
            original = [row.copy() for row in matrix]
            self.assertEqual(longest_increasing_matrix_path(matrix), brute(matrix))
            self.assertEqual(matrix, original)
        with self.assertRaises(ValueError):
            longest_increasing_matrix_path([[1], [2, 3]])
        with self.assertRaises(ValueError):
            longest_increasing_matrix_path([[math.nan]])


    def test_moe_router_closed_form_normalization_and_stable_ties(self):
        row = [math.log(value) for value in (1, 2, 4, 8)]
        for offset in (0., 1000.):
            indices, weights = moe_top_k_router([[value + offset for value in row]], 2)
            self.assertEqual(indices, [[3, 2]])
            for actual, expected in zip(weights[0], (2 / 3, 1 / 3)):
                self.assertAlmostEqual(actual, expected)
        indices, weights = moe_top_k_router([row], 2, renormalize=False)
        for actual, expected in zip(weights[0], (8 / 15, 4 / 15)):
            self.assertAlmostEqual(actual, expected)
        self.assertLess(sum(weights[0]), 1.)
        self.assertEqual(moe_top_k_router([[0., 0., 0.]], 2), ([[0, 1]], [[0.5, 0.5]]))
        self.assertEqual(moe_top_k_router([[5., -10.]], 1)[1], [[1.]])
        rng = random.Random(39)
        for _ in range(100):
            row = [rng.randrange(-5, 6) for _ in range(rng.randrange(1, 10))]
            k = rng.randrange(1, len(row) + 1)
            expected = sorted(range(len(row)), key=lambda e: (-row[e], e))[:k]
            actual, weights = moe_top_k_router([row], k)
            self.assertEqual(actual[0], expected)
            exp_values = [math.exp(row[e]) for e in expected]
            for weight, value in zip(weights[0], exp_values):
                self.assertAlmostEqual(weight, value / sum(exp_values))

    def test_moe_router_invalid_k_nonfinite_and_shapes(self):
        for k in (0, -1, 3, True, 1.5):
            with self.assertRaises(ValueError):
                moe_top_k_router([[0., 1.]], k)
        for bad in ([], [[]], [[1.], [2., 3.]], [[math.inf]], [[math.nan]]):
            with self.assertRaises(ValueError):
                moe_top_k_router(bad, 1)
        with self.assertRaises(TypeError):
            moe_top_k_router([[0., 1.]], 1, renormalize=1)


    def test_layernorm_population_variance_epsilon_affine_and_leading_dims(self):
        x = [[1., 3.], [4., 4.]]
        actual = layer_norm_last_dim(x, gamma=[2., 3.], beta=[0.5, -0.25], epsilon=1.)
        expected = [[0.5 - math.sqrt(2), -0.25 + 3 / math.sqrt(2)], [0.5, -0.25]]
        for row, target in zip(actual, expected):
            for value, goal in zip(row, target):
                self.assertAlmostEqual(value, goal)
        self.assertEqual(layer_norm_last_dim([[[2., 2.]], [[-3., -3.]]], beta=[3., 5.]),
                         [[[3., 5.]], [[3., 5.]]])
        self.assertEqual(layer_norm_last_dim([8.]), [0.])
        from statistics import mean, pvariance
        rng = random.Random(40)
        for _ in range(100):
            row = [rng.uniform(-5, 5) for _ in range(rng.randrange(1, 10))]
            expected = [(value - mean(row)) / math.sqrt(pvariance(row) + 0.1) for value in row]
            for value, goal in zip(layer_norm_last_dim(row, epsilon=0.1), expected):
                self.assertAlmostEqual(value, goal)
            for value, shifted in zip(layer_norm_last_dim(row, epsilon=0.1),
                                      layer_norm_last_dim([v + 1000 for v in row], epsilon=0.1)):
                self.assertAlmostEqual(value, shifted)

    def test_layernorm_invalid_shape_statistics_and_parameters(self):
        for x in ([], [[]], [[1], [2, 3]], [[math.nan]]):
            with self.assertRaises(ValueError):
                layer_norm_last_dim(x)
        for kwargs in ({'epsilon': 0}, {'epsilon': math.inf}, {'gamma': [1.]},
                       {'beta': [0., math.nan]}):
            with self.assertRaises(ValueError):
                layer_norm_last_dim([1., 3.], **kwargs)
        with self.assertRaises(ValueError):
            layer_norm_last_dim([1e308, -1e308])


if __name__ == "__main__":
    unittest.main()
