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


if __name__ == "__main__":
    unittest.main()
