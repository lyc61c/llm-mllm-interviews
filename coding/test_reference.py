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


if __name__ == "__main__":
    unittest.main()
