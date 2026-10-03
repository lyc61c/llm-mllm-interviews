import unittest
try:
    import torch
    import torch_primitives as ops
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "optional PyTorch is not installed")
class TorchTests(unittest.TestCase):
    def test_mlp_affine_baseline_and_token_shapes(self):
        torch.manual_seed(2)
        layer = ops.MLP(4, 6, 3)
        x = torch.randn(2, 5, 4, requires_grad=True)
        hidden = x @ layer.first.weight.T + layer.first.bias
        activated = 0.5 * hidden * (1 + torch.erf(hidden / (2 ** 0.5)))
        expected = activated @ layer.second.weight.T + layer.second.bias
        torch.testing.assert_close(layer(x), expected)
        self.assertEqual(tuple(layer(x).shape), (2, 5, 3))
        layer(x).square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for parameter in layer.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        with self.assertRaises(ValueError):
            ops.MLP(0, 6, 3)

    def test_image_mlp_classification_and_shape_validation(self):
        layer = ops.ImageMLP((3, 4, 5), 7, 2)
        images = torch.randn(2, 3, 4, 5, requires_grad=True).transpose(2, 3).contiguous().transpose(2, 3)
        logits = layer(images)
        self.assertEqual(tuple(logits.shape), (2, 2))
        # Input is deliberately non-contiguous; flatten may create a copy.
        self.assertFalse(images.is_contiguous())
        torch.testing.assert_close(logits, layer.mlp(images.reshape(2, -1)))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
        loss.backward()
        self.assertTrue(torch.isfinite(layer.mlp.first.weight.grad).all())
        for invalid in (torch.randn(2, 3, 5, 4), torch.randn(3, 4, 5)):
            with self.assertRaises(ValueError):
                layer(invalid)
        with self.assertRaises(ValueError):
            ops.ImageMLP((3, 0, 5), 7, 2)

    def test_causal_future_independence_and_gradients(self):
        torch.manual_seed(1)
        layer = ops.MultiHeadAttention(8, 2).eval()
        x = torch.randn(2, 4, 8, requires_grad=True)
        output = layer(x, causal=True)
        other = x.detach().clone(); other[:, 3] += 100
        torch.testing.assert_close(output[:, :3], layer(other, causal=True)[:, :3])
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_rope_preserves_norm(self):
        x = torch.randn(2, 3, 4, 8)
        y = ops.rotary_adjacent(x, torch.arange(4))
        torch.testing.assert_close(x.square().sum(-1), y.square().sum(-1))
        with self.assertRaises(ValueError):
            ops.rotary_adjacent(x, torch.tensor([0]))

    def test_lora_merge_and_freeze(self):
        layer = ops.LoRALinear(5, 3, rank=2)
        with torch.no_grad(): layer.b.normal_()
        x = torch.randn(4,5)
        torch.testing.assert_close(layer(x), torch.nn.functional.linear(x, layer.merged_weight(), layer.base.bias))
        layer(x).sum().backward()
        self.assertIsNone(layer.base.weight.grad)
        self.assertIsNotNone(layer.a.grad)

    def test_info_nce_pairing(self):
        x = torch.eye(3)
        self.assertLess(ops.info_nce(x,x).item(), ops.info_nce(x,x.roll(1,0)).item())

    def test_response_shift_and_padding(self):
        logits = torch.zeros(1,4,5)
        ids = torch.tensor([[1,2,3,-100]])
        mask = torch.tensor([[False,False,True,False]])
        torch.testing.assert_close(ops.response_logprobs(logits,ids,mask), torch.tensor([-torch.log(torch.tensor(5.)).item()]))

    def test_dpo_reference_detached(self):
        pc = torch.tensor([-1.], requires_grad=True)
        pr = torch.tensor([-3.], requires_grad=True)
        ref = torch.tensor([-2.], requires_grad=True)
        loss = ops.dpo_loss(pc,pr,ref,ref)
        loss.backward()
        self.assertLess(pc.grad.item(),0)
        self.assertGreater(pr.grad.item(),0)
        self.assertIsNone(ref.grad)

    def test_vae_reparameterization_gradients(self):
        mean = torch.tensor([[0.5, -0.2]], dtype=torch.float64, requires_grad=True)
        logvar = torch.tensor([[0.0, 1.0]], dtype=torch.float64, requires_grad=True)
        noise = torch.tensor([[2.0, -3.0]], dtype=torch.float64)
        sampled = ops.vae_reparameterize(mean, logvar, noise)
        torch.testing.assert_close(sampled, mean + noise * (0.5 * logvar).exp())
        sampled.sum().backward()
        torch.testing.assert_close(mean.grad, torch.ones_like(mean))
        torch.testing.assert_close(logvar.grad, 0.5 * noise * (0.5 * logvar.detach()).exp())
        with self.assertRaises(ValueError):
            ops.vae_reparameterize(mean, logvar, torch.zeros(1, 3))

    def test_vae_elbo_against_distributions_and_reduction(self):
        logits = torch.tensor([[0.2, -0.8, 0.0], [-2.0, 1.3, 0.4]], dtype=torch.float64, requires_grad=True)
        target = torch.tensor([[0., 1., 0.], [1., 1., 0.]], dtype=torch.float64)
        mean = torch.tensor([[0.3, -0.5], [0., 1.]], dtype=torch.float64, requires_grad=True)
        logvar = torch.tensor([[0.1, -0.4], [0., 0.5]], dtype=torch.float64, requires_grad=True)
        total, reconstruction, kl = ops.vae_loss(logits, target, mean, logvar)
        expected_rec = -torch.distributions.Bernoulli(logits=logits).log_prob(target).sum(1).mean()
        posterior = torch.distributions.Normal(mean, (0.5 * logvar).exp())
        prior = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(mean))
        expected_kl = torch.distributions.kl_divergence(posterior, prior).sum(1).mean()
        torch.testing.assert_close(reconstruction, expected_rec)
        torch.testing.assert_close(kl, expected_kl)
        torch.testing.assert_close(total, expected_rec + expected_kl)
        duplicated = [value.repeat(2, 1) for value in (logits, target, mean, logvar)]
        torch.testing.assert_close(ops.vae_loss(*duplicated)[0], total)
        torch.testing.assert_close(ops.vae_loss(logits, target, mean, logvar, beta=2)[0], total + kl)
        total.backward()
        for value in (logits, mean, logvar):
            self.assertTrue(torch.isfinite(value.grad).all())
        gaussian = ops.vae_loss(logits.detach(), target, mean.detach(), logvar.detach(), likelihood='unit_gaussian')
        expected_gaussian = -torch.distributions.Normal(logits.detach(), 1.).log_prob(target).sum(1).mean()
        torch.testing.assert_close(gaussian[1], expected_gaussian)
        with self.assertRaises(ValueError):
            ops.vae_loss(logits, target + 2, mean, logvar)
        with self.assertRaises(ValueError):
            ops.vae_loss(logits, target, mean, logvar, beta=-1)

    def test_bucket_batch_sampler_dataloader_contract(self):
        from reference import BucketBatchSampler
        dataset = [torch.arange(n) for n in (5, 1, 3, 2, 4)]
        sampler = BucketBatchSampler([len(value) for value in dataset], 2, shuffle=False)
        def collate(samples):
            lengths = torch.tensor([len(value) for value in samples])
            return torch.nn.utils.rnn.pad_sequence(samples, batch_first=True, padding_value=-1), lengths
        loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, collate_fn=collate)
        batches = list(loader)
        self.assertEqual(len(batches), len(sampler))
        self.assertEqual(sorted(length for _, lengths in batches for length in lengths.tolist()), [1, 2, 3, 4, 5])
        for values, lengths in batches:
            self.assertEqual(values.shape[1], lengths.max().item())
            for row, length in zip(values, lengths):
                length = int(length.item())
                torch.testing.assert_close(row[:length], torch.arange(length))
                self.assertTrue((row[length:] == -1).all())

    def test_infonce_temperature_scale_invariance_and_singleton_gradient(self):
        import math
        q = torch.eye(2, requires_grad=True)
        k = torch.tensor([[8., 0.], [0., 0.25]], requires_grad=True)
        for temperature in (0.2, 1., 3.):
            expected = math.log1p(math.exp(-1. / temperature))
            self.assertAlmostEqual(ops.info_nce(q, k, temperature).item(), expected, places=6)
            self.assertAlmostEqual(ops.info_nce(q * torch.tensor([[7.], [11.]]), k, temperature).item(), expected, places=6)
            wrong = 1. / temperature + expected
            self.assertAlmostEqual(ops.info_nce(q, k.flip(0), temperature).item(), wrong, places=6)
        ops.info_nce(q, k, symmetric=True).backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertTrue(torch.isfinite(k.grad).all())
        single_q = torch.tensor([[1., 2.]], requires_grad=True)
        single_k = torch.tensor([[3., 4.]], requires_grad=True)
        loss = ops.info_nce(single_q, single_k)
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        torch.testing.assert_close(single_q.grad, torch.zeros_like(single_q))
        torch.testing.assert_close(single_k.grad, torch.zeros_like(single_k))


    def test_gqa_independent_heads_forward_and_gradients(self):
        torch.manual_seed(36)
        for hq, hkv in ((6, 2), (4, 1), (3, 3)):
            q = torch.randn(2, hq, 2, 3, dtype=torch.float64, requires_grad=True)
            k = torch.randn(2, hkv, 4, 3, dtype=torch.float64, requires_grad=True)
            v = torch.randn(2, hkv, 4, 3, dtype=torch.float64, requires_grad=True)
            actual = ops.grouped_query_attention(q, k, v)
            # Build a separate graph per query head, without any repeated KV tensor.
            expected = torch.stack([torch.stack([
                ((q[b, h] @ k[b, h // (hq // hkv)].T) / (3 ** 0.5)).softmax(-1)
                @ v[b, h // (hq // hkv)] for h in range(hq)]) for b in range(2)])
            torch.testing.assert_close(actual, expected)
            weights = torch.randn_like(actual)
            gradients = torch.autograd.grad((actual * weights).sum(), (q, k, v), retain_graph=True)
            expected_gradients = torch.autograd.grad((expected * weights).sum(), (q, k, v))
            for gradient, target in zip(gradients, expected_gradients):
                torch.testing.assert_close(gradient, target)
                self.assertTrue(torch.isfinite(gradient).all())

    def test_gqa_masks_cache_offset_and_sdpa(self):
        torch.manual_seed(37)
        q = torch.randn(2, 4, 2, 3)
        k = torch.randn(2, 2, 4, 3)
        v = torch.randn(2, 2, 4, 3)
        mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
        actual = ops.grouped_query_attention(q, k, v, causal=True, query_offset=1)
        torch.testing.assert_close(actual, ops.grouped_query_attention(q, k, v, mask))
        expected = torch.nn.functional.scaled_dot_product_attention(
            q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1), attn_mask=mask, dropout_p=0.)
        torch.testing.assert_close(actual, expected)
        per_head_mask = mask.expand(2, 4, 2, 4).clone()
        per_head_mask[0, 1, 1, 2] = False
        self.assertFalse(torch.equal(actual, ops.grouped_query_attention(q, k, v, per_head_mask)))
        with self.assertRaises(ValueError):
            ops.grouped_query_attention(q, k, v, torch.zeros(2, 4, dtype=torch.bool))
        with self.assertRaises(ValueError):
            ops.grouped_query_attention(q, k, v, torch.ones(2, 4))
        with self.assertRaises(ValueError):
            ops.grouped_query_attention(q[:, :3], k, v)
        with self.assertRaises(ValueError):
            ops.grouped_query_attention(q, k, v, torch.ones(3, 5, dtype=torch.bool))

    def test_variation_indices_against_squared_score_oracle(self):
        torch.manual_seed(38)
        before = torch.randn(12, 3, requires_grad=True)
        after = torch.randn(12, 3, requires_grad=True)
        mask = torch.tensor([False, True, True, False, True, True, True, False, True, False, True, False])
        # Squared Euclidean distance has the same order, independent of vector_norm.
        squares = (after.detach() - before.detach()).square().sum(-1).tolist()
        visual = [i for i, flag in enumerate(mask.tolist()) if flag]
        for budget in (0, 1, 4, len(visual)):
            selected = set(sorted(visual, key=lambda i: (-squares[i], i))[:budget])
            expected = [i for i, flag in enumerate(mask.tolist()) if not flag or i in selected]
            actual = ops.variation_keep_indices(before, after, mask, budget)
            self.assertEqual(actual.tolist(), expected)
            self.assertEqual(actual.dtype, torch.int64)
            self.assertFalse(actual.requires_grad)
        self.assertIsNone(before.grad)
        self.assertIsNone(after.grad)
        tied = torch.tensor([[0., 0.], [3., 4.], [5., 0.], [4., 3.], [0., 0.]])
        self.assertEqual(ops.variation_keep_indices(torch.zeros_like(tied), tied,
                         torch.tensor([False, True, True, True, False]), 1).tolist(), [0, 1, 4])

    def test_variation_multistage_gather_empty_and_validation(self):
        before = torch.zeros(6, 1)
        after = torch.tensor([[0.], [2.], [8.], [3.], [9.], [0.]])
        mask = torch.tensor([False, True, True, True, True, False])
        original_positions = torch.tensor([10, 20, 30, 40, 50, 60])
        first = ops.variation_keep_indices(before, after, mask, 3)
        next_before = after[first]
        next_after = next_before + torch.tensor([[0.], [5.], [7.], [1.], [0.]])
        second = ops.variation_keep_indices(next_before, next_after, mask[first], 1)
        self.assertEqual(original_positions[first][second].tolist(), [10, 40, 60])
        self.assertEqual(ops.variation_keep_indices(torch.empty(0, 2), torch.empty(0, 2),
                         torch.empty(0, dtype=torch.bool), 0).numel(), 0)
        with self.assertRaises(ValueError):
            ops.variation_keep_indices(before, after, mask, 5)
        with self.assertRaises(TypeError):
            ops.variation_keep_indices(before, after, mask, True)
        with self.assertRaises(TypeError):
            ops.variation_keep_indices(before.long(), after, mask, 2)
        with self.assertRaises(ValueError):
            ops.variation_keep_indices(before, after, mask.float(), 2)
        nonfinite = after.clone()
        nonfinite[1] = float('inf')
        with self.assertRaises(ValueError):
            ops.variation_keep_indices(before, nonfinite, mask, 0)


    def test_moe_router_analytic_weights_and_selected_gradient(self):
        import math
        logits = torch.tensor([[0., math.log(2), math.log(4), math.log(8)]], dtype=torch.float64, requires_grad=True)
        ids, weights = ops.moe_top_k_router(logits, 2)
        self.assertEqual(ids.tolist(), [[3, 2]])
        torch.testing.assert_close(weights, torch.tensor([[2 / 3, 1 / 3]], dtype=torch.float64))
        loss = 2 * weights[0, 0] - weights[0, 1]
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.tensor([[0., 0., -2 / 3, 2 / 3]], dtype=torch.float64))
        self.assertFalse(ids.requires_grad)
        _, raw_weights = ops.moe_top_k_router(logits.detach(), 2, renormalize=False)
        torch.testing.assert_close(raw_weights, torch.tensor([[8 / 15, 4 / 15]], dtype=torch.float64))
        singleton = torch.tensor([[1., 3., 2.]], requires_grad=True)
        _, gate = ops.moe_top_k_router(singleton, 1, renormalize=True)
        gate.sum().backward()
        torch.testing.assert_close(singleton.grad, torch.zeros_like(singleton))

    def test_moe_router_ties_translation_and_validation(self):
        logits = torch.tensor([[0., 0., 0.], [-1000., 1., 2.]])
        ids, weights = ops.moe_top_k_router(logits, 2)
        shifted_ids, shifted_weights = ops.moe_top_k_router(logits + 1000, 2)
        self.assertEqual(ids[0].tolist(), [0, 1])
        torch.testing.assert_close(ids, shifted_ids)
        torch.testing.assert_close(weights, shifted_weights)
        for k in (0, 4, True):
            with self.assertRaises(ValueError):
                ops.moe_top_k_router(logits, k)
        with self.assertRaises(ValueError):
            ops.moe_top_k_router(torch.tensor([[float('nan')]]), 1)
        with self.assertRaises(TypeError):
            ops.moe_top_k_router(logits.long(), 1)


    def test_layernorm_functional_baseline_affine_and_gradients(self):
        torch.manual_seed(41)
        x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
        gamma = torch.randn(4, dtype=torch.float64, requires_grad=True)
        beta = torch.randn(4, dtype=torch.float64, requires_grad=True)
        actual = ops.layer_norm_last_dim(x, gamma, beta, epsilon=0.2)
        expected = torch.nn.functional.layer_norm(x, (4,), gamma, beta, eps=0.2)
        torch.testing.assert_close(actual, expected)
        weights = torch.randn_like(x)
        actual_grads = torch.autograd.grad((actual * weights).sum(), (x, gamma, beta), retain_graph=True)
        expected_grads = torch.autograd.grad((expected * weights).sum(), (x, gamma, beta))
        for gradient, target in zip(actual_grads, expected_grads):
            torch.testing.assert_close(gradient, target)
        constant = torch.full((2, 3, 4), 5.)
        torch.testing.assert_close(ops.layer_norm_last_dim(constant, beta=torch.arange(4.)),
                                  torch.arange(4.).expand_as(constant))
        half = constant.to(torch.bfloat16)
        self.assertEqual(ops.layer_norm_last_dim(half).dtype, torch.bfloat16)

    def test_layernorm_population_variance_and_validation(self):
        x = torch.tensor([1., 3.], dtype=torch.float64)
        expected = torch.tensor([-1., 1.], dtype=torch.float64) / (2 ** 0.5)
        torch.testing.assert_close(ops.layer_norm_last_dim(x, epsilon=1.), expected)
        for bad in (torch.empty(0, 2), torch.tensor([float('inf')]), torch.ones(2, dtype=torch.long)):
            with self.assertRaises(ValueError):
                ops.layer_norm_last_dim(bad)
        with self.assertRaises(ValueError):
            ops.layer_norm_last_dim(x, gamma=torch.ones(3))
        with self.assertRaises(ValueError):
            ops.layer_norm_last_dim(x, epsilon=-1.)


if __name__ == "__main__":
    unittest.main()
