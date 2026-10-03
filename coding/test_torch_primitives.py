import unittest
try:
    import torch
    import torch_primitives as ops
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "optional PyTorch is not installed")
class TorchTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
