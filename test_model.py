import unittest
import torch
import torch.nn as nn
from model import GPTConfig, CausalSelfAttention, MLP, Block, GPT

class TestGPTConfig(unittest.TestCase):
    def test_config_defaults(self):
        config = GPTConfig()
        self.assertEqual(config.vocab_size, 65)
        self.assertEqual(config.block_size, 256)
        self.assertEqual(config.n_layer, 6)
        self.assertEqual(config.n_head, 6)
        self.assertEqual(config.n_embd, 384)
        self.assertEqual(config.dropout, 0.2)

    def test_config_custom_values(self):
        config = GPTConfig(vocab_size=100, block_size=128, n_layer=3, n_head=4, n_embd=256, dropout=0.1)
        self.assertEqual(config.vocab_size, 100)
        self.assertEqual(config.block_size, 128)
        self.assertEqual(config.n_layer, 3)
        self.assertEqual(config.n_head, 4)
        self.assertEqual(config.n_embd, 256)
        self.assertEqual(config.dropout, 0.1)

class TestCausalSelfAttention(unittest.TestCase):
    def setUp(self):
        self.config = GPTConfig(n_embd=64, n_head=4, block_size=16)
        self.attn = CausalSelfAttention(self.config)

    def test_initialization(self):
        self.assertIsInstance(self.attn.c_attn, nn.Linear)
        self.assertIsInstance(self.attn.c_proj, nn.Linear)
        self.assertEqual(self.attn.n_head, 4)

    def test_forward_pass_shape(self):
        B, T, C = 2, 8, self.config.n_embd
        x = torch.randn(B, T, C)
        output = self.attn(x)
        self.assertEqual(output.shape, (B, T, C))

    def test_causal_masking(self):
        # This test is more conceptual, ensuring the mask is applied.
        # Directly testing the effect of the mask would require mocking or deep inspection.
        # We can check that the bias is correctly registered and its shape.
        self.assertTrue(hasattr(self.attn, 'bias'))
        self.assertEqual(self.attn.bias.shape, (1, 1, self.config.block_size, self.config.block_size))
        self.assertTrue(torch.all(self.attn.bias.squeeze()[torch.triu_indices(self.config.block_size, self.config.block_size, offset=1)[0], torch.triu_indices(self.config.block_size, self.config.block_size, offset=1)[1]] == 0))

class TestMLP(unittest.TestCase):
    def setUp(self):
        self.config = GPTConfig(n_embd=64)
        self.mlp = MLP(self.config)

    def test_initialization(self):
        self.assertIsInstance(self.mlp.c_fc, nn.Linear)
        self.assertIsInstance(self.mlp.c_proj, nn.Linear)

    def test_forward_pass_shape(self):
        B, T, C = 2, 8, self.config.n_embd
        x = torch.randn(B, T, C)
        output = self.mlp(x)
        self.assertEqual(output.shape, (B, T, C))

class TestBlock(unittest.TestCase):
    def setUp(self):
        self.config = GPTConfig(n_embd=64, n_head=4, block_size=16)
        self.block = Block(self.config)

    def test_initialization(self):
        self.assertIsInstance(self.block.ln_1, nn.LayerNorm)
        self.assertIsInstance(self.block.attn, CausalSelfAttention)
        self.assertIsInstance(self.block.ln_2, nn.LayerNorm)
        self.assertIsInstance(self.block.mlp, MLP)

    def test_forward_pass_shape(self):
        B, T, C = 2, 8, self.config.n_embd
        x = torch.randn(B, T, C)
        output = self.block(x)
        self.assertEqual(output.shape, (B, T, C))

class TestGPT(unittest.TestCase):
    def setUp(self):
        self.config = GPTConfig(vocab_size=100, block_size=16, n_layer=2, n_head=4, n_embd=64)
        self.gpt = GPT(self.config)

    def test_initialization_and_params(self):
        self.assertIsInstance(self.gpt.transformer.wte, nn.Embedding)
        self.assertIsInstance(self.gpt.transformer.wpe, nn.Embedding)
        self.assertIsInstance(self.gpt.lm_head, nn.Linear)
        self.assertTrue(self.gpt.num_params() > 0)
        self.assertEqual(self.gpt.transformer.wte.weight.data_ptr(), self.gpt.lm_head.weight.data_ptr())

    def test_forward_pass_shape_no_targets(self):
        B, T = 2, 8
        idx = torch.randint(0, self.config.vocab_size, (B, T))
        logits, loss = self.gpt(idx)
        self.assertEqual(logits.shape, (B, T, self.config.vocab_size))
        self.assertIsNone(loss)

    def test_forward_pass_shape_with_targets(self):
        B, T = 2, 8
        idx = torch.randint(0, self.config.vocab_size, (B, T))
        targets = torch.randint(0, self.config.vocab_size, (B, T))
        logits, loss = self.gpt(idx, targets)
        self.assertEqual(logits.shape, (B, T, self.config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, torch.Tensor)

    def test_forward_pass_block_size_assertion(self):
        B, T = 2, self.config.block_size + 1
        idx = torch.randint(0, self.config.vocab_size, (B, T))
        with self.assertRaisesRegex(AssertionError, "sequence longer than block_size"):
            self.gpt(idx)

    def test_generate_output_shape(self):
        B, T = 2, 5
        idx = torch.randint(0, self.config.vocab_size, (B, T))
        max_new_tokens = 10
        output_idx = self.gpt.generate(idx, max_new_tokens)
        self.assertEqual(output_idx.shape, (B, T + max_new_tokens))

    def test_generate_with_top_k(self):
        B, T = 2, 5
        idx = torch.randint(0, self.config.vocab_size, (B, T))
        max_new_tokens = 3
        top_k = 5
        output_idx = self.gpt.generate(idx, max_new_tokens, top_k=top_k)
        self.assertEqual(output_idx.shape, (B, T + max_new_tokens))

    def test_generate_temperature_effect(self):
        # This test ensures temperature doesn't break generation, 
        # but doesn't test the statistical effect of temperature rigorously.
        B, T = 1, 5
        idx = torch.randint(0, self.config.vocab_size, (B, T))
        max_new_tokens = 2
        temp_low = self.gpt.generate(idx, max_new_tokens, temperature=0.1)
        temp_high = self.gpt.generate(idx, max_new_tokens, temperature=10.0)
        self.assertEqual(temp_low.shape, (B, T + max_new_tokens))
        self.assertEqual(temp_high.shape, (B, T + max_new_tokens))

if __name__ == '__main__':
    unittest.main()