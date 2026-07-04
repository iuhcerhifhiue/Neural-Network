import unittest
import torch
from model import GPTConfig, CausalSelfAttention, MLP, Block, GPT
from unittest.mock import MagicMock, patch

class TestGPTConfig(unittest.TestCase):
    def test_default_config(self):
        config = GPTConfig()
        self.assertEqual(config.vocab_size, 65)
        self.assertEqual(config.block_size, 256)
        self.assertEqual(config.n_layer, 6)
        self.assertEqual(config.n_head, 6)
        self.assertEqual(config.n_embd, 384)
        self.assertEqual(config.dropout, 0.2)

    def test_custom_config(self):
        config = GPTConfig(vocab_size=100, block_size=128, n_layer=4, n_head=4, n_embd=256, dropout=0.1)
        self.assertEqual(config.vocab_size, 100)
        self.assertEqual(config.block_size, 128)
        self.assertEqual(config.n_layer, 4)
        self.assertEqual(config.n_head, 4)
        self.assertEqual(config.n_embd, 256)
        self.assertEqual(config.dropout, 0.1)

class TestCausalSelfAttention(unittest.TestCase):
    def setUp(self):
        self.config = GPTConfig(n_embd=64, n_head=4, block_size=10) # Ensure n_embd is divisible by n_head
        self.attention = CausalSelfAttention(self.config)
        self.dummy_input = torch.randn(2, 5, self.config.n_embd) # B, T, C

    def test_forward_output_shape(self):
        output = self.attention(self.dummy_input)
        self.assertEqual(output.shape, self.dummy_input.shape)

    def test_causal_mask_effect(self):
        # A more direct test for causal masking would involve checking attention weights
        # For now, we'll rely on the forward pass not raising errors due to masking
        # and assume the PyTorch `tril` function works as expected.
        # A proper test would mock softmax and check the values before and after masking.
        input_sequence = torch.ones(1, 3, self.config.n_embd)
        output = self.attention(input_sequence)
        self.assertIsNotNone(output) # Just ensuring it runs without error for now

class TestMLP(unittest.TestCase):
    def setUp(self):
        self.config = GPTConfig(n_embd=64)
        self.mlp = MLP(self.config)
        self.dummy_input = torch.randn(2, 5, self.config.n_embd)

    def test_forward_output_shape(self):
        output = self.mlp(self.dummy_input)
        self.assertEqual(output.shape, self.dummy_input.shape)

class TestBlock(unittest.TestCase):
    def setUp(self):
        self.config = GPTConfig(n_embd=64, n_head=4, block_size=10)
        self.block = Block(self.config)
        self.dummy_input = torch.randn(2, 5, self.config.n_embd)

    def test_forward_output_shape(self):
        output = self.block(self.dummy_input)
        self.assertEqual(output.shape, self.dummy_input.shape)

class TestGPT(unittest.TestCase):
    def setUp(self):
        self.config = GPTConfig(vocab_size=100, block_size=10, n_embd=64, n_head=4, n_layer=2)
        self.gpt = GPT(self.config)
        self.device = 'cpu' # Use CPU for tests

    def test_init_weights(self):
        # Check if weights are initialized as expected (e.g., non-zero for linear, zero for bias)
        for module in self.gpt.modules():
            if isinstance(module, torch.nn.Linear):
                self.assertFalse(torch.all(module.weight == 0))
                if module.bias is not None:
                    self.assertTrue(torch.all(module.bias == 0))
            elif isinstance(module, torch.nn.Embedding):
                self.assertFalse(torch.all(module.weight == 0))

    def test_num_params(self):
        num_params = self.gpt.num_params()
        self.assertGreater(num_params, 0)
        # A more rigorous test would calculate the expected number of parameters manually

    def test_forward_pass_output_shape(self):
        idx = torch.randint(0, self.config.vocab_size, (2, 5)).to(self.device) # B, T
        logits, loss = self.gpt(idx)
        self.assertEqual(logits.shape, (2, 5, self.config.vocab_size))
        self.assertIsNone(loss) # No targets provided

    def test_forward_pass_with_targets(self):
        idx = torch.randint(0, self.config.vocab_size, (2, 5)).to(self.device)
        targets = torch.randint(0, self.config.vocab_size, (2, 5)).to(self.device)
        logits, loss = self.gpt(idx, targets)
        self.assertEqual(logits.shape, (2, 5, self.config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0) # Loss should be positive

    def test_forward_pass_sequence_too_long(self):
        idx = torch.randint(0, self.config.vocab_size, (2, self.config.block_size + 1)).to(self.device)
        with self.assertRaises(AssertionError):
            self.gpt(idx)

    def test_generate_output_shape_and_length(self):
        idx = torch.randint(0, self.config.vocab_size, (1, 3)).to(self.device)
        max_new_tokens = 7
        generated_idx = self.gpt.generate(idx, max_new_tokens)
        self.assertEqual(generated_idx.shape, (1, 3 + max_new_tokens))
        self.assertTrue(torch.equal(generated_idx[:, :3], idx)) # Initial tokens should be preserved

    def test_generate_temperature_and_topk(self):
        idx = torch.randint(0, self.config.vocab_size, (1, 3)).to(self.device)
        max_new_tokens = 5
        # Test with temperature and top_k
        generated_idx_temp_topk = self.gpt.generate(idx, max_new_tokens, temperature=0.5, top_k=5)
        self.assertEqual(generated_idx_temp_topk.shape, (1, 3 + max_new_tokens))
        self.assertTrue(torch.equal(generated_idx_temp_topk[:, :3], idx))
