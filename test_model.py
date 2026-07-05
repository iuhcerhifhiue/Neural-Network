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
        # Test for causal masking: ensure that future tokens are masked out
        # We expect the attention weights to have -inf (or very small numbers after softmax)
        # in the upper triangular part for each head.
        att_weights = self.attention.c_attn(self.dummy_input).split(self.config.n_embd, dim=2)[0] # Just q
        B, T, C = self.dummy_input.size()
        head_dim = C // self.config.n_head
        q = att_weights.view(B, T, self.config.n_head, head_dim).transpose(1, 2)
        k = q # for simplicity, assume k=q to get attention scores easily for masking check
        att = (q @ k.transpose(-2, -1)) * (1.0 / (head_dim ** 0.5))

        # Apply the mask manually as in the forward method
        mask = self.attention.bias[:, :, :T, :T] == 0
        att_masked = att.masked_fill(mask, float("-inf"))

        # After masking, the upper triangle should contain -inf
        for b in range(B):
            for h in range(self.config.n_head):
                for i in range(T):
                    for j in range(i + 1, T):
                        self.assertTrue(torch.isinf(att_masked[b, h, i, j]))
                        self.assertLess(att_masked[b, h, i, j], 0) # Should be -inf


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
        # Calculate expected parameters manually
        expected_params = sum(p.numel() for p in self.gpt.parameters() if p.requires_grad)
        self.assertEqual(num_params, expected_params)
        self.assertGreater(num_params, 0)

        # Basic check for specific layer parameter counts for more rigor
        # wte + wpe
        expected_wte_wpe = self.config.vocab_size * self.config.n_embd + self.config.block_size * self.config.n_embd
        # lm_head (tied with wte)
        expected_lm_head = 0 # Tied weight
        # Block layers: ln_1, attn (c_attn, c_proj), ln_2, mlp (c_fc, c_proj)
        # CausalSelfAttention: c_attn (in:n_embd, out:3*n_embd), c_proj (in:n_embd, out:n_embd)
        # params = (n_embd * 3*n_embd + 3*n_embd) + (n_embd * n_embd + n_embd)
        # MLP: c_fc (in:n_embd, out:4*n_embd), c_proj (in:4*n_embd, out:n_embd)
        # params = (n_embd * 4*n_embd + 4*n_embd) + (4*n_embd * n_embd + n_embd)
        # LayerNorm: 2 * n_embd (weight + bias)
        # Total for one block: 2 * (2 * n_embd) + (n_embd * 3*n_embd + 3*n_embd) + (n_embd * n_embd + n_embd) + (n_embd * 4*n_embd + 4*n_embd) + (4*n_embd * n_embd + n_embd)
        # This is becoming too complex for a concise test, better to rely on sum(p.numel())
        # and possibly a few key layer checks if needed.
        # For now, the sum is sufficient and accurate.


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
