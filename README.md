# Neural Network — a GPT from scratch

A decoder-only **transformer language model** (the same architecture family
behind GPT / ChatGPT), implemented from scratch in PyTorch. It learns to
generate text one character at a time by predicting the next token, and it's
small enough to train on a laptop.

This is a *real* neural network — attention, embeddings, backprop and all — not
a wrapper around someone's API.

## What's inside

| File | Purpose |
|------|---------|
| `model.py` | The GPT: token/positional embeddings, causal multi-head self-attention, transformer blocks, and autoregressive sampling. |
| `data.py`  | Character-level tokenizer + auto-downloading text corpus (tiny Shakespeare). |
| `train.py` | Training loop with train/val loss reporting and checkpointing. |
| `sample.py`| Load a checkpoint and generate text from a prompt. |

## Quickstart

```bash
pip install -r requirements.txt

# Train (downloads the corpus on first run). CPU works; a GPU is much faster.
python train.py --iters 3000

# Generate text from your trained model
python sample.py --prompt "ROMEO:" --tokens 500 --temperature 0.8
```

On a GPU the default config (~10M parameters) reaches readable Shakespeare-style
text in a few minutes. On CPU, start smaller and shorter:

```bash
python train.py --n-layer 4 --n-embd 256 --block-size 128 --iters 2000
```

## How it works

1. **Tokenize** — each character becomes an integer id.
2. **Embed** — ids map to learned vectors, plus a positional embedding so the
   model knows token order.
3. **Attention blocks** — each token gathers information from earlier tokens via
   causal self-attention (it can't see the future), followed by a feed-forward
   MLP. Residual connections + LayerNorm keep training stable.
4. **Predict** — a final linear layer produces a probability distribution over
   the vocabulary for the next token.
5. **Generate** — sample the next token, append it, and repeat.

## Configuration

All hyperparameters are CLI flags on `train.py` — `--n-layer`, `--n-head`,
`--n-embd`, `--block-size`, `--batch-size`, `--lr`, `--dropout`, `--iters`.
Swap in your own `data/input.txt` to train on any text you like.
