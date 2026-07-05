# Neural Network

This repo contains **two related projects**, built to explore the honest
question: *how close to a "real AI chatbot" can you get on a normal laptop?*

1. **A GPT from scratch** (`model.py` etc.) — a real transformer, trained from
   nothing. Great for learning how these models actually work, but at a few
   million parameters it can only ever be a narrow specialist.
2. **A working local chatbot** (`chat.py` etc.) — a genuinely conversational
   assistant with live fact-lookup, by standing on top of an already-trained
   open model instead of training one from scratch.

The two together tell the real story: training your own model teaches you the
mechanics; getting ChatGPT-like behavior requires a model trained at a scale no
laptop can match, so you *use* one rather than *build* one.

---

## Project 1 — a GPT from scratch

A decoder-only **transformer language model** (the same architecture family
behind GPT / ChatGPT), implemented from scratch in PyTorch. It learns to
generate text one character at a time by predicting the next token, and it's
small enough to train on a laptop.

This is a *real* neural network — attention, embeddings, backprop and all — not
a wrapper around someone's API.

### What's inside

| File | Purpose |
|------|---------|
| `model.py` | The GPT: token/positional embeddings, causal multi-head self-attention, transformer blocks, and autoregressive sampling. |
| `data.py`  | Character-level tokenizer + auto-downloading text corpus (tiny Shakespeare). |
| `train.py` | Training loop with train/val loss reporting and checkpointing. Supports `--dataset shakespeare` or `--dataset math`. |
| `sample.py`| Load a checkpoint and generate text from a prompt. |
| `math_data.py` | Generates a synthetic 5th-grade math corpus (addition, subtraction, fractions) to train the model as an arithmetic solver. |
| `eval_math.py` | Measures exact-match accuracy of a math-trained model on held-out problems. |

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

### Configuration

All hyperparameters are CLI flags on `train.py` — `--n-layer`, `--n-head`,
`--n-embd`, `--block-size`, `--batch-size`, `--lr`, `--dropout`, `--iters`.
Swap in your own `data/input.txt` to train on any text you like.

Train it on math instead of Shakespeare:

```bash
python train.py --dataset math --iters 3500 --save ckpt_math.pt
python eval_math.py --ckpt ckpt_math.pt        # measure accuracy
```

---

## Project 2 — a real local chatbot (`chat.py`)

A from-scratch model can't hold a general conversation — that needs a model
trained on far more data than a laptop can handle. So this part takes the
practical route: it uses an **already-trained open model** (Llama 3.2, run
locally and free via [Ollama](https://ollama.com)) as the conversational brain,
and gives it a **tool** for live facts. This is the same pattern real
assistants use when they "search the web": the language model decides when to
call a tool, then phrases a natural answer from the result.

| File | Purpose |
|------|---------|
| `chat.py` | The chatbot: local LLM + conversation memory + tool-calling, with a deterministic gate that decides when a fact lookup is actually needed. |
| `ask.py` | The fact-lookup tool: real answers from the World Bank API (GDP, population, …), Wikipedia summaries, and DuckDuckGo. No API key needed. |
| `Modelfile` | Builds a customized model (`oikos-chat`) on top of Llama 3.2 — a baked-in persona, tuned sampling, and few-shot examples. Not gradient fine-tuning (that needs a GPU), but real behavior shaping. |

### Setup

```bash
# 1. Install Ollama from https://ollama.com, then pull the base model:
ollama pull llama3.2

# 2. Build the polished custom model:
ollama create oikos-chat -f Modelfile

# 3. Chat:
python chat.py
```

Ask it anything — it chats normally, does arithmetic itself, writes a haiku on
request, and looks up real cited facts ("what is the GDP of China?") only when
a question actually needs one.

The fact lookup also works standalone:

```bash
python ask.py "what is the population of japan"
```

### Honest limitations

This is a genuinely conversational assistant, but it is **not** ChatGPT/Claude
quality — the local 3B model's fluency and reasoning are well below frontier
models. That gap comes from training scale (frontier models cost millions of
dollars in compute to train) and can't be closed on a personal machine. What
*is* solved here is reliable behavior: knowing when to look things up, doing
math directly, and staying conversational.
