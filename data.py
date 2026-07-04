"""Character-level tokenizer and dataset loading.

Uses the "tiny Shakespeare" corpus by default (~1MB of text). It downloads
automatically the first time you train. A character-level tokenizer keeps the
whole pipeline dependency-free and easy to read.
"""

import os
import urllib.request

import torch

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)


class CharTokenizer:
    """Maps characters <-> integer ids, built from the training text."""

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids)


def load_text(data_dir: str = "./data") -> str:
    """Return the training corpus, downloading it if necessary."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "input.txt")
    if not os.path.exists(path):
        print(f"Downloading corpus -> {path}")
        try:
            urllib.request.urlretrieve(DATA_URL, path)
        except urllib.error.URLError as e:
            print(f"Error downloading corpus: {e}")
            raise IOError(f"Failed to download corpus: {e}")
        except urllib.error.HTTPError as e:
            print(f"HTTP Error downloading corpus: {e.code} - {e.reason}")
            raise IOError(f"Failed to download corpus (HTTP {e.code}): {e.reason}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_dataset(data_dir: str = "./data", val_frac: float = 0.1):
    """Load text, build the tokenizer, and split into train/val tensors."""
    text = load_text(data_dir)
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int((1 - val_frac) * len(data))
    return tokenizer, data[:n], data[n:]


def get_batch(data, block_size: int, batch_size: int, device: str):
    """Sample a random batch of (input, target) sequences.

    Targets are the inputs shifted one position to the right, i.e. for each
    position the model learns to predict the next character.
    """
    if len(data) <= block_size:
        raise ValueError(
            f"Data length ({len(data)}) must be greater than block_size "
            f"({block_size}) to sample a batch."
        )
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)
