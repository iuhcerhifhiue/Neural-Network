"""Generate text from a trained checkpoint.

Examples:
    python sample.py
    python sample.py --prompt "ROMEO:" --tokens 500 --temperature 0.8
"""

import argparse

import torch

from model import GPT, GPTConfig


def main():
    p = argparse.ArgumentParser(description="Generate text from a trained GPT.")
    p.add_argument("--ckpt", default="ckpt.pt")
    p.add_argument("--prompt", default="\n", help="text to condition on")
    p.add_argument("--tokens", type=int, default=500, help="new tokens to generate")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=200)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device)
    cfg = GPTConfig(**ckpt["config"])
    stoi, itos = ckpt["stoi"], ckpt["itos"]

    model = GPT(cfg).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join(itos[i] for i in ids)

    start = args.prompt if args.prompt else "\n"
    idx = torch.tensor([encode(start)], dtype=torch.long, device=args.device)
    out = model.generate(
        idx, max_new_tokens=args.tokens,
        temperature=args.temperature, top_k=args.top_k,
    )[0].tolist()
    print(decode(out))


if __name__ == "__main__":
    main()
