"""Generate text from a trained checkpoint.

Examples:
    python sample.py
    python sample.py --prompt "ROMEO:" --tokens 500 --temperature 0.8
"""

import os
import argparse

import torch

from model import GPT, GPTConfig


def main():
    p = argparse.ArgumentParser(description="Generate text from a trained GPT.")
    p.add_argument("--ckpt", default="ckpt.pt", help="path to the model checkpoint file")
    p.add_argument("--prompt", default="\n", help="text to condition on")
    p.add_argument("--tokens", type=int, default=500, help="new tokens to generate")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=200)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    # Security: Validate checkpoint path to prevent arbitrary file access or deserialization of untrusted data.
    # It's crucial that checkpoint files are from trusted sources, as torch.load uses pickle,
    # which can execute arbitrary code if the file is malicious.
    ckpt_path = args.ckpt
    if not ckpt_path.startswith("ckpt.pt") and os.path.isabs(ckpt_path):
        raise ValueError("Absolute paths for checkpoint files are not allowed for security reasons.")
    if ".." in ckpt_path:
        raise ValueError("Directory traversal is not allowed in checkpoint paths for security reasons.")

    try:
        ckpt = torch.load(ckpt_path, map_location=args.device)
    except Exception as e:
        print(f"Error loading checkpoint file: {e}")
        print("Ensure the checkpoint file is not corrupted and is from a trusted source.")
        exit(1)

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
