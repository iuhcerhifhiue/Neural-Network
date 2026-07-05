"""Measure how well a trained model actually does math.

Generates fresh problems (different random seed than training), feeds the model
everything up to the '=', lets it produce an answer, and checks exact match.

Usage:
    python eval_math.py --ckpt ckpt.pt --n 300
"""

import argparse

import torch

from math_data import generate_lines
from model import GPT, GPTConfig


def main():
    p = argparse.ArgumentParser(description="Evaluate math accuracy.")
    p.add_argument("--ckpt", default="ckpt.pt")
    p.add_argument("--n", type=int, default=300, help="problems to test")
    p.add_argument("--seed", type=int, default=99999, help="held-out problem seed")
    p.add_argument("--show", type=int, default=12, help="examples to print")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device)
    cfg = GPTConfig(**ckpt["config"])
    stoi, itos = ckpt["stoi"], ckpt["itos"]
    model = GPT(cfg).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    def encode(s):
        return [stoi[c] for c in s]

    def decode(ids):
        return "".join(itos[int(i)] for i in ids)

    problems = generate_lines(args.n, seed=args.seed)
    correct = 0
    shown = 0
    per_kind = {"add": [0, 0], "sub": [0, 0], "frac": [0, 0]}

    for line in problems:
        prompt, gold = line.rsplit(" = ", 1)
        prompt += " = "
        kind = "frac" if "/" in prompt else ("add" if "+" in prompt else "sub")

        # Any unseen character would break encoding; skip defensively.
        if any(c not in stoi for c in prompt):
            continue

        idx = torch.tensor([encode(prompt)], dtype=torch.long, device=args.device)
        out = model.generate(idx, max_new_tokens=12, temperature=1.0, top_k=1)[0].tolist()
        generated = decode(out[len(prompt):])
        pred = generated.split("\n", 1)[0].strip()

        ok = pred == gold
        correct += ok
        per_kind[kind][0] += ok
        per_kind[kind][1] += 1
        if shown < args.show:
            mark = "OK " if ok else "XX "
            print(f"{mark}{prompt}{pred}    (gold {gold})")
            shown += 1

    total = sum(v[1] for v in per_kind.values())
    print("\n=== accuracy ===")
    for kind, (c, t) in per_kind.items():
        if t:
            print(f"  {kind:4}: {c}/{t} = {c / t:.1%}")
    print(f"  ALL : {correct}/{total} = {correct / total:.1%}")


if __name__ == "__main__":
    main()
