"""Train the GPT language model.

Examples:
    python train.py                        # sensible defaults
    python train.py --iters 5000 --device cuda
    python train.py --n-layer 4 --n-embd 256   # smaller/faster model
"""

import argparse

import torch

from data import build_dataset, get_batch
from math_data import generate_text
from model import GPT, GPTConfig


@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg, batch_size, device, eval_iters=100):
    """Average the loss over a few batches of train and val data."""
    out = {}
    model.eval()
    for split, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, cfg.block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    p = argparse.ArgumentParser(description="Train a GPT on character-level text.")
    p.add_argument("--dataset", default="shakespeare", choices=["shakespeare", "math"])
    p.add_argument("--math-samples", type=int, default=150000,
                   help="number of unique problems to generate for --dataset math")
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--n-embd", type=int, default=384)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save", default="ckpt.pt")
    args = p.parse_args()

    torch.manual_seed(1337)
    print(f"Using device: {args.device}")

    if args.dataset == "math":
        print(f"Generating {args.math_samples} math problems...")
        corpus = generate_text(args.math_samples)
        tokenizer, train_data, val_data = build_dataset(text=corpus)
    else:
        tokenizer, train_data, val_data = build_dataset()
    cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(cfg).to(args.device)
    print(f"Model parameters: {model.num_params() / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for it in range(1, args.iters + 1):
        if it % args.eval_interval == 0 or it == 1:
            stats = estimate_loss(
                model, train_data, val_data, cfg, args.batch_size, args.device
            )
            print(f"iter {it:>5}: train {stats['train']:.4f} | val {stats['val']:.4f}")

        x, y = get_batch(train_data, cfg.block_size, args.batch_size, args.device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Save weights + tokenizer + config so sample.py can reconstruct the model.
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg.__dict__,
            "stoi": tokenizer.stoi,
            "itos": tokenizer.itos,
        },
        args.save,
    )
    print(f"Saved checkpoint to {args.save}")

    # Quick generation preview.
    context = torch.zeros((1, 1), dtype=torch.long, device=args.device)
    sample = model.generate(context, max_new_tokens=300)[0].tolist()
    print("\n--- sample ---")
    print(tokenizer.decode(sample))


if __name__ == "__main__":
    main()
