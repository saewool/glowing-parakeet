#!/usr/bin/env python3
"""Train a tiny character-level LLM (GPT-style) in PyTorch.

Dataset is automatically downloaded from the internet (Tiny Shakespeare).
"""

from __future__ import annotations

import argparse
import math
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


@dataclass
class TrainConfig:
    block_size: int = 128
    batch_size: int = 64
    n_embed: int = 256
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.2
    lr: float = 3e-4
    max_steps: int = 2000
    eval_interval: int = 200
    eval_iters: int = 50
    train_split: float = 0.9
    seed: int = 1337


class Head(nn.Module):
    def __init__(self, n_embed: int, head_size: int, block_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed: int, n_heads: int, block_size: int, dropout: float):
        super().__init__()
        head_size = n_embed // n_heads
        self.heads = nn.ModuleList(
            [Head(n_embed, head_size, block_size, dropout) for _ in range(n_heads)]
        )
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embed: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed: int, n_heads: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.sa = MultiHeadAttention(n_embed, n_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embed)
        self.ffwd = FeedForward(n_embed, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embed: int,
        n_heads: int,
        n_layers: int,
        block_size: int,
        dropout: float,
    ):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.position_embedding = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            *[Block(n_embed, n_heads, block_size, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        b, t = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(t, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            bsz, tsz, channels = logits.shape
            loss = F.cross_entropy(logits.view(bsz * tsz, channels), targets.view(bsz * tsz))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def download_dataset(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading dataset from {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, path)
    return path.read_text(encoding="utf-8")


def build_vocab(text: str):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos


def encode(text: str, stoi: dict[str, int]) -> torch.Tensor:
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)


def get_batch(data: torch.Tensor, cfg: TrainConfig, device: str):
    ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))
    x = torch.stack([data[i : i + cfg.block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + cfg.block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg: TrainConfig, device: str):
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = get_batch(data, cfg, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a tiny GPT with PyTorch")
    p.add_argument("--data-path", default="data/tiny_shakespeare.txt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--n-embed", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--eval-iters", type=int, default=50)
    p.add_argument("--sample-tokens", type=int, default=300)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        block_size=args.block_size,
        batch_size=args.batch_size,
        n_embed=args.n_embed,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        lr=args.lr,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
    )

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    text = download_dataset(Path(args.data_path))
    stoi, itos = build_vocab(text)
    data = encode(text, stoi)
    split_idx = int(cfg.train_split * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    model = TinyGPT(
        vocab_size=len(stoi),
        n_embed=cfg.n_embed,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        block_size=cfg.block_size,
        dropout=cfg.dropout,
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    for step in range(cfg.max_steps):
        if step % cfg.eval_interval == 0 or step == cfg.max_steps - 1:
            losses = estimate_loss(model, train_data, val_data, cfg, args.device)
            ppl = math.exp(losses["val"])
            print(
                f"step {step:4d} | train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | val ppl {ppl:.2f}"
            )

        xb, yb = get_batch(train_data, cfg, args.device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    context = torch.zeros((1, 1), dtype=torch.long, device=args.device)
    generated = model.generate(context, max_new_tokens=args.sample_tokens)[0].tolist()
    sample_text = "".join(itos[i] for i in generated)
    print("\n--- Sample generation ---")
    print(sample_text)


if __name__ == "__main__":
    main()
