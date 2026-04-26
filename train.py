"""Train LocalMixer with forward gradient + local losses, weight-perturbation
forward gradient, or backprop. See README.md for the algorithm description.

Examples:
  python train.py --dataset mnist --mode activity --epochs 20
  python train.py --dataset mnist --mode weight --epochs 20
  python train.py --dataset mnist --mode backprop --epochs 5     # baseline
  python train.py --dataset cifar10 --mode activity --epochs 50
"""
import argparse
import math
import time

import torch
from torch.optim.lr_scheduler import LambdaLR

from data import get_datasets, make_loaders
from forward_grad import TRAIN_STEPS
from model import LocalMixer


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(x).argmax(-1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    p.add_argument("--mode", default="activity",
                   choices=["activity", "activity_guess", "weight", "backprop"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_blocks", type=int, default=4)
    p.add_argument("--num_groups", type=int, default=4)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--token_hidden", type=int, default=64)
    p.add_argument("--channel_hidden", type=int, default=128)
    p.add_argument("--patch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n_samples", type=int, default=1,
                   help="Forward-gradient samples per step (variance reduction).")
    p.add_argument("--warmup_frac", type=float, default=0.05,
                   help="Fraction of total steps used for linear LR warmup.")
    p.add_argument("--log_every", type=int, default=0,
                   help="If >0, print step-level loss/lr every N steps.")
    p.add_argument("--max_steps_per_epoch", type=int, default=0,
                   help="If >0, cap training steps per epoch (smoke-test).")
    return p.parse_args()


def make_scheduler(optimizer, total_steps: int, warmup_frac: float) -> LambdaLR:
    warmup = max(1, int(warmup_frac * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    train_ds, test_ds, info = get_datasets(args.dataset)
    train_loader, test_loader = make_loaders(
        train_ds, test_ds, args.batch_size, num_workers=args.num_workers)

    model = LocalMixer(
        image_size=info["image_size"],
        patch_size=args.patch_size,
        in_channels=info["in_channels"],
        num_groups=args.num_groups,
        channels=args.channels,
        token_hidden=args.token_hidden,
        channel_hidden=args.channel_hidden,
        num_blocks=args.num_blocks,
        num_classes=info["num_classes"],
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"mode={args.mode} dataset={args.dataset} params={n_params:,} "
          f"n_samples={args.n_samples}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_step = TRAIN_STEPS[args.mode]

    steps_per_epoch = (args.max_steps_per_epoch
                       if args.max_steps_per_epoch
                       else len(train_loader))
    total_steps = args.epochs * steps_per_epoch
    scheduler = make_scheduler(optimizer, total_steps, args.warmup_frac)

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        loss_sum, n = 0.0, 0
        for step, (x, y) in enumerate(train_loader):
            if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                break
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)
            loss = train_step(model, x, y, optimizer, n_samples=args.n_samples)
            scheduler.step()
            bs = x.size(0)
            loss_sum += loss.item() * bs
            n += bs
            global_step += 1
            if args.log_every and global_step % args.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"  step={global_step:5d} lr={lr:.2e} "
                      f"loss={loss.item():.4f}")
        train_loss = loss_sum / max(n, 1)
        acc = evaluate(model, test_loader, args.device)
        print(f"epoch={epoch:3d} loss={train_loss:.4f} test_acc={acc:.4f} "
              f"t={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
