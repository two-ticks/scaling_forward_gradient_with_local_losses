"""Sanity check: forward-gradient estimator should be an unbiased estimator of
the true gradient. Average many JVP samples and compare with backprop.
"""
import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp

from model import LocalMixer


def true_grad(model, x, y):
    model.zero_grad(set_to_none=True)
    z = x
    losses = []
    for blk, head in zip(model.blocks, model.heads):
        z = blk(z)
        losses.append(F.cross_entropy(head(z), y))
    sum(losses).backward()
    return {n: p.grad.detach().clone() for n, p in model.named_parameters()}


def forward_grad_block0(model, x, y, n_samples=2000):
    """Average MC forward-gradient samples for block 0 + head 0 only."""
    block, head = model.blocks[0], model.heads[0]
    p_block = {k: v.detach() for k, v in block.named_parameters()}
    p_head = {k: v.detach() for k, v in head.named_parameters()}
    acc_block = {k: torch.zeros_like(v) for k, v in p_block.items()}
    acc_head = {k: torch.zeros_like(v) for k, v in p_head.items()}

    def loss_fn(pb, ph):
        z = functional_call(block, pb, x)
        return F.cross_entropy(functional_call(head, ph, z), y)

    for _ in range(n_samples):
        t_block = {k: torch.randn_like(v) for k, v in p_block.items()}
        t_head = {k: torch.randn_like(v) for k, v in p_head.items()}
        _, c = jvp(loss_fn, (p_block, p_head), (t_block, t_head))
        for k in acc_block:
            acc_block[k] += c.detach() * t_block[k]
        for k in acc_head:
            acc_head[k] += c.detach() * t_head[k]
    for k in acc_block:
        acc_block[k] /= n_samples
    for k in acc_head:
        acc_head[k] /= n_samples
    return acc_block, acc_head


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LocalMixer(
        image_size=8, patch_size=4, in_channels=1,
        num_groups=2, channels=4, token_hidden=4, channel_hidden=8,
        num_blocks=2, num_classes=3,
    ).to(device).eval()
    x = torch.randn(4, 1, 8, 8, device=device)
    y = torch.tensor([0, 1, 2, 0], device=device)

    grads = true_grad(model, x, y)
    fg_block, fg_head = forward_grad_block0(model, x, y, n_samples=4000)

    # The local-loss backprop accumulates gradients from *all* heads into
    # block 0 (since it's upstream). For an apples-to-apples check we redo the
    # true gradient using only block 0's head loss.
    model.zero_grad(set_to_none=True)
    z = model.blocks[0](x)
    F.cross_entropy(model.heads[0](z), y).backward()
    truth_block = {n: p.grad.detach().clone()
                   for n, p in model.blocks[0].named_parameters()}
    truth_head = {n: p.grad.detach().clone()
                  for n, p in model.heads[0].named_parameters()}

    def cos(a, b):
        return (a.flatten() @ b.flatten()) / (a.norm() * b.norm() + 1e-12)

    print("Cosine similarity between forward-gradient MC estimate and exact "
          "gradient (should approach 1 with more samples):")
    for k in truth_block:
        c = cos(fg_block[k], truth_block[k]).item()
        print(f"  block.{k:25s} cos={c: .4f}")
    for k in truth_head:
        c = cos(fg_head[k], truth_head[k]).item()
        print(f"  head .{k:25s} cos={c: .4f}")


if __name__ == "__main__":
    main()
