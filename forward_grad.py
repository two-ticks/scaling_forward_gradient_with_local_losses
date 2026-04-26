"""Forward-gradient training steps with local losses.

For an unbiased forward-gradient estimator the recipe is:
  sample tangent  v ~ N(0, I)
  c = JVP of L along v   (a single scalar, the directional derivative)
  g_hat = c * v
  E[g_hat] = E[(grad L . v) v] = grad L

Two variants are implemented per the paper:

* Weight perturbation: v lives on the parameters; g_hat is a per-param gradient
  estimate. Variance ~ #params, so the local-losses split is what makes this
  scale: each block only sees its own parameters.
* Activity perturbation: v lives on the block's output activations. The JVP
  through the local head gives c, then the activity-gradient estimate
  c*eps is propagated through a single block via standard autograd
  (one step of backprop, not full backprop) to populate weight grads. The
  local head's own parameters are trained with exact gradients (it's a single
  linear layer).

Between blocks we always detach activations to keep losses local.

`n_samples > 1` averages K independent JVP samples per step. Each sample is
unbiased; averaging cuts variance by a factor of K at K-times the cost.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp


def _detached_params(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach() for k, v in module.named_parameters()}


def _normal_like(params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: torch.randn_like(v) for k, v in params.items()}


def train_step_weight_pert(model, x, y, optimizer, n_samples: int = 1):
    """Weight-perturbation forward gradient applied independently per block."""
    optimizer.zero_grad(set_to_none=True)
    losses = []
    z = x  # block 0 owns the patch embedding

    for block, head in zip(model.blocks, model.heads):
        z_in = z.detach()
        p_block = _detached_params(block)
        p_head = _detached_params(head)

        def loss_and_z(pb, ph):
            z_out = functional_call(block, pb, z_in)
            logits = functional_call(head, ph, z_out)
            return F.cross_entropy(logits, y), z_out

        g_block = {k: torch.zeros_like(v) for k, v in p_block.items()}
        g_head = {k: torch.zeros_like(v) for k, v in p_head.items()}
        loss_acc = torch.zeros((), device=x.device)
        z_next = None
        for _ in range(n_samples):
            t_block = _normal_like(p_block)
            t_head = _normal_like(p_head)
            (loss, z_out), (c, _) = jvp(
                loss_and_z, (p_block, p_head), (t_block, t_head),
            )
            c_d = c.detach()
            for k in g_block:
                g_block[k].add_(c_d * t_block[k])
            for k in g_head:
                g_head[k].add_(c_d * t_head[k])
            loss_acc = loss_acc + loss.detach()
            z_next = z_out

        scale = 1.0 / n_samples
        for name, p in block.named_parameters():
            p.grad = (g_block[name] * scale).detach()
        for name, p in head.named_parameters():
            p.grad = (g_head[name] * scale).detach()

        losses.append(loss_acc / n_samples)
        z = z_next

    optimizer.step()
    return torch.stack(losses).mean()


def train_step_activity_pert(model, x, y, optimizer, n_samples: int = 1):
    """Activity-perturbation forward gradient.

    For each block:
      1. Forward pass with autograd on block params.
      2. Average c*eps over n_samples to get an activity-gradient estimate
         g_z at the block output.
      3. z_out.backward(g_z) gives an unbiased estimate of the block-parameter
         gradient via *one* layer of backprop.
      4. The (small) local head is trained with an exact backward.
    """
    optimizer.zero_grad(set_to_none=True)
    losses = []
    z = x

    for block, head in zip(model.blocks, model.heads):
        z_in = z.detach()
        z_out = block(z_in)
        head_p = _detached_params(head)

        def head_loss(z_):
            logits = functional_call(head, head_p, z_)
            return F.cross_entropy(logits, y)

        g_z = torch.zeros_like(z_out)
        for _ in range(n_samples):
            eps = torch.randn_like(z_out)
            _, c = jvp(head_loss, (z_out.detach(),), (eps,))
            g_z.add_(c.detach() * eps)
        g_z.div_(n_samples)

        z_out.backward(g_z)

        logits = head(z_out.detach())
        loss = F.cross_entropy(logits, y)
        loss.backward()

        losses.append(loss.detach())
        z = z_out

    optimizer.step()
    return torch.stack(losses).mean()


def train_step_backprop(model, x, y, optimizer, n_samples: int = 1):
    """End-to-end backprop baseline over the sum of per-block local losses."""
    del n_samples  # exact gradient; no MC sampling
    optimizer.zero_grad(set_to_none=True)
    losses = []
    z = x
    for block, head in zip(model.blocks, model.heads):
        z = block(z)
        losses.append(F.cross_entropy(head(z), y))
    total = torch.stack(losses).sum()
    total.backward()
    optimizer.step()
    return (total / len(losses)).detach()


TRAIN_STEPS = {
    "weight": train_step_weight_pert,
    "activity": train_step_activity_pert,
    "backprop": train_step_backprop,
}
