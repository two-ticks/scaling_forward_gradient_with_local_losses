# Scaling Forward Gradient With Local Losses -- PyTorch port

A from-scratch PyTorch implementation of the algorithm in
[Ren, Kornblith, Liao, Hinton (ICLR 2023)](https://arxiv.org/abs/2210.03310).
The reference implementation
([google-research/local_forward_gradient](https://github.com/google-research/google-research/tree/master/local_forward_gradient))
is in JAX. This port answers a natural question: **is this now feasible in
PyTorch?** Yes -- with `torch.func`.

## What changed in PyTorch that makes this easy now

The core operation in forward-gradient learning is a forward-mode JVP of a
loss function w.r.t. parameters or activations. JAX has had `jax.jvp` since
day one, but until PyTorch 2.0 the equivalent (`torch.autograd.forward_ad`)
was experimental and awkward to use with `nn.Module`. PyTorch 2.x ships:

- `torch.func.jvp` -- forward-mode JVP, structurally identical to `jax.jvp`.
- `torch.func.functional_call` -- call an `nn.Module` with an explicit
  parameter dict (analogous to flax's functional API).
- `torch.func.vmap` -- JAX-style vectorisation, useful if you want to estimate
  gradients with multiple noise samples per step.

That trio is sufficient to express the paper's algorithm idiomatically. We
use `torch.func.jvp` + `functional_call` directly in
[`forward_grad.py`](forward_grad.py).

## Algorithm in one paragraph

For an unbiased forward-gradient estimate, sample tangent `v ~ N(0, I)`,
compute the directional derivative `c = (grad L . v)` via a JVP, and emit
`g_hat = c . v`. Then `E[g_hat] = grad L`. Variance grows with the dimensionality of
`v`, so naive forward gradient on full-network weights does not scale. Two
ideas in the paper fix that:

1. **Activity perturbation.** Put the tangent on activations rather than
   weights. The per-element variance no longer scales with the parameter
   count.
2. **Local losses.** Attach a small classifier head to every block and
   compute the JVP only through that block + head. Each block sees a tiny
   parameter set, so the per-block gradient estimate is well behaved.

Between blocks we always detach activations, so each block is trained
independently -- no global backward pass.

## Files

| File | What it has |
|---|---|
| `model.py` | `LocalMixer`: `PatchEmbed`, `MixerBlock` (token + channel mixers), `GroupLayerNorm`, `LocalHead`. Per-group weights make local losses cheap. |
| `forward_grad.py` | Three training steps: `weight`, `activity`, `backprop` baseline. |
| `data.py` | MNIST + CIFAR-10 with the same normalisation as the paper. |
| `train.py` | CLI driver. |
| `test_unbiased.py` | Sanity: average many JVP samples, compare with the true gradient. |

## Usage

```
pip install -r requirements.txt

# Activity perturbation (paper's main method)
python train.py --dataset mnist   --mode activity --epochs 20

# Weight perturbation forward gradient
python train.py --dataset mnist   --mode weight   --epochs 20

# Backprop baseline (sum of local losses, end-to-end)
python train.py --dataset mnist   --mode backprop --epochs 5

# CIFAR-10
python train.py --dataset cifar10 --mode activity --epochs 50
```

`--max_steps_per_epoch N` caps each epoch to N batches -- useful for smoke
tests.

## Implementation notes

- Each `MixerBlock` is one "trainable unit"; block 0 owns the patch embedding
  so the embedding is perturbed/updated together with that block's weights.
- For **weight perturbation** we JVP the closure
  `(p_block, p_head) -> CE(head(block(z_in)), y)` along Gaussian tangents,
  multiply the resulting scalar by the tangent, and assign to `.grad`.
- For **activity perturbation** we run a normal forward through the block,
  sample `eps ~ N(0, I)` at the block output, JVP the head along `eps` to
  get `c`, and call `z_out.backward(c . eps)`. Because `z_in` is detached,
  this populates only that block's parameters -- one layer of backprop, not
  full backprop. The local head's own weights get an exact gradient (it is
  one linear layer; cheap).
- The paper uses many more local losses (block x patch group x channel
  group). We keep one head per block for clarity; the per-group weights
  inside each block already give the architecture the structure local
  learning needs. Adding patch/channel-group heads is straightforward -- fold
  them into `LocalHead` and produce one cross-entropy term per group.

## Defaults

`--num_blocks=4 --num_groups=4 --channels=64 --token_hidden=64
--channel_hidden=128 --patch_size=4`. ~386k params on MNIST. Smaller than
the paper's networks; the goal here is to demonstrate the algorithm, not to
reproduce ImageNet numbers.

## Sanity check

`python test_unbiased.py` averages 4000 JVP samples on a tiny LocalMixer and
compares with the true gradient (computed via backprop). Cosine similarity
approaches 1 for the large-magnitude parameters, which is the expected
behaviour of an unbiased high-variance estimator.
