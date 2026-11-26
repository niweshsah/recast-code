"""Recast additions to Octo (Flax): object-point tokenizer, proprio tokenizer and the auxiliary phase head.

All three follow Octo's component interfaces, so they are configured with ``ModuleSpec`` and trained jointly with
the pretrained backbone:
  * tokenizers are called as ``tok(observations, tasks, train=...)`` and return a ``TokenGroup`` with tokens of
    shape (batch, window, n_tokens, dim); Octo projects them to the transformer width and adds positional
    embeddings;
  * heads are called as ``head(transformer_outputs, train=...)`` and read their readout token group.

Octo adds a learned positional embedding per token slot. Keypoints have no fixed semantic order across
demonstrations, so the point tokenizer shuffles them during training: the model cannot rely on slot identity.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
from octo.model.components.base import TokenGroup

from recast import PHASES
from recast.data.episodes import PROPRIO_DIM


class PointTokenizer(nn.Module):
    """s^o_t in R^{M x 3} -> M tokens through a shared two-layer MLP (one token per keypoint)."""

    num_points: int = 16
    hidden: int = 256
    out_dim: int = 256
    obs_key: str = "proprio"
    shuffle_in_training: bool = True

    @nn.compact
    def __call__(self, observations, *unused_args, train: bool = True, **unused_kwargs):
        if self.obs_key not in observations:
            return None
        state = observations[self.obs_key]
        pts = state[..., PROPRIO_DIM:PROPRIO_DIM + 3 * self.num_points]
        pts = pts.reshape(*pts.shape[:-1], self.num_points, 3)          # (B, W, M, 3)
        if train and self.shuffle_in_training:
            perm = jax.random.permutation(self.make_rng("dropout"), self.num_points)
            pts = pts[..., perm, :]
        x = nn.gelu(nn.Dense(self.hidden, name="point_mlp_0")(pts))
        x = nn.Dense(self.out_dim, name="point_mlp_1")(x)
        return TokenGroup(x, jnp.ones(x.shape[:-1]))


class ProprioTokenizer(nn.Module):
    """[q_t, g_t] -> one token through an MLP."""

    hidden: int = 256
    out_dim: int = 256
    obs_key: str = "proprio"

    @nn.compact
    def __call__(self, observations, *unused_args, train: bool = True, **unused_kwargs):
        if self.obs_key not in observations:
            return None
        x = observations[self.obs_key][..., :PROPRIO_DIM]
        x = nn.gelu(nn.Dense(self.hidden, name="proprio_mlp_0")(x))
        x = nn.Dense(self.out_dim, name="proprio_mlp_1")(x)[..., None, :]   # (B, W, 1, D)
        return TokenGroup(x, jnp.ones(x.shape[:-1]))


class PhaseHead(nn.Module):
    """p_theta(phi_{t:t+T_a} | z_t): a phase distribution for each step of the action chunk, from the action readout."""

    readout_key: str = "readout_action"
    action_horizon: int = 4
    num_phases: int = len(PHASES)

    @nn.compact
    def __call__(self, transformer_outputs, train: bool = True):
        tokens = transformer_outputs[self.readout_key].tokens          # (B, W, n_readout, D)
        z = tokens.mean(axis=-2)
        logits = nn.Dense(self.action_horizon * self.num_phases, name="phase_logits")(z)
        return logits.reshape(*logits.shape[:-1], self.action_horizon, self.num_phases)

    def loss(self, transformer_outputs, phases, timestep_pad_mask, train: bool = True):
        """Cross entropy over the chunk; label -1 (hand-only data / padding) is ignored."""
        logits = self(transformer_outputs, train=train)                  # (B, W, H, P)
        labels = phases.astype(jnp.int32)
        valid = (labels >= 0) & timestep_pad_mask[..., None]
        logp = jax.nn.log_softmax(logits, axis=-1)
        nll = -jnp.take_along_axis(logp, jnp.clip(labels, 0)[..., None], axis=-1)[..., 0]
        loss = (nll * valid).sum() / jnp.maximum(valid.sum(), 1)
        acc = ((logits.argmax(-1) == labels) * valid).sum() / jnp.maximum(valid.sum(), 1)
        return loss, {"phase_loss": loss, "phase_accuracy": acc}

