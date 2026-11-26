"""Fine-tune Octo-Base on Recast demonstrations (paper eq. 7):

    L = E_{k,eps} || eps - eps_theta(z_t, A^k_t, k) ||^2  +  beta * sum_j CE(phi_{t+j}, p_theta(. | z_t)_j)

Variants (same checkpoint, same number of full-parameter updates):
  * recast     primary image + language + object-point tokens + proprio tokens, diffusion head + phase head
  * hand_only  Phantom-style baseline: primary image + language + proprio, diffusion head only

Run in the training environment (envs/train.txt):
    python -m recast.policy.finetune --data_dir ~/tensorflow_datasets --dataset recast_cube \
        --variant recast --save_dir checkpoints/recast_cube
"""

from __future__ import annotations

from absl import app, flags, logging

FLAGS = flags.FLAGS
flags.DEFINE_string("pretrained", "hf://rail-berkeley/octo-base-1.5", "Octo checkpoint to start from.")
flags.DEFINE_string("data_dir", None, "TFDS data dir holding the RLDS dataset.")
flags.DEFINE_string("dataset", None, "RLDS dataset name (see recast.data.episodes).")
flags.DEFINE_enum("variant", "recast", ["recast", "hand_only"], "Model variant.")
flags.DEFINE_string("save_dir", None, "Checkpoint directory.")
flags.DEFINE_integer("steps", 20000, "Gradient steps (identical for every variant).")
flags.DEFINE_integer("batch_size", 128, "Global batch size.")
flags.DEFINE_float("lr", 3e-5, "Peak learning rate.")
flags.DEFINE_float("beta", 0.1, "Phase-loss weight.")
flags.DEFINE_integer("num_points", 16, "Object keypoints M.")
flags.DEFINE_integer("action_horizon", 4, "Action chunk length T_a.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_bool("wandb", False, "Log to Weights & Biases.")


def main(_):
    import flax
    import jax
    import optax
    import tensorflow as tf
    import tqdm
    from octo.data.dataset import make_single_dataset
    from octo.model.components.action_heads import DiffusionActionHead
    from octo.model.octo_model import OctoModel
    from octo.utils.jax_utils import initialize_compilation_cache
    from octo.utils.spec import ModuleSpec
    from octo.utils.train_utils import TrainState, merge_params, process_text

    from recast.data.episodes import ACTION_NORMALIZATION_MASK
    from recast.policy.modules import PhaseHead, PointTokenizer, ProprioTokenizer

    assert FLAGS.batch_size % jax.device_count() == 0, "batch size must divide the device count"
    initialize_compilation_cache()
    tf.config.set_visible_devices([], "GPU")      # TF only feeds data
    recast = FLAGS.variant == "recast"
    if FLAGS.wandb:
        import wandb

        wandb.init(project="recast", name=f"{FLAGS.dataset}_{FLAGS.variant}")

    pretrained = OctoModel.load_pretrained(FLAGS.pretrained)
    dataset = make_single_dataset(
        dataset_kwargs=dict(name=FLAGS.dataset, data_dir=FLAGS.data_dir, image_obs_keys={"primary": "image_primary"},
                            proprio_obs_key="state", language_key="language_instruction",
                            action_normalization_mask=ACTION_NORMALIZATION_MASK),
        traj_transform_kwargs=dict(window_size=2, action_horizon=FLAGS.action_horizon),
        frame_transform_kwargs=dict(resize_size={"primary": (256, 256)}),
        train=True,
    )
    it = dataset.repeat().unbatch().shuffle(10000).batch(FLAGS.batch_size).iterator()

    def process(batch):
        batch = process_text(batch, pretrained.text_processor)
        del batch["dataset_name"]
        return batch

    it = map(process, it)
    example = next(it)

    # ---- model surgery: drop the wrist camera, add Recast tokenizers and heads
    config = pretrained.config
    config["model"]["observation_tokenizers"].pop("wrist", None)
    config["model"]["observation_tokenizers"]["proprio"] = ModuleSpec.create(ProprioTokenizer)
    if recast:
        config["model"]["observation_tokenizers"]["points"] = ModuleSpec.create(PointTokenizer,
                                                                                num_points=FLAGS.num_points)
        config["model"]["heads"]["phase"] = ModuleSpec.create(PhaseHead, readout_key="readout_action",
                                                              action_horizon=FLAGS.action_horizon)
    config["model"]["heads"]["action"] = ModuleSpec.create(DiffusionActionHead, readout_key="readout_action",
                                                           use_map=False, action_horizon=FLAGS.action_horizon,
                                                           action_dim=7, n_diffusion_samples=1)
    model = OctoModel.from_config(config, example, pretrained.text_processor, verbose=True,
                                  dataset_statistics=dataset.dataset_statistics)
    model = model.replace(params=merge_params(model.params, pretrained.params))   # new modules stay from scratch
    del pretrained

    schedule = optax.join_schedules([optax.linear_schedule(0, FLAGS.lr, 500), optax.cosine_decay_schedule(
        FLAGS.lr, FLAGS.steps - 500)], [500])
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=0.01))
    state = TrainState.create(rng=jax.random.PRNGKey(FLAGS.seed), model=model, tx=tx)

    def loss_fn(params, batch, rng, train=True):
        module = model.module.bind({"params": params}, rngs={"dropout": rng})
        obs, pad = batch["observation"], batch["observation"]["timestep_pad_mask"]
        emb = module.octo_transformer(obs, batch["task"], pad, train=train)
        actions = batch["action"]                                     # (B, W, H, 8)
        loss, metrics = module.heads["action"].loss(emb, actions[..., :7], pad, batch["action_pad_mask"][..., :7],
                                                    train=train)
        if recast:
            ph_loss, ph_metrics = module.heads["phase"].loss(emb, actions[..., 7], pad, train=train)
            loss = loss + FLAGS.beta * ph_loss
            metrics = {**metrics, **ph_metrics}
        return loss, {**metrics, "total_loss": loss}

    @jax.jit
    def train_step(state, batch):
        rng, drng = jax.random.split(state.rng)
        (_, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.model.params, batch, drng)
        return state.apply_gradients(grads=grads, rng=rng), info

    for step in tqdm.trange(FLAGS.steps, dynamic_ncols=True):
        state, info = train_step(state, next(it))
        if (step + 1) % 100 == 0:
            info = jax.device_get(info)
            logging.info("step %d: %s", step + 1, {k: float(v) for k, v in info.items()})
            if FLAGS.wandb:
                wandb.log(flax.traverse_util.flatten_dict({"train": info}, sep="/"), step=step)
        if (step + 1) % 5000 == 0 or step + 1 == FLAGS.steps:
            state.model.save_pretrained(step=step + 1, checkpoint_path=FLAGS.save_dir)


if __name__ == "__main__":
    flags.mark_flags_as_required(["data_dir", "dataset", "save_dir"])
    app.run(main)
