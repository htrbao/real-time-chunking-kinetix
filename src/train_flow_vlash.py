import concurrent.futures
import dataclasses
import functools
import pathlib
import pickle
from typing import Sequence

import einops
from flax import struct
import flax.nnx as nnx
import imageio
import jax
import jax.numpy as jnp
import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import tyro
import wandb

import eval_flow_vlash as _eval
import generate_data
import model as _model
from model import get_vlash_history_size, compute_vlash_state, get_vlash_state_dim
import train_expert

WANDB_PROJECT = "rtc-kinetix-bc"
LOG_DIR = pathlib.Path("logs-bc")


@dataclasses.dataclass(frozen=True)
class Config:
    run_path: str
    level_paths: Sequence[str] = (
        # "worlds/l/grasp_easy.json",
        # "worlds/l/catapult.json",
        # "worlds/l/cartpole_thrust.json",
        "worlds/l/hard_lunar_lander.json",
        # "worlds/l/mjc_half_cheetah.json",
        # "worlds/l/mjc_swimmer.json",
        # "worlds/l/mjc_walker.json",
        # "worlds/l/h17_unicycle.json",
        # "worlds/l/chain_lander.json",
        # "worlds/l/catcher_v3.json",
        # "worlds/l/trampoline.json",
        # "worlds/l/car_launch.json",
    )
    batch_size: int = 512
    num_epochs: int = 32
    seed: int = 0

    eval: _eval.EvalConfig = _eval.EvalConfig()

    learning_rate: float = 3e-4
    grad_norm_clip: float = 10.0
    weight_decay: float = 1e-2
    lr_warmup_steps: int = 1000

    load_dir: str | None = None


@struct.dataclass
class EpochCarry:
    rng: jax.Array
    train_state: nnx.State
    graphdef: nnx.GraphDef[tuple[_model.FlowPolicy, nnx.Optimizer]]


def main(config: Config):
    static_env_params = kenv_state.StaticEnvParams(**train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP)
    env_params = kenv_state.EnvParams()
    levels = train_expert.load_levels(config.level_paths, static_env_params, env_params)
    static_env_params = static_env_params.replace(screen_dim=train_expert.SCREEN_DIM)

    env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)

    mesh = jax.make_mesh((jax.local_device_count(),), ("level",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("level"))

    action_chunk_size = config.eval.model.action_chunk_size

    # load data
    def load_data(level_path: str):
        level_name = level_path.replace("/", "_").replace(".json", "")
        print("Loading data for level:", level_name)
        return dict(np.load(pathlib.Path(config.run_path) / "data" / f"{level_name}.npz"))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        data = list(executor.map(load_data, config.level_paths))
    with jax.default_device(jax.devices("cpu")[0]):
        num_steps, num_episodes = data[0]["obs"].shape[:-1]
        # data has shape: (num_levels, num_steps, num_envs, ...)
        # flatten envs and steps together for learning
        data = jax.tree.map(lambda *x: einops.rearrange(jnp.stack(x), "l s e ... -> l (e s) ..."), *data)
        # truncate to multiple of batch size
        valid_steps = data["obs"].shape[1] - action_chunk_size + 1
        data = jax.tree.map(
            lambda x: x[:, : (valid_steps // config.batch_size) * config.batch_size + action_chunk_size - 1], data
        )
        # put on device
        data = jax.tree.map(
            lambda x: jax.make_array_from_single_device_arrays(
                x.shape,
                sharding,
                [
                    jax.device_put(y, d)
                    for y, d in zip(jnp.split(x, jax.local_device_count()), jax.local_devices(), strict=True)
                ],
            ),
            data,
        )

    data: generate_data.Data = generate_data.Data(**data)
    print(f"Truncated data to {data.obs.shape[1]:_} steps ({valid_steps // config.batch_size:_} batches)")

    obs_dim = data.obs.shape[-1]
    action_dim = env.action_space(env_params).shape[0]
    vlash_state_dim = get_vlash_state_dim(action_dim, config.eval.model.vlash_order)

    # VLASH
    max_delta = action_chunk_size - 1
    len_data = data.obs.shape[1]
    ids = jnp.arange(len_data)
    episode_ids = jnp.arange(num_episodes)
    meta_episode_ids = ids // num_steps
    episode_from_ids = episode_ids * num_steps
    episode_to_ids = jnp.minimum(episode_from_ids + num_steps, len_data)

    if config.load_dir is not None:
        state_dicts = []
        for level_path in config.level_paths:
            level_name = level_path.replace("/", "_").replace(".json", "")
            with (pathlib.Path(config.load_dir) / "policies" / f"{level_name}.pkl").open("rb") as f:
                state_dicts.append(pickle.load(f))
        state_dicts = jax.device_put(jax.tree.map(lambda *x: jnp.array(x), *state_dicts))
    else:
        state_dicts = None

    @functools.partial(jax.jit, in_shardings=sharding, out_shardings=sharding)
    @jax.vmap
    def init(rng: jax.Array, state_dict: dict | None) -> EpochCarry:
        rng, key = jax.random.split(rng)
        policy = _model.FlowPolicy(
            obs_dim=obs_dim + vlash_state_dim,
            action_dim=action_dim,
            config=config.eval.model,
            rngs=nnx.Rngs(key),
        )
        if state_dict is not None:
            graphdef, state = nnx.split(policy)
            state.replace_by_pure_dict(state_dict)
            policy = nnx.merge(graphdef, state)
        total_params = sum(x.size for x in jax.tree.leaves(nnx.state(policy, nnx.Param)))
        print(f"Total params: {total_params:,}")
        optimizer = nnx.Optimizer(
            policy,
            optax.chain(
                optax.clip_by_global_norm(config.grad_norm_clip),
                optax.adamw(
                    optax.warmup_constant_schedule(0, config.learning_rate, config.lr_warmup_steps),
                    weight_decay=config.weight_decay,
                ),
            ),
        )
        graphdef, train_state = nnx.split((policy, optimizer))
        return EpochCarry(rng, train_state, graphdef)

    @functools.partial(jax.jit, donate_argnums=(0,), in_shardings=sharding, out_shardings=sharding)
    @jax.vmap
    def train_epoch(epoch_carry: EpochCarry, level: kenv_state.EnvState, data: generate_data.Data):
        def train_minibatch(carry: tuple[jax.Array, nnx.State], batch_idxs: jax.Array):
            rng, train_state = carry
            policy, optimizer = nnx.merge(epoch_carry.graphdef, train_state)

            rng, key = jax.random.split(rng)

            def loss_fn(policy: _model.FlowPolicy):
                # VLASH with full-order dynamics support
                # For each sample in batch, we compute states at multiple delay offsets
                
                # simulated_delay must be set for VLASH training
                assert policy.simulated_delay is not None, \
                    "simulated_delay must be set in ModelConfig for VLASH training"
                
                vlash_history = get_vlash_history_size(policy.vlash_order)
                
                episodes = meta_episode_ids[batch_idxs]     # (B,)
                episodes_start = episode_from_ids[episodes] # (B,)
                episodes_end = episode_to_ids[episodes]     # (B,)
                
                # Compute valid offsets (how many delays we can apply)
                max_offsets = jnp.minimum(
                    jnp.maximum(
                        episodes_end - 1 - (batch_idxs + max_delta),
                        0
                    ),
                    policy.simulated_delay
                ) # (B,)
                num_offsets = max_offsets + 1 # (B,)
                
                # Create offset indices: [0, 1, ..., simulated_delay-1]
                offsets = jnp.arange(policy.simulated_delay) # (Moff,)
                batch_offsets = jnp.broadcast_to(offsets, (config.batch_size, policy.simulated_delay)) # (B, Moff)
                
                # Compute VLASH states for each offset
                # For offset k at batch_idx i, the observation point is at i + k
                # The state should be computed from actions BEFORE that observation point
                # 
                # For position order: state = action[obs_point - 1]
                # For velocity order: state = [action[obs_point - 1], action[obs_point - 1] - action[obs_point - 2]]
                # For acceleration: + second derivative
                
                # Indices for gathering action history for state computation
                # Shape: (B, Moff, vlash_history)
                # For each (batch, offset), we need vlash_history consecutive previous actions
                obs_points = batch_idxs[:, None] + batch_offsets  # (B, Moff) - observation point indices
                
                # Action history indices: for state at obs_point, we need actions at
                # obs_point - vlash_history, obs_point - vlash_history + 1, ..., obs_point - 1
                history_offsets = jnp.arange(-vlash_history, 0)  # (-H, ..., -1)
                action_history_idxs = obs_points[:, :, None] + history_offsets[None, None, :]  # (B, Moff, H)
                
                # Clamp to episode boundaries
                action_history_idxs = jnp.maximum(
                    jnp.minimum(action_history_idxs, (episodes_end - 1)[:, None, None]),
                    episodes_start[:, None, None]
                )  # (B, Moff, H)
                
                # Gather action history
                action_history = data.action[action_history_idxs]  # (B, Moff, H, action_dim)
                
                # Zero out actions for invalid offsets and handle episode start
                is_valid_offset = batch_offsets < num_offsets[:, None]  # (B, Moff)
                is_at_episode_start = obs_points <= episodes_start[:, None]  # (B, Moff)
                
                action_history = jnp.where(
                    (is_valid_offset & ~is_at_episode_start)[:, :, None, None],
                    action_history,
                    0.0
                )  # (B, Moff, H, action_dim)
                
                # Compute VLASH states from action history
                states = jax.vmap(jax.vmap(
                    lambda ah: compute_vlash_state(ah, policy.vlash_order)
                ))(action_history)  # (B, Moff, state_dim)
                
                # Compute action chunk indices for each offset
                # For offset k, the chunk starts at batch_idx + k
                chunk_start_idxs = batch_idxs[:, None] + batch_offsets  # (B, Moff)
                chunk_batch_idxs = jnp.maximum(
                    jnp.minimum(
                        chunk_start_idxs[:, :, None] + jnp.arange(action_chunk_size)[None, None, :],
                        (episodes_end - 1)[:, None, None]
                    ),
                    episodes_start[:, None, None]
                )  # (B, Moff, chunk_size)
                
                action_chunks = data.action[chunk_batch_idxs]  # (B, Moff, chunk_size, action_dim)
                
                # Zero out invalid offsets
                action_chunks = jnp.where(
                    is_valid_offset[:, :, None, None],
                    action_chunks,
                    0.0
                )  # (B, Moff, chunk_size, action_dim)
                
                # Get observations for each offset point (not just base observation)
                obs_points = batch_idxs[:, None] + batch_offsets  # (B, Moff)
                obs_points = jnp.minimum(obs_points, episodes_end[:, None] - 1)  # Clamp to episode end
                obs_all = data.obs[obs_points]  # (B, Moff, obs_dim)
                
                # Zero out invalid offset observations
                obs_all = jnp.where(
                    is_valid_offset[:, :, None],
                    obs_all,
                    0.0
                )  # (B, Moff, obs_dim)
                
                return policy.forward_shared_observation(key, obs_all, states, action_chunks)

            loss, grads = nnx.value_and_grad(loss_fn)(policy)
            info = {"loss": loss, "grad_norm": optax.global_norm(grads)}
            optimizer.update(grads)
            _, train_state = nnx.split((policy, optimizer))
            return (rng, train_state), info

        # shuffle
        rng, key = jax.random.split(epoch_carry.rng)
        permutation = jax.random.permutation(key, data.obs.shape[0] - action_chunk_size + 1)
        # batch
        permutation = permutation.reshape(-1, config.batch_size)
        # train
        (rng, train_state), train_info = jax.lax.scan(
            train_minibatch, (epoch_carry.rng, epoch_carry.train_state), permutation
        )
        train_info = jax.tree.map(lambda x: x.mean(), train_info)
        # eval
        rng, key = jax.random.split(rng)
        eval_policy, _ = nnx.merge(epoch_carry.graphdef, train_state)
        eval_info = {}
        for horizon in range(1, config.eval.model.action_chunk_size + 1):
            eval_config = dataclasses.replace(config.eval, execute_horizon=horizon)
            info, _ = _eval.eval(eval_config, env, key, level, eval_policy, env_params, static_env_params)
            eval_info.update({f"{k}_{horizon}": v for k, v in info.items()})
        video = None
        return EpochCarry(rng, train_state, epoch_carry.graphdef), ({**train_info, **eval_info}, video)

    wandb.init(project=WANDB_PROJECT)
    rng = jax.random.key(config.seed)
    epoch_carry = init(jax.random.split(rng, len(config.level_paths)), state_dicts)
    for epoch_idx in tqdm.tqdm(range(config.num_epochs)):
        epoch_carry, (info, video) = train_epoch(epoch_carry, levels, data)

        for i in range(len(config.level_paths)):
            level_name = config.level_paths[i].replace("/", "_").replace(".json", "")
            wandb.log({f"{level_name}/{k}": v[i] for k, v in info.items()}, step=epoch_idx)

            log_dir = LOG_DIR / wandb.run.name / str(epoch_idx)

            if video is not None:
                video_dir = log_dir / "videos"
                video_dir.mkdir(parents=True, exist_ok=True)
                imageio.mimwrite(video_dir / f"{level_name}.mp4", video[i], fps=15)

            policy_dir = log_dir / "policies"
            policy_dir.mkdir(parents=True, exist_ok=True)
            level_train_state = jax.tree.map(lambda x: x[i], epoch_carry.train_state)
            with (policy_dir / f"{level_name}.pkl").open("wb") as f:
                policy, _ = nnx.merge(epoch_carry.graphdef, level_train_state)
                state_dict = nnx.state(policy).to_pure_dict()
                pickle.dump(state_dict, f)


if __name__ == "__main__":
    tyro.cli(main)
