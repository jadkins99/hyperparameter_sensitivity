# Oringinal repo forked from:
# https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/ppo_continuous_action.py
import argparse
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
from PyExpUtils.models.ExperimentDescription import ExperimentDescription
import json

from wrappers import (
    LogWrapper,
    BraxGymnaxWrapper,
    VecEnv,
    NormalizeVecObservation,
    ClipAction,
)

# jax.config.update("jax_enable_x64", True)


class ActorTrainState(TrainState):
    advn_stats: dict[float, float]


class Critic(nn.Module):
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return jnp.squeeze(critic, axis=-1)


class Actor(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        return pi


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    if config["ENV_NAME"] == "swimmer":
        env, env_params = (
            BraxGymnaxWrapper(config["ENV_NAME"], backend="generalized"),
            None,
        )

    else:
        env, env_params = (
            BraxGymnaxWrapper(config["ENV_NAME"], backend="positional"),
            None,
        )
    env = LogWrapper(env)
    env = ClipAction(env)
    env = VecEnv(env)
    if config["NORMALIZE_OBS"]:
        env = NormalizeVecObservation(env)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        actor_network = Actor(
            env.action_space(env_params).shape[0], activation=config["ACTIVATION"]
        )
        critic_network = Critic(activation=config["ACTIVATION"])

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        actor_network_params = actor_network.init(_rng, init_x)
        rng, _rng = jax.random.split(rng)
        critic_network_params = critic_network.init(_rng, init_x)

        if config["ANNEAL_LR"]:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["ACTOR_LR"], eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["CRITIC_LR"], eps=1e-5),
            )
        actor_train_state = ActorTrainState.create(
            apply_fn=actor_network.apply,
            params=actor_network_params,
            tx=actor_tx,
            advn_stats={
                "advn_per_5": 0.0,
                "advn_per_95": 0.0,
                "advn_mean": 0.0,
                "advn_std": 0.0,
            },
        )
        critic_train_state = TrainState.create(
            apply_fn=critic_network.apply,
            params=critic_network_params,
            tx=critic_tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    actor_train_state,
                    critic_train_state,
                    env_state,
                    last_obs,
                    rng,
                ) = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi = actor_network.apply(actor_train_state.params, last_obs)
                value = critic_network.apply(critic_train_state.params, last_obs)

                if config["SYMLOG_CRITIC_TARGETS"]:
                    symexp = lambda x: jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)
                    value = symexp(value)

                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = env.step(
                    rng_step, env_state, action, env_params
                )

                if config["SYMLOG_OBS"]:
                    symlog = lambda x: jnp.sign(x) * jnp.log(1 + jnp.abs(x))
                    obsv = symlog(obsv)

                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (
                    actor_train_state,
                    critic_train_state,
                    env_state,
                    obsv,
                    rng,
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            (
                actor_train_state,
                critic_train_state,
                env_state,
                last_obs,
                rng,
            ) = runner_state
            last_val = critic_network.apply(critic_train_state.params, last_obs)
            if config["SYMLOG_CRITIC_TARGETS"]:
                symexp = lambda x: jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)
                last_val = symexp(last_val)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            actor_train_state.advn_stats["advn_per_5"] = jax.lax.cond(
                actor_train_state.advn_stats["advn_per_5"] == 0.0,
                lambda x: jnp.percentile(x, 5),
                lambda x: config["EMA_RATE"] * jnp.percentile(x, 5)
                + (1 - config["EMA_RATE"]) * actor_train_state.advn_stats["advn_per_5"],
                advantages,
            )

            actor_train_state.advn_stats["advn_per_95"] = jax.lax.cond(
                actor_train_state.advn_stats["advn_per_95"] == 0.0,
                lambda x: jnp.percentile(x, 95),
                lambda x: config["EMA_RATE"] * jnp.percentile(x, 95)
                + (1 - config["EMA_RATE"])
                * actor_train_state.advn_stats["advn_per_95"],
                advantages,
            )

            actor_train_state.advn_stats["advn_mean"] = jax.lax.cond(
                actor_train_state.advn_stats["advn_mean"] == 0.0,
                lambda x: x.mean(),
                lambda x: config["EMA_RATE"] * x.mean()
                + (1 - config["EMA_RATE"]) * actor_train_state.advn_stats["advn_mean"],
                advantages,
            )

            actor_train_state.advn_stats["advn_std"] = jax.lax.cond(
                actor_train_state.advn_stats["advn_std"] == 0.0,
                lambda x: x.std(),
                lambda x: config["EMA_RATE"] * x.std()
                + (1 - config["EMA_RATE"]) * actor_train_state.advn_stats["advn_std"],
                advantages,
            )

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info
                    actor_train_state, critic_train_state = train_state

                    def _critic_loss_fn(critic_params, traj_batch, gae, targets):

                        if config["SYMLOG_CRITIC_TARGETS"]:
                            symlog = lambda x: jnp.sign(x) * jnp.log(1 + jnp.abs(x))
                            targets = symlog(targets)

                        # RERUN NETWORK
                        value = critic_network.apply(critic_params, traj_batch.obs)

                        # CALCULATE VALUE LOSS

                        value_losses = jnp.square(
                            value - jax.lax.stop_gradient(targets)
                        )
                        value_loss = value_losses.mean()

                        total_loss = value_loss

                        return total_loss, (value_loss,)

                    def _actor_loss_fn(actor_params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi = actor_network.apply(actor_params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        if config["ADVN_NORM"] == "MEAN":
                            gae = (gae - gae.mean()) / (gae.std() + 1e-8)

                        elif config["ADVN_NORM"] == "EMA_MEAN":
                            gae = (gae - actor_train_state.advn_stats["advn_mean"]) / (
                                actor_train_state.advn_stats["advn_std"] + 1e-8
                            )

                        elif config["ADVN_NORM"] == "EMA_PERC":
                            gae = gae / (
                                actor_train_state.advn_stats["advn_per_95"]
                                - actor_train_state.advn_stats["advn_per_5"]
                                + 1e-8
                            )
                        elif config["ADVN_NORM"] == "MAX_EMA_PERC":
                            gae = gae / (
                                jnp.maximum(
                                    1.0,
                                    actor_train_state.advn_stats["advn_per_95"]
                                    - actor_train_state.advn_stats["advn_per_5"],
                                )
                            )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = loss_actor - config["ENT_COEF"] * entropy

                        return total_loss, (loss_actor, entropy)

                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)

                    critic_loss, critic_grads = critic_grad_fn(
                        critic_train_state.params, traj_batch, advantages, targets
                    )

                    actor_loss, actor_grads = actor_grad_fn(
                        actor_train_state.params, traj_batch, advantages, targets
                    )

                    total_loss = actor_loss + critic_loss

                    critic_train_state = critic_train_state.apply_gradients(
                        grads=critic_grads
                    )

                    actor_train_state = actor_train_state.apply_gradients(
                        grads=actor_grads
                    )

                    train_state = (actor_train_state, critic_train_state)
                    return train_state, total_loss

                (
                    actor_train_state,
                    critic_train_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state = (actor_train_state, critic_train_state)
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                actor_train_state, critic_train_state = train_state
                update_state = (
                    actor_train_state,
                    critic_train_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            update_state = (
                actor_train_state,
                critic_train_state,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            actor_train_state = update_state[0]
            critic_train_state = update_state[1]

            metric = traj_batch.info
            rng = update_state[-1]
            if config.get("DEBUG"):

                def callback(info):

                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]

                    timesteps = (
                        info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    )
                    for t in range(len(timesteps)):
                        print(
                            f"global step={timesteps[t]}, episodic return={return_values[t]}"
                        )

                jax.debug.callback(callback, metric)

            runner_state = (
                actor_train_state,
                critic_train_state,
                env_state,
                last_obs,
                rng,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (actor_train_state, critic_train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--sweep_idx", action="store", default=-1, type=int)
    parser.add_argument("--num_seeds", action="store", default=200, type=int)
    parser.add_argument("--start_seed", action="store", default=42, type=int)
    parser.add_argument("--actor_lr", action="store", default=3e-4, type=float)
    parser.add_argument("--critic_lr", action="store", default=3e-4, type=float)
    parser.add_argument("--ent_coef", action="store", default=0.1, type=float)
    parser.add_argument("--gae_lambda", action="store", default=0.9, type=float)
    parser.add_argument("--env_name", action="store", default="swimmer", type=str)
    parser.add_argument("--alg_type", action="store", default="lambda_ac", type=str)

    args = parser.parse_args()

    if args.sweep_idx == -1:
        hypers = dict(
            alg_type=args.alg_type,
            env_name=args.env_name,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
        )
    else:
        exp_path = "../experiments/ppo_variants_brax.json"
        with open(exp_path, "r") as f:
            d = json.load(f)
        exp = ExperimentDescription(d)
        hypers = exp.getPermutation(args.sweep_idx)["metaParameters"]

    config = {
        "ACTOR_LR": hypers["actor_lr"],
        "CRITIC_LR": hypers["critic_lr"],
        "NUM_ENVS": 128,
        "NUM_STEPS": 10,
        "TOTAL_TIMESTEPS": 3e6,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 32,
        "GAMMA": 0.99,
        "GAE_LAMBDA": hypers["gae_lambda"],
        "CLIP_EPS": 0.2,
        "ENT_COEF": hypers["ent_coef"],
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ENV_NAME": hypers["env_name"],
        "ANNEAL_LR": False,
        "DEBUG": True,
        "EMA_RATE": 0.02,
        "ADVN_NORM": "OFF",
        "SYMLOG_CRITIC_TARGETS": False,
        "NORMALIZE_OBS": False,
        "SYMLOG_OBS": False,
    }

    if hypers["alg_type"] == "advn_norm_ema":
        config["ADVN_NORM"] = "EMA_PERC"
    elif hypers["alg_type"] == "advn_norm_max_ema":
        config["ADVN_NORM"] = "MAX_EMA_PERC"
    elif hypers["alg_type"] == "advn_norm_mean":
        config["ADVN_NORM"] = "MEAN"
    elif hypers["alg_type"] == "symlog_critic_targets":
        config["SYMLOG_CRITIC_TARGETS"] = True
    elif hypers["alg_type"]:
        config["SYMLOG_OBS"] = True
    elif hypers["alg_type"] == "norm_obs":
        config["NORMALIZE_OBS"] = True

    rng = jax.random.PRNGKey(args.start_seed)

    rngs = jax.random.split(rng, args.num_seeds)

    train = make_train(config)
    train_jit = jax.jit(jax.vmap(train))
    out = train_jit(rngs)

    if args.sweep_idx == -1:
        returns = out["metrics"]["returned_episode_returns"]
        jnp.save("./returns.npy", returns)

    else:

        file_context = exp.buildSaveContext(args.sweep_idx)
        file_context.ensureExists()
        path_returns = file_context.resolve("returns.npy")
        path_timestep = file_context.resolve("timestep.npy")
        path_lengths = file_context.resolve("lengths.npy")
        path_completed = file_context.resolve("completed.npy")

        print("saving file to: " + path_returns)

        returns = out["metrics"]["returned_episode_returns"]
        jnp.save(path_returns, returns)
        timestep = out["metrics"]["timestep"]
        jnp.save(path_timestep, timestep)
        lengths = out["metrics"]["returned_episode_lengths"]
        jnp.save(path_lengths, lengths)
        completed_episodes = out["metrics"]["returned_episode"]
        jnp.save(path_completed, completed_episodes)
