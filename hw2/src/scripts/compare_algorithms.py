"""A quick, single-seed PG/PPO/TRPO experiment. No W&B or videos."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def run_one(args, env_name, algorithm, output, run_name=None):
    import gym
    import numpy as np
    import torch
    from agents.pg_agent import PGAgent
    from infrastructure import pytorch_util as ptu
    from infrastructure import utils

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = gym.make(env_name, new_step_api=True, disable_env_checker=True)
    eval_env = gym.make(env_name, new_step_api=True, disable_env_checker=True)
    env.reset(seed=args.seed)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    policy_kwargs = {}
    if algorithm == "ppo":
        policy_kwargs = dict(clip_ratio=0.2, update_epochs=5, minibatch_size=256)
    elif algorithm == "trpo":
        policy_kwargs = dict(max_kl=0.01)
    learning_rate = args.ppo_lr if algorithm == "ppo" else 3e-3 if algorithm == "pg" else 3e-4
    agent = PGAgent(
        ob_dim=env.observation_space.shape[0],
        ac_dim=env.action_space.n if discrete else env.action_space.shape[0],
        discrete=discrete, n_layers=2, layer_size=64,
        gamma=0.99, learning_rate=learning_rate,
        use_baseline=True, use_reward_to_go=True,
        baseline_learning_rate=1e-3, baseline_gradient_steps=20,
        gae_lambda=0.95, normalize_advantages=True,
        algorithm=algorithm, policy_kwargs=policy_kwargs,
    )

    class EvaluationPolicy:
        @torch.no_grad()
        def get_action(self, observation):
            distribution = agent.actor(ptu.from_numpy(observation[None]))
            action = distribution.probs.argmax(-1) if discrete else distribution.mean
            return ptu.to_numpy(action)[0]

    def evaluate():
        trajectory = utils.sample_trajectory(
            eval_env, EvaluationPolicy(), env.spec.max_episode_steps, seed=100_000
        )
        return float(trajectory["reward"].sum())

    folder = output / f"{env_name}_{run_name or algorithm}"
    folder.mkdir()
    config = dict(vars(args), env=env_name, algorithm=algorithm,
                  actor_learning_rate=learning_rate,
                  policy_kwargs=policy_kwargs, gamma=0.99, gae_lambda=0.95,
                  use_baseline=True, normalize_advantages=True,
                  baseline_learning_rate=1e-3, baseline_gradient_steps=20,
                  n_layers=2, layer_size=64)
    (folder / "config.json").write_text(json.dumps(config, indent=2))
    steps = iteration = 0
    train_seconds = 0.0
    started = time.perf_counter()
    rows = []
    fields = ["iteration", "train_steps", "train_seconds", "elapsed_seconds",
              "train_return", "eval_return"]
    with (folder / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()

        def record(train_return=""):
            value = evaluate()
            row = dict(iteration=iteration, train_steps=steps, train_seconds=train_seconds,
                       elapsed_seconds=time.perf_counter()-started,
                       train_return=train_return, eval_return=value)
            writer.writerow(row)
            stream.flush()
            rows.append(row)
            print(f"  {algorithm.upper():4} steps={steps:6} eval_return={value:8.1f}", flush=True)

        record()
        # Evaluate every ~2000 training steps, including the final update.
        next_eval = min(2000, args.steps)
        while steps < args.steps:
            tick = time.perf_counter()
            trajectories, count = utils.sample_trajectories(
                env, agent.actor, min(args.batch_size, args.steps-steps), env.spec.max_episode_steps
            )
            agent.update([t["observation"] for t in trajectories],
                         [t["action"] for t in trajectories],
                         [t["reward"] for t in trajectories],
                         [t["terminated"] for t in trajectories])
            if args.device == "mps":
                torch.mps.synchronize()
            train_seconds += time.perf_counter()-tick
            steps += count
            iteration += 1
            if steps >= next_eval or steps >= args.steps:
                record(float(np.mean([t["reward"].sum() for t in trajectories])))
                next_eval = steps + 2000
            torch.save(agent.state_dict(), folder / "agent.pt")
    env.close()
    eval_env.close()
    return rows


def plot(env_name, runs, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for algorithm, rows in runs.items():
        rewards = [row["eval_return"] for row in rows]
        axes[0].plot([row["train_steps"] for row in rows], rewards, marker=".", label=algorithm.upper())
        axes[1].plot([row["train_seconds"] for row in rows], rewards, marker=".", label=algorithm.upper())
    for axis in axes:
        axis.set_ylabel("Evaluation return (one deterministic episode)")
        axis.grid(alpha=0.2)
        axis.legend()
    axes[0].set_xlabel("Training environment steps")
    axes[1].set_xlabel("Training time (s, excluding evaluation)")
    figure.suptitle(env_name)
    figure.tight_layout()
    figure.savefig(output / f"{env_name}.png", dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--algorithms", nargs="+", choices=["pg", "ppo", "trpo"],
                        default=["pg", "ppo", "trpo"])
    parser.add_argument("--ppo-lr", type=float, default=3e-4)
    parser.add_argument("--ppo-lrs", nargs="+", type=float,
                        help="Compare these PPO learning rates; runs PPO only")
    parser.add_argument("--envs", nargs="+", default=["LunarLander-v2", "HalfCheetah-v4"])
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.steps <= 0 or args.batch_size <= 0:
        parser.error("steps and batch-size must be positive")
    if args.ppo_lr <= 0:
        parser.error("ppo-lr must be positive")
    if args.ppo_lrs and any(lr <= 0 for lr in args.ppo_lrs):
        parser.error("ppo-lrs must be positive")
    import torch
    from infrastructure import pytorch_util as ptu

    torch.set_num_threads(1)
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    ptu.device = torch.device(args.device)
    output = Path(args.output or ROOT / "exp" / time.strftime("comparison_%Y%m%d_%H%M%S")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    print(f"Results: {output}\nDevice: {args.device}; seed: {args.seed}; steps/run: {args.steps}", flush=True)
    experiments = ([("ppo", lr) for lr in args.ppo_lrs] if args.ppo_lrs else
                   [(algorithm, args.ppo_lr) for algorithm in args.algorithms])
    started = time.perf_counter()
    with (output / "summary.csv").open("w", newline="") as stream:
        fields = ["env", "algorithm", "ppo_lr", "train_steps", "train_seconds", "elapsed_seconds", "eval_return"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for env_name in args.envs:
            runs = {}
            for algorithm, learning_rate in experiments:
                args.ppo_lr = learning_rate
                label = f"ppo lr={learning_rate:g}" if args.ppo_lrs else algorithm
                run_name = f"ppo_lr{learning_rate:g}" if args.ppo_lrs else None
                print(f"\n{env_name} / {label.upper()}", flush=True)
                rows = run_one(args, env_name, algorithm, output, run_name=run_name)
                runs[label] = rows
                last = rows[-1]
                row = dict(env=env_name, algorithm=algorithm,
                           ppo_lr=learning_rate if algorithm == "ppo" else "",
                           **{key: last[key] for key in fields[3:]})
                writer.writerow(row)
                stream.flush()
            plot(env_name, runs, output)
    print(f"\nDone in {time.perf_counter()-started:.1f}s. Plots and summary: {output}", flush=True)


if __name__ == "__main__":
    main()
