import argparse
import json
import os
from datetime import datetime
import time

import gym
import numpy as np
import torch
import tqdm

from agents.pg_agent import PGAgent
from infrastructure import utils
from infrastructure import pytorch_util as ptu
from infrastructure.log_utils import setup_wandb, Logger, dump_log

MAX_NVIDEO = 2


def run_training_loop(logger, args):
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    env = gym.make(args.env_name, render_mode=None)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    max_ep_len = args.ep_len or env.spec.max_episode_steps

    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    # simulation timestep, will be used for video saving
    if hasattr(env, "model"):
        fps = 1 / env.dt
    else:
        fps = env.env.metadata["render_fps"]

    # initialize agent
    policy_kwargs = {}
    if args.algo == "ppo":
        policy_kwargs = dict(
            clip_ratio=args.clip_ratio, update_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
        )
    elif args.algo == "trpo":
        policy_kwargs = dict(
            max_kl=args.max_kl,
        )
    agent = PGAgent(
        ob_dim,
        ac_dim,
        discrete,
        n_layers=args.n_layers,
        layer_size=args.layer_size,
        gamma=args.discount,
        learning_rate=args.learning_rate,
        use_baseline=args.use_baseline,
        use_reward_to_go=args.use_reward_to_go,
        normalize_advantages=args.normalize_advantages,
        baseline_learning_rate=args.baseline_learning_rate,
        baseline_gradient_steps=args.baseline_gradient_steps,
        gae_lambda=args.gae_lambda,
        algorithm=args.algo,
        policy_kwargs=policy_kwargs,
        optimizer=args.optimizer,
        optimizer_kwargs=args.optimizer_kwargs,
    )

    total_envsteps = 0
    start_time = time.time()

    for itr in range(args.n_iter):
        print(f"\n********** Iteration {itr} ************")
        # Collect at least `args.batch_size` transitions with the current policy.
        trajs, envsteps_this_batch = utils.sample_trajectories(
            env, agent.actor, args.batch_size, max_ep_len
        )
        total_envsteps += envsteps_this_batch

        # trajs should be a list of dictionaries of NumPy arrays, where each dictionary corresponds to a trajectory.
        # this line converts this into a single dictionary of lists of NumPy arrays.
        trajs_dict = {k: [traj[k] for traj in trajs] for k in trajs[0]}

        # Update the actor and, when enabled, the value-function baseline.
        train_info: dict = agent.update(
            trajs_dict["observation"],
            trajs_dict["action"],
            trajs_dict["reward"],
            trajs_dict["terminated"],
        )

        if itr % args.scalar_log_freq == 0:
            # save eval metrics
            print("\nCollecting data for eval...")
            eval_trajs, eval_envsteps_this_batch = utils.sample_trajectories(
                env, agent.actor, args.eval_batch_size, max_ep_len
            )

            logs = utils.compute_metrics(trajs, eval_trajs)
            # compute additional metrics
            logs.update(train_info)
            logs["Train_EnvstepsSoFar"] = total_envsteps
            logs["TimeSinceStart"] = time.time() - start_time
            if itr == 0:
                logs["Initial_DataCollection_AverageReturn"] = logs[
                    "Train_AverageReturn"
                ]

            # perform the logging
            for key, value in logs.items():
                print("{} : {}".format(key, value))
            logger.log(logs, itr)
            print("Done logging...\n\n", flush=True)

        if args.video_log_freq != -1 and itr % args.video_log_freq == 0:
            print("\nCollecting video rollouts...")
            eval_video_trajs = utils.sample_n_trajectories(
                env, agent.actor, MAX_NVIDEO, max_ep_len, render=True
            )

            logger.log_trajs_as_videos(
                eval_video_trajs,
                itr,
                fps=fps,
                max_videos_to_save=MAX_NVIDEO,
                video_title="eval_rollouts",
            )

    dump_log(agent, logger, args, args.save_dir)


def setup_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=("pg", "ppo", "trpo"), default="pg")
    parser.add_argument("--env_name", type=str, default='CartPole-v0')
    parser.add_argument("--exp_name", type=str, default='exp')
    parser.add_argument("--n_iter", "-n", type=int, default=200)

    parser.add_argument("--use_reward_to_go", "-rtg", action="store_true")
    parser.add_argument("--use_baseline", action="store_true")
    parser.add_argument("--baseline_learning_rate", "-blr", type=float, default=None)
    parser.add_argument("--baseline_gradient_steps", "-bgs", type=int, default=None)
    parser.add_argument("--gae_lambda", type=float, default=None)
    parser.add_argument("--normalize_advantages", "-na", action="store_true")
    parser.add_argument(
        "--batch_size", "-b", type=int, default=1000
    )  # steps collected per train iteration
    parser.add_argument(
        "--eval_batch_size", "-eb", type=int, default=400
    )  # steps collected per eval iteration

    parser.add_argument("--discount", type=float, default=None)
    parser.add_argument("--learning_rate", "-lr", type=float, default=None)
    parser.add_argument("--optimizer", choices=("gd", "adam"), default="adam")
    parser.add_argument("--optimizer_kwargs", type=json.loads, default=None)
    parser.add_argument("--n_layers", "-l", type=int, default=2)
    parser.add_argument("--layer_size", "-s", type=int, default=64)

    parser.add_argument(
        "--ep_len", type=int
    )  # students shouldn't change this away from env's default
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--video_log_freq", type=int, default=-1)
    parser.add_argument("--scalar_log_freq", type=int, default=1)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--ppo_epochs", type=int, default=10)
    parser.add_argument("--minibatch_size", type=int, default=64)
    parser.add_argument("--max_kl", type=float, default=0.01)

    args = parser.parse_args(args=args)
    modern = args.algo != "pg"
    defaults = dict(discount=0.99 if modern else 1.0,
                    learning_rate=3e-4 if modern else 5e-3,
                    baseline_learning_rate=1e-3 if modern else 5e-3,
                    baseline_gradient_steps=20 if modern else 5)
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def main(args):
    # Create directory for logging
    logdir_prefix = "exp"  # Keep for autograder

    exp_name = f"{args.env_name}_{args.exp_name}_sd{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    config = vars(args)
    setup_wandb(project='cs285_hw2', name=exp_name, config=config)
    args.save_dir = os.path.join(logdir_prefix, exp_name)
    os.makedirs(args.save_dir, exist_ok=True)
    logger = Logger(os.path.join(args.save_dir, 'log.csv'))

    run_training_loop(logger, args)


if __name__ == "__main__":
    args = setup_arguments()
    main(args)
