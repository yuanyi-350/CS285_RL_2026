from typing import Optional, Sequence
import numpy as np
import torch

from networks.critics import ValueCritic
from networks.policies import MLPPolicyPG
from networks.ppo_policy import MLPPolicyPPO
from networks.trpo_policy import MLPPolicyTRPO
from infrastructure import pytorch_util as ptu
from torch import nn


class PGAgent(nn.Module):
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        gamma: float,
        learning_rate: float,
        use_baseline: bool,
        use_reward_to_go: bool,
        baseline_learning_rate: Optional[float],
        baseline_gradient_steps: Optional[int],
        gae_lambda: Optional[float],
        normalize_advantages: bool,
        algorithm: str = "pg",
        policy_kwargs: Optional[dict] = None,
        optimizer: str = "adam",
        optimizer_kwargs: Optional[dict] = None,
    ):
        super().__init__()

        actor_classes = {"pg": MLPPolicyPG, "ppo": MLPPolicyPPO, "trpo": MLPPolicyTRPO}
        if algorithm not in actor_classes:
            raise ValueError("algorithm must be pg, ppo, or trpo")
        self.algorithm = algorithm

        # create the actor (policy) network
        actor_class = actor_classes[algorithm]
        self.actor = actor_class(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            **(policy_kwargs or {}),
        )

        # create the critic (baseline) network, if needed
        if use_baseline:
            self.critic = ValueCritic(
                ob_dim, n_layers, layer_size, baseline_learning_rate,
                optimizer=optimizer, optimizer_kwargs=optimizer_kwargs,
            )
            self.baseline_gradient_steps = baseline_gradient_steps
        else:
            self.critic = None

        # other agent parameters
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

    def update(
        self,
        # 每条轨迹保存 s_0, ..., s_T，shape=(T_i+1, ob_dim)，比动作和奖励多一个最终状态。
        obs: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        # 真正终止为 1，时间/长度截断为 0；shape=(T_i,)，轨迹边界由分组确定。
        terminated: Sequence[np.ndarray],
    ) -> dict:
        """The train step for PG involves updating its actor using the given observations/actions and the calculated
        qvals/advantages that come from the seen rewards.

        Each input is a list of NumPy arrays, where each array corresponds to a single trajectory. The batch size is the
        total number of transitions across all trajectories (i.e. the sum of the lengths of the reward arrays).
        """

        # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards)

        # step 2: calculate advantages from Q values
        advantages, value_targets = self._estimate_advantage(obs, rewards, q_values, terminated)

        # Flatten the lists of trajectories into one batch. If there are N
        # trajectories with lengths T_1, ..., T_N, then
        # batch_size = T_1 + ... + T_N. Before concatenation:
        #   obs[i].shape       == (T_i+1, ob_dim)
        #   actions[i].shape   == (T_i,)         for discrete actions
        #                      == (T_i, ac_dim)   for continuous actions
        #   rewards[i].shape   == (T_i,)
        #   terminated[i].shape == (T_i,)
        #   q_values[i].shape  == (T_i,)
        # Actor/Critic 更新使用 s_0, ..., s_{T-1}，最终状态仅用于 GAE 的 bootstrap。
        obs = np.concatenate([path_obs[:-1] for path_obs in obs], axis=0)
        # obs.shape == (batch_size, ob_dim)

        actions = np.concatenate(actions, axis=0)
        # actions.shape == (batch_size,) for discrete actions, or
        #                  (batch_size, ac_dim) for continuous actions

        # step 3: use all datapoints (s_t, a_t, adv_t) to update the PG actor/policy
        # Update the PG actor/policy network once using the advantages.
        info = self.actor.update(obs, actions, advantages)

        # step 4: if needed, use all datapoints (s_t, a_t, q_t) to update the PG critic/baseline
        if self.critic is not None:
            # Perform `self.baseline_gradient_steps` updates to the critic/baseline network.
            assert self.baseline_gradient_steps is not None
            critic_info = {}
            for _ in range(self.baseline_gradient_steps):
                critic_info = self.critic.update(obs, value_targets)

            info.update(critic_info)

        return info

    def _gae_recursion(self, deltas):
        advantages = np.zeros_like(deltas, dtype=np.float32)
        running_advantage = 0.0
        for i in reversed(range(len(deltas))):
            # Recursively compute advantage estimates starting from timestep T.
            # 每条轨迹单独递推，从末尾的 running_advantage=0 开始，不跨越轨迹边界。
            running_advantage = (
                deltas[i] + self.gamma * self.gae_lambda * running_advantage
            )
            advantages[i] = running_advantage
        return advantages

    def _discounted_return(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns
        a list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}

        Note that all entries of the output list should be the exact same because each sum is from 0 to T (and doesn't
        involve t)!
        """
        rewards = np.asarray(rewards, dtype=np.float32)
        discounts = self.gamma ** np.arange(len(rewards))
        discounted_return = np.sum(discounts * rewards)
        return np.full(rewards.shape, discounted_return, dtype=np.float32)

    def _discounted_reward_to_go(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns a list where the entry
        in each index t is sum_{t'=t}^T gamma^(t'-t) * r_{t'}.
        """
        rewards = np.asarray(rewards, dtype=np.float32)
        reward_to_go = np.zeros_like(rewards)
        running_return = 0.0

        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return
            reward_to_go[t] = running_return

        return reward_to_go

    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """Monte Carlo estimation of the Q function."""

        if not self.use_reward_to_go:
            # Case 1: in trajectory-based PG, we ignore the timestep and instead use the discounted return for the entire
            # trajectory at each point.
            # In other words: Q(s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}
            # Use the full discounted return for every timestep in the trajectory.
            q_values = [self._discounted_return(path_rewards) for path_rewards in rewards]
        else:
            # Case 2: in reward-to-go PG, we only use the rewards after timestep t to estimate the Q-value for (s_t, a_t).
            # In other words: Q(s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            # Use the discounted reward-to-go starting from each timestep.
            q_values = [
                self._discounted_reward_to_go(path_rewards)
                for path_rewards in rewards
            ]

        return q_values

    def _estimate_advantage(
        self,
        obs: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        q_values: Sequence[np.ndarray],
        terminated: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values.

        Computes GAE within each trajectory, then returns flat advantage and value-target arrays.
        """
        value_targets = np.concatenate(q_values)
        if self.critic is None:
            # Without a baseline, use the Monte Carlo Q estimates directly.
            advantages = value_targets.copy()
        else:
            # Run the critic and use its value predictions as a baseline.
            with torch.no_grad():
                path_values = [ptu.to_numpy(self.critic(ptu.from_numpy(path_obs))) for path_obs in obs]
            values = np.concatenate([v[:-1] for v in path_values])
            assert values.shape == value_targets.shape

            if self.gae_lambda is None:
                # Use the Monte Carlo advantage when GAE is disabled.
                advantages = value_targets - values
            else:
                # Compute generalized advantage estimates recursively.
                # v[:-1] = V(s_t)，v[1:] = V(s_{t+1})；只有真正终止时禁用 bootstrap。
                advantages = np.concatenate([
                    self._gae_recursion(r + self.gamma * v[1:] * (1 - term) - v[:-1])
                    for r, v, term in zip(rewards, path_values, terminated)
                ])
                # 在优势归一化之前，用 A_GAE + V_old 固定 Critic 目标。
                value_targets = advantages + values

        # Normalize advantages to mean zero and standard deviation one within the batch.
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8
            )

        return advantages, value_targets
