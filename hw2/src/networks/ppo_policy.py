"""PPO-Clip actor update."""
import torch

from infrastructure import pytorch_util as ptu
from networks.policies import MLPPolicy


class MLPPolicyPPO(MLPPolicy):
    def __init__(self, *args, clip_ratio=0.2, update_epochs=10,
                 minibatch_size=64, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip_ratio = clip_ratio
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size

    def update(self, obs, actions, advantages):
        # 输入是同一批旧策略采样的数据；advantages 已计算好，更新期间保持固定。
        # 离散动作 shape 为 (N,)，连续动作为 (N, ac_dim)；log_prob 都是 (N,)。
        obs, actions, advantages = map(ptu.from_numpy, (obs, actions, advantages))
        if self.discrete:
            actions = actions.long()
        # 固定 log pi_old(a|s)，作为所有 epoch 中概率比的分母。
        # 采样后 Actor 尚未改变，因此在更新开始时计算一次即可。
        with torch.no_grad():
            old_log_probs = self(obs).log_prob(actions)
        for _ in range(self.update_epochs):
            # 重复使用这一批数据，每个 epoch 重新打乱 minibatch。
            indices = torch.randperm(len(obs), device=obs.device)
            for batch in indices.split(self.minibatch_size):
                # 用当前参数重新构造分布，得到可求导的 log pi_theta(a|s)。
                log_probs = self(obs[batch]).log_prob(actions[batch])
                # r_t(theta) = pi_theta(a_t|s_t) / pi_old(a_t|s_t)。
                ratio = (log_probs - old_log_probs[batch]).exp()
                surrogate = ratio * advantages[batch]
                clipped = ratio.clamp(1 - self.clip_ratio, 1 + self.clip_ratio) * advantages[batch]
                # 最大化 min(r_t A_t, clip(r_t) A_t)，因此最小化它的负均值。
                # min 同时处理正、负优势；并不保证新旧策略的 KL 满足硬约束。
                loss = -torch.minimum(surrogate, clipped).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return {"Actor Loss": loss.item()}
