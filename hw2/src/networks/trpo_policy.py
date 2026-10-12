"""TRPO: implicit Fisher products, conjugate gradient, and KL-constrained search."""
import math
import torch
from torch.distributions import kl_divergence
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from infrastructure import pytorch_util as ptu
from networks.policies import MLPPolicy


def conjugate_gradient(matrix_vector_product, b, iterations=10, tolerance=1e-10):
    """Approximately solve F x = b without constructing the matrix F."""
    # 只需要能计算 Fv 的函数，不需要存储完整矩阵 F。
    # residual = b - Fx；direction 是本轮的共轭搜索方向。
    x = torch.zeros_like(b)
    residual = b.clone()
    direction = residual.clone()
    residual_squared = torch.dot(residual, residual)
    for _ in range(iterations):
        if residual_squared <= tolerance:
            break
        product = matrix_vector_product(direction)
        curvature = torch.dot(direction, product)
        # 正定系统应有 direction^T F direction > 0；异常时停止，避免错误步长。
        if not torch.isfinite(curvature) or curvature <= 0:
            break
        alpha = residual_squared / curvature
        x += alpha * direction
        residual -= alpha * product
        next_squared = torch.dot(residual, residual)
        # 用新残差构造下一条共轭方向，而不是每次都沿梯度方向走。
        direction = residual + (next_squared / residual_squared) * direction
        residual_squared = next_squared
    return x


class MLPPolicyTRPO(MLPPolicy):
    def __init__(self, *args, max_kl=0.01, cg_iters=10, cg_damping=0.1,
                 backtrack_iters=10, backtrack_coeff=0.8,
                 accept_ratio=0.1, **kwargs):
        # TRPO Actor 直接更新参数，不调用 Adam 等 optimizer.step()。
        kwargs["optimizer"] = None
        super().__init__(*args, **kwargs)
        if max_kl <= 0 or cg_iters < 1 or cg_damping < 0 or backtrack_iters < 1:
            raise ValueError("Invalid TRPO KL/CG/backtracking settings")
        if not 0 < backtrack_coeff < 1 or not 0 <= accept_ratio < 1:
            raise ValueError("Invalid TRPO backtracking/acceptance ratio")
        self.max_kl = max_kl
        self.cg_iters = cg_iters
        self.cg_damping = cg_damping
        self.backtrack_iters = backtrack_iters
        self.backtrack_coeff = backtrack_coeff
        self.accept_ratio = accept_ratio

    def update(self, obs, actions, advantages):
        # 动作、优势固定；优化的是采样后更新前的策略参数。
        obs, actions, advantages = map(ptu.from_numpy, (obs, actions, advantages))
        if self.discrete:
            actions = actions.long()
        parameters = tuple(self.parameters())
        with torch.no_grad():
            # 保存旧分布，用于解析 KL(pi_old || pi_theta)，不参与梯度计算。
            # 这里保存的是分布的数值快照，不需要复制整个网络。
            old_distribution = self(obs)
            # 采样后 Actor 尚未改变，在本次更新开始时固定旧概率。
            old_log_probs = old_distribution.log_prob(actions)
            # 将参数展平成向量，方便添加搜索步长以及失败时恢复。
            old_parameters = parameters_to_vector(parameters).clone()

        def objective():
            # L(theta) = mean[(pi_theta / pi_old) * A]，TRPO 要最大化这个目标。
            ratio = (self(obs).log_prob(actions) - old_log_probs).exp()
            return (ratio * advantages).mean()

        def mean_kl():
            # 对每个采样状态的完整动作分布计算 KL，再取状态样本均值。
            return kl_divergence(old_distribution, self(obs)).mean()

        def flat_gradient(value, create_graph=False):
            # 将各参数张量的梯度拼接，顺序与参数向量一致。
            gradients = torch.autograd.grad(value, parameters, create_graph=create_graph)
            return torch.cat([gradient.reshape(-1) for gradient in gradients])

        def fisher_vector_product(vector):
            # F 是旧参数处平均 KL 的 Hessian，等于这里使用的 Fisher 矩阵。
            # Fv = grad_theta[(grad_theta KL)^T v]：两次求导得到矩阵向量积。
            # create_graph=True 保留第一次求导的图，才能继续求二阶导数。
            kl_gradient = flat_gradient(mean_kl(), create_graph=True)
            product = flat_gradient(torch.dot(kl_gradient, vector)).detach()
            # 实际求解 (F + damping * I)x = g，阻尼改善数值稳定性。
            return product + self.cg_damping * vector

        initial_objective = objective()
        # g 是目标梯度；共轭梯度近似求阻尼后的自然梯度方向 (F + damping * I)^{-1}g。
        gradient = flat_gradient(initial_objective).detach()
        initial_value = initial_objective.item()
        accepted, attempts, expected_improvement = False, 0, 0.0
        if torch.isfinite(gradient).all() and gradient.norm() > 1e-12:
            direction = conjugate_gradient(fisher_vector_product, gradient, self.cg_iters)
            curvature = torch.dot(direction, fisher_vector_product(direction)).item()
            if math.isfinite(curvature) and curvature > 0:
                # KL 的局部近似为 1/2 * step^T F step。
                # 按二次约束缩放方向；这里使用加入阻尼的 F 来计算曲率。
                full_step = direction * math.sqrt(2 * self.max_kl / curvature)
                expected_improvement = torch.dot(gradient, full_step).item()
                # 局部近似不保证实际 KL 可行，因此需要回溯检查真实目标和 KL。
                try:
                    for attempt in range(self.backtrack_iters):
                        attempts = attempt + 1
                        fraction = self.backtrack_coeff ** attempt
                        with torch.no_grad():
                            vector_to_parameters(old_parameters + fraction * full_step, parameters)
                            candidate = objective().item()
                            candidate_kl = mean_kl().item()
                        improvement = candidate - initial_value
                        # 候选步必须改善目标、满足实测 KL 上限，并达到预期改善的一定比例。
                        if (math.isfinite(candidate) and math.isfinite(candidate_kl)
                                and candidate_kl <= self.max_kl and improvement > 0
                                and improvement >= self.accept_ratio * fraction * expected_improvement):
                            accepted = True
                            break
                finally:
                    # 没有可接受的候选步（或搜索异常）时，完整恢复旧参数。
                    if not accepted:
                        with torch.no_grad():
                            vector_to_parameters(old_parameters, parameters)
        with torch.no_grad():
            final_value = objective().item()
            final_kl = mean_kl().item()
        return {
            "Actor Loss": -final_value,
            "KL": final_kl,
            "Surrogate Improvement": final_value - initial_value,
            "Expected Improvement": expected_improvement,
            "TRPO Step Accepted": int(accepted),
            "Backtrack Attempts": attempts,
        }
