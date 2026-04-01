from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


Tensor = torch.Tensor


@dataclass
class TRPOConfig:
    gamma: float = 0.99
    lam: float = 0.97
    max_kl: float = 0.01
    damping: float = 0.1
    cg_iters: int = 10
    residual_tol: float = 1e-10
    backtrack_iters: int = 10
    backtrack_coeff: float = 0.5
    value_lr: float = 1e-3
    value_epochs: int = 50
    init_log_std: float = -0.5
    hidden_sizes: Tuple[int, ...] = (64, 32, 32)



def flat_grad(grads: Sequence[Tensor | None], params: Sequence[nn.Parameter]) -> Tensor:
    flat: List[Tensor] = []
    for grad, p in zip(grads, params):
        if grad is None:
            flat.append(torch.zeros_like(p).view(-1))
        else:
            flat.append(grad.contiguous().view(-1))
    return torch.cat(flat)



def flat_params(params: Sequence[nn.Parameter]) -> Tensor:
    return torch.cat([p.data.view(-1) for p in params])



def set_params(params: Sequence[nn.Parameter], flat: Tensor) -> None:
    idx = 0
    for p in params:
        num = p.numel()
        p.data.copy_(flat[idx : idx + num].view_as(p))
        idx += num



def conjugate_gradient(
    f_ax: Callable[[Tensor], Tensor],
    b: Tensor,
    cg_iters: int,
    residual_tol: float,
) -> Tensor:
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(cg_iters):
        z = f_ax(p)
        denom = torch.dot(p, z)
        if torch.abs(denom) < 1e-12:
            break
        v = rdotr / denom
        x = x + v * p
        r = r - v * z
        new_rdotr = torch.dot(r, r)
        if new_rdotr < residual_tol:
            break
        mu = new_rdotr / (rdotr + 1e-12)
        p = r + mu * p
        rdotr = new_rdotr
    return x


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_sizes:
            linear = nn.Linear(prev, h)
            nn.init.orthogonal_(linear.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.Tanh())
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Sequence[int], init_log_std: float):
        super().__init__()
        self.backbone = MLP(obs_dim, hidden_sizes)
        self.mean = nn.Linear(self.backbone.out_dim, act_dim)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.zeros_(self.mean.bias)
        self.log_std = nn.Parameter(torch.ones(act_dim) * float(init_log_std))

    def forward(self, obs: Tensor) -> Tuple[Tensor, Tensor]:
        if obs.dtype != torch.float32:
            obs = obs.float()
        feat = self.backbone(obs)
        mean = self.mean(feat)
        std = torch.exp(self.log_std)
        return mean, std


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        self.backbone = MLP(obs_dim, hidden_sizes)
        self.value = nn.Linear(self.backbone.out_dim, 1)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.zeros_(self.value.bias)

    def forward(self, obs: Tensor) -> Tensor:
        if obs.dtype != torch.float32:
            obs = obs.float()
        feat = self.backbone(obs)
        return self.value(feat).squeeze(-1)


class TRPOAgent:
    def __init__(self, obs_dim: int, act_dim: int, config: TRPOConfig | None = None, device: str = "cpu"):
        self.config = config or TRPOConfig()
        self.device = torch.device(device)
        self.policy = PolicyNetwork(obs_dim, act_dim, self.config.hidden_sizes, self.config.init_log_std).to(self.device)
        self.value_net = ValueNetwork(obs_dim, self.config.hidden_sizes).to(self.device)
        self.value_optim = optim.Adam(self.value_net.parameters(), lr=self.config.value_lr)

    def policy_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mean, std = self.policy(obs_t)
            dist = torch.distributions.Normal(mean, std)
            if deterministic:
                raw = mean
            else:
                raw = dist.sample()
            action = torch.tanh(raw)
        return action.cpu().numpy()[0], raw.cpu().numpy()[0]

    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        return self.policy_action(obs, deterministic=deterministic)

    def _prepare_batch(self, rollouts: Sequence[Dict[str, object]]) -> Dict[str, Tensor]:
        obs = torch.as_tensor(np.asarray([r["obs"] for r in rollouts], dtype=np.float32), device=self.device)
        raw_actions = torch.as_tensor(np.asarray([r["raw_action"] for r in rollouts], dtype=np.float32), device=self.device)
        rewards = torch.as_tensor(np.asarray([r["reward"] for r in rollouts], dtype=np.float32), device=self.device)
        masks = torch.as_tensor(np.asarray([r["mask"] for r in rollouts], dtype=np.float32), device=self.device)
        return {"obs": obs, "raw_actions": raw_actions, "rewards": rewards, "masks": masks}

    def compute_returns_advantages(self, rewards: Tensor, values: Tensor, masks: Tensor) -> Tuple[Tensor, Tensor]:
        n = rewards.shape[0]
        returns = torch.zeros_like(rewards)
        adv = torch.zeros_like(rewards)
        gae = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
        running_return = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
        next_value = torch.zeros((), dtype=rewards.dtype, device=rewards.device)

        for t in reversed(range(n)):
            running_return = rewards[t] + self.config.gamma * running_return * masks[t]
            returns[t] = running_return
            delta = rewards[t] + self.config.gamma * next_value * masks[t] - values[t]
            gae = delta + self.config.gamma * self.config.lam * masks[t] * gae
            adv[t] = gae
            next_value = values[t]
        adv_std = adv.std(unbiased=False)
        if not torch.isfinite(adv_std) or float(adv_std.item()) < 1e-12:
            adv = adv - adv.mean()
        else:
            adv = (adv - adv.mean()) / (adv_std + 1e-8)
        return returns, adv

    def update(self, rollouts: Sequence[Dict[str, object]]) -> Dict[str, float]:
        if not rollouts:
            return {"updated": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "kl": 0.0}

        batch = self._prepare_batch(rollouts)
        obs = batch["obs"]
        raw_actions = batch["raw_actions"]
        rewards = batch["rewards"]
        masks = batch["masks"]

        with torch.no_grad():
            values_old = self.value_net(obs)
            returns, advantages = self.compute_returns_advantages(rewards, values_old, masks)
            old_mean, old_std = self.policy(obs)
            old_dist = torch.distributions.Normal(old_mean, old_std)
            old_log_probs = old_dist.log_prob(raw_actions).sum(dim=-1)

        def surrogate_loss(no_grad: bool = False) -> Tensor:
            if no_grad:
                with torch.no_grad():
                    mean, std = self.policy(obs)
            else:
                mean, std = self.policy(obs)
            dist = torch.distributions.Normal(mean, std)
            log_probs = dist.log_prob(raw_actions).sum(dim=-1)
            ratio = torch.exp(log_probs - old_log_probs)
            return -(ratio * advantages).mean()

        def mean_kl() -> Tensor:
            mean, std = self.policy(obs)
            new_dist = torch.distributions.Normal(mean, std)
            old_dist_detached = torch.distributions.Normal(old_mean.detach(), old_std.detach())
            kl = torch.distributions.kl_divergence(old_dist_detached, new_dist).sum(dim=-1).mean()
            return kl

        loss = surrogate_loss()
        params = list(self.policy.parameters())
        grads = torch.autograd.grad(loss, params)
        loss_grad = flat_grad(grads, params)

        if torch.norm(loss_grad) < 1e-12:
            return {"updated": 0.0, "policy_loss": float(loss.item()), "value_loss": 0.0, "kl": 0.0}

        def fisher_vector_product(v: Tensor) -> Tensor:
            kl = mean_kl()
            grad_kl = flat_grad(torch.autograd.grad(kl, params, create_graph=True), params)
            kl_v = (grad_kl * v).sum()
            grad2 = flat_grad(torch.autograd.grad(kl_v, params), params)
            return grad2 + self.config.damping * v

        step_dir = conjugate_gradient(
            fisher_vector_product,
            -loss_grad,
            cg_iters=self.config.cg_iters,
            residual_tol=self.config.residual_tol,
        )

        fvp_step = fisher_vector_product(step_dir)
        shs = 0.5 * (step_dir * fvp_step).sum()
        if torch.isnan(shs) or shs <= 0:
            return {"updated": 0.0, "policy_loss": float(loss.item()), "value_loss": 0.0, "kl": 0.0}

        scale = torch.sqrt(shs / self.config.max_kl)
        full_step = step_dir / (scale + 1e-12)
        old_params = flat_params(params).clone()
        old_loss = float(loss.item())

        accepted = False
        final_kl = 0.0
        for j in range(self.config.backtrack_iters):
            stepfrac = self.config.backtrack_coeff ** j
            new_params = old_params + stepfrac * full_step
            set_params(params, new_params)
            new_loss = float(surrogate_loss(no_grad=True).item())
            new_kl = float(mean_kl().item())
            improvement = old_loss - new_loss
            if improvement > 0 and new_kl <= self.config.max_kl:
                accepted = True
                final_kl = new_kl
                break
        if not accepted:
            set_params(params, old_params)
            final_kl = float(mean_kl().item())

        value_loss_scalar = 0.0
        for _ in range(self.config.value_epochs):
            pred = self.value_net(obs)
            value_loss = torch.mean((pred - returns) ** 2)
            self.value_optim.zero_grad(set_to_none=True)
            value_loss.backward()
            self.value_optim.step()
            value_loss_scalar = float(value_loss.item())

        return {
            "updated": 1.0 if accepted else 0.0,
            "policy_loss": old_loss,
            "value_loss": value_loss_scalar,
            "kl": final_kl,
        }

    def state_dict(self) -> Dict[str, object]:
        return {
            "policy": self.policy.state_dict(),
            "value_net": self.value_net.state_dict(),
            "value_optim": self.value_optim.state_dict(),
            "config": self.config.__dict__,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.policy.load_state_dict(state["policy"])
        self.value_net.load_state_dict(state["value_net"])
        self.value_optim.load_state_dict(state["value_optim"])
