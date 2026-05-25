import argparse
import csv
import math
import multiprocessing as mp
import os
import pathlib
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from sac_agent import GaussianPolicy, DiscretePolicy, ReplayBuffer
from iq_learn import ExpertDataset, evaluate


# ── Networks ──────────────────────────────────────────────────────────────────

class EnsembleQ(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, discrete, L):
        super().__init__()
        self.discrete = discrete
        self.L = L
        in_dim  = state_dim if discrete else state_dim + action_dim
        out_dim = action_dim if discrete else 1
        self.nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            ) for _ in range(L)
        ])

    def forward(self, s, a=None):
        x = s if self.discrete else torch.cat([s, a], -1)
        return [net(x) for net in self.nets]

    def mean_std(self, s, a=None):
        qs = torch.stack(self.forward(s, a), 0)
        return qs.mean(0), qs.std(0)


# ── Agent ─────────────────────────────────────────────────────────────────────

class CSILAgent:
    def __init__(self, state_dim, action_dim, discrete, action_low, action_high,
                 hidden_dim=256, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2,
                 batch_size=256, buffer_size=100_000,
                 soar=False, ensemble_size=4, soar_beta=1.0,
                 bc_steps=5_000, scale_factor=0.1, grad_norm_sf=0.1):

        self.discrete = discrete
        self.gamma, self.tau, self.alpha = gamma, tau, alpha
        self.batch_size = batch_size
        self.soar, self.soar_beta = soar, soar_beta
        self.bc_steps = bc_steps
        self.scale_factor = scale_factor
        self.grad_norm_sf = grad_norm_sf

        L = ensemble_size if soar else 2
        self.critic        = EnsembleQ(state_dim, action_dim, hidden_dim, discrete, L)
        self.critic_target = EnsembleQ(state_dim, action_dim, hidden_dim, discrete, L)
        self.critic_target.load_state_dict(self.critic.state_dict())

        Policy = DiscretePolicy if discrete else GaussianPolicy
        self.actor     = Policy(state_dim, action_dim, hidden_dim)
        self.bc_policy = Policy(state_dim, action_dim, hidden_dim)

        self.opt_critic = optim.Adam(self.critic.parameters(),   lr=lr)
        self.opt_actor  = optim.Adam(self.actor.parameters(),    lr=lr)
        self.opt_bc     = optim.Adam(self.bc_policy.parameters(), lr=lr)
        self.replay     = ReplayBuffer(buffer_size)

        if not discrete and action_low is not None:
            self._lo = torch.FloatTensor(action_low)
            self._hi = torch.FloatTensor(action_high)

    def _scale(self, a_unit):
        return self._lo + (a_unit + 1.0) * 0.5 * (self._hi - self._lo)

    def _bc_logp(self, states, a_unit):
        a_unit = a_unit.clamp(-1 + 1e-6, 1 - 1e-6)
        mean, log_std = self.bc_policy.forward(states)
        std = log_std.exp()
        z = torch.atanh(a_unit)
        lp = (-0.5 * ((z - mean) / std) ** 2
              - log_std
              - 0.5 * math.log(2 * math.pi)
              - torch.log(1 - a_unit.pow(2) + 1e-6))
        return lp.sum(-1, keepdim=True)

    def _prior_logp(self, a_unit):
        a_unit = a_unit.clamp(-1 + 1e-6, 1 - 1e-6)
        z = torch.atanh(a_unit)
        lp = (-0.5 * z ** 2
              - 0.5 * math.log(2 * math.pi)
              - torch.log(1 - a_unit.pow(2) + 1e-6))
        return lp.sum(-1, keepdim=True)

    def _shaped_reward(self, states, actions_env):
        with torch.no_grad():
            if self.discrete:
                lp_bc    = self.bc_policy.get_action_probs(states)[1].gather(1, actions_env.long().view(-1,1))
                lp_prior = torch.full((states.shape[0], 1),
                                      -math.log(self.actor.net[-1].out_features),
                                      device=states.device)
            else:
                a_unit = ((actions_env - self._lo) / (0.5 * (self._hi - self._lo)) - 1.0)
                lp_bc    = self._bc_logp(states, a_unit)
                lp_prior = self._prior_logp(a_unit)
        return self.alpha * (lp_bc - lp_prior)

    def _soft_V(self, states, use_target=True):
        critic = self.critic_target if use_target else self.critic
        if self.discrete:
            probs, log_probs = self.actor.get_action_probs(states)
            bc_lp = self.bc_policy.get_action_probs(states)[1].detach()
            q = torch.stack(critic.forward(states), 0).min(0).values
            return (probs * (q - self.alpha * (log_probs - bc_lp))).sum(-1, keepdim=True)
        else:
            a_unit, log_probs = self.actor.sample(states)
            bc_lp = self._bc_logp(states, a_unit.detach()).detach()
            q = torch.stack(critic.forward(states, self._scale(a_unit)), 0).min(0).values
            return q - self.alpha * (log_probs - bc_lp)

    def _critic_loss(self, e_s, e_a, e_ns, e_d, o_s=None, o_a=None, o_ns=None, o_d=None):
        r_e = self._shaped_reward(e_s, e_a)

        if o_s is not None:
            r_o   = self._shaped_reward(o_s, o_a)
            all_s  = torch.cat([e_s, o_s])
            all_a  = torch.cat([e_a, o_a])
            all_ns = torch.cat([e_ns, o_ns])
            all_d  = torch.cat([e_d, o_d])
            all_r  = torch.cat([r_e, r_o])
        else:
            r_o = None
            all_s, all_a, all_ns, all_d, all_r = e_s, e_a, e_ns, e_d, r_e

        with torch.no_grad():
            v_next  = self._soft_V(all_ns, use_target=True).clamp(-1000, 1000)
            target  = (all_r + self.gamma * (1 - all_d) * v_next).clamp(-1000, 1000)

        qs = self.critic.forward(all_s, None if self.discrete else all_a)
        if self.discrete:
            qs = [q.gather(1, all_a.long().view(-1,1)) for q in qs]
        bellman = sum(((q - target)**2).mean() for q in qs) / len(qs)

        r_reg = r_o if r_o is not None else r_e
        kl_est = (torch.exp(-r_reg.clamp(min=-5)) - 1 + r_reg).mean()

        grad_loss = torch.tensor(0.0)
        if not self.discrete and self.grad_norm_sf > 0:
            a_req = all_a.detach().requires_grad_(True)
            q_sum = torch.stack(self.critic.forward(all_s, a_req), 0).mean(0).sum()
            grads = torch.autograd.grad(q_sum, a_req, create_graph=True)[0]
            grad_loss = (grads**2).sum(-1).mean().sqrt()

        return bellman - r_e.mean() + self.scale_factor * kl_est + self.grad_norm_sf * grad_loss

    def _actor_loss(self, states):
        if self.discrete:
            probs, log_probs = self.actor.get_action_probs(states)
            bc_lp = self.bc_policy.get_action_probs(states)[1].detach()
            if self.soar:
                mean_q, std_q = self.critic.mean_std(states)
                q = mean_q + self.soar_beta * std_q
            else:
                q = torch.stack(self.critic.forward(states), 0).min(0).values
            return (probs * (self.alpha * (log_probs - bc_lp) - q)).sum(-1).mean()
        else:
            a_unit, log_probs = self.actor.sample(states)
            bc_lp = self._bc_logp(states, a_unit.detach()).detach()
            if self.soar:
                mean_q, std_q = self.critic.mean_std(states, self._scale(a_unit))
                q = mean_q + self.soar_beta * std_q
            else:
                q = torch.stack(self.critic.forward(states, self._scale(a_unit)), 0).min(0).values
            return (self.alpha * (log_probs - bc_lp) - q).mean()

    def pretrain_bc(self, expert, n_steps=None):
        n_steps = n_steps or self.bc_steps
        for _ in range(n_steps):
            e_s, e_a, _, _ = expert.sample(self.batch_size)
            if self.discrete:
                loss = -self.bc_policy.get_action_probs(e_s)[1].gather(1, e_a.long().view(-1,1)).mean()
            else:
                a_unit = ((e_a - self._lo) / (0.5 * (self._hi - self._lo)) - 1.0).clamp(-1+1e-6, 1-1e-6)
                loss = -self._bc_logp(e_s, a_unit).mean()
            self.opt_bc.zero_grad(); loss.backward(); self.opt_bc.step()
        self.actor.load_state_dict(self.bc_policy.state_dict())

    def update(self, expert):
        e_s, e_a, e_ns, e_d = expert.sample(self.batch_size)
        has_online = len(self.replay) >= self.batch_size

        if has_online:
            o_s, o_a_raw, _, o_ns, o_d = self.replay.sample(self.batch_size)
            o_a = o_a_raw.long().view(-1,1) if self.discrete else o_a_raw
            closs = self._critic_loss(e_s, e_a, e_ns, e_d, o_s, o_a, o_ns, o_d)
            actor_states = o_s
        else:
            closs = self._critic_loss(e_s, e_a, e_ns, e_d)
            actor_states = e_s

        self.opt_critic.zero_grad()
        closs.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.opt_critic.step()

        aloss = self._actor_loss(actor_states)
        self.opt_actor.zero_grad()
        aloss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.opt_actor.step()

        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)

        return {"critic_loss": closs.item(), "actor_loss": aloss.item()}

    def select_action(self, state, evaluate=False):
        s = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            a = self.actor.deterministic(s) if evaluate else self.actor.sample(s)[0]
        if self.discrete:
            return a.item()
        return self._lo.numpy() + (a.squeeze(0).numpy() + 1.0) * 0.5 * (self._hi.numpy() - self._lo.numpy())


# ── Training loop ─────────────────────────────────────────────────────────────

def train_csil(env_name, expert_path, seed=42, total_steps=50_000,
               eval_interval=1000, eval_episodes=5,
               hidden_dim=256, batch_size=256, lr=3e-4, alpha=0.2,
               soar=False, ensemble_size=4, soar_beta=1.0,
               bc_steps=5_000, scale_factor=0.1, grad_norm_sf=0.1,
               verbose=True):

    np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)

    env      = gym.make(env_name)
    eval_env = gym.make(env_name)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n if discrete else env.action_space.shape[0]

    expert = ExpertDataset(expert_path, discrete=discrete)
    agent  = CSILAgent(
        state_dim, action_dim, discrete,
        action_low  = None if discrete else env.action_space.low,
        action_high = None if discrete else env.action_space.high,
        hidden_dim=hidden_dim, lr=lr, alpha=alpha,
        batch_size=batch_size,
        soar=soar, ensemble_size=ensemble_size, soar_beta=soar_beta,
        bc_steps=bc_steps, scale_factor=scale_factor, grad_norm_sf=grad_norm_sf,
    )

    algo = "CSIL+SOAR" if soar else "CSIL"
    if verbose:
        print(f"[{algo}] BC pre-training ({bc_steps} steps)…")
    agent.pretrain_bc(expert)
    if verbose:
        print(f"[{algo}] BC reward = {evaluate(agent, eval_env, eval_episodes):.2f}")

    log = {"step": [], "eval_reward": [], "critic_loss": [], "actor_loss": []}
    state, _ = env.reset(seed=seed)

    for step in range(1, total_steps + 1):
        action = agent.select_action(state)
        next_state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.replay.push(state, [action] if discrete else action, 0.0, next_state, float(terminated))
        state = env.reset()[0] if done else next_state

        m = agent.update(expert)

        if step % eval_interval == 0:
            r = evaluate(agent, eval_env, eval_episodes)
            log["step"].append(step)
            log["eval_reward"].append(r)
            log["critic_loss"].append(m["critic_loss"])
            log["actor_loss"].append(m["actor_loss"])
            if verbose:
                print(f"[{algo}] step {step:6d}  eval={r:7.2f}  "
                      f"crit={m['critic_loss']:+.4f}  actor={m['actor_loss']:+.4f}")

    env.close(); eval_env.close()
    return agent, log


# ── I/O ───────────────────────────────────────────────────────────────────────

def save_log_csv(log, path, seed, K):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "step", "eval_reward", "K", "critic_loss", "actor_loss"])
        for s, r, c, a in zip(log["step"], log["eval_reward"], log["critic_loss"], log["actor_loss"]):
            w.writerow([seed, s, r, K, c, a])


def save_agent(agent, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"critic": agent.critic.state_dict(),
                "actor":  agent.actor.state_dict(),
                "bc":     agent.bc_policy.state_dict()}, path)


# ── Experiment sweep ──────────────────────────────────────────────────────────

ENV_CONFIGS = {
    "CartPole": {
        "env": "CartPole-v1", "total_steps": 20_000, "hidden_dim": 128,
        "batch_size": 128, "eval_interval": 1000, "lr": 3e-4, "alpha": 0.2,
        "bc_steps": 5_000, "scale_factor": 0.1,
    },
    "Pendulum": {
        "env": "Pendulum-v1", "total_steps": 30_000, "hidden_dim": 256,
        "batch_size": 256, "eval_interval": 1000, "lr": 3e-4, "alpha": 0.2,
        "bc_steps": 5_000, "scale_factor": 0.1,
    },
}
K_VALUES = [1, 3, 5, 10, 15]
SEEDS    = [42, 43, 44]

# Find project root by walking up to find expert_data/.
# Works whether csil.py lives in imitation_learning/ or imitation_learning/src/.
def _find_root():
    p = pathlib.Path(__file__).resolve().parent
    for candidate in [p, p.parent, p.parent.parent]:
        if (candidate / "expert_data").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate expert_data/ relative to csil.py")

_ROOT = _find_root()


def run_one(env_key, K, seed, soar=False, verbose=False):
    cfg  = ENV_CONFIGS[env_key]
    algo = "csilsoar" if soar else "csil"
    expert_path = str(_ROOT / f"expert_data/{cfg['env']}_K{K}.npz")
    print(f"\n[{algo.upper()} | {env_key} | K={K} | seed={seed}]")

    t0 = time.time()
    agent, log = train_csil(
        env_name=cfg["env"], expert_path=expert_path,
        seed=seed, total_steps=cfg["total_steps"],
        eval_interval=cfg["eval_interval"],
        hidden_dim=cfg["hidden_dim"], batch_size=cfg["batch_size"],
        lr=cfg["lr"], alpha=cfg["alpha"],
        soar=soar, ensemble_size=4, soar_beta=1.0,
        bc_steps=cfg["bc_steps"], scale_factor=cfg["scale_factor"],
        verbose=verbose,
    )

    csv_path   = str(_ROOT / f"logs/{algo}_{env_key}_K{K}_seed{seed}.csv")
    model_path = str(_ROOT / f"models/{algo}_{env_key}_K{K}_seed{seed}.pt")
    save_log_csv(log, csv_path, seed, K)
    save_agent(agent, model_path)

    final = log["eval_reward"][-1] if log["eval_reward"] else float("nan")
    best  = max(log["eval_reward"]) if log["eval_reward"] else float("nan")
    print(f"  {(time.time()-t0)/60:.1f} min | final={final:.1f} | best={best:.1f}")


def _run_one_star(args):
    run_one(*args)


def full_sweep(n_workers=None, env_filter=None, k_filter=None):
    jobs = [
        (env_key, K, seed, soar)
        for env_key in ENV_CONFIGS
        for K in K_VALUES
        for seed in SEEDS
        for soar in [False, True]
        if (env_filter is None or env_key in env_filter)
        and (k_filter is None or K in k_filter)
    ]
    n_workers = n_workers or min(mp.cpu_count(), len(jobs))
    print(f"Running {len(jobs)} jobs on {n_workers} workers")
    with mp.Pool(processes=n_workers, maxtasksperchild=1) as pool:
        pool.map(_run_one_star, jobs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",  default="CartPole-v1")
    parser.add_argument("--K",    type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--soar", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--workers",    type=int,  default=None)
    parser.add_argument("--env-filter", nargs="+", default=None, metavar="ENV")
    parser.add_argument("--k-filter",   nargs="+", type=int, default=None, metavar="K")
    args = parser.parse_args()

    if args.full:
        full_sweep(n_workers=args.workers, env_filter=args.env_filter, k_filter=args.k_filter)
    else:
        env_key = args.env.split("-")[0]
        run_one(env_key, args.K, args.seed, soar=args.soar, verbose=True)