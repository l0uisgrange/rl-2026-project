"""IQ-Learn implementation built on the project SAC backbone."""

import csv
import os
import pathlib
import random
import sys

import numpy as np
import torch
import torch.optim as optim
import gymnasium as gym

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from sac_agent import QNetwork, GaussianPolicy, DiscretePolicy, ReplayBuffer


class ExpertDataset:
    def __init__(self, npz_path, discrete):
        data = np.load(npz_path)

        self.states = torch.FloatTensor(data["states"])
        self.next_states = torch.FloatTensor(data["next_states"])
        self.dones = torch.FloatTensor(data["dones"]).unsqueeze(1)

        if discrete:
            self.actions = torch.LongTensor(data["actions"]).view(-1, 1)
        else:
            self.actions = torch.FloatTensor(data["actions"])

        self.traj_starts = data["traj_starts"]
        self.initial_states = self.states[self.traj_starts]
        self.n_transitions = len(self.states)
        self.discrete = discrete

    def sample(self, batch_size):
        idx = np.random.randint(0, self.n_transitions, size=batch_size)
        return self.states[idx], self.actions[idx], self.next_states[idx], self.dones[idx]

    def sample_initial_states(self, batch_size):
        idx = np.random.randint(0, len(self.initial_states), size=batch_size)
        return self.initial_states[idx]


class IQLearnAgent:
    def __init__(self, state_dim, action_dim, discrete=False,
                 action_low=None, action_high=None,
                 hidden_dim=128, lr=3e-4, gamma=0.99, tau=0.005,
                 alpha=0.2, auto_alpha=False, chi2_coef=0.5,
                 buffer_size=100_000, batch_size=128):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.action_low = action_low
        self.action_high = action_high
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_alpha = auto_alpha
        self.chi2_coef = chi2_coef
        self.batch_size = batch_size

        self.critic = QNetwork(state_dim, action_dim, hidden_dim, discrete)
        self.critic_target = QNetwork(state_dim, action_dim, hidden_dim, discrete)
        self.critic_target.load_state_dict(self.critic.state_dict())

        if discrete:
            self.actor = DiscretePolicy(state_dim, action_dim, hidden_dim)
        else:
            self.actor = GaussianPolicy(state_dim, action_dim, hidden_dim)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        if auto_alpha:
            if discrete:
                self.target_entropy = -np.log(1.0 / action_dim) * 0.98
            else:
                self.target_entropy = -float(action_dim)
            self.log_alpha = torch.tensor([np.log(alpha)], requires_grad=True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = float(self.log_alpha.exp().item())

        self.replay_buffer = ReplayBuffer(buffer_size)

        if not discrete and action_low is not None:
            self._lo = torch.FloatTensor(action_low)
            self._hi = torch.FloatTensor(action_high)

    def _scale_action(self, a_unit):
        return self._lo + (a_unit + 1.0) * 0.5 * (self._hi - self._lo)

    def _soft_value(self, states, use_target):
        critic = self.critic_target if use_target else self.critic
        if self.discrete:
            probs, log_probs = self.actor.get_action_probs(states)
            q1, q2 = critic(states)
            q = torch.min(q1, q2)
            return (probs * (q - self.alpha * log_probs)).sum(dim=-1, keepdim=True)
        actions, log_probs = self.actor.sample(states)
        q1, q2 = critic(states, self._scale_action(actions))
        q = torch.min(q1, q2)
        return q - self.alpha * log_probs

    def select_action(self, state, evaluate=False):
        s_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            if evaluate:
                a = self.actor.deterministic(s_t)
            else:
                a, _ = self.actor.sample(s_t)
        if self.discrete:
            return a.item()
        a_np = a.squeeze(0).numpy()
        return self.action_low + (a_np + 1.0) * 0.5 * (self.action_high - self.action_low)

    def actor_loss(self, states):
        if self.discrete:
            probs, log_probs = self.actor.get_action_probs(states)
            q1, q2 = self.critic(states)
            q = torch.min(q1, q2)
            loss = (probs * (self.alpha * log_probs - q)).sum(dim=-1).mean()
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            return loss, entropy
        actions, log_probs = self.actor.sample(states)
        q1, q2 = self.critic(states, self._scale_action(actions))
        q = torch.min(q1, q2)
        loss = (self.alpha * log_probs - q).mean()
        entropy = -log_probs.mean()
        return loss, entropy

    def update(self, expert_dataset):
        expert_batch = expert_dataset.sample(self.batch_size)

        if len(self.replay_buffer) >= self.batch_size:
            r_s, _, _, r_s_next, r_d = self.replay_buffer.sample(self.batch_size)
            policy_batch = (r_s, r_s_next, r_d)
        else:
            policy_batch = None

        c_loss = self.iq_critic_loss(expert_batch, policy_batch)
        self.critic_optimizer.zero_grad()
        c_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        self.critic_optimizer.step()

        states = r_s if policy_batch is not None else expert_batch[0]

        a_loss, entropy = self.actor_loss(states)
        self.actor_optimizer.zero_grad()
        a_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        if self.auto_alpha:
            alpha_loss = self.log_alpha * (entropy.detach() + self.target_entropy)
            self.alpha_optimizer.zero_grad()
            alpha_loss.mean().backward()
            self.alpha_optimizer.step()
            self.alpha = float(self.log_alpha.exp().item())

        for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_t.data.copy_(self.tau * p.data + (1 - self.tau) * p_t.data)

        return {"critic_loss": c_loss.item(), "actor_loss": a_loss.item(), "alpha": self.alpha}

    def iq_critic_loss(self, expert_batch, policy_batch):
        e_s, e_a, e_s_next, e_d = expert_batch

        if self.discrete:
            q1_all, q2_all = self.critic(e_s)
            q1 = q1_all.gather(1, e_a)
            q2 = q2_all.gather(1, e_a)
        else:
            q1, q2 = self.critic(e_s, e_a)

        with torch.no_grad():
            v_next_e = self._soft_value(e_s_next, use_target=True)

        res1 = q1 - self.gamma * (1 - e_d) * v_next_e
        res2 = q2 - self.gamma * (1 - e_d) * v_next_e

        reward_term = -(res1.mean() + res2.mean())
        chi2_term = (res1.pow(2).mean() + res2.pow(2).mean()) / (4 * self.chi2_coef)

        if policy_batch is not None:
            p_s, p_s_next, p_d = policy_batch
            all_s = torch.cat([e_s, p_s], dim=0)
            all_s_next = torch.cat([e_s_next, p_s_next], dim=0)
            all_d = torch.cat([e_d, p_d], dim=0)
        else:
            all_s, all_s_next, all_d = e_s, e_s_next, e_d

        v_curr = self._soft_value(all_s, use_target=False)
        with torch.no_grad():
            v_next_all = self._soft_value(all_s_next, use_target=True)

        value_term = 2 * (v_curr - self.gamma * (1 - all_d) * v_next_all).mean()

        return reward_term + value_term + chi2_term


def save_log_csv(log, out_path, seed, K):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "step", "eval_reward", "K", "critic_loss", "actor_loss"])
        for s, r, c, a in zip(log["step"], log["eval_reward"],
                              log["critic_loss"], log["actor_loss"]):
            writer.writerow([seed, s, r, K, c, a])


def evaluate(agent, eval_env, n_episodes=5):
    rewards = []
    for _ in range(n_episodes):
        s, _ = eval_env.reset()
        ep_r = 0.0
        done = False
        while not done:
            a = agent.select_action(s, evaluate=True)
            s, r, term, trunc, _ = eval_env.step(a)
            ep_r += r
            done = term or trunc
        rewards.append(ep_r)
    return float(np.mean(rewards))


def train_iq_learn(env_name, expert_npz_path, seed=42, total_steps=50_000,
                   eval_interval=1000, eval_episodes=5,
                   hidden_dim=128, batch_size=128,
                   lr=3e-4, chi2_coef=0.5, alpha=0.2, auto_alpha=False,
                   verbose=True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n if discrete else env.action_space.shape[0]

    expert = ExpertDataset(expert_npz_path, discrete=discrete)

    agent = IQLearnAgent(
        state_dim=state_dim, action_dim=action_dim, discrete=discrete,
        action_low=None if discrete else env.action_space.low,
        action_high=None if discrete else env.action_space.high,
        hidden_dim=hidden_dim, batch_size=batch_size,
        lr=lr, chi2_coef=chi2_coef, alpha=alpha, auto_alpha=auto_alpha,
    )

    log = {"step": [], "eval_reward": [], "critic_loss": [], "actor_loss": []}

    state, _ = env.reset(seed=seed)
    for step in range(1, total_steps + 1):
        action = agent.select_action(state)
        next_state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        stored_action = [action] if discrete else action
        agent.replay_buffer.push(state, stored_action, 0.0, next_state, float(terminated))

        state, _ = env.reset() if done else (next_state, None)

        m = agent.update(expert)

        if step % eval_interval == 0:
            ev = evaluate(agent, eval_env, eval_episodes)
            log["step"].append(step)
            log["eval_reward"].append(ev)
            log["critic_loss"].append(m["critic_loss"])
            log["actor_loss"].append(m["actor_loss"])
            if verbose:
                print(f"  step {step:6d}  eval={ev:7.2f}  "
                      f"crit={m['critic_loss']:+.3f}  "
                      f"actor={m['actor_loss']:+.3f}  "
                      f"alpha={m['alpha']:.3f}")

    env.close()
    eval_env.close()
    return agent, log
