"""
IQ-Learn: Inverse soft-Q Learning for Imitation
Paper: Garg et al., NeurIPS 2021 — https://arxiv.org/abs/2106.12142

The reward is recovered from Q via the inverse soft Bellman operator:
    r(s,a) = Q(s,a) - gamma * V(s'),   V(s) = E_{a~pi}[Q(s,a) - alpha * log pi(a|s)]
So IQ-Learn is SAC with a critic loss that consumes expert data.
"""

import csv
import os
import random

import numpy as np
import torch
import torch.optim as optim
import gymnasium as gym

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
    def __init__(
        self,
        state_dim,
        action_dim,
        discrete=False,
        action_low=None,
        action_high=None,
        hidden_dim=128,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_alpha=False,
        chi2_coef=0.5,
        buffer_size=100_000,
        batch_size=128,
    ):
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
            self._action_low_t = torch.FloatTensor(action_low)
            self._action_high_t = torch.FloatTensor(action_high)

    def _scale_action(self, action_unit):
        """Map tanh output in [-1, 1] to env action range [low, high]."""
        return self._action_low_t + (action_unit + 1.0) * 0.5 * (self._action_high_t - self._action_low_t)

    def _soft_value(self, states, use_target):
        """V(s) = E_{a~pi}[Q(s,a) - alpha * log pi(a|s)] using min(Q1, Q2)."""
        critic = self.critic_target if use_target else self.critic
        if self.discrete:
            probs, log_probs = self.actor.get_action_probs(states)
            q1, q2 = critic(states)
            q = torch.min(q1, q2)
            return (probs * (q - self.alpha * log_probs)).sum(dim=-1, keepdim=True)
        else:
            actions, log_probs = self.actor.sample(states)
            q1, q2 = critic(states, self._scale_action(actions))
            q = torch.min(q1, q2)
            return q - self.alpha * log_probs

    def select_action(self, state, evaluate=False):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            if evaluate:
                action = self.actor.deterministic(state_t)
            else:
                action, _ = self.actor.sample(state_t)
        if self.discrete:
            return action.item()
        action_np = action.squeeze(0).numpy()
        return self.action_low + (action_np + 1.0) * 0.5 * (self.action_high - self.action_low)

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

        critic_loss = self.iq_critic_loss(expert_batch, policy_batch)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        self.critic_optimizer.step()

        if policy_batch is not None:
            states = r_s
        else:
            states = expert_batch[0]

        actor_loss, entropy = self.actor_loss(states)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
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

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item(), "alpha": self.alpha}

    def iq_critic_loss(self, expert_batch, policy_batch):
        """IQ-Learn loss in 'value' mode (official default).

        - reward & chi2 terms: on expert (s,a,s') only
        - value term: Bellman residual of V averaged over expert + policy states
        """
        e_states, e_actions, e_next_states, e_dones = expert_batch

        if self.discrete:
            q1_all, q2_all = self.critic(e_states)
            q1 = q1_all.gather(1, e_actions)
            q2 = q2_all.gather(1, e_actions)
        else:
            q1, q2 = self.critic(e_states, e_actions)

        with torch.no_grad():
            v_next_e = self._soft_value(e_next_states, use_target=True)

        residual_1 = q1 - self.gamma * (1 - e_dones) * v_next_e
        residual_2 = q2 - self.gamma * (1 - e_dones) * v_next_e

        reward_term = -(residual_1.mean() + residual_2.mean())
        chi2_term = (residual_1.pow(2).mean() + residual_2.pow(2).mean()) / (4 * self.chi2_coef)

        if policy_batch is not None:
            p_states, p_next_states, p_dones = policy_batch
            all_states = torch.cat([e_states, p_states], dim=0)
            all_next_states = torch.cat([e_next_states, p_next_states], dim=0)
            all_dones = torch.cat([e_dones, p_dones], dim=0)
        else:
            all_states = e_states
            all_next_states = e_next_states
            all_dones = e_dones

        v_curr = self._soft_value(all_states, use_target=False)
        with torch.no_grad():
            v_next_all = self._soft_value(all_next_states, use_target=True)

        value_term = 2 * (v_curr - self.gamma * (1 - all_dones) * v_next_all).mean()

        return reward_term + value_term + chi2_term


def save_log_csv(log, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "eval_reward", "critic_loss", "actor_loss"])
        for s, r, c, a in zip(log["step"], log["eval_reward"], log["critic_loss"], log["actor_loss"]):
            writer.writerow([s, r, c, a])


def evaluate(agent, eval_env, n_episodes=5):
    rewards = []
    for _ in range(n_episodes):
        s, _ = eval_env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            a = agent.select_action(s, evaluate=True)
            s, r, term, trunc, _ = eval_env.step(a)
            ep_reward += r
            done = term or trunc
        rewards.append(ep_reward)
    return float(np.mean(rewards))


def train_iq_learn(
    env_name,
    expert_npz_path,
    seed=42,
    total_steps=50_000,
    eval_interval=1000,
    eval_episodes=5,
    hidden_dim=128,
    batch_size=128,
    lr=3e-4,
    chi2_coef=0.5,
    alpha=0.2,
    auto_alpha=False,
    verbose=True,
):
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
        state_dim=state_dim,
        action_dim=action_dim,
        discrete=discrete,
        action_low=None if discrete else env.action_space.low,
        action_high=None if discrete else env.action_space.high,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        lr=lr,
        chi2_coef=chi2_coef,
        alpha=alpha,
        auto_alpha=auto_alpha,
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

        metrics = agent.update(expert)

        if step % eval_interval == 0:
            eval_reward = evaluate(agent, eval_env, eval_episodes)
            log["step"].append(step)
            log["eval_reward"].append(eval_reward)
            log["critic_loss"].append(metrics["critic_loss"])
            log["actor_loss"].append(metrics["actor_loss"])
            if verbose:
                print(
                    f"  step {step:6d}  eval={eval_reward:7.2f}  "
                    f"crit={metrics['critic_loss']:+.3f}  "
                    f"actor={metrics['actor_loss']:+.3f}  "
                    f"alpha={metrics['alpha']:.3f}"
                )

    env.close()
    eval_env.close()
    return agent, log


if __name__ == "__main__":
    print("Loading CartPole expert data (K=5)...")
    cp = ExpertDataset("expert_data/CartPole-v1_K5.npz", discrete=True)
    print(f"  states:        {cp.states.shape}")
    print(f"  actions:       {cp.actions.shape}  dtype={cp.actions.dtype}")
    print(f"  initial_states:{cp.initial_states.shape}")

    print("\nLoading Pendulum expert data (K=5)...")
    pd = ExpertDataset("expert_data/Pendulum-v1_K5.npz", discrete=False)
    print(f"  states:        {pd.states.shape}")
    print(f"  actions:       {pd.actions.shape}  dtype={pd.actions.dtype}")

    print("\nBuilding IQLearnAgent (CartPole)...")
    cp_agent = IQLearnAgent(state_dim=4, action_dim=2, discrete=True, hidden_dim=128)

    print("Computing IQ-Learn loss on CartPole batch...")
    expert_batch = cp.sample(batch_size=128)
    loss = cp_agent.iq_critic_loss(expert_batch, policy_batch=None)
    print(f"  loss = {loss.item():.4f}  requires_grad={loss.requires_grad}")
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in cp_agent.critic.parameters() if p.grad is not None)
    print(f"  backward OK, critic grad-norm sum = {grad_norm:.4f}")

    print("\nBuilding IQLearnAgent (Pendulum)...")
    pd_agent = IQLearnAgent(
        state_dim=3, action_dim=1, discrete=False,
        action_low=np.array([-2.0]), action_high=np.array([2.0]),
        hidden_dim=256,
    )
    print("Computing IQ-Learn loss on Pendulum batch...")
    expert_batch = pd.sample(batch_size=128)
    loss = pd_agent.iq_critic_loss(expert_batch, policy_batch=None)
    loss.backward()
    print(f"  loss = {loss.item():.4f}, backward OK")

    print("\nAll components OK. Use run_experiments.py to launch training runs.")
