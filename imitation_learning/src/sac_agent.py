"""
SAC (Soft Actor-Critic) Implementation
=======================================
Implements the entropy-regularized RL objective from Lecture 6, Slide 10:

    max_π  E[ Σ γ^t ( r(s_t, a_t) + λ · H(π(·|s_t)) ) ]

where H(π(·|s)) = E_{a~π}[-log π(a|s)] is the entropy of the policy.

This file provides:
  - Continuous SAC (squashed Gaussian) for Pendulum-v1
  - Discrete SAC (softmax policy) for CartPole-v1
  - Replay buffer (experience replay, Lecture 6 Slide 19-20)
  - Automatic entropy coefficient (α) tuning

Key design decisions connected to the course:
  - Twin Q-networks to combat overestimation (Double DQN idea, Lecture 6 Slide 22)
  - Target networks with soft (Polyak) updates (DQN stabilization, Lecture 6 Slide 20)
  - Actor-critic architecture (Lecture 6 Slide 4-5): actor = policy π_θ, critic = Q_w
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
import random

# ============================================================
# Replay Buffer
# ============================================================
# This implements experience replay (Lecture 6, Slide 19):
# "Reduce correlation, allow mini-batch update"
# We store (s, a, r, s', done) transitions and sample uniformly.

class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


# ============================================================
# Neural Network Architectures
# ============================================================
# These are the function approximators for Q_w(s,a) and π_θ(a|s).
# From Lecture 6: "Use neural networks for value function approximation"

class QNetwork(nn.Module):
    """Twin Q-Network: outputs two Q-values to reduce overestimation bias.
    
    For continuous actions: takes (state, action) as input.
    For discrete actions: takes state, outputs Q-value for each action.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, discrete=False):
        super().__init__()
        self.discrete = discrete
        
        if discrete:
            input_dim = state_dim
            output_dim = action_dim
        else:
            input_dim = state_dim + action_dim
            output_dim = 1

        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        # Q2
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, state, action=None):
        if self.discrete:
            return self.q1(state), self.q2(state)
        else:
            x = torch.cat([state, action], dim=-1)
            return self.q1(x), self.q2(x)


class GaussianPolicy(nn.Module):
    """Squashed Gaussian policy for Pendulum environment.
    
    π_θ(a|s) = tanh(μ_θ(s) + σ_θ(s) · ε),  ε ~ N(0,1)
    
    The tanh squashing ensures actions stay in [-1, 1], which we then
    rescale to the environment's action bounds.
    
    Connected to Lecture 4 Slide 6: this is a "neural softmax parameterization"
    generalized to continuous actions via the reparameterization trick.
    """
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        """Sample action using reparameterization trick plus tanh squashing."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        # Reparameterization: z = μ + σ * ε
        z = normal.rsample()
        action = torch.tanh(z) # force output into [-1,1]

        # Log-prob with tanh correction (change of variables formula):
        # log π(a|s) = log N(z; μ, σ) - log(1 - tanh²(z))
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def deterministic(self, state):
        """For evaluation: use the mean action (no sampling)."""
        mean, _ = self.forward(state)
        return torch.tanh(mean)


class DiscretePolicy(nn.Module):
    """Softmax policy for discrete action spaces.
    
    π_θ(a|s) = softmax(h_θ(s))
    
    "neural softmax parameterization" from Lecture 4, Slide 6:
        π_θ(a|s) = exp(h_θ(s,a)) / Σ_{a'} exp(h_θ(s,a'))
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        logits = self.net(state)
        return logits

    def sample(self, state):
        """Sample action from categorical distribution."""
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        # Log probability of the sampled action
        log_prob = dist.log_prob(action).unsqueeze(-1)
        return action, log_prob

    def get_action_probs(self, state):
        """Return action probabilities and log probabilities (for Q-value weighting)."""
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs

    def deterministic(self, state):
        """For evaluation: pick the most likely action."""
        logits = self.forward(state)
        return logits.argmax(dim=-1)


# ============================================================
# SAC Agent
# ============================================================
class SACAgent:
    """
    Soft Actor-Critic agent.
    
    The update rule follows the actor-critic framework from Lecture 6 Slide 4-5:
      - Critic update: minimize Bellman error (TD learning, Lecture 6 Slide 19)
      - Actor update: maximize expected Q-value + entropy bonus (policy gradient)
      - α update: automatic tuning of the entropy coefficient
    
    The soft Bellman equation that the critic targets:
        Q(s,a) = r + γ E_{s'}[ E_{a'~π}[ Q(s',a') - α log π(a'|s') ] ]
    
    This is the entropy-regularized version of Q-learning (Lecture 6 Slide 10).
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        discrete=False,
        action_low=None,
        action_high=None,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_alpha=False,
        buffer_size=100_000,
        batch_size=256,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.action_low = action_low
        self.action_high = action_high
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Networks
        self.critic = QNetwork(state_dim, action_dim, hidden_dim, discrete)
        self.critic_target = QNetwork(state_dim, action_dim, hidden_dim, discrete)
        # Initialize target = critic (Lecture 6 Slide 20: target network)
        self.critic_target.load_state_dict(self.critic.state_dict())

        if discrete:
            self.actor = DiscretePolicy(state_dim, action_dim, hidden_dim)
        else:
            self.actor = GaussianPolicy(state_dim, action_dim, hidden_dim)

        # Optimizers (Adam: adaptive SGD, Lecture 6 Slide 18)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Automatic entropy tuning
        # α controls the exploration-exploitation tradeoff in the SAC objective
        self.auto_alpha = auto_alpha
        if auto_alpha:
            if discrete:
                # Target entropy for discrete: -log(1/|A|) * ratio
                self.target_entropy = -np.log(1.0 / action_dim) * 0.98
            else:
                # Target entropy for continuous: -dim(A)
                self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, state, evaluate=False):
        """Select action for environment interaction."""
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            if evaluate:
                action = self.actor.deterministic(state_t)
            else:
                action, _ = self.actor.sample(state_t)

        if self.discrete:
            return action.item()
        else:
            # Scale from [-1, 1] to [action_low, action_high]
            action_np = action.squeeze(0).numpy()
            scaled = self.action_low + (action_np + 1.0) * 0.5 * (self.action_high - self.action_low)
            return scaled

    def update(self):
        """One gradient step on critic, actor, and α."""
        if len(self.replay_buffer) < self.batch_size:
            return {}

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # ---- Critic update ----
        # Compute target: y = r + γ(1-d) * (Q_target(s', a') - α log π(a'|s'))
        with torch.no_grad():
            if self.discrete:
                # For discrete: sum over all actions weighted by π
                next_probs, next_log_probs = self.actor.get_action_probs(next_states)
                q1_next, q2_next = self.critic_target(next_states)
                q_next = torch.min(q1_next, q2_next)
                # V(s') = Σ_a π(a|s') [Q(s',a) - α log π(a|s')]
                v_next = (next_probs * (q_next - self.alpha * next_log_probs)).sum(dim=-1, keepdim=True)
            else:
                next_actions, next_log_probs = self.actor.sample(next_states)
                q1_next, q2_next = self.critic_target(next_states, next_actions)
                q_next = torch.min(q1_next, q2_next)
                v_next = q_next - self.alpha * next_log_probs

            target_q = rewards + self.gamma * (1 - dones) * v_next

        # Current Q estimates
        if self.discrete:
            q1, q2 = self.critic(states)
            actions_idx = actions.long()
            q1 = q1.gather(1, actions_idx)
            q2 = q2.gather(1, actions_idx)
        else:
            q1, q2 = self.critic(states, actions)

        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---- Actor update ----
        # Maximize: E_{a~π}[Q(s, a) - α log π(a|s)]
        # Equivalent to: minimize E_{a~π}[α log π(a|s) - Q(s, a)]
        if self.discrete:
            probs, log_probs = self.actor.get_action_probs(states)
            q1_pi, q2_pi = self.critic(states)
            q_pi = torch.min(q1_pi, q2_pi)
            # Policy loss: E_π[α log π - Q]
            actor_loss = (probs * (self.alpha * log_probs - q_pi)).sum(dim=-1).mean()
            # For α update: expected entropy
            entropy = -(probs * log_probs).sum(dim=-1).mean()
        else:
            actions_pi, log_probs_pi = self.actor.sample(states)
            q1_pi, q2_pi = self.critic(states, actions_pi)
            q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (self.alpha * log_probs_pi - q_pi).mean()
            entropy = -log_probs_pi.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---- Alpha (temperature) update ----
        if self.auto_alpha:
            # Adjust α so that the policy entropy stays near target_entropy
            # Dual variable update: increase α when entropy < target, decrease when entropy > target
            # J(α) = α * (H(π) + H̄), where H̄ = target_entropy (negative)
            alpha_loss = self.log_alpha * (entropy + self.target_entropy).detach()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # ---- Target network soft update (Polyak averaging) ----
        # θ_target ← τ·θ + (1-τ)·θ_target
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha,
            "entropy": entropy.item(),
        }

    def save(self, path):
        """Save all model weights."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }, path)

    def load(self, path):
        """Load model weights."""
        checkpoint = torch.load(path, weights_only=True)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])


# ============================================================
# Training Loop
# ============================================================
def train_sac(
    env_name,
    max_episodes=500,
    max_steps=None,
    eval_interval=10,
    eval_episodes=5,
    seed=42,
    save_path=None,
    hidden_dim=256,
    lr=3e-4,
    batch_size=256,
    updates_per_step=1,
    start_steps=1000,
    verbose=True,
):
    """
    Train SAC on a given environment.
    
    Args:
        env_name: Gymnasium environment name
        max_episodes: Number of training episodes
        max_steps: Override for max steps per episode
        eval_interval: Evaluate every N episodes
        eval_episodes: Number of episodes for evaluation
        seed: Random seed for reproducibility
        save_path: Where to save the trained model
        start_steps: Random actions for initial exploration
    
    Returns:
        agent: Trained SACAgent
        log: Dictionary of training metrics
    """
    # Setup
    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    state_dim = env.observation_space.shape[0]
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    if discrete:
        action_dim = env.action_space.n
        action_low, action_high = None, None
    else:
        action_dim = env.action_space.shape[0]
        action_low = env.action_space.low
        action_high = env.action_space.high

    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        discrete=discrete,
        action_low=action_low,
        action_high=action_high,
        hidden_dim=hidden_dim,
        lr=lr,
        batch_size=batch_size,
    )

    # Logging
    log = {
        "episode": [],
        "train_reward": [],
        "eval_reward": [],
        "critic_loss": [],
        "actor_loss": [],
        "alpha": [],
        "total_steps": [],
    }

    total_steps = 0
    best_eval_reward = -float("inf")

    for episode in range(1, max_episodes + 1):
        state, _ = env.reset(seed=seed + episode)
        episode_reward = 0
        done = False
        ep_steps = 0

        while not done:
            # Initial random exploration to fill the buffer
            if total_steps < start_steps:
                if discrete:
                    action = env.action_space.sample()
                else:
                    action = env.action_space.sample()
            else:
                action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition
            if discrete:
                agent.replay_buffer.push(state, [action], reward, next_state, float(terminated))
            else:
                agent.replay_buffer.push(state, action, reward, next_state, float(terminated))

            state = next_state
            episode_reward += reward
            total_steps += 1
            ep_steps += 1

            # Update after each step (once buffer has enough data)
            if total_steps >= start_steps:
                for _ in range(updates_per_step):
                    agent.update()

        log["episode"].append(episode)
        log["train_reward"].append(episode_reward)
        log["total_steps"].append(total_steps)

        # Evaluation
        if episode % eval_interval == 0:
            eval_rewards = []
            for _ in range(eval_episodes):
                s, _ = eval_env.reset()
                er = 0
                d = False
                while not d:
                    a = agent.select_action(s, evaluate=True)
                    s, r, term, trunc, _ = eval_env.step(a)
                    er += r
                    d = term or trunc
                eval_rewards.append(er)
            
            mean_eval = np.mean(eval_rewards)
            log["eval_reward"].append(mean_eval)

            if verbose:
                print(
                    f"Episode {episode:4d} | "
                    f"Steps {total_steps:6d} | "
                    f"Train: {episode_reward:7.1f} | "
                    f"Eval: {mean_eval:7.1f} | "
                    f"α: {agent.alpha:.4f}"
                )

            # Save best model
            if mean_eval > best_eval_reward and save_path:
                best_eval_reward = mean_eval
                agent.save(save_path)
                if verbose:
                    print(f"  -> New best! Saved to {save_path}")

    env.close()
    eval_env.close()

    # Final save
    if save_path:
        agent.save(save_path)

    return agent, log


# ============================================================
# Main: Train experts for both environments
# ============================================================
if __name__ == "__main__":
    import os
    
    os.makedirs("../models", exist_ok=True)
    os.makedirs("../logs", exist_ok=True)

    # --- Train CartPole expert ---
    print("=" * 60)
    print("Training SAC Expert on CartPole-v1")
    print("=" * 60)
    cartpole_agent, cartpole_log = train_sac(
        env_name="CartPole-v1",
        max_episodes=300,
        eval_interval=10,
        seed=42,
        save_path="../models/sac_cartpole_expert.pt",
        hidden_dim=128,       # smaller net for this simple env
        lr=3e-4,
        batch_size=128,
        start_steps=500,      # less random exploration needed
    )

    # --- Train Pendulum expert ---
    print("\n" + "=" * 60)
    print("Training SAC Expert on Pendulum-v1")
    print("=" * 60)
    pendulum_agent, pendulum_log = train_sac(
        env_name="Pendulum-v1",
        max_episodes=200,
        eval_interval=10,
        seed=42,
        save_path="../models/sac_pendulum_expert.pt",
        hidden_dim=256,
        lr=3e-4,
        batch_size=256,
        start_steps=1000,
    )

    # Save logs for later plotting
    np.savez(
        "../logs/training_logs.npz",
        cartpole_episodes=cartpole_log["episode"],
        cartpole_train_rewards=cartpole_log["train_reward"],
        cartpole_eval_rewards=cartpole_log["eval_reward"],
        pendulum_episodes=pendulum_log["episode"],
        pendulum_train_rewards=pendulum_log["train_reward"],
        pendulum_eval_rewards=pendulum_log["eval_reward"],
    )
    print("\nTraining complete! Logs saved to logs/training_logs.npz")
