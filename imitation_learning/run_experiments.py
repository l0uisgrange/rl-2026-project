"""Sweep runner for IQ-Learn on CartPole-v1 and Pendulum-v1."""

import argparse
import time

from src.iq_learn import train_iq_learn, save_log_csv

ENV_CONFIGS = {
    "CartPole-v1": {"total_steps": 20_000, "hidden_dim": 128, "batch_size": 128,
                    "eval_interval": 1000, "lr": 3e-4, "chi2_coef": 0.5,
                    "alpha": 0.2, "auto_alpha": False},
    "Pendulum-v1": {"total_steps": 50_000, "hidden_dim": 256, "batch_size": 256,
                    "eval_interval": 2000, "lr": 1e-4, "chi2_coef": 0.5,
                    "alpha": 0.2, "auto_alpha": True},
}

K_VALUES = [1, 3, 5, 10, 15]
SEEDS = [42, 43, 44]


def run_one(env_name, K, seed, log_dir="logs", verbose=False):
    cfg = ENV_CONFIGS[env_name]
    expert_path = f"expert_data/{env_name}_K{K}.npz"

    print(f"\n[{env_name} | K={K} | seed={seed}] starting ({cfg['total_steps']} steps)")
    t0 = time.time()
    _, log = train_iq_learn(
        env_name=env_name,
        expert_npz_path=expert_path,
        seed=seed,
        total_steps=cfg["total_steps"],
        eval_interval=cfg["eval_interval"],
        eval_episodes=5,
        hidden_dim=cfg["hidden_dim"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        chi2_coef=cfg["chi2_coef"],
        alpha=cfg["alpha"],
        auto_alpha=cfg["auto_alpha"],
        verbose=verbose,
    )
    elapsed = time.time() - t0
    final_r = log["eval_reward"][-1] if log["eval_reward"] else float("nan")
    max_r = max(log["eval_reward"]) if log["eval_reward"] else float("nan")

    out_path = f"{log_dir}/iqlearn_{env_name}_K{K}_seed{seed}.csv"
    save_log_csv(log, out_path, seed=seed, K=K)

    print(f"  done in {elapsed/60:.1f} min | final={final_r:.1f} | max={max_r:.1f}")
    return elapsed, final_r, max_r


def smoke_test():
    print("Smoke test: CartPole K=5 and Pendulum K=5 at seed 42")
    results = []
    for env_name in ["CartPole-v1", "Pendulum-v1"]:
        results.append((env_name, *run_one(env_name, K=5, seed=42, verbose=True)))
    print("\nSummary:")
    for env, elapsed, final, mx in results:
        print(f"  {env:14s}  {elapsed/60:5.1f} min  final={final:7.1f}  max={mx:7.1f}")


def full_sweep():
    n_runs = len(K_VALUES) * len(SEEDS) * len(ENV_CONFIGS)
    print(f"Full sweep: {n_runs} runs total")
    t_total = time.time()
    for env_name in ENV_CONFIGS:
        for K in K_VALUES:
            for seed in SEEDS:
                run_one(env_name, K, seed)
    print(f"\nDone in {(time.time() - t_total)/60:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    if args.full:
        full_sweep()
    else:
        smoke_test()
