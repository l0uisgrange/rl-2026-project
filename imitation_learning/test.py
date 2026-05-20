from run_experiments import run_one

for env in ["CartPole-v1", "Pendulum-v1"]:
    for seed in [42, 43, 44]:
        run_one(env, K=5, seed=seed, verbose=True)