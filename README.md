# Reinforcement learning

Repository for the EE-556 Reinforcement learning course at EPFL. The evaluation and work was divided into

| Task                                                                                                    | Grade obtained | 
|:--------------------------------------------------------------------------------------------------------|:---------------|
| [Notebook 1](https://github.com/l0uisgrange/rl-2026-project/blob/main/notebook1_2026/Assignment.ipynb)  | 98%[^1]        |  
| [Notebook 2](https://github.com/l0uisgrange/rl-2026-project/blob/main/notebook2_2026/Assignment.ipynb)  | 92%[^2]        | 
| [Notebook 3](https://github.com/l0uisgrange/rl-2026-project/blob/main/notebook3_2026/Assignment.ipynb)  |                | 
| [Project](https://github.com/l0uisgrange/rl-2026-project/blob/main/imitation_learning/README.md)        |                |

[^1]: 2: Need to mention when epsilon goes to 0
[^2]: Q4.1 The λ-gradient should be ∇_λ L = r + γP V^k − E V^k. Q5: The Bellman residual = 0 argument actually yields L(λ^k, V^πλk) = (1-γ)⟨μ, V^πλk⟩ directly; the subsequent converting ⟨λ^k, r⟩ to (1-γ)⟨μ, V^πλk⟩ is invalid because λ^k is not a valid occupancy measure