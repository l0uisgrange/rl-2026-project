# EE-568 Reinforcement Learning: Project 4 - Imitation Learning

## 1. Summary + setup
This document explains the execution plan for Project 4: Imitation Learning. The objective is to implement and compare different imitation learning algorithms using expert trajectories.

**Selected Environments:**
* **CartPole-v1 (Discrete Control):** Our "Hello World" baseline. It trains extremely fast, making it ideal for rapid prototyping and debugging algorithms.
* **Pendulum-v1 (Continuous Control):** A continuous control problem. It showcases the capabilities of continuous-action algorithms like SAC and SOAR.

**Selected Algorithms:**
* **Soft Actor-Critic (SAC):** Used for trajectory generation. We will train an SAC agent to optimality to act as the "expert" and collect $K$ trajectories. SAC is chosen, since it aligns architecturally with the SOAR critic we will build later.
* **IQ-Learn:** Learns a soft state-action value function directly from expert data, bypassing the unstable min-max adversarial training loops.
* **CSIL (Coherent Soft Imitation Learning):** Our baseline imitation learning method. It is highly stable and integrates seamlessly with the SAC framework.
* **CSIL + SOAR:** The enhanced algorithm. It adds an optimistic critic ensemble to CSIL to use disagreement as an exploration signal, theoretically improving sample efficiency.

---

## 2. Work Breakdown Structure (3 Members)

### **Person 1: The Data & Environment Architect**
* **Setup:** Initialize the OpenAI Gym environments (`CartPole-v1`, `Pendulum-v1`).
* **Expert Generation:** Train the SAC agent to reach the maximum reward in both environments.
* **Data Pipeline:** Write scripts to roll out the expert policy, collect $K$ trajectories, and save them in a reusable format (e.g., NumPy arrays or PyTorch tensors containing state, action, and next state).
* **Evaluation Engine:** Build the plotting scripts that will take the logs from the other 2 people and generate the required performance graphs (averaging across 3 random seeds).

### **Person 2: The IQ-Learn Specialist**
* **Implementation:** Understand and implement the IQ-Learn objective. Focus on how it uses the expert data directly in the Q-function update without an explicit reward function.
* **Integration:** Connect the IQ-Learn agent to the data pipeline built by Person 1.
* **Tuning:** Run experiments across the required seeds, tune hyperparameters, and log the performance metrics.

### **Person 3: The CSIL & SOAR Engineer**
* **Base Implementation:** Implement the standard CSIL algorithm using the SAC actor-critic architecture.
* **SOAR Modification:** Modify the CSIL critic to be an *ensemble* of $L$ critics (e.g., 3 to 5 networks). Implement the optimism bonus (mean value + standard deviation of the ensemble) to encourage exploration.
* **Tuning:** Run the standard CSIL and CSIL+SOAR experiments across the 3 seeds, ensuring a fair comparison (e.g., matching the number of network updates).

---

## 3. Evaluation & Deliverables

To fulfill the project requirements, we will generate the following plots (each averaged across at least 3 random seeds):

1.  **Sample Efficiency (Reward vs. $K$):** X-axis is the number of expert trajectories ($K$ = e.g., 1, 3, 5, 10, 15), Y-axis is the final average reward. This will directly demonstrate if SOAR improves sample efficiency over base CSIL when expert data is scarce.
2.  **Learning Curves:** X-axis is training steps, Y-axis is the moving average of the episode reward. Keep $K$ fixed (e.g., $K=5$) to show how quickly each algorithm converges.

---

### 4. The Skipped Algorithms
* **f-IRL & ML-IRL:** These are strict Inverse Reinforcement Learning (IRL) methods. They literally try to reverse-engineer the reward function by solving brutal min-max optimization problems. They take forever to run and are a headache to tune. CSIL + SOAR is a way cleaner engineering path for us.
* **HyPE:** Another imitation method, but CSIL just maps way better to the SOAR upgrades we need to do.

### The Skipped Environments
* **MountainCar & MountainCarContinuous:** These are fun, but the momentum-based physics can be really annoying to tune. CartPole is a much better discrete baseline, and Pendulum is a cleaner continuous challenge.
* **Acrobot-v1:** This is a crazy non-linear double-pendulum. It’s way harder to solve than CartPole. We want to spend our time testing and comparing algorithms, not spending three weeks pulling our hair out just trying to solve one environment!