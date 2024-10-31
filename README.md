# Optimized-Backward-Policies-for-Effective-Reward-Matching-in-Generative-Flow-Networks

Optimized Backward Policies for Learning Dynamics in Stochastic Environments
Here's an updated GitHub project description tailored to the new focus on time series data and stochastic environments:

---

## Optimized Backward Policies for Learning Dynamics in Stochastic Environments

### Project Overview

This repository contains the implementation of **Optimized Backward Policies for Generative Flow Networks (OBP-GFN)**, designed to learn the dynamics of **stochastic environments** through time series data. Generative Flow Networks (GFlowNets) are used to incrementally construct sequences of state transitions, which represent the evolution of a dynamic system. However, GFlowNets may under-exploit high-reward sequences due to insufficient exploration. Our proposed approach, OBP-GFN, mitigates these limitations, resulting in more effective learning of system dynamics and better modeling of temporal patterns.

### Key Features

- **Optimized Backward Policy (OBP-GFN)**: A novel approach to optimizing backward policies in stochastic environments to align backward flows with true reward values, improving learning of the system dynamics.
- **Enhanced Temporal Dynamics Learning**: OBP-GFN aims to learn the stochastic transitions of dynamic systems effectively, optimizing backward flows to enhance reward discovery and temporal accuracy.
- **Extensive Benchmarks**: Evaluated across diverse benchmarks, including synthetic time series datasets, dynamic system modeling, and stochastic prediction tasks, to validate the ability of OBP-GFN to capture complex system dynamics.

### Repository Structure

- **`src/`**: Contains the core implementation of OBP-GFN for time series and stochastic environments, including forward and backward policies.
- **`experiments/`**: Scripts for running experiments on synthetic and real-world time series datasets.
- **`docs/`**: Documentation to help understand how to use this repository to model stochastic system dynamics.
- **`notebooks/`**: Jupyter notebooks for visualization and analysis of results from experiments involving time series data.

### Background

In **stochastic environments**, the goal of GFlowNets is to learn the underlying system dynamics, which are represented through sequences of state transitions similar to time series data. The **forward policy** generates potential state transitions, while the **backward policy** helps optimize the distribution of rewards across these sequences, enhancing the learning of temporal patterns and system evolution.

A major challenge in GFlowNets when applied to time series data is the **under-exploitation of high-reward sequences**—typically due to incomplete observations of high-reward trajectories. OBP-GFN addresses this issue by optimizing the backward flow, allowing it to better align with observed high-reward states, ultimately leading to improved learning of the dynamics of stochastic environments.

### Methodology

The core idea behind **OBP-GFN** is to optimize the backward policy to improve the learning of stochastic system transitions. By maximizing the likelihood of observed backward flows, OBP-GFN helps improve learning in stochastic time series environments:

\[
\ell*{OBP} = -\mathbb{E}*{\tau \in B(x)}\left[\log P_B(\tau | x)\right],
\]

where \( B(x) \) represents the set of observed trajectories ending at state \( x \), and \( P_B(\tau | x) \) is the probability of trajectory \( \tau \) given the terminal state \( x \). This optimization helps in aligning the backward flow with high-reward trajectories, enhancing the model's ability to learn the system dynamics effectively.

### Experimentation

The efficacy of OBP-GFN is demonstrated through extensive experiments across various benchmarks involving time series and stochastic environments. Key metrics used for evaluation include:

- **Reward Coverage**: Evaluating how well the learned policy captures high-reward sequences in time series data.
- **Dynamic Accuracy**: Measuring the accuracy of learned system dynamics against the true system transitions.
- **Top-100 Reward Score**: The average score of the top-100 high-reward trajectories, indicating the effectiveness of the learning process.

### Pseudo-Code for OBP-GFN Training

```python
# Training Optimized Backward Policy for GFlowNets in Stochastic Environments
Initialize replay buffer B, forward policy P_F, backward policy P_B, and parameter Z_theta
while not converged:
    Sample batch of trajectories {tau^(k)} from behavior policy
    Update B with sampled trajectories
    for n = 1 to N:
        Update P_B to minimize the loss over B using stochastic gradients
    Update P_F and Z_theta to minimize the trajectory balance loss
```

### Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/vincehass/Optimized-Backward-Policies-for-Effective-Reward-Matching-in-Generative-Flow-Networks.git
   cd OBP-GFN-timeseries
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run experiments**:
   ```bash
   python experiments/run_time_series.py
   ```

### Contributions

The key contributions of OBP-GFN for stochastic environments include:

- **Optimized Backward Flow for Time Series Data**: Addressing under-exploitation by optimizing backward policies to align better with true system rewards, effectively learning the dynamic system transitions.
- **Enhanced System Dynamics Learning**: OBP-GFN improves the ability to model and learn complex temporal relationships in stochastic environments, allowing for better prediction and understanding of system evolution.
- **Broad Applicability**: OBP-GFN's reward alignment strategy has been validated across various time series benchmarks, demonstrating its versatility and robustness in handling stochastic dynamics.

### Future Work

- **Exploration-Exploitation Balance**: Explore methods to combine OBP-GFN with exploration-oriented policies for better learning in complex systems.
- **Real-World System Applications**: Extend OBP-GFN to model real-world systems in domains such as finance, healthcare, and climate modeling, where stochastic environments are prevalent.

### License

This project is licensed under the MIT License.

### Contact

For any questions or feedback, please reach out to the authors at:

- Nadhir Hassen: \texttt{nadhir.hassen@umontreal.ca}
