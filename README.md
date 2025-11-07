# 🧠 Deep Q-Learning Agent — Atari *Crazy Climber*

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)
![Gymnasium](https://img.shields.io/badge/Env-Gymnasium-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

---

## 🎮 Project Overview

This project implements a **Deep Q-Learning (DQN)** agent to master the Atari game **CrazyClimber-v5**, built with **PyTorch**, **Gymnasium**, and **Atari Preprocessing** wrappers.

The agent learns to climb using reinforcement learning — observing frames, predicting Q-values for each action, and maximizing long-term rewards.  
It is part of the **LLM Agents & Deep Q-Learning** academic assignment, focusing on agent optimization, hyperparameter analysis, and policy exploration.

---

## 🚀 Key Features

✅ **DQN Implementation**
- Experience Replay + Target Network updates  
- ε-Greedy and Softmax exploration  
- Huber loss and Adam optimizer  
- FrameStack (4 × 84×84 grayscale frames)

✅ **Experimentation Framework**
- Configurable γ, α, ε decay rates, learning rate  
- Variant runs for hyperparameter sweeps  
- Visualization of metrics (returns, loss, steps)

✅ **Portfolio-Ready Deliverables**
- Jupyter notebook with analysis and plots  
- Scripts for environment inspection & reward probing  
- Documentation, licensing, and attribution files  

---

## 🧩 Results Summary

| Variant | Avg Return (last 10 eps) | Avg Steps | Highlights |
|----------|--------------------------|------------|-------------|
| **Baseline (γ = 0.99)** | **15,010** | 2,557 | Stable convergence |
| **γ = 0.95** | 18,340 | 2,861 | Faster initial learning |
| **γ = 0.999 + lr 5e-5** | 8,220 | 2,348 | Slower, underfitted |
| **Softmax (τ = 1.0)** | 24–30 k | 3,000 | Strong exploration & reward spikes |
| **Fast ε Decay (20 ep)** | 11,380 | 2,834 | Rapid exploration, less stability |
| **Slow ε Decay (80 ep)** | 12,010 | 2,367 | Gradual learning, consistent returns |

---

## 📊 Visual Insights

Key metrics visualized in the notebook:
- 📈 **Episode Return vs. Steps**
- 📉 **Loss Convergence Curve**
- 🎯 **Epsilon Decay Schedule**
- 🪜 **FrameStack Visualization (84×84 grayscale)**  
- 🧩 **Reward Distribution Probe**
- 🧠 **LLM-Agent Integration Diagram**

---

## 🧠 Concepts Demonstrated
- Markov Decision Processes (MDPs)  
- Bellman Optimality & Q-Value updates  
- Experience Replay & Stability in Training  
- Exploration–Exploitation trade-off  
- Hyperparameter tuning & analysis  

---

## ⚙️ Quick Start

```bash
# 1️⃣ Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
# or .venv\Scripts\activate   # Windows

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Verify setup
python -m src.smoke_test

# 4️⃣ Train the baseline agent
python train_dqn.py

# 5️⃣ Evaluate performance
python eval_baseline.py

# 6️⃣ Run variants (optional)
python scripts/run_variant.py --out_dir runs/gamma_0_95 --gamma 0.95
```

## 📂 Project Structure

crazyclimber-dqn/
├── src/           # Core implementation
├── scripts/       # Experiment and analysis scripts
├── runs/          # Saved checkpoints & logs
├── notebook.ipynb # Main experiment notebook
├── LICENSE
├── ATTRIBUTION.md
└── README.md


---

## 🪪 License & Attribution
MIT License © 2025 **Akshaya Gavhane**  
Developed for the *LLM Agents & Deep Q-Learning* course at Northeastern University.  
See [LICENSE](./LICENSE) and [ATTRIBUTION.md](./ATTRIBUTION.md) for details.
