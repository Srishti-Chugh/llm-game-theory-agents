# 🧠 LLM Agents in Repeated Strategic Games: A Game-Theoretic and Entropy-Based Study

## 📌 Project Overview

This project investigates the behavior of Large Language Model (LLM) agents in repeated strategic games using concepts from **Game Theory**, **Information Theory (Entropy)**, and **Multi-Agent Systems**.

We study how LLM agents:

* Choose actions in repeated Prisoner’s Dilemma and Trust Games
* Converge to Nash Equilibrium strategies
* Exhibit **entropy collapse** (loss of strategic diversity)
* Respond to different **linguistic framings** (neutral vs moral prompts)

The project aims to provide empirical and theoretical insights into:

> How language-based reasoning systems behave as rational agents in strategic environments.

---

## 🎯 Motivation & Novelty

While prior work benchmarks LLMs on games, this project introduces:

* ✅ **Entropy-based analysis** of LLM strategies over time
* ✅ Study of **language framing (moral vs neutral)** as a utility modifier
* ✅ Empirical demonstration of **Nash equilibrium convergence**
* ✅ Modular framework for experimenting with LLM-based agents
* ✅ Foundation for future work in Bayesian games and multi-agent orchestration

This bridges:

* Game Theory
* Human-AI Interaction
* Cognitive Systems
* Multi-Agent Learning

---

## 🧩 Key Components

### 1. Games Implemented

* Prisoner’s Dilemma
* Trust Game (extendable)

Each game defines:

* Action space
* Payoff matrix
* History tracking

---

### 2. Agents

* **SimpleAgent / RandomAgent** (baseline)
* **LLMAgent** (uses prompts + game history)

LLM agents:

* Read game history
* Reason using prompt templates
* Output structured actions (C or D)

---

### 3. Metrics

Implemented in `metrics/`:

* **Action Entropy** (strategy randomness)
* Cooperation rate
* Payoff tracking
* Round-by-round logging

Entropy is used to detect:

> Strategic collapse vs sustained diversity.

---

## 📁 Project Structure

```
llm-game-theory-agents/
│
├── games/
│   ├── prisoners_dilemma.py
│   ├── trust_game.py
│
├── agents/
│   ├── llm_agent.py
│
├── prompts/
│   ├── neutral.txt
│   ├── moral.txt
│
├── experiments/
│   └── run_two_agent_game.py
│
├── metrics/
│   ├── entropy.py
│   ├── logger.py
│
├── plots/
│   └── plot_results.py
│
├── results/
│
├── config.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/your-username/llm-game-theory-agents.git
cd llm-game-theory-agents
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Add API Key

Create `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

---

## ▶️ Running Experiments

Run a two-agent Prisoner’s Dilemma experiment:

```bash
python experiments/run_two_agent_game.py
```

This will:

* Run repeated game rounds
* Log actions and payoffs
* Compute entropy for each agent
* Save results to `results/`

---

## 📊 Example Output

Sample result:

```
Round, Action_A, Action_B, Payoff_A, Payoff_B
1, D, C, 5, 0
2, D, D, 1, 1
...
Entropy Agent A: 0.0
Entropy Agent B: 0.46
```

Interpretation:

* LLM agents converge to defection (Nash equilibrium)
* Entropy collapses → deterministic strategy
* Moral framing can alter cooperation levels

---

## 🧪 Experiments Supported

* LLM vs LLM
* LLM vs Random Agent
* Neutral vs Moral framing
* Multiple runs with entropy analysis
* Strategy convergence tracking

---

## 🔬 Research Questions

This project explores:

1. Do LLM agents converge to Nash Equilibrium in repeated games?
2. Does linguistic framing influence cooperation?
3. How fast does entropy collapse?
4. Are LLM strategies stable across runs?
5. How does reasoning differ from payoff maximization?

---

## 🚀 Future Extensions

Planned enhancements:

* Bayesian incomplete-information games
* Multi-agent societies (3+ agents)
* Entropy dynamics over long horizons
* Test-time scaling (longer reasoning chains)
* Evolutionary agent populations
* Visualization dashboards

---

## 🧠 Scientific Relevance

This project contributes to:

* Game Theory in AI
* Cognitive Systems & Human-AI Interaction
* Multi-Agent Systems
* Empirical analysis of LLM reasoning
* Strategic alignment research

---
