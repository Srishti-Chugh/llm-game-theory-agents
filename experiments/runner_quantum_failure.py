# runner_quantum_failure.py

import os
import json
import pandas as pd

from agents.llm_agent_quantum import QuantumLLMAgent
from games.quantum_game import QuantumGame

from metrics.quantum_entropy import quantum_entropy
from metrics.quantum_volatility import quantum_volatility
from metrics.collapse_instability import collapse_instability
from metrics.welfare import social_welfare
from metrics.coherence_loss import coherence_loss
from metrics.adaptation_delay import adaptation_delay
from metrics.exploitation_vulnerability import exploitation_vulnerability

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

ROUNDS = 10

EXPERIMENTS = [
    {"name": "baseline"},
    {"name": "decoherence", "decoherence": True},
    {"name": "irrational"},
    {"name": "deceptive"},
    {"name": "non_stationary", "non_stationary": True}
]

baseline_thetasA = None   # store baseline strategy for coherence loss

for exp in EXPERIMENTS:

    print("Running:", exp["name"])

    game = QuantumGame(
        rounds=ROUNDS,
        decoherence=exp.get("decoherence", False),
        non_stationary=exp.get("non_stationary", False)
    )

    if exp["name"] == "irrational":
        agentB_prompt = "prompts/quantum_irrational.txt"
    elif exp["name"] == "deceptive":
        agentB_prompt = "prompts/quantum_deceptive.txt"
    else:
        agentB_prompt = "prompts/quantum_neutral.txt"

    agentA = QuantumLLMAgent("AgentA", "prompts/quantum_neutral.txt")
    agentB = QuantumLLMAgent("AgentB", agentB_prompt)

    thetasA, thetasB, outcomes, payA, payB = [], [], [], [], []
    rows = []

    for r in range(ROUNDS):

        if exp["name"] == "non_stationary":
            if r < ROUNDS // 2:
                agentB = QuantumLLMAgent("AgentB", "prompts/quantum_neutral.txt")
            else:
                agentB = QuantumLLMAgent("AgentB", "prompts/quantum_deceptive.txt")

        thetaA = agentA.act(game.history)
        thetaB = agentB.act(game.history)

        outcome, p1, p2 = game.step(thetaA, thetaB, r)

        thetasA.append(thetaA)
        thetasB.append(thetaB)
        outcomes.append(outcome)
        payA.append(p1)
        payB.append(p2)

        rows.append({
            "round": r + 1,
            "theta_a": thetaA,
            "theta_b": thetaB,
            "outcome": outcome,
            "payoff_a": p1,
            "payoff_b": p2
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS_DIR}/quantum_{exp['name']}.csv", index=False)

    # ----- coherence loss only meaningful for decoherence -----
    if exp["name"] == "decoherence" and baseline_thetasA is not None:
        coherence = coherence_loss(baseline_thetasA, thetasA)
    else:
        coherence = 0.0

    metrics = {
        "entropy_A": float(quantum_entropy(thetasA)),
        "volatility_A": float(quantum_volatility(thetasA)),
        "collapse_instability": float(collapse_instability(thetasA, outcomes)),
        "social_welfare": float(social_welfare(payA, payB)),
        "coherence_loss": float(coherence),
        "adaptation_delay": float(adaptation_delay(thetasA, ROUNDS // 2)),
        "exploitation_vulnerability": float(exploitation_vulnerability(payA, payB))
    }

    failures = {
        # Strategy randomness
        "high_entropy": metrics["entropy_A"] > 1.5,

        # Strategy oscillation
        "unstable": metrics["volatility_A"] > 0.3,

        # Measurement-induced flip
        "collapse": metrics["collapse_instability"] > 0.5,

        # Loss of quantum coherence
        "coherence_loss": metrics["coherence_loss"] > 0.1,

        # Failure to adapt after environment shift
        "adaptation_delay": metrics["adaptation_delay"] > (ROUNDS // 4),

        # Opponent advantage
        "exploitation_vulnerability": metrics["exploitation_vulnerability"] > 0.3
    }

    summary = {
        "experiment": exp["name"],
        "metrics": metrics,
        "failures": failures
    }

    with open(f"{RESULTS_DIR}/quantum_{exp['name']}_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    # store baseline strategy
    if exp["name"] == "baseline":
        baseline_thetasA = thetasA.copy()

    print(metrics)
    print(failures)