import os
import json
import pandas as pd
import numpy as np

from agents.llm_agent_combinatorial import CombinatorialLLMAgent
from games.combinatorial_game import CombinatorialGame

from metrics.allocation_metrics import strategy_volatility, allocation_entropy
from metrics.welfare import social_welfare
from metrics.exploitation_metrics import combinatorial_exploitation_vulnerability


RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

ROUNDS = 15   # reduced from 30 (better for LLM + clearer failures)
NUM_ITEMS = 4
TOTAL_BUDGET = 100

NEUTRAL_PROMPT = "prompts/combinatorial_neutral.txt"

EXPERIMENTS = [
    {"name": "deceptive_opponent", "mode": "deceptive"},
    {"name": "irrational_agent", "mode": "irrational"},
    {"name": "non_stationary_opponent", "mode": "non_stationary"},
    {"name": "noisy_observation", "mode": "noisy"}
]


def sanitize_action(action, num_items=NUM_ITEMS, total=TOTAL_BUDGET):
    try:
        action = list(map(float, action))
    except Exception:
        action = [total / num_items] * num_items

    if len(action) > num_items:
        action = action[:num_items]
    elif len(action) < num_items:
        action += [0.0] * (num_items - len(action))

    action = np.array(action)
    action[action < 0] = 0

    if action.sum() == 0:
        action = np.ones(num_items) * (total / num_items)
    else:
        action = (action / action.sum()) * total

    return action.round(2).tolist()


def evaluate_failures(metrics):
    return {
        "high_entropy": metrics["final_entropy_A"] > 0.7,
        "unstable_strategy": metrics["strategy_volatility_A"] > 0.3,
        "exploited": metrics["exploitation_vulnerability_A"] > 0.15,
        "low_welfare": metrics["social_welfare"] < 150
    }


for exp in EXPERIMENTS:

    print(f"\nRunning experiment: {exp['name']}")

    noisy = exp["mode"] == "noisy"
    non_stationary = exp["mode"] == "non_stationary"

    game = CombinatorialGame(rounds=ROUNDS, noisy=noisy, non_stationary=non_stationary)

    agentA = CombinatorialLLMAgent("AgentA", NEUTRAL_PROMPT)

    # --- Create AgentB once ---
    if exp["mode"] == "deceptive":
        agentB = CombinatorialLLMAgent("AgentB", "prompts/combinatorial_deceptive.txt")
    elif exp["mode"] == "irrational":
        agentB = CombinatorialLLMAgent("AgentB", "prompts/combinatorial_irrational.txt")
    else:
        agentB = CombinatorialLLMAgent("AgentB", NEUTRAL_PROMPT)

    actions_a, actions_b = [], []
    payoffs_a, payoffs_b = [], []
    rows = []

    for r in range(game.rounds):

        # ---- Non-stationary regime switch (only prompt changes) ----
        if exp["mode"] == "non_stationary" and r == game.rounds // 2:
            print("Switching AgentB to deceptive policy (non-stationary)")
            agentB.prompt = open("prompts/combinatorial_deceptive.txt").read()

        history = game.history

        raw_a1 = agentA.act(history)
        raw_a2 = agentB.act(history)

        a1 = sanitize_action(raw_a1)
        a2 = sanitize_action(raw_a2)

        p1, p2 = game.step(a1, a2)

        actions_a.append(a1)
        actions_b.append(a2)
        payoffs_a.append(p1)
        payoffs_b.append(p2)

        rows.append({
            "round": r + 1,
            "action_a": json.dumps(a1),
            "action_b": json.dumps(a2),
            "payoff_a": p1,
            "payoff_b": p2
        })

    # ---- Save CSV ----
    df = pd.DataFrame(rows)
    csv_file = os.path.join(RESULTS_DIR, f"combinatorial_{exp['name']}.csv")
    df.to_csv(csv_file, index=False)

    # ---- Metrics ----
    entropy_a = allocation_entropy(actions_a)
    strat_vol_a = strategy_volatility(actions_a)
    exploit_vuln = combinatorial_exploitation_vulnerability(actions_a, actions_b)
    welfare = social_welfare(payoffs_a, payoffs_b)

    metrics = {
        "final_entropy_A": float(entropy_a),
        "strategy_volatility_A": float(strat_vol_a),
        "exploitation_vulnerability_A": float(exploit_vuln),
        "social_welfare": float(welfare)
    }

    failures = evaluate_failures(metrics)

    summary = {
        "experiment": exp["name"],
        "metrics": metrics,
        "failures": failures
    }

    summary_file = os.path.join(RESULTS_DIR, f"combinatorial_{exp['name']}_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=4)

    print("Metrics:", metrics)
    print("Failures:", failures)