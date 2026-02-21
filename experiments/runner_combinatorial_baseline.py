from agents.llm_agent_combinatorial import CombinatorialLLMAgent
from games.combinatorial_game import CombinatorialGame
from metrics.allocation_metrics import strategy_volatility, allocation_entropy
from metrics.welfare import social_welfare

import os
import json
import pandas as pd


# ================= CONFIG =================

ROUNDS = 10

agent1 = CombinatorialLLMAgent("Agent1", "prompts/combinatorial_neutral.txt")
agent2 = CombinatorialLLMAgent("Agent2", "prompts/combinatorial_neutral.txt")

game = CombinatorialGame(rounds=ROUNDS)

os.makedirs("results", exist_ok=True)

rows = []
allocations_a = []
allocations_b = []
payoffs_a = []
payoffs_b = []


# ================= RUN GAME =================

for r in range(game.rounds):
    a1 = agent1.act(game.history)
    a2 = agent2.act(game.history)

    p1, p2 = game.step(a1, a2)

    allocations_a.append(a1)
    allocations_b.append(a2)
    payoffs_a.append(p1)
    payoffs_b.append(p2)

    rows.append({
        "round": r + 1,
        "allocation_a": a1,
        "allocation_b": a2,
        "payoff_a": p1,
        "payoff_b": p2
    })


# ================= SAVE CSV =================

csv_file = "results/combinatorial_baseline.csv"
df = pd.DataFrame(rows)
df.to_csv(csv_file, index=False)


# ================= METRICS =================

volatility_a = strategy_volatility(allocations_a)
volatility_b = strategy_volatility(allocations_b)

entropy_a = allocation_entropy(allocations_a)
entropy_b = allocation_entropy(allocations_b)

welfare = social_welfare(payoffs_a, payoffs_b)

avg_payoff_a = sum(payoffs_a) / len(payoffs_a)
avg_payoff_b = sum(payoffs_b) / len(payoffs_b)


# ================= SUMMARY JSON =================

summary_data = {
    "experiment": "combinatorial_baseline",
    "rounds": ROUNDS,

    "strategy_volatility_A": volatility_a,
    "strategy_volatility_B": volatility_b,

    "allocation_entropy_A": entropy_a,
    "allocation_entropy_B": entropy_b,

    "social_welfare": welfare,

    "avg_payoff_A": avg_payoff_a,
    "avg_payoff_B": avg_payoff_b
}

summary_file = "results/summary_combinatorial_baseline.json"

with open(summary_file, "w") as f:
    json.dump(summary_data, f, indent=4)


# ================= PRINT =================

print("=== Combinatorial Baseline Results ===")
print("Strategy Volatility A:", volatility_a)
print("Strategy Volatility B:", volatility_b)
print("Allocation Entropy A:", entropy_a)
print("Allocation Entropy B:", entropy_b)
print("Social Welfare:", welfare)
print("Avg Payoff A:", avg_payoff_a)
print("Avg Payoff B:", avg_payoff_b)

print("\nSaved CSV to:", csv_file)
print("Saved summary JSON to:", summary_file)