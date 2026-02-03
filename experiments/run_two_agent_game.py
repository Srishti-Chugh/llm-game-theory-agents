import os
import pandas as pd
from agents.llm_agent import LLMAgent
from games.prisoners_dilemma import PrisonersDilemma
from metrics.logger import Logger
from metrics.entropy import entropy_over_time, action_entropy
from metrics.cooperation import cooperation_rate
from metrics.nash import nash_deviation
from metrics.payoff import average_payoff
from plots.plot_results import plot_entropy, plot_cooperation

agent1 = LLMAgent("Agent1", "prompts/neutral.txt")
agent2 = LLMAgent("Agent2", "prompts/moral.txt")

game = PrisonersDilemma(rounds=20)
logger = Logger()

actions1 = []
actions2 = []
payoffs_a = []
payoffs_b = []

for r in range(game.rounds):
    a1 = agent1.act(game.history)
    a2 = agent2.act(game.history)

    p1, p2 = game.step(a1, a2)

    actions1.append(a1)
    actions2.append(a2)
    payoffs_a.append(p1)
    payoffs_b.append(p2)

    logger.log_round(r + 1, a1, a2, p1, p2)

output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

logger.save(os.path.join(output_dir, "pd_test.csv"))

entropy_a = entropy_over_time(actions1)
entropy_b = entropy_over_time(actions2)

print("Final Entropy A:", entropy_a[-1])
print("Final Entropy B:", entropy_b[-1])

print("Cooperation A:", cooperation_rate(actions1))
print("Cooperation B:", cooperation_rate(actions2))
print("Nash Deviation:", nash_deviation(actions1, actions2))

print("Avg Payoff A:", average_payoff(payoffs_a))
print("Avg Payoff B:", average_payoff(payoffs_b))

plot_entropy(entropy_a, entropy_b)
plot_cooperation(actions1, actions2)

coop_a = cooperation_rate(actions1)
coop_b = cooperation_rate(actions2)
nash_dev = nash_deviation(actions1, actions2)
avg_payoff_a = average_payoff(payoffs_a)
avg_payoff_b = average_payoff(payoffs_b)

summary_data = {
    "coop_a": coop_a,
    "coop_b": coop_b,
    "entropy_a_final": entropy_a[-1],
    "entropy_b_final": entropy_b[-1],
    "nash_deviation": nash_dev
}

summary_file = "results/summary.csv"

# If file exists, append; else create
if os.path.exists(summary_file):
    df_existing = pd.read_csv(summary_file)
    df_new = pd.DataFrame([summary_data])
    df_all = pd.concat([df_existing, df_new], ignore_index=True)
else:
    df_all = pd.DataFrame([summary_data])

df_all.to_csv(summary_file, index=False)