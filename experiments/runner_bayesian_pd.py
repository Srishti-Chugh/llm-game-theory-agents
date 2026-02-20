# experiments/runner_bayesian_pd.py
import os
import pandas as pd
from agents.llm_agent_bayesian import BayesianLLMAgent
from games.bayesian_pd import BayesianPrisonersDilemma
from metrics.logger_bayesian import LoggerBayesian

from metrics.belief_metrics import belief_entropy, belief_volatility, belief_convergence_time
from metrics.utility_metrics import regret
from metrics.policy_metrics import action_belief_alignment, strategy_volatility
from metrics.exploitation_metrics import exploitation_vulnerability
from metrics.welfare import social_welfare

game = BayesianPrisonersDilemma(rounds=20, type_a="cooperative", type_b="selfish")

agent1 = BayesianLLMAgent("Agent1", "prompts/bayesian_neutral.txt", reasoning_steps=3)
agent2 = BayesianLLMAgent("Agent2", "prompts/bayesian_neutral.txt", reasoning_steps=3)

logger = LoggerBayesian()

actions1 = []
actions2 = []
payoffs_a = []
payoffs_b = []
beliefs_a = []
beliefs_b = []

for r in range(game.rounds):
    a1 = agent1.act(game.history)
    a2 = agent2.act(game.history)

    p1, p2 = game.step(a1, a2)

    agent1.update_belief(a2)
    agent2.update_belief(a1)

    actions1.append(a1)
    actions2.append(a2)
    payoffs_a.append(p1)
    payoffs_b.append(p2)
    beliefs_a.append(agent1.current_belief)
    beliefs_b.append(agent2.current_belief)

    logger.log_round(r+1, a1, a2, p1, p2,
                     agent1.current_belief, agent2.current_belief)

logger.save("results/bayesian_pd_run.csv")

# ================= METRICS =================

print("Final belief entropy A:", belief_entropy(beliefs_a[-1]))
print("Final belief entropy B:", belief_entropy(beliefs_b[-1]))

print("Belief Volatility A:", belief_volatility(beliefs_a))
print("Belief Convergence A:", belief_convergence_time(beliefs_a))

regrets = [regret(b, a) for b, a in zip(beliefs_a, actions1)]
print("Avg Regret A:", sum(regrets) / len(regrets))

print("Action-belief alignment A:",
      action_belief_alignment(actions1, beliefs_a))

print("Strategy volatility A:",
      strategy_volatility(actions1))

print("Exploitation vulnerability A:",
      exploitation_vulnerability(actions1, beliefs_a, actions2))

print("Social welfare:",
      social_welfare(payoffs_a, payoffs_b))


summary_data = {
    "belief_entropy_final_A": belief_entropy(beliefs_a[-1]),
    "belief_entropy_final_B": belief_entropy(beliefs_b[-1]),
    "belief_volatility_A": belief_volatility(beliefs_a),
    "belief_convergence_A": belief_convergence_time(beliefs_a),
    "avg_regret_A": sum(regrets) / len(regrets),
    "action_belief_alignment_A": action_belief_alignment(actions1, beliefs_a),
    "strategy_volatility_A": strategy_volatility(actions1),
    "exploitation_vulnerability_A": exploitation_vulnerability(actions1, beliefs_a, actions2),
    "social_welfare": social_welfare(payoffs_a, payoffs_b)
}

summary_file = "results/summary_bayesian_pd.csv"
os.makedirs("results", exist_ok=True)

# Append if exists, else create
if os.path.exists(summary_file):
    df_existing = pd.read_csv(summary_file)
    df_new = pd.DataFrame([summary_data])
    df_all = pd.concat([df_existing, df_new], ignore_index=True)
else:
    df_all = pd.DataFrame([summary_data])

df_all.to_csv(summary_file, index=False)

print("✅ Summary saved to", summary_file)