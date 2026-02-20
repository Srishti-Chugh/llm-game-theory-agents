# experiments/runner_bayesian_pd.py

from agents.llm_agent_bayesian import BayesianLLMAgent
from games.bayesian_pd import BayesianPrisonersDilemma
from metrics.logger_bayesian import LoggerBayesian
from metrics.belief_metrics import belief_entropy

game = BayesianPrisonersDilemma(rounds=20, type_a="cooperative", type_b="selfish")

agent1 = BayesianLLMAgent("Agent1", "prompts/bayesian_neutral.txt", reasoning_steps=3)
agent2 = BayesianLLMAgent("Agent2", "prompts/bayesian_neutral.txt", reasoning_steps=3)

logger = LoggerBayesian()

for r in range(game.rounds):
    a1 = agent1.act(game.history)
    a2 = agent2.act(game.history)

    p1, p2 = game.step(a1, a2)

    agent1.update_belief(a2)
    agent2.update_belief(a1)

    logger.log_round(r+1, a1, a2, p1, p2, agent1.current_belief, agent2.current_belief)

logger.save("results/bayesian_pd_run.csv")

print("Final belief entropy A:", belief_entropy(agent1.current_belief))
print("Final belief entropy B:", belief_entropy(agent2.current_belief))