from games.prisoners_dilemma import PrisonersDilemma
from agents.simple_agent import RandomAgent
from metrics.logger import Logger
from metrics.entropy import action_entropy

game = PrisonersDilemma(rounds=20)
agent1 = RandomAgent("A")
agent2 = RandomAgent("B")
logger = Logger()

actions_a = []
actions_b = []

for r in range(game.rounds):
    action_a = agent1.act(game.history)
    action_b = agent2.act(game.history)

    actions_a.append(action_a)
    actions_b.append(action_b)

    payoff_a, payoff_b = game.step(action_a, action_b)
    logger.log_round(r, action_a, action_b, payoff_a, payoff_b)

entropy_a = action_entropy(actions_a)
entropy_b = action_entropy(actions_b)

print("Entropy Agent A:", entropy_a)
print("Entropy Agent B:", entropy_b)

logger.save("results/pd_test.csv")
