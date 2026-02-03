from agents.llm_agent import LLMAgent
from games.prisoners_dilemma import PrisonersDilemma
from metrics.entropy import action_entropy
from metrics.logger import Logger
import os

agent1 = LLMAgent("Agent1", "prompts/neutral.txt")
agent2 = LLMAgent("Agent2", "prompts/moral.txt")

game = PrisonersDilemma(rounds=20)
logger = Logger()

actions1 = []
actions2 = []

for r in range(game.rounds):
    a1 = agent1.act(game.history)
    a2 = agent2.act(game.history)

    actions1.append(a1)
    actions2.append(a2)

    p1, p2 = game.step(a1, a2)
    logger.log_round(r + 1, a1, a2, p1, p2)

output_dir = "results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

logger.save(os.path.join(output_dir, "pd_test.csv"))

print("Entropy A1:", action_entropy(actions1))
print("Entropy A2:", action_entropy(actions2))
