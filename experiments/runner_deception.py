import random
import json
import os

from agents.llm_agent_bayesian import BayesianLLMAgent
from games.bayesian_pd import BayesianPrisonersDilemma
from metrics.logger_bayesian import LoggerBayesian

from metrics.belief_metrics import belief_entropy, belief_volatility, belief_convergence_time
from metrics.utility_metrics import regret
from metrics.policy_metrics import action_belief_alignment, strategy_volatility
from metrics.exploitation_metrics import exploitation_vulnerability
from metrics.welfare import social_welfare


# =========================
# Opponent types
# =========================

class DeceptiveAgent:
    def __init__(self):
        self.round = 0

    def act(self, history):
        self.round += 1
        if self.round <= 10:
            return "C"
        return "D"

    def update_belief(self, action):
        pass


class NonStationaryAgent:
    def __init__(self):
        self.round = 0

    def act(self, history):
        self.round += 1
        if self.round <= 10:
            return "C"
        elif self.round <= 20:
            return "D"
        else:
            return random.choice(["C", "D"])

    def update_belief(self, action):
        pass


# =========================
# Failure detector
# =========================

def detect_failures(metrics):
    return {
        "high_entropy": metrics["final_entropy_A"] > 0.7,
        "high_regret": metrics["avg_regret_A"] > 3,
        "exploited": metrics["exploitation_vulnerability_A"] > 0,
        "unstable_strategy": metrics["strategy_volatility_A"] > 0.2
    }


# =========================
# Experiment runner
# =========================

def run_experiment(experiment_name, noisy=False, irrational=False, opponent_type="baseline"):
    game = BayesianPrisonersDilemma(rounds=30, type_a="cooperative", type_b="selfish")

    if irrational:
        prompt = "prompts/bayesian_irrational.txt"
    else:
        prompt = "prompts/bayesian_neutral.txt"

    agent1 = BayesianLLMAgent("Agent1", prompt, reasoning_steps=3)

    if opponent_type == "deceptive":
        agent2 = DeceptiveAgent()
    elif opponent_type == "non_stationary":
        agent2 = NonStationaryAgent()
    else:
        agent2 = BayesianLLMAgent("Agent2", prompt, reasoning_steps=3)

    logger = LoggerBayesian()

    actions1, actions2 = [], []
    payoffs_a, payoffs_b = [], []
    beliefs_a = []

    NOISE_PROB = 0.3

    for r in range(game.rounds):
        a1 = agent1.act(game.history)
        a2 = agent2.act(game.history)

        p1, p2 = game.step(a1, a2)

        observed_a2 = a2
        if noisy and random.random() < NOISE_PROB:
            observed_a2 = "D" if a2 == "C" else "C"

        agent1.update_belief(observed_a2)
        if hasattr(agent2, "update_belief"):
            agent2.update_belief(a1)

        logger.log_round(r+1, a1, a2, p1, p2,
                         agent1.current_belief,
                         getattr(agent2, "current_belief", None))

        actions1.append(a1)
        actions2.append(a2)
        payoffs_a.append(p1)
        payoffs_b.append(p2)
        beliefs_a.append(agent1.current_belief)

    os.makedirs("results", exist_ok=True)
    logger.save(f"results/{experiment_name}_log.csv")

    # =========================
    # Metrics
    # =========================

    final_entropy_A = belief_entropy(beliefs_a[-1])
    volatility_A = belief_volatility(beliefs_a)
    convergence_A = belief_convergence_time(beliefs_a)

    regrets = [regret(b, a) for b, a in zip(beliefs_a, actions1)]
    avg_regret_A = sum(regrets) / len(regrets)

    alignment_A = action_belief_alignment(actions1, beliefs_a)
    strategy_vol_A = strategy_volatility(actions1)

    exploit_A = exploitation_vulnerability(actions1, beliefs_a, actions2)
    welfare = social_welfare(payoffs_a, payoffs_b)

    metrics = {
        "final_entropy_A": final_entropy_A,
        "belief_volatility_A": volatility_A,
        "belief_convergence_A": convergence_A,
        "avg_regret_A": avg_regret_A,
        "action_belief_alignment_A": alignment_A,
        "strategy_volatility_A": strategy_vol_A,
        "exploitation_vulnerability_A": exploit_A,
        "social_welfare": welfare
    }

    failures = detect_failures(metrics)

    summary = {
        "experiment": experiment_name,
        "metrics": metrics,
        "failures": failures
    }

    with open(f"results/{experiment_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Finished {experiment_name}")
    print(json.dumps(summary, indent=2))


# =========================
# Main
# =========================

if __name__ == "__main__":
    run_experiment("baseline")

    run_experiment("deceptive_opponent", opponent_type="deceptive")

    run_experiment("non_stationary_opponent", opponent_type="non_stationary")

    run_experiment("noisy_observation", noisy=True)

    run_experiment("irrational_agent", irrational=True)