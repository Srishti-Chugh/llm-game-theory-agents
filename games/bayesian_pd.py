# games/bayesian_pd.py

import random

class BayesianPrisonersDilemma:
    def __init__(self, rounds=20, type_a="cooperative", type_b="selfish"):
        self.rounds = rounds
        self.type_a = type_a
        self.type_b = type_b
        self.history = []

        # Standard PD payoff matrix
        self.payoff_matrix = {
            ("C", "C"): (3, 3),
            ("D", "C"): (5, 0),
            ("C", "D"): (0, 5),
            ("D", "D"): (1, 1),
        }

    def step(self, action_a, action_b):
        payoff_a, payoff_b = self.payoff_matrix[(action_a, action_b)]
        self.history.append((action_a, action_b, payoff_a, payoff_b))
        return payoff_a, payoff_b

    def get_true_types(self):
        return self.type_a, self.type_b