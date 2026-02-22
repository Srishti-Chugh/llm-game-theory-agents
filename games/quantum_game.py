# games/quantum_game.py

import numpy as np
import random

class QuantumGame:
    def __init__(self, rounds=10, decoherence=False, non_stationary=False):
        self.rounds = rounds
        self.decoherence = decoherence
        self.non_stationary = non_stationary
        self.history = []

        # Payoff matrices
        self.payoff_matrix_1 = {
            (0,0):(3,3),
            (0,1):(0,5),
            (1,0):(5,0),
            (1,1):(1,1)
        }

        self.payoff_matrix_2 = {
            (0,0):(2,2),
            (0,1):(1,4),
            (1,0):(4,1),
            (1,1):(0,0)
        }

    def apply_noise(self, theta):
        return theta + random.uniform(-0.2, 0.2)

    def measure(self, theta1, theta2):
        p1 = np.cos(theta1)**2
        p2 = np.cos(theta2)**2

        o1 = 0 if random.random() < p1 else 1
        o2 = 0 if random.random() < p2 else 1

        return o1, o2

    def payoff(self, outcome, round_id):
        if self.non_stationary and round_id > self.rounds // 2:
            return self.payoff_matrix_2[outcome]
        return self.payoff_matrix_1[outcome]

    def step(self, theta1, theta2, round_id):

        if self.decoherence:
            theta1 = self.apply_noise(theta1)
            theta2 = self.apply_noise(theta2)

        outcome = self.measure(theta1, theta2)
        p1, p2 = self.payoff(outcome, round_id)

        self.history.append((theta1, theta2, outcome, p1, p2))
        return outcome, p1, p2