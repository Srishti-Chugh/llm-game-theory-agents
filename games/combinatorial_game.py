# games/combinatorial_game.py

import random
import numpy as np


class CombinatorialGame:
    def __init__(self, projects=4, budget=100, rounds=15,
                 noisy=False, non_stationary=False):
        self.projects = projects
        self.budget = budget
        self.rounds = rounds
        self.noisy = noisy
        self.non_stationary = non_stationary

        self.history = []          # what agents observe
        self.true_history = []     # ground truth

        # Initial project values (stationary environment)
        self.project_values = np.array([1.0, 1.2, 0.8, 1.5])

    # -------------------------
    # Payoff function
    # -------------------------
    def payoff(self, alloc1, alloc2):
        payoff1 = 0
        payoff2 = 0

        for i in range(self.projects):
            total = alloc1[i] + alloc2[i]

            if total == 0:
                continue

            # congestion penalty (competition hurts)
            share1 = alloc1[i] / total
            share2 = alloc2[i] / total

            value = self.project_values[i]

            payoff1 += share1 * value * alloc1[i]
            payoff2 += share2 * value * alloc2[i]

        return round(payoff1, 2), round(payoff2, 2)

    # -------------------------
    # Noise model
    # -------------------------
    def _add_noise(self, alloc):
        noisy_alloc = []
        for v in alloc:
            noise = random.randint(-3, 3)
            noisy_alloc.append(max(0, v + noise))
        return noisy_alloc

    # -------------------------
    # Non-stationary dynamics
    # -------------------------
    def _update_environment(self, round_idx):
        if self.non_stationary and round_idx == self.rounds // 2:
            # Change project values suddenly
            self.project_values = np.array([1.5, 0.7, 1.3, 0.6])
            print("Environment changed (non-stationary)")

    # -------------------------
    # Step
    # -------------------------
    def step(self, a1, a2):
        round_idx = len(self.true_history)

        # update environment if needed
        self._update_environment(round_idx)

        p1, p2 = self.payoff(a1, a2)

        # record true history
        self.true_history.append((a1, a2, p1, p2))

        # noisy observation
        if self.noisy:
            obs_a1 = self._add_noise(a1)
            obs_a2 = self._add_noise(a2)
        else:
            obs_a1, obs_a2 = a1, a2

        self.history.append((obs_a1, obs_a2, p1, p2))

        return p1, p2