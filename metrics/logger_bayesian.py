# metrics/logger_bayesian.py

import csv

class LoggerBayesian:
    def __init__(self):
        self.rows = []

    def log_round(self, r, a1, a2, p1, p2, belief1, belief2):
        self.rows.append([
            r, a1, a2, p1, p2,
            belief1, round(1 - belief1, 4),
            belief2, round(1 - belief2, 4)
        ])

    def save(self, filename):
        header = ["round","action_a","action_b","payoff_a","payoff_b",
                  "belief1_coop","belief1_selfish","belief2_coop","belief2_selfish"]
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(self.rows)