# metrics/logger_bayesian.py

import csv

class LoggerBayesian:
    def __init__(self):
        self.rows = []

    def log_round(self, r, a1, a2, p1, p2, belief1, belief2):
        b1_coop = belief1 if belief1 is not None else None
        b1_self = round(1 - belief1, 4) if belief1 is not None else None
        b2_coop = belief2 if belief2 is not None else None
        b2_self = round(1 - belief2, 4) if belief2 is not None else None
        self.rows.append([
            r, a1, a2, p1, p2,
            b1_coop, b1_self,
            b2_coop, b2_self
        ])

    def save(self, filename):
        header = ["round","action_a","action_b","payoff_a","payoff_b",
                  "belief1_coop","belief1_selfish","belief2_coop","belief2_selfish"]
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(self.rows)