import pandas as pd

class Logger:
    def __init__(self):
        self.data = []

    def log_round(self, round_no, action_a, action_b, payoff_a, payoff_b, reasoning1=None, reasoning2=None):
        self.data.append({
            "round": round_no,
            "action_a": action_a,
            "action_b": action_b,
            "payoff_a": payoff_a,
            "payoff_b": payoff_b,
            "reasoning1": reasoning1,
            "reasoning2": reasoning2
        })

    def save(self, filename):
        df = pd.DataFrame(self.data)
        df.to_csv(filename, index=False)