class BaseGame:
    def __init__(self, rounds=10):
        self.rounds = rounds
        self.history = []

    def get_actions(self):
        raise NotImplementedError

    def get_payoff(self, action_a, action_b):
        raise NotImplementedError

    def step(self, action_a, action_b):
        payoff_a, payoff_b = self.get_payoff(action_a, action_b)
        self.history.append((action_a, action_b, payoff_a, payoff_b))
        return payoff_a, payoff_b
