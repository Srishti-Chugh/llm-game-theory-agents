from games.base_game import BaseGame

class TrustGame(BaseGame):
    def __init__(self, rounds=10):
        super().__init__(rounds)

    def get_payoff(self, sent, returned):
        payoff_a = 10 - sent + returned
        payoff_b = sent * 3 - returned
        return payoff_a, payoff_b
