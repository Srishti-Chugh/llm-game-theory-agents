from games.base_game import BaseGame

class PrisonersDilemma(BaseGame):
    def __init__(self, rounds=10):
        super().__init__(rounds)
        self.actions = ["C", "D"]

    def get_actions(self):
        return self.actions

    def get_payoff(self, action_a, action_b):
        if action_a == "C" and action_b == "C":
            return 3, 3
        elif action_a == "C" and action_b == "D":
            return 0, 5
        elif action_a == "D" and action_b == "C":
            return 5, 0
        elif action_a == "D" and action_b == "D":
            return 1, 1
