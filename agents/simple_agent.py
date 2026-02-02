import random

class RandomAgent:
    def __init__(self, name):
        self.name = name

    def act(self, history):
        return random.choice(["C", "D"])
