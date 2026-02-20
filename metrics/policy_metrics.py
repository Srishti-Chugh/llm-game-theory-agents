import numpy as np

def action_belief_alignment(actions, beliefs, threshold=0.7):
    """
    Measures P(C | belief_coop > threshold)
    """
    coop_when_high_belief = 0
    count = 0

    for a, b in zip(actions, beliefs):
        if b > threshold:
            count += 1
            if a == "C":
                coop_when_high_belief += 1

    return coop_when_high_belief / count if count > 0 else 0


def strategy_volatility(actions):
    changes = 0
    for i in range(1, len(actions)):
        if actions[i] != actions[i-1]:
            changes += 1
    return changes / len(actions)