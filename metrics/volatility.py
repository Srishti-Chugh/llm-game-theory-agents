# metrics/volatility.py

def strategy_volatility(actions):
    """
    Fraction of rounds where the agent changes action.
    """
    if len(actions) < 2:
        return 0.0

    changes = 0
    for i in range(1, len(actions)):
        if actions[i] != actions[i-1]:
            changes += 1

    return changes / (len(actions) - 1)
