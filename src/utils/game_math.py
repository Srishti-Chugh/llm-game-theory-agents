import numpy as np
from scipy.stats import entropy

def calculate_nash_deviation(current_payoff, equilibrium_payoff):
    """Quantifies deviation from the Nash Equilibrium"""
    return abs(current_payoff - equilibrium_payoff)

def calculate_entropy(move_history):
    """Measures the randomness/unpredictability of opponent moves"""
    if not move_history: return 0.5
    # Frequency of moves (e.g., [1, 0, 1, 1])
    _, counts = np.unique(move_history, return_counts=True)
    probs = counts / len(move_history)
    return entropy(probs, base=2)

def calculate_social_welfare(player_a_payoff, player_b_payoff):
    """Calculates the sum of all players' utilities"""
    return player_a_payoff + player_b_payoff

def calculate_volatility(metric_history):
    """Standard deviation of metrics to detect Non-Stationarity"""
    if len(metric_history) < 2: return 0.0
    return np.std(metric_history)
