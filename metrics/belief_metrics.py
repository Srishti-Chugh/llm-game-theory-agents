# metrics/belief_metrics.py
# belief is a float: P(opponent is cooperative type)

import math
import numpy as np

def belief_entropy(belief):
    """Binary entropy of the belief distribution (cooperative vs selfish)."""
    probs = [belief, 1 - belief]
    entropy = 0
    for p in probs:
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

def kl_divergence(belief, true_type):
    """KL divergence from belief distribution to the true type (one-hot)."""
    belief_dist = {"cooperative": belief, "selfish": 1 - belief}
    true_dist = {"cooperative": 0.0, "selfish": 0.0}
    true_dist[true_type] = 1.0

    kl = 0
    for k in belief_dist:
        if belief_dist[k] > 0 and true_dist[k] > 0:
            kl += belief_dist[k] * math.log2(belief_dist[k] / true_dist[k])
    return kl



def belief_volatility(beliefs):
    """Std deviation of belief changes"""
    diffs = np.diff(beliefs)
    return float(np.std(diffs))


def belief_convergence_time(beliefs, eps=0.05):
    """
    Round when belief stabilizes
    """
    for i in range(len(beliefs)-1):
        if abs(beliefs[i+1] - beliefs[i]) < eps:
            return i+1
    return len(beliefs)