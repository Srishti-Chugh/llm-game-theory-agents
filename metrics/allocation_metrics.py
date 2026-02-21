# metrics/allocation_metrics.py

import numpy as np

def strategy_volatility(allocations):
    changes = 0
    for i in range(1, len(allocations)):
        if allocations[i] != allocations[i-1]:
            changes += 1
    return changes / len(allocations)

def allocation_entropy(allocations):
    unique = [tuple(a) for a in allocations]
    probs = [unique.count(x)/len(unique) for x in set(unique)]
    return -sum(p*np.log2(p) for p in probs if p > 0)