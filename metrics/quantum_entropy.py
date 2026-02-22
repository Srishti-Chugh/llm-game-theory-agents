# metrics/quantum_entropy.py

import numpy as np
from collections import Counter

def quantum_entropy(actions):
    """
    actions: list of discrete actions (0/1 or small integers)
    """
    counts = Counter(actions)
    total = sum(counts.values())

    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * np.log2(p)

    return entropy