from collections import Counter
import numpy as np

def action_entropy(actions):
    counts = Counter(actions)
    probs = np.array(list(counts.values())) / len(actions)
    return -np.sum(probs * np.log2(probs))
