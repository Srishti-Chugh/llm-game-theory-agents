from collections import Counter
import numpy as np

def action_entropy(actions):
    counts = Counter(actions)
    probs = np.array(list(counts.values())) / len(actions)
    return -np.sum(probs * np.log2(probs + 1e-9))


def entropy_over_time(actions, window=5):
    entropies = []
    for i in range(len(actions)):
        start = max(0, i-window)
        window_actions = actions[start:i+1]
        entropies.append(action_entropy(window_actions))
    return entropies
