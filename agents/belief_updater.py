# agents/belief_updater.py

import math

def normalize(dist):
    total = sum(dist.values())
    return {k: v / total for k, v in dist.items()}

def update_belief(prior_belief, observed_action):
    """
    prior_belief: dict {"cooperative":0.5, "selfish":0.5}
    observed_action: "C" or "D"
    """

    likelihood = {
        "cooperative": 0.8 if observed_action == "C" else 0.2,
        "selfish": 0.2 if observed_action == "C" else 0.8
    }

    posterior = {}
    for t in prior_belief:
        posterior[t] = prior_belief[t] * likelihood[t]

    return normalize(posterior)