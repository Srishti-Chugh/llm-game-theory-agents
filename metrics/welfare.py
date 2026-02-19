# metrics/welfare.py

def social_welfare(payoffs_a, payoffs_b):
    """
    Average sum of payoffs per round (social welfare).
    """
    total = 0
    for a, b in zip(payoffs_a, payoffs_b):
        total += (a + b)

    return total / len(payoffs_a)
