from metrics.quantum_entropy import quantum_entropy

def coherence_loss(list_baseline, list_decoherence):
    """
    Normalized coherence loss due to decoherence.
    """
    H_base = quantum_entropy(list_baseline)
    H_deco = quantum_entropy(list_decoherence)

    if H_base == 0:
        return 0.0

    return max(0.0, (H_deco - H_base) / H_base)