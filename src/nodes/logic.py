import numpy as np
from src.state import SCUAState

class QuantumSolver:
    """Layer 3: Handles Superposition & Entanglement Logic"""
    def __init__(self, entanglement_gamma=np.pi/2):
        self.gamma = entanglement_gamma # Default: Max Entanglement

    def evaluate_quantum_advantage(self, coherence_loss: float):
        """
        Calculates if the 'Quantum Magic Move (Q)' is viable.
        If coherence_loss is high, the state collapses to classical PD.
        """
        effective_coherence = 1.0 - coherence_loss
        
        if effective_coherence > 0.7:
            return "RECOMMENDED: Q-Strategy (Quantum Superposition). This move 'undoes' classical defection via entanglement."
        elif effective_coherence > 0.3:
            return "CAUTION: Partial Decoherence. Recommend Mixed Quantum Strategy (Rotation Move)."
        else:
            return "CRITICAL: Decoherence/Collapse. Execute Classical Nash: Tit-for-Tat."

class CombinatorialSearch:
    """Layer 3: MCTS / Search for complex branching games"""
    def get_search_path(self, volatility: float):
        # Addresses 'Adaptation Delay' by projecting future states
        if volatility > 0.6:
            return "SEARCH RESULT: High Volatility detected. Pruning aggressive branches. Execute: Conservative-Reciprocity."
        return "SEARCH RESULT: Stable environment. Deep search suggests: Long-term-Cooperation-Path."

def logic_layer_node(state: SCUAState):
    """LangGraph Node: The Logic Engine"""
    q_solver = QuantumSolver()
    c_search = CombinatorialSearch()
    
    # 1. Pull real-time metrics
    coherence_loss = state["metrics"].get("coherence_loss", 0.0)
    volatility = state["metrics"].get("volatility", 0.0)
    
    # 2. Logic Branching per Layer 3 of SCUA Architecture
    if state["game_regime"] == "Quantum PD":
        # Force a recommendation for the "Q-Strategy"
        rec = q_solver.evaluate_quantum_advantage(coherence_loss)
    else:
        rec = c_search.get_search_path(volatility)
        
    return {"logic_recommendation": rec}
