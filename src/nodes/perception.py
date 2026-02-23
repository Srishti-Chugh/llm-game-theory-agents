from src.state import SCUAState
from src.utils.game_math import calculate_entropy, calculate_volatility

def perception_node(state: SCUAState):
    # Extract historical moves from the state's message history or dedicated history list
    history = state.get("history", [])
    opponent_moves = [h.get('opponent_move') for h in history]
    
    # Calculate real-time metrics for the "Metric Tracker"
    ent = calculate_entropy(opponent_moves)
    
    # Map moves to numbers for numerical analysis (e.g., Cooperate=1, Defect=0)
    move_values = [1.0 if m == "Cooperate" else 0.0 for m in opponent_moves]
    vol = calculate_volatility(move_values)
    
    # Logic for the "Contextual Classifier"
    # For now, we use a simple heuristic; this could be an LLM call later
    regime = "Bayesian PD" if len(history) < 10 else "Combinatorial"
    
    return {
        "metrics": {
            "entropy": ent, 
            "volatility": vol, 
            "nash_dev": 0.15 # Baseline example
        },
        "game_regime": regime
    }
