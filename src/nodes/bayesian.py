from src.state import SCUAState

def bayesian_engine_node(state: SCUAState):
    metrics = state["metrics"]
    history = state.get("history", [])
    
    # Calculate how many times the opponent has defected recently
    recent_defs = sum(1 for h in history[-3:] if h.get('opponent_move') == 'Defect')
    
    # Logic: If opponent defects 2+ times in a row, spike Deceptive probability
    p_deceptive = 0.10 # Base
    if recent_defs >= 2:
        p_deceptive = 0.85  # CRITICAL: This must change to trigger Gemma
    elif recent_defs == 1:
        p_deceptive = 0.40

    beliefs = {
        "Deceptive": p_deceptive,
        "Noisy": 0.10 if recent_defs < 2 else 0.05,
        "Non-Stationary": 0.70 if metrics['volatility'] > 0.4 else 0.1,
        "Irrational": 0.10
    }
        
    return {"beliefs": beliefs}
