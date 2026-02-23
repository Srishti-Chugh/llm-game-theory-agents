from src.state import SCUAState

def bayesian_engine_node(state: SCUAState):
    history = state.get("history", [])
    recent_moves = [h.get('opponent_move') for h in history[-3:]]
    
    # COUNT DEFECTIONS
    total_defects = recent_moves.count("Defect")
    
    # 1. NOISY CHECK: If it's just ONE defect after long cooperation
    if total_defects == 1 and len(history) > 3:
        p_noisy = 0.80      # High probability of a 'slip' or noise
        p_deceptive = 0.15   # Low probability of a calculated betrayal
    
    # 2. DECEPTION CHECK: Multiple defects in a row
    elif total_defects >= 2:
        p_noisy = 0.10
        p_deceptive = 0.85
        
    else:
        p_noisy = 0.05
        p_deceptive = 0.05

    return {
        "beliefs": {
            "Noisy": p_noisy, 
            "Deceptive": p_deceptive,
            "Non-Stationary": 0.1, 
            "Irrational": 0.1
        }
    }
