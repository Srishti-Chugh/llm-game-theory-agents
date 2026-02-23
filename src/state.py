from typing import TypedDict, Annotated, Dict, List, Union
from langgraph.graph.message import add_messages

class SCUAState(TypedDict):
    # History & Feedback Loop
    messages: Annotated[list, add_messages]
    history: List[Dict]
    
    # Layer 1: Perception
    game_regime: str           # "Bayesian", "Quantum", "Combinatorial"
    metrics: Dict[str, float]  # entropy, volatility, nash_dev
    
    # Layer 2: Beliefs
    beliefs: Dict[str, float]  # Probability: {Deceptive, Noisy, Non-Stationary, Irrational}
    trust_score: float
    
    # Layer 3: Logic
    logic_recommendation: str  # Optimal move from MCTS or Quantum solver
    
    # Output
    final_action: str
