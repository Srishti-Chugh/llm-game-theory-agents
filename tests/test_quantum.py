from src.graph import create_scua_graph
from src.utils.persistent_store import get_checkpointer

def run_quantum_trial_fixed():
    # 1. Initialize with a checkpointer to allow state injection
    checkpointer = get_checkpointer()
    app = create_scua_graph() # Ensure compile(checkpointer=checkpointer) is in graph.py
    
    config = {"configurable": {"thread_id": "quantum_override_test"}}

    # 2. MANUALLY INJECT the "Quantum" State as if the Perception Node just finished
    # This bypasses the logic that defaults to "Bayesian PD"
    app.update_state(config, {
        "game_regime": "Quantum PD",
        "metrics": {"coherence_loss": 0.05, "entropy": 0.1, "volatility": 0.1},
        "beliefs": {"Noisy": 0.01, "Deceptive": 0.01},
        "history": []
    }, as_node="perception") # Tells LangGraph: "Act as if 'perception' just returned this"

    print("🚀 Starting FORCED Quantum PD Trial...")

    # 3. Invoke with None to resume from the Bayesian/Logic layers
    for event in app.stream(None, config):
        for node, data in event.items():
            if 'final_action' in data:
                print(f"🤖 SCUA Action: {data['final_action']}")

if __name__ == "__main__":
    run_quantum_trial_fixed()
