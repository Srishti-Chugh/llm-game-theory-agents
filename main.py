from src.graph import create_scua_graph
from src.utils.persistent_store import get_checkpointer

# 1. Initialize Graph with Persistence
checkpointer = get_checkpointer()
app = create_scua_graph() # Note: Ensure you pass checkpointer in graph.py compile()

# 2. Define a Test Case (e.g., Baseline PD) — executive node uses ChatNVIDIA (default: Llama 3.1 8B; see SCUA_NVIDIA_MODEL)
config = {"configurable": {"thread_id": "experiment_001"}}
initial_state = {
    "messages": [],
    "history": [{"opponent_move": "Defect", "my_move": "Cooperate"}],
    # Force combinatorial regime (else perception uses Bayesian PD when len(history) < 10).
    "game_regime": "Combinatorial",
    "beliefs": {"Deceptive": 0.0, "Noisy": 0.0, "Non-Stationary": 0.0, "Irrational": 0.0},
    "logic_recommendation": "Waiting for Logic Layer..."
}

# 3. Stream the execution
for event in app.stream(initial_state, config):
    for node, data in event.items():
        print(f"\n--- NODE: {node} ---")
        if 'metrics' in data: print(f"Metrics: {data['metrics']}")
        if 'final_action' in data: print(f"ACTION: {data['final_action']}")
