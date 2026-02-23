import time
from src.graph import create_scua_graph

def run_betrayal_experiment():
    app = create_scua_graph()
    config = {"configurable": {"thread_id": "betrayal_test_001"}}
    
    # Starting state: High trust, mutual cooperation history
    state = {
        "history": [
            {"opponent_move": "Cooperate", "my_move": "Cooperate"},
            {"opponent_move": "Cooperate", "my_move": "Cooperate"},
            {"opponent_move": "Cooperate", "my_move": "Cooperate"}
        ],
        "beliefs": {"Deceptive": 0.05, "Noisy": 0.05, "Non-Stationary": 0.0, "Irrational": 0.0},
        "messages": []
    }

    print("🚀 Starting Betrayal Experiment: Opponent switches to constant DEFECT at Round 4.")

    # Simulate 5 rounds of opponent BETRAYAL
    for round_num in range(4, 9):
        print(f"\n--- ROUND {round_num} ---")
        
        # 1. Run the SCUA Agent
        output = app.invoke(state, config)
        
        # 2. Extract metrics and beliefs from the engine
        metrics = output.get("metrics", {})
        beliefs = output.get("beliefs", {})
        action = output.get("final_action", "No Action")

        print(f"📈 Metrics -> Entropy: {metrics.get('entropy'):.2f} | Volatility: {metrics.get('volatility'):.2f}")
        print(f"🧠 Beliefs -> Deceptive Prob: {beliefs.get('Deceptive'):.2f}")
        print(f"🤖 SCUA Action: {action}")

        # 3. Simulate Opponent Defection (The "Deceptive" behavior)
        opponent_move = "Defect"
        
        # 4. Update state for next round feedback loop
        state["history"].append({"opponent_move": opponent_move, "my_move": action})
        
        # Logic Check: If Deceptive prob > 0.5, the engine is working!
        if beliefs.get("Deceptive", 0) > 0.5:
            print("✅ SUCCESS: Bayesian Engine flagged Deception and adjusted strategy.")

if __name__ == "__main__":
    run_betrayal_experiment()
