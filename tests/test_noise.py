from src.graph import create_scua_graph

def run_noise_experiment():
    app = create_scua_graph()
    config = {"configurable": {"thread_id": "noise_test_001"}}
    
    # Start with a clean history of cooperation
    state = {
        "history": [{"opponent_move": "Cooperate", "my_move": "Cooperate"}] * 4,
        "beliefs": {"Deceptive": 0.05, "Noisy": 0.05, "Non-Stationary": 0.0, "Irrational": 0.0},
        "messages": []
    }

    print("🚀 Starting Noise Experiment: Opponent defects ONCE at Round 5, then returns to Cooperate.")

    for round_num in range(5, 9):
        print(f"\n--- ROUND {round_num} ---")
        
        # 1. Simulate the "Noise" at Round 5
        opponent_move = "Defect" if round_num == 5 else "Cooperate"
        
        # 2. Run the SCUA Agent
        output = app.invoke(state, config)
        
        beliefs = output.get("beliefs", {})
        action = output.get("final_action", "")

        print(f"🧠 Beliefs -> Noisy: {beliefs.get('Noisy'):.2f} | Deceptive: {beliefs.get('Deceptive'):.2f}")
        print(f"🤖 SCUA Action: {action}")

        # 3. Update state for next round
        state["history"].append({"opponent_move": opponent_move, "my_move": action})

if __name__ == "__main__":
    run_noise_experiment()
