"""
test_baseline_pd.py
===================
Test 1 — Baseline Prisoner's Dilemma vs a Neutral Opponent
-----------------------------------------------------------
The neutral opponent always cooperates.  This establishes the agent's
default cooperate/defect tendency and validates the full pipeline (Perception
→ Bayesian → Logic → LLM Executive) in the simplest possible setting.

Run:
    cd c:\\Users\\srish\\llm-game-theory-agents
    python -m tests.test_baseline_pd
"""

import uuid
from src.graph import create_scua_graph
from evaluation.opponent_generators import neutral
from evaluation.metrics_utils import parse_action, payoff, print_section, print_round_row

# ── Configuration ────────────────────────────────────────────────────────────
NUM_ROUNDS   = 10
GAME_REGIME  = "Bayesian PD"
THREAD_ID    = f"baseline_pd_{uuid.uuid4().hex[:8]}"

# ── PD payoff constants (for scoring) ────────────────────────────────────────
NASH_EQUILIBRIUM_ACTION = "Defect"   # Classical Nash: both defect
OPTIMAL_SOCIAL_ACTION   = "Cooperate"  # Pareto-optimal: both cooperate


def run_baseline_pd():
    app    = create_scua_graph()
    config = {"configurable": {"thread_id": THREAD_ID}}

    # Seed state — begin with zero history (fresh game)
    state = {
        "messages":  [],
        "history":   [],
        "beliefs":   {"Deceptive": 0.0, "Noisy": 0.0,
                      "Non-Stationary": 0.0, "Irrational": 0.0},
        "logic_recommendation": "Initialising…",
    }

    print_section(f"BASELINE PD — Neutral Opponent  (thread={THREAD_ID})")
    print(f"  {'Rnd':<5} {'Opp Move':<13} {'Agent Action':<14} "
          f"{'Dec':>5} {'Noi':>5} {'Ent':>6} {'Vol':>6} {'MyPayoff':>9}")
    print("  " + "-" * 70)

    cooperative_rounds = 0
    total_payoff       = 0
    round_results      = []

    for rnd in range(1, NUM_ROUNDS + 1):
        # 1. Opponent move (neutral → always cooperate)
        opp_move = neutral(rnd, state["history"])

        # 2. Run SCUA pipeline
        output = app.invoke(state, config)

        metrics = output.get("metrics", {})
        beliefs = output.get("beliefs", {})
        raw_action = output.get("final_action", "")
        agent_action = parse_action(raw_action) or "Cooperate"  # default cooperate

        # 3. Compute payoff
        my_pf, _ = payoff(agent_action, opp_move)
        total_payoff += my_pf
        if agent_action == "Cooperate":
            cooperative_rounds += 1

        # 4. Log
        dec = beliefs.get("Deceptive", 0.0)
        noi = beliefs.get("Noisy", 0.0)
        ent = metrics.get("entropy", 0.0)
        vol = metrics.get("volatility", 0.0)
        print(f"  R{rnd:02d}  | {opp_move:<13} | {agent_action:<14} | "
              f"{dec:>5.2f} {noi:>5.2f} {ent:>6.2f} {vol:>6.2f} {my_pf:>9}")

        round_results.append({
            "round": rnd, "opp_move": opp_move,
            "agent_action": agent_action, "payoff": my_pf,
            "beliefs": beliefs, "metrics": metrics,
        })

        # 5. Advance history
        state["history"].append({"opponent_move": opp_move, "my_move": agent_action})
        state["messages"] = []   # reset messages so graph doesn't duplicate

    # ── Summary ──────────────────────────────────────────────────────────────
    coop_rate      = cooperative_rounds / NUM_ROUNDS * 100
    avg_payoff     = total_payoff / NUM_ROUNDS
    pareto_optimal = coop_rate >= 80  # ≥80% cooperation = broadly Pareto-optimal

    print_section("BASELINE SUMMARY")
    print(f"  {'Cooperation Rate':<40} {coop_rate:.1f}%")
    print(f"  {'Average Payoff per Round':<40} {avg_payoff:.2f}")
    print(f"  {'Total Cumulative Payoff':<40} {total_payoff}")
    print(f"  {'Reached Pareto-Optimal Zone (≥80% coop)':<40} {'✅ YES' if pareto_optimal else '❌ NO'}")
    print(f"  {'Nash Equilibrium Avoidance':<40} {'✅ PASS' if coop_rate > 50 else '❌ FAIL'}")
    print()


if __name__ == "__main__":
    run_baseline_pd()
