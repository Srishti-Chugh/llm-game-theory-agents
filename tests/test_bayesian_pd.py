"""
test_bayesian_pd.py
===================
Test 2 — Bayesian Prisoner's Dilemma vs 4 Opponent Types
---------------------------------------------------------
Runs the SCUA agent through 10-round Bayesian PD episodes against:
  • Neutral      — always cooperates
  • Deceptive    — cooperates 3 rounds then permanently defects
  • Noisy        — ~20% random defection
  • Irrational   — 50/50 random each round

Each episode uses a fresh LangGraph thread so state doesn't bleed across runs.

Run:
    cd c:\\Users\\srish\\llm-game-theory-agents
    python -m tests.test_bayesian_pd
"""

import uuid
from src.graph import create_scua_graph
from evaluation.opponent_generators import neutral, deceptive, noisy, irrational
from evaluation.metrics_utils import (
    parse_action, payoff, print_section, print_round_row
)

# ── Configuration ────────────────────────────────────────────────────────────
NUM_ROUNDS  = 10
GAME_REGIME = "Bayesian PD"

OPPONENT_PROFILES = [
    ("Neutral",    neutral,    {}),
    ("Deceptive",  deceptive,  {"grace_rounds": 3}),
    ("Noisy",      noisy,      {"noise_rate": 0.20, "seed": 42}),
    ("Irrational", irrational, {"seed": 99}),
]

# Belief-type → expected dominant belief label
EXPECTED_TYPE_MAP = {
    "Neutral":    None,           # no threat signal expected
    "Deceptive":  "Deceptive",
    "Noisy":      "Noisy",
    "Irrational": "Irrational",
}


def _fresh_state():
    return {
        "messages": [],
        "history":  [],
        "beliefs":  {"Deceptive": 0.0, "Noisy": 0.0,
                     "Non-Stationary": 0.0, "Irrational": 0.0},
        "logic_recommendation": "Initialising…",
    }


def run_episode(profile_name: str, opp_fn, opp_kwargs: dict) -> dict:
    """Run one 10-round episode and return summary statistics."""
    thread_id = f"bayesian_{profile_name.lower()}_{uuid.uuid4().hex[:8]}"
    app    = create_scua_graph()
    config = {"configurable": {"thread_id": thread_id}}
    state  = _fresh_state()

    print_section(f"BAYESIAN PD — {profile_name} Opponent  (thread={thread_id})")
    print(f"  {'Rnd':<5} {'Opp':<12} {'Agent':<12} "
          f"{'Dec':>5} {'Noi':>5} {'Irr':>5} {'MyPf':>6}")
    print("  " + "-" * 58)

    coop_rounds        = 0
    total_payoff       = 0
    correct_type_round = None        # first round correct type flagged
    detection_round    = None        # first round dominant belief matches threat
    expected_dom       = EXPECTED_TYPE_MAP[profile_name]

    for rnd in range(1, NUM_ROUNDS + 1):
        opp_move = opp_fn(rnd, state["history"], **opp_kwargs)

        output = app.invoke(state, config)
        metrics      = output.get("metrics", {})
        beliefs      = output.get("beliefs", {})
        raw_action   = output.get("final_action", "")
        agent_action = parse_action(raw_action) or "Cooperate"

        my_pf, _ = payoff(agent_action, opp_move)
        total_payoff += my_pf
        if agent_action == "Cooperate":
            coop_rounds += 1

        dec = beliefs.get("Deceptive", 0.0)
        noi = beliefs.get("Noisy", 0.0)
        irr = beliefs.get("Irrational", 0.0)
        print(f"  R{rnd:02d}  | {opp_move:<12} | {agent_action:<12} | "
              f"{dec:>5.2f} {noi:>5.2f} {irr:>5.2f}  {my_pf:>4}")

        # Track type identification
        if expected_dom and correct_type_round is None:
            dom = max(beliefs, key=beliefs.get)
            if dom == expected_dom and beliefs[expected_dom] >= 0.5:
                correct_type_round = rnd
                detection_round    = rnd

        state["history"].append({"opponent_move": opp_move, "my_move": agent_action})
        state["messages"] = []

    # ── Episode summary ───────────────────────────────────────────────────────
    coop_rate  = coop_rounds / NUM_ROUNDS * 100
    avg_payoff = total_payoff / NUM_ROUNDS

    print(f"\n  ┌─ Episode Summary: {profile_name}")
    print(f"  │  Cooperation Rate  : {coop_rate:.1f}%")
    print(f"  │  Avg Payoff/Round  : {avg_payoff:.2f}")
    if expected_dom:
        detected = f"Round {correct_type_round}" if correct_type_round else "Not detected"
        print(f"  │  {expected_dom} type identified: {detected}")
    print(f"  └─ Thread: {thread_id}")

    return {
        "profile":             profile_name,
        "coop_rate":           coop_rate,
        "avg_payoff":          avg_payoff,
        "correct_type_round":  correct_type_round,
    }


def run_bayesian_pd():
    print("\n" + "█" * 60)
    print("  BAYESIAN PD — Full Opponent Suite")
    print("█" * 60)

    all_results = []
    for name, fn, kwargs in OPPONENT_PROFILES:
        result = run_episode(name, fn, kwargs)
        all_results.append(result)

    # ── Cross-profile report ──────────────────────────────────────────────────
    print_section("BAYESIAN PD — CROSS-PROFILE REPORT")
    header = f"  {'Profile':<14} {'CoopRate':>10} {'AvgPayoff':>10} {'TypeDetected':>14}"
    print(header)
    print("  " + "-" * 52)
    for r in all_results:
        detected = (f"Rnd {r['correct_type_round']}"
                    if r["correct_type_round"] else "N/A")
        print(f"  {r['profile']:<14} {r['coop_rate']:>9.1f}% "
              f"{r['avg_payoff']:>10.2f} {detected:>14}")
    print()


if __name__ == "__main__":
    run_bayesian_pd()
