"""
test_quantum_pd.py
==================
Test 3 — Quantum Prisoner's Dilemma vs 4 Opponent Types
--------------------------------------------------------
Forces game_regime = "Quantum PD" by injecting state after the perception
node (using update_state), then streams the remaining pipeline.

Opponents:
  • Neutral      — always cooperates
  • Deceptive    — cooperates 3 rounds then defects
  • Noisy        — 20% random defection
  • Irrational   — 50/50 random

Key quantum metrics tracked per round:
  • coherence_loss     — starts low, increases as opponent becomes adversarial
  • Q-strategy rate    — how often the LLM recommends the Q/Quantum path
  • Decoherence events — rounds where coherence_loss > 0.7 (classical collapse)

Run:
    cd c:\\Users\\srish\\llm-game-theory-agents
    python -m tests.test_quantum_pd
"""

import uuid
import math
from src.graph import create_scua_graph
from evaluation.opponent_generators import neutral, deceptive, noisy, irrational
from evaluation.metrics_utils import parse_action, payoff, print_section

# ── Configuration ────────────────────────────────────────────────────────────
NUM_ROUNDS = 10

OPPONENT_PROFILES = [
    ("Neutral",    neutral,    {}),
    ("Deceptive",  deceptive,  {"grace_rounds": 3}),
    ("Noisy",      noisy,      {"noise_rate": 0.20, "seed": 7}),
    ("Irrational", irrational, {"seed": 13}),
]

# Coherence degrades as defections accumulate
def _coherence_loss(history):
    """
    Computes coherence_loss in [0, 1] based on opponent defection frequency.
    A purely cooperative history = 0 coherence_loss.
    All defections = 1.0 coherence_loss.
    """
    if not history:
        return 0.05
    defections = sum(1 for h in history if h.get("opponent_move") == "Defect")
    return min(1.0, defections / len(history))


def _q_strategy_mentioned(text: str) -> bool:
    """True if the LLM response mentions Q-Strategy or quantum language."""
    keywords = ["q-strategy", "quantum", "entanglement", "superposition",
                "quantum nash", "q strategy"]
    return any(kw in text.lower() for kw in keywords)


def _fresh_state():
    return {
        "messages": [],
        "history":  [],
        "beliefs":  {"Deceptive": 0.0, "Noisy": 0.0,
                     "Non-Stationary": 0.0, "Irrational": 0.0},
        "logic_recommendation": "Initialising…",
    }


def run_quantum_episode(profile_name: str, opp_fn, opp_kwargs: dict) -> dict:
    """Run one 10-round Quantum PD episode; returns summary statistics."""
    thread_id = f"quantum_{profile_name.lower()}_{uuid.uuid4().hex[:8]}"
    app    = create_scua_graph()
    config = {"configurable": {"thread_id": thread_id}}
    state  = _fresh_state()

    print_section(f"QUANTUM PD — {profile_name} Opponent  (thread={thread_id})")
    print(f"  {'Rnd':<5} {'Opp':<12} {'Agent':<12} "
          f"{'CoherLoss':>10} {'Q-Strategy':>12} {'Dec':>6}")
    print("  " + "-" * 62)

    coop_rounds       = 0
    total_payoff      = 0
    q_strategy_rounds = 0
    decoherence_events = 0

    for rnd in range(1, NUM_ROUNDS + 1):
        opp_move = opp_fn(rnd, state["history"], **opp_kwargs)
        coh_loss = _coherence_loss(state["history"])

        # ── Inject Quantum PD state after the perception node ─────────────────
        app.update_state(config, {
            "game_regime": "Quantum PD",
            "metrics": {
                "coherence_loss": coh_loss,
                "entropy":        0.5 + coh_loss * 0.5,
                "volatility":     coh_loss * 0.8,
                "nash_dev":       coh_loss * 0.3,
            },
            "beliefs": state["beliefs"],
            "history": state["history"],
        }, as_node="perception")

        # ── Stream from Bayesian node onwards (skip perception) ───────────────
        raw_action  = ""
        agent_beliefs = state["beliefs"]

        for event in app.stream(None, config):
            for node_name, data in event.items():
                if "beliefs" in data:
                    agent_beliefs = data["beliefs"]
                if "final_action" in data:
                    raw_action = data.get("final_action", "")

        agent_action = parse_action(raw_action) or "Cooperate"
        q_mentioned  = _q_strategy_mentioned(raw_action)

        my_pf, _ = payoff(agent_action, opp_move)
        total_payoff += my_pf
        if agent_action == "Cooperate":
            coop_rounds += 1
        if q_mentioned:
            q_strategy_rounds += 1
        if coh_loss > 0.7:
            decoherence_events += 1

        dec = agent_beliefs.get("Deceptive", 0.0)
        print(f"  R{rnd:02d}  | {opp_move:<12} | {agent_action:<12} | "
              f"{coh_loss:>10.3f} {'✅ Yes' if q_mentioned else '❌ No':>12} "
              f"{dec:>6.2f}")

        state["history"].append({"opponent_move": opp_move, "my_move": agent_action})
        state["beliefs"]  = agent_beliefs
        state["messages"] = []

    # ── Episode summary ───────────────────────────────────────────────────────
    coop_rate     = coop_rounds / NUM_ROUNDS * 100
    avg_payoff    = total_payoff / NUM_ROUNDS
    q_rate        = q_strategy_rounds / NUM_ROUNDS * 100
    coh_final     = _coherence_loss(state["history"])

    print(f"\n  ┌─ Quantum Episode Summary: {profile_name}")
    print(f"  │  Cooperation Rate    : {coop_rate:.1f}%")
    print(f"  │  Avg Payoff/Round    : {avg_payoff:.2f}")
    print(f"  │  Q-Strategy Rate     : {q_rate:.1f}%")
    print(f"  │  Final Coherence Loss: {coh_final:.3f}")
    print(f"  │  Decoherence Events  : {decoherence_events}")
    print(f"  └─ Thread: {thread_id}")

    return {
        "profile":             profile_name,
        "coop_rate":           coop_rate,
        "avg_payoff":          avg_payoff,
        "q_rate":              q_rate,
        "final_coherence_loss": coh_final,
        "decoherence_events":  decoherence_events,
    }


def run_quantum_pd():
    print("\n" + "█" * 60)
    print("  QUANTUM PD — Full Opponent Suite")
    print("█" * 60)

    all_results = []
    for name, fn, kwargs in OPPONENT_PROFILES:
        result = run_quantum_episode(name, fn, kwargs)
        all_results.append(result)

    # ── Cross-profile report ──────────────────────────────────────────────────
    print_section("QUANTUM PD — CROSS-PROFILE REPORT")
    print(f"  {'Profile':<14} {'CoopRate':>10} {'QRate':>8} "
          f"{'FinalCoh':>10} {'DecoheEvts':>12}")
    print("  " + "-" * 58)
    for r in all_results:
        print(f"  {r['profile']:<14} {r['coop_rate']:>9.1f}% "
              f"{r['q_rate']:>7.1f}% "
              f"{r['final_coherence_loss']:>10.3f} "
              f"{r['decoherence_events']:>12}")
    print()


if __name__ == "__main__":
    run_quantum_pd()
