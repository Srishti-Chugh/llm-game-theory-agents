"""
eval_metagame.py
================
Layer 3 — Meta-Game Controller Evaluation
-----------------------------------------
Simulates game sequences where the opponent strategy *shifts* mid-game,
then measures how quickly the Bayesian Engine + Perception layer detects
and reacts to the non-stationarity.

Metrics
-------
1. Detection Latency
   — Rounds elapsed from the strategy shift until Deceptive belief > 0.5.

2. Non-Stationarity Score
   — Volatility reading at the moment of shift detection (high = correctly
     capturing instability).

3. Strategy Reset Trigger
   — Rounds from the shift until the agent consistently changes its action
     (e.g., switches from Cooperate to Defect for 2+ consecutive rounds).

Run:
    cd c:\\Users\\srish\\llm-game-theory-agents
    python -m evaluation.eval_metagame
"""

from src.nodes.bayesian    import bayesian_engine_node
from src.nodes.perception  import perception_node
from evaluation.metrics_utils import print_section

# ── Configuration ────────────────────────────────────────────────────────────
COOPERATIVE_ROUNDS   = 5   # rounds before the shift
POST_SHIFT_ROUNDS    = 10  # rounds after the shift
DETECTION_THRESHOLD  = 0.5  # belief prob to count as "detected"
RESET_CONSECUTIVE    = 2    # runs of Defect needed to count as a strategy reset

# My simulated action heuristic:
# COOPERATE if dominant belief ≠ Deceptive, else DEFECT
def _heuristic_action(beliefs: dict) -> str:
    dom = max(beliefs, key=beliefs.get)
    p_d = beliefs.get("Deceptive", 0.0)
    if dom == "Deceptive" and p_d >= DETECTION_THRESHOLD:
        return "Defect"
    return "Cooperate"


def _run_phase(history: list, opp_moves_schedule: list, start_rnd: int,
               label: str, shift_rnd: int = None):
    """
    Run rounds, returning updated history + per-round records.
    """
    records = []
    for i, opp_move in enumerate(opp_moves_schedule):
        rnd = start_rnd + i
        state = {"history": history, "messages": []}

        # Perception
        perc   = perception_node(state)
        metrics = perc["metrics"]

        # Bayesian
        state_with_metrics = {**state, "metrics": metrics, "game_regime": "Bayesian PD"}
        bayes  = bayesian_engine_node(state_with_metrics)
        beliefs = bayes["beliefs"]

        # Heuristic action (stand-in for LLM for offline eval)
        action = _heuristic_action(beliefs)

        records.append({
            "round":    rnd,
            "opp":      opp_move,
            "action":   action,
            "beliefs":  beliefs,
            "metrics":  metrics,
        })

        history.append({"opponent_move": opp_move, "my_move": action})

    return history, records


def run_scenario(scenario_name: str, shift_round: int = 6):
    """
    Cooperative phase → shift at `shift_round` → defect phase.
    Returns detection_latency, volatility_at_shift, reset_trigger_round.
    """
    history = []

    # Phase 1: Cooperative (pre-shift)
    coop_schedule = ["Cooperate"] * COOPERATIVE_ROUNDS
    history, pre_records = _run_phase(
        history, coop_schedule,
        start_rnd=1, label="Pre-Shift"
    )

    # Phase 2: Defection (post-shift) — opponent permanently defects
    defect_schedule = ["Defect"] * POST_SHIFT_ROUNDS
    history, post_records = _run_phase(
        history, defect_schedule,
        start_rnd=shift_round, label="Post-Shift", shift_rnd=shift_round
    )

    # Combine for display
    all_records = pre_records + post_records

    print_section(f"Meta-Game Controller — {scenario_name}  (shift@Rnd {shift_round})")
    print(f"  {'Rnd':<5} {'Opp':<12} {'Action':<12} "
          f"{'Dec':>6} {'Noi':>6} {'Vol':>6}  {'Shift':>6}")
    print("  " + "-" * 65)

    for rec in all_records:
        dec  = rec["beliefs"].get("Deceptive", 0.0)
        noi  = rec["beliefs"].get("Noisy", 0.0)
        vol  = rec["metrics"].get("volatility", 0.0)
        is_s = "← SHIFT" if rec["round"] == shift_round else ""
        print(f"  R{rec['round']:02d}  | {rec['opp']:<12} | {rec['action']:<12} | "
              f"{dec:>6.2f} {noi:>6.2f} {vol:>6.2f}  {is_s}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    detection_round    = None
    volatility_at_det  = None
    reset_trigger_round = None
    consec_defect      = 0

    for rec in post_records:
        dec = rec["beliefs"].get("Deceptive", 0.0)
        if detection_round is None and dec >= DETECTION_THRESHOLD:
            detection_round   = rec["round"]
            volatility_at_det  = rec["metrics"].get("volatility", 0.0)

        if rec["action"] == "Defect":
            consec_defect += 1
            if consec_defect >= RESET_CONSECUTIVE and reset_trigger_round is None:
                reset_trigger_round = rec["round"]
        else:
            consec_defect = 0

    latency = (detection_round - shift_round) if detection_round else None

    print(f"\n  ┌─ Results: {scenario_name}")
    print(f"  │  Detection Round           : "
          f"{detection_round if detection_round else 'Never'}")
    print(f"  │  Detection Latency (rounds): "
          f"{latency if latency is not None else 'N/A'}")
    print(f"  │  Volatility at Detection   : "
          f"{volatility_at_det:.3f}" if volatility_at_det else "  │  Volatility at Detection   : N/A")
    print(f"  │  Strategy Reset Round      : "
          f"{reset_trigger_round if reset_trigger_round else 'Never'}")

    if detection_round and reset_trigger_round:
        reset_latency = reset_trigger_round - shift_round
        print(f"  │  Rounds to Strategy Reset  : {reset_latency}")
    print(f"  └─────────────────────────────────────────")

    return {
        "scenario":           scenario_name,
        "shift_round":        shift_round,
        "detection_round":    detection_round,
        "detection_latency":  latency,
        "volatility_at_det":  volatility_at_det,
        "reset_trigger_round": reset_trigger_round,
    }


def run_metagame_eval():
    print("\n" + "█" * 60)
    print("  LAYER 3: META-GAME CONTROLLER — EVALUATION")
    print("█" * 60)

    scenarios = [
        ("Early Shift (Round 3)",  3),
        ("Mid Shift (Round 6)",    6),
        ("Late Shift (Round 10)", 10),
    ]

    all_results = []
    for name, shift_rnd in scenarios:
        r = run_scenario(name, shift_round=shift_rnd)
        all_results.append(r)

    print_section("META-GAME CONTROLLER — AGGREGATE RESULTS")
    hdr = (f"  {'Scenario':<24} {'ShiftRnd':>9} {'DetRnd':>8} "
           f"{'Latency':>9} {'VolAtDet':>10} {'ResetRnd':>10}")
    print(hdr)
    print("  " + "-" * 74)
    for r in all_results:
        det  = str(r["detection_round"])   if r["detection_round"]    else "Never"
        lat  = str(r["detection_latency"]) if r["detection_latency"] is not None else "N/A"
        vol  = f"{r['volatility_at_det']:.3f}" if r["volatility_at_det"] else "N/A"
        rst  = str(r["reset_trigger_round"]) if r["reset_trigger_round"] else "Never"
        print(f"  {r['scenario']:<24} {r['shift_round']:>9} {det:>8} "
              f"{lat:>9} {vol:>10} {rst:>10}")
    print()


if __name__ == "__main__":
    run_metagame_eval()
