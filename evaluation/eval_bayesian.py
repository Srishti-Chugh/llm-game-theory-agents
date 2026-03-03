"""
eval_bayesian.py
================
Layer 1 — Bayesian Belief Engine Evaluation
-------------------------------------------
Tests the `bayesian_engine_node` in isolation (no LLM calls).

Metrics
-------
1. Prior Accuracy
   — % of rounds where the highest-probability belief type matches
     the actual opponent type.

2. Trust Traps / Betrayal Detection
   — After betrayal starts, how many rounds pass before
     Deceptive belief crosses the detection threshold (0.5)?

3. Correct Type-Identification Round
   — The first round where dominant belief == true type AND prob >= 0.5.

Run:
    cd c:\\Users\\srish\\llm-game-theory-agents
    python -m evaluation.eval_bayesian
"""

from src.nodes.bayesian import bayesian_engine_node
from evaluation.opponent_generators import (
    neutral, deceptive, noisy, irrational
)
from evaluation.metrics_utils import print_section, print_metric

# ── Constants ────────────────────────────────────────────────────────────────
ROUNDS             = 15
DETECTION_THRESHOLD = 0.5   # belief probability considered "flagged"

SCENARIOS = [
    {
        "name":          "Neutral",
        "opp_fn":        neutral,
        "opp_kwargs":    {},
        "true_type":     "Neutral",     # no threat type — check low Dec & Noi
        "threat_belief": None,
    },
    {
        "name":          "Deceptive",
        "opp_fn":        deceptive,
        "opp_kwargs":    {"grace_rounds": 3},
        "true_type":     "Deceptive",
        "threat_belief": "Deceptive",
    },
    {
        "name":          "Noisy",
        "opp_fn":        noisy,
        "opp_kwargs":    {"noise_rate": 0.20, "seed": 42},
        "true_type":     "Noisy",
        "threat_belief": "Noisy",
    },
    {
        "name":          "Irrational",
        "opp_fn":        irrational,
        "opp_kwargs":    {"seed": 7},
        "true_type":     "Irrational",
        "threat_belief": "Irrational",
    },
]


def _dominant(beliefs: dict) -> str:
    return max(beliefs, key=beliefs.get)


def evaluate_scenario(scenario: dict) -> dict:
    """Run one scenario and return metric results."""
    name          = scenario["name"]
    opp_fn        = scenario["opp_fn"]
    opp_kwargs    = scenario["opp_kwargs"]
    true_type     = scenario["true_type"]
    threat_belief = scenario["threat_belief"]

    history             = []
    correct_rounds      = 0
    detection_round     = None          # first round belief crosses threshold
    id_round            = None          # first round dominant == true_type
    betrayal_start_rnd  = None

    print_section(f"Bayesian Engine — {name} Opponent ({ROUNDS} rounds)")
    print(f"  {'Rnd':<5} {'OppMove':<12} {'Dec':>6} {'Noi':>6} "
          f"{'Dom':<16} {'Correct':>8}")
    print("  " + "-" * 58)

    for rnd in range(1, ROUNDS + 1):
        opp_move = opp_fn(rnd, history, **opp_kwargs)

        # Build minimal state for the bayesian node
        state = {"history": history, "messages": []}
        result = bayesian_engine_node(state)
        beliefs = result["beliefs"]

        dom     = _dominant(beliefs)
        dec     = beliefs.get("Deceptive", 0.0)
        noi     = beliefs.get("Noisy", 0.0)

        # ── Correct classification logic ──────────────────────────────────────
        # For Neutral: correct = both Dec and Noi are low
        if true_type == "Neutral":
            correct = dec < DETECTION_THRESHOLD and noi < DETECTION_THRESHOLD
        else:
            correct = dom == true_type and beliefs.get(true_type, 0.0) >= DETECTION_THRESHOLD

        if correct:
            correct_rounds += 1
            if id_round is None:
                id_round = rnd

        # ── Track betrayal latency (Deceptive scenario only) ─────────────────
        if threat_belief == "Deceptive" and betrayal_start_rnd is None:
            if opp_move == "Defect" and rnd > 3:
                betrayal_start_rnd = rnd
        if threat_belief and detection_round is None:
            if beliefs.get(threat_belief, 0.0) >= DETECTION_THRESHOLD:
                detection_round = rnd

        print(f"  R{rnd:02d}  | {opp_move:<12} | {dec:>6.2f} {noi:>6.2f} "
              f" {dom:<16} {'✅' if correct else '❌':>8}")

        # Advance history
        history.append({"opponent_move": opp_move, "my_move": "Cooperate"})

    # ── Metrics ───────────────────────────────────────────────────────────────
    prior_accuracy = correct_rounds / ROUNDS * 100

    if betrayal_start_rnd and detection_round:
        betrayal_latency = detection_round - betrayal_start_rnd
    elif betrayal_start_rnd and not detection_round:
        betrayal_latency = ROUNDS - betrayal_start_rnd + 1  # never detected
    else:
        betrayal_latency = None

    print(f"\n  ┌─ Results: {name}")
    print(f"  │  Prior Accuracy              : {prior_accuracy:.1f}%")
    print(f"  │  Type Identified (first round): {id_round if id_round else 'Never'}")
    if betrayal_latency is not None:
        print(f"  │  Betrayal Detection Latency  : {betrayal_latency} rounds")
    print(f"  └─────────────────────────────────────────")

    return {
        "scenario":           name,
        "prior_accuracy":     prior_accuracy,
        "id_round":           id_round,
        "detection_round":    detection_round,
        "betrayal_latency":   betrayal_latency,
    }


def run_bayesian_eval():
    print("\n" + "█" * 60)
    print("  LAYER 1: BAYESIAN BELIEF ENGINE — EVALUATION")
    print("█" * 60)

    all_results = []
    for scenario in SCENARIOS:
        r = evaluate_scenario(scenario)
        all_results.append(r)

    # ── Summary table ─────────────────────────────────────────────────────────
    print_section("BAYESIAN ENGINE — AGGREGATE RESULTS")
    print(f"  {'Scenario':<14} {'PriorAcc':>10} {'IdRound':>9} "
          f"{'DetRound':>10} {'BetrayalLat':>13}")
    print("  " + "-" * 62)
    for r in all_results:
        id_r  = str(r["id_round"])  if r["id_round"]  else "Never"
        det_r = str(r["detection_round"]) if r["detection_round"] else "Never"
        bl    = str(r["betrayal_latency"]) if r["betrayal_latency"] else "N/A"
        print(f"  {r['scenario']:<14} {r['prior_accuracy']:>9.1f}% "
              f"{id_r:>9} {det_r:>10} {bl:>13}")
    print()


if __name__ == "__main__":
    run_bayesian_eval()
