"""
eval_kalman.py
==============
Layer 4 — Kalman Filter (Noise Suppression) Evaluation
-------------------------------------------------------
The SCUA codebase implements noise filtering inside `bayesian_engine_node`:
  - One defect in a long cooperation streak → high Noisy probability (0.80)
  - Two+ defects → high Deceptive probability (0.85)

This script evaluates how well that logic suppresses noise and avoids
over-retaliation, which maps to what a Kalman filter module would do.

Metrics
-------
1. Noise Suppression Rate
   — After a single-defect "noise event", % of subsequent rounds where
     Noisy belief stays dominant (i.e., the filter correctly absorbs noise).

2. Over-Sensitivity Rate
   — % of rounds where action heuristic = Defect despite Noisy being dominant
     (agent retaliates unnecessarily to probable noise).

3. % Retaliation for Single-Noise Events
   — Of all single-turn noise events, what fraction result in an agent Defect
     in the *immediate next round*?

Run:
    cd c:\\Users\\srish\\llm-game-theory-agents
    python -m evaluation.eval_kalman
"""

from src.nodes.bayesian   import bayesian_engine_node
from src.nodes.perception import perception_node
from evaluation.metrics_utils import print_section, is_retaliation

# ── Configuration ────────────────────────────────────────────────────────────
COOPERATION_PRE   = 6    # rounds of cooperation before the noise event
POST_NOISE_ROUNDS = 6    # rounds after the noise defect to observe recovery
NUM_NOISE_TRIALS  = 5    # repeat the experiment with varied pre-history lengths

NOISY_THRESHOLD   = 0.5  # belief prob considered "noise dominant"


def _heuristic_action(beliefs: dict) -> str:
    """Simple stand-in for the LLM: Defect only if Deceptive is dominant and > 0.5."""
    dom = max(beliefs, key=beliefs.get)
    p_d = beliefs.get("Deceptive", 0.0)
    if dom == "Deceptive" and p_d >= 0.5:
        return "Defect"
    return "Cooperate"


def run_noise_trial(cooperation_rounds: int, trial_num: int) -> dict:
    """
    Runs: cooperation_rounds × Cooperate, then 1 Defect, then POST_NOISE_ROUNDS × Cooperate.
    Returns per-trial statistics.
    """
    history = []

    # Phase 1: Pre-noise cooperation
    for i in range(cooperation_rounds):
        history.append({"opponent_move": "Cooperate", "my_move": "Cooperate"})

    noise_round  = cooperation_rounds + 1
    noise_injection_idx = len(history)  # index where noise happens

    # Phase 2: Single noise defection
    history.append({"opponent_move": "Defect", "my_move": "Cooperate"})

    # Phase 3: Post-noise cooperation (opponent returns to cooperate)
    recovery_records = []
    for i in range(POST_NOISE_ROUNDS):
        rnd = noise_round + 1 + i
        opp  = "Cooperate"  # returns to cooperate

        state = {"history": history, "messages": []}
        perc  = perception_node(state)
        metrics = perc["metrics"]

        state_full = {**state, "metrics": metrics, "game_regime": "Bayesian PD"}
        bayes      = bayesian_engine_node(state_full)
        beliefs    = bayes["beliefs"]
        action     = _heuristic_action(beliefs)

        retaliated = action == "Defect"
        noisy_dom  = (beliefs.get("Noisy", 0.0) >= NOISY_THRESHOLD)

        recovery_records.append({
            "round":       rnd,
            "beliefs":     beliefs,
            "action":      action,
            "retaliated":  retaliated,
            "noisy_dom":   noisy_dom,
        })

        history.append({"opponent_move": opp, "my_move": action})

    # Immediate retaliation = did agent Defect in the very first post-noise round?
    immediate_retaliation = (recovery_records[0]["action"] == "Defect"
                              if recovery_records else False)

    # Noise suppression: rounds in recovery phase where Noisy stays dominant
    suppressed_rounds   = sum(1 for r in recovery_records if r["noisy_dom"])
    noise_suppression   = suppressed_rounds / POST_NOISE_ROUNDS * 100

    # Over-sensitivity: rounds where action = Defect but Noisy was dominant
    over_sensitive_rounds = sum(
        1 for r in recovery_records if r["retaliated"] and r["noisy_dom"]
    )
    over_sensitivity = over_sensitive_rounds / POST_NOISE_ROUNDS * 100

    print(f"\n  Trial {trial_num} (pre-noise coop = {cooperation_rounds} rounds,  "
          f"noise @ R{noise_round})")
    print(f"  {'Rnd':<5} {'Noi':>6} {'Dec':>6} {'Action':<12} "
          f"{'NoisyDom':>10} {'Retaliate':>11}")
    print("  " + "-" * 55)
    for rec in recovery_records:
        noi = rec["beliefs"].get("Noisy", 0.0)
        dec = rec["beliefs"].get("Deceptive", 0.0)
        print(f"  R{rec['round']:02d}  | {noi:>6.2f} {dec:>6.2f} {rec['action']:<12} "
              f"{'Yes' if rec['noisy_dom'] else 'No':>10} "
              f"{'⚠️ Yes' if rec['retaliated'] else 'No':>11}")

    return {
        "trial":                trial_num,
        "coop_pre":             cooperation_rounds,
        "noise_suppression":    noise_suppression,
        "over_sensitivity":     over_sensitivity,
        "immediate_retaliation": immediate_retaliation,
    }


def run_kalman_eval():
    print("\n" + "█" * 60)
    print("  LAYER 4: KALMAN FILTER (NOISE SUPPRESSION) — EVALUATION")
    print("█" * 60)

    print_section("Noise Trials — Single-Turn Defect in Cooperation Stream")

    all_results = []
    pre_lengths = [3, 4, 5, 6, 8]   # vary cooperate-streak length

    for t, coop_len in enumerate(pre_lengths[:NUM_NOISE_TRIALS], start=1):
        r = run_noise_trial(cooperation_rounds=coop_len, trial_num=t)
        all_results.append(r)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    avg_suppression    = sum(r["noise_suppression"] for r in all_results) / len(all_results)
    avg_over_sensitive = sum(r["over_sensitivity"] for r in all_results) / len(all_results)
    retaliation_rate   = (sum(1 for r in all_results if r["immediate_retaliation"])
                          / len(all_results) * 100)

    print_section("KALMAN FILTER — AGGREGATE RESULTS")
    print(f"  {'Noise Suppression Rate (avg across trials)':<46} {avg_suppression:.1f}%")
    print(f"  {'Over-Sensitivity Rate (avg across trials)':<46} {avg_over_sensitive:.1f}%")
    print(f"  {'Retaliation Rate (single-noise events)':<46} {retaliation_rate:.1f}%")
    print()

    print_section("Per-Trial Breakdown")
    print(f"  {'Trial':<7} {'CoopPre':>8} {'Suppression':>13} "
          f"{'OverSens':>10} {'ImmediateRetl':>15}")
    print("  " + "-" * 58)
    for r in all_results:
        print(f"  T{r['trial']:<6} {r['coop_pre']:>8} "
              f"{r['noise_suppression']:>12.1f}% "
              f"{r['over_sensitivity']:>9.1f}% "
              f"{'Yes' if r['immediate_retaliation'] else 'No':>15}")
    print()


if __name__ == "__main__":
    run_kalman_eval()
