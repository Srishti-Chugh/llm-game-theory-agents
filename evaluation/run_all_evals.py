"""
run_all_evals.py
================
Orchestrator — Runs all 4 layer evaluation scripts and prints a consolidated
summary report. All evaluations run offline (no LLM calls required).

Usage:
    cd c:\\Users\\srish\\llm-game-theory-agents
    python -m evaluation.run_all_evals
"""

import time
import traceback

from evaluation.eval_bayesian import run_bayesian_eval
from evaluation.eval_logic    import evaluate_logic
from evaluation.eval_metagame import run_metagame_eval
from evaluation.eval_kalman   import run_kalman_eval


EVAL_SUITE = [
    ("Layer 1 — Bayesian Belief Engine",          run_bayesian_eval),
    ("Layer 2 — Logic Layer (MCTS/Q-Solver)",     evaluate_logic),
    ("Layer 3 — Meta-Game Controller",            run_metagame_eval),
    ("Layer 4 — Kalman Filter (Noise Suppressor)", run_kalman_eval),
]

_BANNER_WIDTH = 60


def _banner(text: str, char: str = "═") -> None:
    print("\n" + char * _BANNER_WIDTH)
    print(f"  {text}")
    print(char * _BANNER_WIDTH)


def run_all():
    _banner("SCUA AGENT — FULL EVALUATION SUITE", "█")
    print(f"\n  Timestamp : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Layers    : {len(EVAL_SUITE)}")
    print(f"  LLM calls : ❌ None (offline evaluation only)")

    results_summary = []   # (name, status, elapsed_s)

    for name, fn in EVAL_SUITE:
        _banner(f"RUNNING: {name}")
        t0 = time.time()
        status = "PASS"
        try:
            fn()
        except Exception as exc:
            status = "FAIL"
            print(f"\n  ⚠️  Error in '{name}':")
            traceback.print_exc()
        elapsed = time.time() - t0
        results_summary.append((name, status, elapsed))

    # ── Master summary Table ──────────────────────────────────────────────────
    _banner("EVALUATION SUITE — MASTER SUMMARY")
    print(f"  {'Layer':<45} {'Status':>8} {'Time':>8}")
    print("  " + "-" * 65)
    all_pass = True
    for name, status, elapsed in results_summary:
        icon = "✅" if status == "PASS" else "❌"
        print(f"  {icon} {name:<43} {status:>8} {elapsed:>6.1f}s")
        if status == "FAIL":
            all_pass = False

    total_time = sum(e for _, _, e in results_summary)
    print("  " + "-" * 65)
    print(f"  {'OVERALL':<45} {'PASS' if all_pass else 'FAIL':>8} {total_time:>6.1f}s")
    _banner("Evaluation Complete", "─")
    print()


if __name__ == "__main__":
    run_all()
