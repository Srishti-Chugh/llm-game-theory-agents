"""
eval_logic.py
=============
Layer 2 — Logic Layer (MCTS / Quantum Solver) Evaluation
---------------------------------------------------------
Tests the `logic_layer_node` in isolation (no LLM) across a parameter sweep
of (game_regime, volatility, coherence_loss) combinations.

Metrics
-------
1. Search Depth Efficiency
   — In Bayesian/Combinatorial regime: does the node correctly recommend
     Conservative-Reciprocity when volatility > 0.6?
     (1 = deep search triggered correctly; 0 = wrong path chosen)

2. Logical Hallucination Rate
   — % of outputs containing neither expected MCTS keywords nor Q-Strategy
     keywords (gibberish / unexpected output).

3. % Adherence to Optimal Paths
   — How often the recommendation matches the mathematically expected path
     given the input parameters?

Run:
    cd c:\\Users\\srish\\llm-game-theory-agents
    python -m evaluation.eval_logic
"""

from src.nodes.logic import logic_layer_node
from evaluation.metrics_utils import print_section

# ── Expected output keywords per scenario ────────────────────────────────────
MCTS_KEYWORDS      = ["search result", "conservative", "cooperation-path",
                       "long-term-cooperation", "cooperation", "reciprocity",
                       "pruning", "stable"]
Q_KEYWORDS         = ["q-strategy", "quantum", "entanglement", "superposition",
                       "q strategy", "quantum nash", "rotation"]
CLASSICAL_KEYWORDS = ["tit-for-tat", "classical nash", "decoherence", "collapse"]

# ── Parameter sweep definition ────────────────────────────────────────────────
# (regime, volatility, coherence_loss, expected_path_keywords)
SWEEP = [
    # Bayesian PD — stable (low volatility) → deep cooperation path
    ("Bayesian PD",  0.2, 0.0, ["long-term-cooperation", "cooperation"]),
    # Bayesian PD — high volatility → conservative / pruning
    ("Bayesian PD",  0.7, 0.0, ["conservative", "pruning"]),
    ("Bayesian PD",  0.9, 0.0, ["conservative", "pruning"]),
    # Quantum PD — high coherence (low loss) → Q-Strategy
    ("Quantum PD",   0.1, 0.05, Q_KEYWORDS),
    # Quantum PD — partial decoherence → mixed quantum strategy
    ("Quantum PD",   0.1, 0.50, ["partial", "mixed", "caution"]),
    # Quantum PD — near-total decoherence → classical Nash
    ("Quantum PD",   0.1, 0.85, CLASSICAL_KEYWORDS),
]


def _keywords_found(text: str, keywords: list) -> bool:
    t = text.lower()
    return any(kw.lower() in t for kw in keywords)


def _is_hallucination(text: str) -> bool:
    t = text.lower()
    all_known = MCTS_KEYWORDS + Q_KEYWORDS + CLASSICAL_KEYWORDS
    return not any(kw.lower() in t for kw in all_known)


def evaluate_logic():
    print("\n" + "█" * 60)
    print("  LAYER 2: LOGIC LAYER (MCTS / QUANTUM SOLVER) — EVALUATION")
    print("█" * 60)

    results = []

    print_section("Logic Layer — Parameter Sweep")
    hdr = (f"  {'Regime':<14} {'Vol':>5} {'CohLoss':>8} "
           f"{'Halluc':>8} {'Optimal':>8}  Output (truncated)")
    print(hdr)
    print("  " + "-" * 80)

    for regime, vol, coh, expected_kws in SWEEP:
        # Build minimal state
        state = {
            "game_regime": regime,
            "metrics": {"volatility": vol, "coherence_loss": coh},
            "history": [],
            "messages": [],
            "beliefs":  {},
        }
        result_state = logic_layer_node(state)
        rec = result_state.get("logic_recommendation", "")

        hallucination = _is_hallucination(rec)
        optimal       = _keywords_found(rec, expected_kws)

        trunc = rec[:55].replace("\n", " ")
        print(f"  {regime:<14} {vol:>5.2f} {coh:>8.2f} "
              f"  {'❌' if hallucination else '✅':>6}   "
              f"{'✅' if optimal else '❌':>6}  {trunc}…")

        results.append({
            "regime":       regime,
            "volatility":   vol,
            "coh_loss":     coh,
            "hallucination": hallucination,
            "optimal":      optimal,
            "output":       rec,
        })

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    n              = len(results)
    halluc_rate    = sum(1 for r in results if r["hallucination"]) / n * 100
    adherence_rate = sum(1 for r in results if r["optimal"]) / n * 100

    # Search depth efficiency: high-volatility Bayesian cases that correctly
    # triggered Conservative-Reciprocity / pruning
    high_vol_bayes = [r for r in results
                      if r["regime"] == "Bayesian PD" and r["volatility"] > 0.6]
    depth_eff = (sum(1 for r in high_vol_bayes if r["optimal"]) /
                 max(len(high_vol_bayes), 1) * 100)

    print_section("LOGIC LAYER — AGGREGATE RESULTS")
    print(f"  {'Search Depth Efficiency (high-vol Bayesian)':<45} {depth_eff:.1f}%")
    print(f"  {'Logical Hallucination Rate':<45} {halluc_rate:.1f}%")
    print(f"  {'% Adherence to Optimal Path':<45} {adherence_rate:.1f}%")
    print()


if __name__ == "__main__":
    evaluate_logic()
