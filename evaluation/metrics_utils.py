"""
metrics_utils.py
================
Shared metric helpers for the SCUA evaluation suite.

Provides:
- PD payoff matrix
- LLM response parser (extracts Cooperate/Defect)
- Belief accuracy helper
- Retaliation detector
- Pretty console report printer
"""

import re
from typing import List, Dict, Optional, Tuple


# --------------------------------------------------------------------------- #
#  Prisoner's Dilemma — payoff matrix
# --------------------------------------------------------------------------- #
# Standard parameterisation: T > R > P > S
#   T (Temptation)  = 5   — defect while other cooperates
#   R (Reward)      = 3   — mutual cooperation
#   P (Punishment)  = 1   — mutual defection
#   S (Sucker)      = 0   — cooperate while other defects

PD_PAYOFF = {
    ("Cooperate", "Cooperate"): (3, 3),   # (my_payoff, opp_payoff)
    ("Cooperate", "Defect"):    (0, 5),
    ("Defect",    "Cooperate"): (5, 0),
    ("Defect",    "Defect"):    (1, 1),
}


def payoff(my_move: str, opp_move: str) -> Tuple[int, int]:
    """
    Returns (my_payoff, opponent_payoff) for the given move pair.
    Raises KeyError for invalid moves.
    """
    key = (my_move, opp_move)
    if key not in PD_PAYOFF:
        raise KeyError(f"Invalid move pair: {key}")
    return PD_PAYOFF[key]


# --------------------------------------------------------------------------- #
#  LLM response parser
# --------------------------------------------------------------------------- #

_ACTION_PATTERN = re.compile(r"\b(cooperate|defect)\b", re.IGNORECASE)


def parse_action(text: str) -> Optional[str]:
    """
    Extracts the first occurrence of 'Cooperate' or 'Defect' from the LLM
    response text (case-insensitive).  Returns None if neither is found.
    """
    if not text:
        return None
    match = _ACTION_PATTERN.search(text)
    if match:
        return match.group(1).capitalize()
    return None


# --------------------------------------------------------------------------- #
#  Belief helpers
# --------------------------------------------------------------------------- #

def dominant_belief(beliefs: Dict[str, float]) -> str:
    """Returns the belief key with the highest probability."""
    if not beliefs:
        return "Unknown"
    return max(beliefs, key=beliefs.get)


def classification_accuracy(records: List[Dict]) -> float:
    """
    Given a list of dicts with keys 'predicted' and 'actual',
    returns the fraction of correct classifications.
    """
    if not records:
        return 0.0
    correct = sum(1 for r in records if r["predicted"] == r["actual"])
    return correct / len(records)


# --------------------------------------------------------------------------- #
#  Retaliation detector
# --------------------------------------------------------------------------- #

def is_retaliation(agent_action: str, beliefs: Dict[str, float],
                   noisy_threshold: float = 0.5) -> bool:
    """
    Returns True if the agent chose Defect even though the dominant belief
    was Noisy (i.e. the agent over-retaliated against probable noise).
    """
    if agent_action != "Defect":
        return False
    return beliefs.get("Noisy", 0.0) >= noisy_threshold


# --------------------------------------------------------------------------- #
#  Pretty report printer
# --------------------------------------------------------------------------- #

def print_section(title: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_metric(name: str, value, unit: str = "") -> None:
    print(f"  {name:<40} {value}{unit}")


def print_round_row(round_num: int, opp_move: str, agent_action: str,
                    beliefs: Dict[str, float], metrics: Dict[str, float]) -> None:
    """One-line tabular row for per-round output."""
    dom = dominant_belief(beliefs)
    dec = beliefs.get("Deceptive", 0.0)
    noi = beliefs.get("Noisy", 0.0)
    ent = metrics.get("entropy", 0.0)
    vol = metrics.get("volatility", 0.0)
    print(
        f"  R{round_num:02d} | Opp:{opp_move:<11} → Agent:{agent_action:<11} | "
        f"DomBelief:{dom:<14} Dec:{dec:.2f} Noi:{noi:.2f} | "
        f"Ent:{ent:.2f} Vol:{vol:.2f}"
    )
