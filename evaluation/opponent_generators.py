"""
opponent_generators.py
======================
Pure-Python opponent behaviour simulators for the SCUA evaluation suite.

Each generator is a callable:
    move = generator(round_num: int, history: list[dict]) -> str

Returns "Cooperate" or "Defect".

Opponent Types
--------------
neutral         — Always cooperates.
deceptive       — Cooperates for the first `grace_rounds`, then always defects.
noisy           — Cooperates most of the time; defects with probability `noise_rate`.
irrational      — Completely random 50/50 each round.
non_stationary  — Switches strategy every `switch_every` rounds (Cooperate → Defect → ...).
"""

import random
from typing import List, Dict


# --------------------------------------------------------------------------- #
#  Base helpers
# --------------------------------------------------------------------------- #

def neutral(round_num: int, history: List[Dict], **kwargs) -> str:
    """Always cooperates — the ideal cooperative baseline."""
    return "Cooperate"


def deceptive(round_num: int, history: List[Dict],
              grace_rounds: int = 3, **kwargs) -> str:
    """
    Lulls the agent into cooperation for `grace_rounds`, then permanently
    defects — the classic backstabber pattern.
    """
    if round_num <= grace_rounds:
        return "Cooperate"
    return "Defect"


def noisy(round_num: int, history: List[Dict],
          noise_rate: float = 0.20, seed: int = None, **kwargs) -> str:
    """
    Mostly cooperative but occasionally defects due to 'error' or
    environmental noise (default 20% defection probability).
    """
    rng = random.Random(seed + round_num if seed is not None else None)
    return "Defect" if rng.random() < noise_rate else "Cooperate"


def irrational(round_num: int, history: List[Dict],
               seed: int = None, **kwargs) -> str:
    """
    Completely random — 50% Cooperate, 50% Defect.
    Models a boundedly rational or emotionally driven opponent.
    """
    rng = random.Random(seed + round_num if seed is not None else None)
    return random.choice(["Cooperate", "Defect"])


def non_stationary(round_num: int, history: List[Dict],
                   switch_every: int = 5, **kwargs) -> str:
    """
    Alternates between a cooperative phase and a defective phase every
    `switch_every` rounds — models a regime-shifting opponent.
    """
    phase = (round_num - 1) // switch_every  # 0-indexed phase number
    if phase % 2 == 0:
        return "Cooperate"
    return "Defect"


# --------------------------------------------------------------------------- #
#  Registry
# --------------------------------------------------------------------------- #

OPPONENT_REGISTRY = {
    "neutral": neutral,
    "deceptive": deceptive,
    "noisy": noisy,
    "irrational": irrational,
    "non_stationary": non_stationary,
}


def get_opponent(name: str):
    """
    Returns the opponent callable by name.
    Raises KeyError if the name is not recognised.
    """
    if name not in OPPONENT_REGISTRY:
        raise KeyError(
            f"Unknown opponent '{name}'. "
            f"Choose from: {list(OPPONENT_REGISTRY.keys())}"
        )
    return OPPONENT_REGISTRY[name]
