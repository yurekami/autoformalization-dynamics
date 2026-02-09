"""
Coordination Game for Formalization Incentives

Models N mathematicians deciding whether to invest in formalizing their
research area. Captures:

  - Network effects (formalization is more valuable when others do it)
  - Coordination failure (Nash equilibrium != social optimum)
  - Threshold dynamics (tipping point for adoption)
  - First-mover advantage (reputation bonus decays with adoption)

The key insight: the chicken-and-egg problem from the dynamical model
manifests here as a coordination game with multiple equilibria — a low
equilibrium (nobody formalizes) and a high equilibrium (everyone does),
separated by an unstable tipping point.
"""

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class GameParams:
    """Parameters for the formalization coordination game. Immutable."""
    n_players: int = 50            # number of mathematicians
    base_cost: float = 0.4         # cost of formalization effort
    prover_benefit: float = 0.6    # benefit from autoproving (at full network)
    network_exp: float = 1.5       # superlinear network effect exponent
    reputation_bonus: float = 0.2  # first-mover reputational advantage
    threshold: float = 0.3         # fraction needed for prover to be useful


def payoff(formalize: bool, frac_others: float, gp: GameParams) -> float:
    """
    Payoff to a single mathematician.

    If formalize:
      - Pay cost
      - Get prover benefit scaled by network effects (if above threshold)
      - Get reputation bonus scaled by how early (fewer others = more bonus)

    If don't formalize:
      - No cost
      - Partial free-rider benefit if others formalized enough
    """
    if formalize:
        network = frac_others ** gp.network_exp if frac_others > gp.threshold else 0.0
        prover = gp.prover_benefit * network
        reputation = gp.reputation_bonus * max(0.0, 1.0 - frac_others * 2)
        return prover + reputation - gp.base_cost

    # Free-rider: diminished benefit, no reputation
    if frac_others > gp.threshold:
        return 0.3 * gp.prover_benefit * (frac_others ** gp.network_exp)
    return 0.0


def best_response(frac_others: float, gp: GameParams) -> bool:
    """Should a player formalize given the fraction of others formalizing?"""
    return payoff(True, frac_others, gp) > payoff(False, frac_others, gp)


def find_nash_equilibria(gp: GameParams, resolution: int = 1000) -> list[dict]:
    """
    Find symmetric Nash equilibria.

    A fraction f* is Nash if the best response at f* is consistent:
      - f*=1 is Nash if formalize payoff > skip payoff at f=1
      - f*=0 is Nash if skip payoff > formalize payoff at f=0
      - Mixed: f* where the two payoff curves cross
    """
    fracs = np.linspace(0, 1, resolution)
    pay_form = np.array([payoff(True, f, gp) for f in fracs])
    pay_skip = np.array([payoff(False, f, gp) for f in fracs])
    advantage = pay_form - pay_skip

    equilibria = []

    # Pure: everyone formalizes
    if advantage[-1] > 0:
        equilibria.append({
            'fraction': 1.0,
            'type': 'pure_formalize',
            'payoff_formalize': float(pay_form[-1]),
            'payoff_skip': float(pay_skip[-1]),
            'stable': True,
        })

    # Pure: nobody formalizes
    if advantage[0] < 0:
        equilibria.append({
            'fraction': 0.0,
            'type': 'pure_skip',
            'payoff_formalize': float(pay_form[0]),
            'payoff_skip': float(pay_skip[0]),
            'stable': True,
        })

    # Mixed: zero crossings of advantage function
    for i in range(1, len(advantage)):
        if advantage[i - 1] * advantage[i] < 0:
            f_cross = fracs[i - 1] + (fracs[i] - fracs[i - 1]) * (
                -advantage[i - 1]
            ) / (advantage[i] - advantage[i - 1])
            # Crossing from + to - means unstable threshold
            equilibria.append({
                'fraction': float(f_cross),
                'type': 'mixed',
                'payoff_formalize': float(payoff(True, f_cross, gp)),
                'payoff_skip': float(payoff(False, f_cross, gp)),
                'stable': advantage[i - 1] < 0,  # stable if restoring
            })

    return equilibria


def social_welfare(frac: float, gp: GameParams) -> float:
    """Total welfare at a given formalization fraction."""
    n_form = int(frac * gp.n_players)
    n_skip = gp.n_players - n_form
    return n_form * payoff(True, frac, gp) + n_skip * payoff(False, frac, gp)


def find_social_optimum(gp: GameParams, resolution: int = 1000) -> dict:
    """Find the fraction that maximizes total welfare."""
    fracs = np.linspace(0, 1, resolution)
    welfare = np.array([social_welfare(f, gp) for f in fracs])
    idx = int(np.argmax(welfare))
    return {
        'fraction': float(fracs[idx]),
        'welfare': float(welfare[idx]),
    }


def coordination_gap(gp: GameParams) -> dict:
    """Quantify the gap between Nash equilibria and social optimum."""
    nash = find_nash_equilibria(gp)
    optimum = find_social_optimum(gp)

    stable_nash = [eq for eq in nash if eq['stable']]

    if not stable_nash:
        worst_nash_welfare = 0.0
    else:
        worst_nash_welfare = min(
            social_welfare(eq['fraction'], gp) for eq in stable_nash
        )

    opt_welfare = max(abs(optimum['welfare']), 1e-10)
    return {
        'nash_equilibria': nash,
        'social_optimum': optimum,
        'gap': optimum['welfare'] - worst_nash_welfare,
        'gap_fraction': (optimum['welfare'] - worst_nash_welfare) / opt_welfare,
    }


def tipping_point_analysis(
    gp: GameParams, resolution: int = 500,
) -> dict[str, np.ndarray]:
    """Analyze how the tipping point shifts with autoprover power."""
    prover_levels = np.linspace(0.1, 2.0, resolution)
    thresholds = []

    for pb in prover_levels:
        modified = GameParams(
            n_players=gp.n_players,
            base_cost=gp.base_cost,
            prover_benefit=pb,
            network_exp=gp.network_exp,
            reputation_bonus=gp.reputation_bonus,
            threshold=gp.threshold,
        )
        equilibria = find_nash_equilibria(modified)
        unstable = [
            eq for eq in equilibria
            if not eq['stable'] and eq['type'] == 'mixed'
        ]
        if unstable:
            thresholds.append((pb, unstable[0]['fraction']))
        elif best_response(0.0, modified):
            thresholds.append((pb, 0.0))
        else:
            thresholds.append((pb, 1.0))

    return {
        'prover_levels': np.array([t[0] for t in thresholds]),
        'thresholds': np.array([t[1] for t in thresholds]),
    }
