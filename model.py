"""
Autoformalization Ecosystem Dynamics — Core Model

A formal dynamical system modeling the co-evolution of formalization,
autoproving, verification, and human interpretation in mathematics.

Based on structural analysis of first principles from the autoformalization
discourse (Abouzaid et al., 2026).

Three Load-Bearing Principles modeled:
  1. Verification Axiom: Trust reduces to verification environment integrity
  2. Translation Gap: informal->formal is irreducibly semantic
  3. Bottleneck Migration: automation shifts constraints to adjacent stages

State Variables:
  F(t) - Formalization rate
  P(t) - Autoproving power
  K(t) - Formalized knowledge base size
  H(t) - Human interpretation capacity
  V(t) - Verification integrity
  I(t) - Incentive to formalize
"""

import numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


# ---------------------------------------------------------------------------
# State indices & labels
# ---------------------------------------------------------------------------
F, P, K, H, V, I = 0, 1, 2, 3, 4, 5

LABELS = [
    'Formalization (F)', 'Autoproving (P)', 'Knowledge Base (K)',
    'Interpretation (H)', 'Verification (V)', 'Incentive (I)',
]
SHORT = ['F', 'P', 'K', 'H', 'V', 'I']


# ---------------------------------------------------------------------------
# Parameters (immutable)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Params:
    """Ecosystem parameters governing coupled ODE dynamics.

    Tuned so the baseline sits near the bifurcation boundary —
    the system can tip toward sustainable growth OR collapse
    depending on verification resilience.
    """

    # Formalization: dF/dt = alpha * I*H/(1+I) - beta * F
    #   Saturating response to incentive, bounded by human capacity H
    alpha: float = 0.5       # incentive -> formalization coupling
    beta: float = 0.05       # formalization friction

    # Autoproving: dP/dt = gamma * K * V / (1 + sigma*K) - delta * P
    #   Diminishing returns: can't double prover power indefinitely
    #   P_max ~ gamma * V / (delta * sigma) — bounded even as K -> inf
    gamma: float = 0.15      # knowledge -> proving coupling
    delta: float = 0.02      # proving capability decay
    sigma: float = 0.3       # autoproving saturation rate

    # Knowledge: dK/dt = F - epsilon * K
    epsilon: float = 0.02    # knowledge deprecation

    # Human interpretation: dH/dt = zeta*H*(1-H/h_max) - eta*P*max(0,1-V)*H
    #   Overwhelm is multiplicative with H -> can't go negative
    h_max: float = 1.0       # maximum human interpretation capacity
    zeta: float = 0.08       # capacity growth rate
    eta: float = 0.2         # overwhelm from unverified proofs

    # Verification: dV/dt = theta*(v_0-V) - iota*P^2/(K+0.1)*V + mu*K*(1-V)
    #   Hacking pressure multiplicative with V -> stays in [0,1]
    #   Knowledge->verification: more proofs = easier cross-referencing
    v_0: float = 1.0         # natural verification level
    theta: float = 0.15      # restoration rate
    iota: float = 0.08       # reward-hacking pressure coefficient
    mu: float = 0.025        # knowledge -> verification reinforcement

    # Incentive: dI/dt = kappa*P/(1+0.5*P) - lamda*I
    #   Saturating response to P, natural decay (no 1/F singularity)
    kappa: float = 0.4       # proving -> incentive coupling
    lamda: float = 0.1       # incentive decay rate


# ---------------------------------------------------------------------------
# ODE system
# ---------------------------------------------------------------------------
def derivatives(state: np.ndarray, t: float, p: Params) -> np.ndarray:
    """
    RHS of the autoformalization ODE system.

    All terms are constructed to be self-consistently non-negative:
      - H overwhelm is proportional to H (multiplicative barrier)
      - V degradation is proportional to V (stays in [0,1])
      - I has natural decay (no 1/F singularity)
      - F saturates with incentive (bounded by human capacity)

    Captures:
      - Chicken-and-egg: K needs F, P needs K, F needs I, I needs P
      - Verification crisis: V degrades as P grows faster than K
      - Bottleneck migration: system constraint shifts between variables
      - Translation gap: F is bounded by H through saturating coupling
    """
    s = np.maximum(state, 1e-10)

    # Formalization: saturating response to incentive, bounded by H
    dF = p.alpha * s[I] * s[H] / (1.0 + s[I]) - p.beta * s[F]

    # Autoproving: grows with knowledge * verification, saturating
    # Diminishing returns prevent unbounded growth (can't just add more data)
    dP = p.gamma * s[K] * s[V] / (1.0 + p.sigma * s[K]) - p.delta * s[P]

    # Knowledge base: accumulates from formalization, depreciates
    dK = s[F] - p.epsilon * s[K]

    # Human interpretation: logistic growth, overwhelmed by unverified proofs
    # Multiplicative H term ensures H >= 0 (barrier at zero)
    overwhelm = p.eta * s[P] * max(0.0, 1.0 - s[V])
    dH = p.zeta * s[H] * (1.0 - s[H] / p.h_max) - overwhelm * s[H]

    # Verification: restores toward v_0, degraded by reward hacking,
    # reinforced by knowledge base (more proofs = easier cross-referencing).
    # This creates the critical race condition: can K grow fast enough
    # to offset hacking pressure from P?
    hacking = p.iota * s[P] ** 2 / (s[K] + 0.1)
    knowledge_boost = p.mu * s[K] * (1.0 - s[V])
    dV = p.theta * (p.v_0 - s[V]) - hacking * s[V] + knowledge_boost

    # Incentive: saturating response to prover power, natural decay
    dI = p.kappa * s[P] / (1.0 + 0.5 * s[P]) - p.lamda * s[I]

    return np.array([dF, dP, dK, dH, dV, dI])


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------
def initial_state() -> np.ndarray:
    """Present-day initial conditions (circa 2026)."""
    return np.array([
        0.05,   # F: low formalization rate
        0.15,   # P: modest autoproving (AlphaProof-era)
        0.10,   # K: ~10% of active math formalized (Mathlib)
        0.80,   # H: high but finite human capacity
        0.95,   # V: high verification integrity (Lean kernel is trustworthy)
        0.15,   # I: modest incentive to formalize
    ])


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def simulate(
    p: Params,
    y0: np.ndarray = None,
    t_span: tuple = (0, 200),
    n_points: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the system with a stiff-capable solver.

    Returns (t, states) arrays. States are clamped to [0, inf)
    after integration for physical consistency.
    """
    if y0 is None:
        y0 = initial_state()
    t_eval = np.linspace(*t_span, n_points)

    sol = solve_ivp(
        fun=lambda t, y: derivatives(y, t, p),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method='Radau',
        rtol=1e-8,
        atol=1e-10,
        max_step=1.0,
    )

    states = np.maximum(sol.y.T, 0.0)  # enforce non-negativity
    return sol.t, states


# ---------------------------------------------------------------------------
# Bottleneck analysis
# ---------------------------------------------------------------------------
def bottleneck(state: np.ndarray) -> int:
    """Index of the binding constraint (minimum state variable)."""
    return int(np.argmin(state))


def bottleneck_series(states: np.ndarray) -> np.ndarray:
    """Bottleneck index at each timestep."""
    return np.array([bottleneck(s) for s in states])


def detect_transitions(t: np.ndarray, states: np.ndarray) -> list[dict]:
    """Detect moments when the system bottleneck shifts."""
    bn = bottleneck_series(states)
    transitions = []
    for i in range(1, len(bn)):
        if bn[i] != bn[i - 1]:
            transitions.append({
                'time': float(t[i]),
                'from': SHORT[bn[i - 1]],
                'to': SHORT[bn[i]],
                'state': states[i].copy(),
            })
    return transitions


# ---------------------------------------------------------------------------
# Fixed-point & stability analysis
# ---------------------------------------------------------------------------
def jacobian(state: np.ndarray, p: Params, eps: float = 1e-7) -> np.ndarray:
    """Numerical Jacobian at a given state."""
    n = len(state)
    J = np.zeros((n, n))
    f0 = derivatives(state, 0, p)
    for i in range(n):
        perturbed = state.copy()
        perturbed[i] += eps
        J[:, i] = (derivatives(perturbed, 0, p) - f0) / eps
    return J


def find_equilibria(p: Params, n_tries: int = 500) -> list[dict]:
    """Find and classify fixed points by searching from random seeds."""
    rng = np.random.default_rng(42)
    found = []

    for _ in range(n_tries):
        x0 = rng.uniform(0.01, 3.0, size=6)
        try:
            sol, _info, ier, _msg = fsolve(
                lambda x: derivatives(x, 0, p), x0, full_output=True,
            )
        except Exception:
            continue

        if ier != 1 or np.any(sol < -0.05):
            continue
        sol = np.maximum(sol, 0)

        if np.linalg.norm(derivatives(sol, 0, p)) > 1e-7:
            continue

        # deduplicate
        if any(np.linalg.norm(fp['state'] - sol) < 1e-3 for fp in found):
            continue

        J = jacobian(sol, p)
        eigs = np.linalg.eigvals(J)
        reals = eigs.real

        if np.all(reals < -1e-8):
            stability = 'stable'
        elif np.all(reals > 1e-8):
            stability = 'unstable'
        elif np.any(reals > 1e-8) and np.any(reals < -1e-8):
            stability = 'saddle'
        else:
            stability = 'marginal'

        found.append({
            'state': sol,
            'eigenvalues': eigs,
            'stability': stability,
            'bottleneck': SHORT[bottleneck(sol)],
            'max_real_eig': float(max(reals)),
        })

    return sorted(found, key=lambda fp: fp['max_real_eig'])


# ---------------------------------------------------------------------------
# Scenario library
# ---------------------------------------------------------------------------
def scenario_params() -> dict[str, Params]:
    """Named parameter scenarios representing distinct futures.

    Key variable: mu (knowledge->verification reinforcement).
    This is the structural lever that determines survival vs collapse.
    """
    return {
        'baseline': Params(),
        'verification_crisis': Params(theta=0.05, iota=0.15, mu=0.005),
        'formalization_boom': Params(alpha=1.0, kappa=0.6),
        'rubber_stamping': Params(eta=0.03, theta=0.05, mu=0.005),
        'knowledge_fortress': Params(mu=0.08, theta=0.3, sigma=0.2),
    }
