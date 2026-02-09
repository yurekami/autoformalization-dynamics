"""
Autoformalization Dynamics — Simulation & Visualization Engine

Generates seven publication-quality figures analyzing the autoformalization
ecosystem from first principles:

  1. Dashboard: time evolution of all 6 state variables
  2. Phase portraits: flow fields in key 2D projections
  3. Bottleneck waterfall: which variable constrains the system over time
  4. Bifurcation diagram: how verification resilience shapes outcomes
  5. Game theory: coordination failure, Nash equilibria, tipping points
  6. Scenario comparison: five distinct futures overlaid
  7. Dependency graph: causal topology of the first principles

Usage:
  python run.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from model import (
    Params, simulate, initial_state, find_equilibria,
    detect_transitions, bottleneck_series, derivatives,
    scenario_params, LABELS, SHORT, F, P, K, H, V, I,
)
from game import (
    GameParams, payoff, find_nash_equilibria, find_social_optimum,
    coordination_gap, tipping_point_analysis, social_welfare,
)


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'text.color': '#c9d1d9',
    'axes.labelcolor': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'axes.edgecolor': '#30363d',
    'grid.color': '#21262d',
    'grid.alpha': 0.5,
    'font.family': 'monospace',
    'font.size': 10,
})

COLORS = ['#58a6ff', '#f0883e', '#3fb950', '#d2a8ff', '#f85149', '#79c0ff']
OUT = Path('output')


# ---------------------------------------------------------------------------
# Figure 1: Dashboard
# ---------------------------------------------------------------------------
def fig1_dashboard(t, states, params):
    """2x3 dashboard: time evolution of all state variables."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        'Autoformalization Ecosystem Dynamics',
        fontsize=16, fontweight='bold', y=0.98,
    )

    bn = bottleneck_series(states)

    for idx, ax in enumerate(axes.flat):
        ax.plot(t, states[:, idx], color=COLORS[idx], linewidth=2)
        ax.set_title(
            LABELS[idx], fontsize=11, fontweight='bold', color=COLORS[idx],
        )
        ax.set_xlabel('Time (years)')
        ax.grid(True, alpha=0.3)

        # shade regions where this variable is the bottleneck
        is_bn = bn == idx
        for i in range(1, len(is_bn)):
            if is_bn[i]:
                ax.axvspan(t[i - 1], t[i], alpha=0.08, color=COLORS[idx])

        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(bottom=0)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / '1_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  [1/7] Dashboard')


# ---------------------------------------------------------------------------
# Figure 2: Phase Portraits
# ---------------------------------------------------------------------------
def fig2_phase_portraits(t, states, params):
    """Phase portraits: Autoproving vs Formalization with flow arrows."""
    projections = [
        (F, P, 'Formalization', 'Autoproving'),
        (P, V, 'Autoproving', 'Verification'),
        (F, H, 'Formalization', 'Interpretation'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(
        'Phase Portraits with Flow Fields',
        fontsize=14, fontweight='bold', y=0.98,
    )

    for ax, (xi, yi, xl, yl) in zip(axes, projections):
        x_max = max(states[:, xi]) * 1.3
        y_max = max(states[:, yi]) * 1.3
        xr = np.linspace(0.01, x_max, 18)
        yr = np.linspace(0.01, y_max, 18)
        X, Y = np.meshgrid(xr, yr)

        U = np.zeros_like(X)
        W = np.zeros_like(Y)

        base = initial_state().copy()
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                s = base.copy()
                s[xi] = X[i, j]
                s[yi] = Y[i, j]
                d = derivatives(s, 0, params)
                U[i, j] = d[xi]
                W[i, j] = d[yi]

        speed = np.sqrt(U ** 2 + W ** 2)
        ax.streamplot(
            xr, yr, U, W,
            color=speed, cmap='cool', density=1.2, linewidth=0.8, arrowsize=1,
        )

        ax.plot(
            states[:, xi], states[:, yi],
            color='#f0883e', linewidth=2.5, zorder=5,
        )
        ax.plot(
            states[0, xi], states[0, yi], 'o',
            color='#3fb950', markersize=10, zorder=6, label='Start (2026)',
        )
        ax.plot(
            states[-1, xi], states[-1, yi], 's',
            color='#f85149', markersize=10, zorder=6, label='Steady state',
        )

        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / '2_phase_portraits.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  [2/7] Phase portraits')


# ---------------------------------------------------------------------------
# Figure 3: Bottleneck Waterfall
# ---------------------------------------------------------------------------
def fig3_bottleneck_waterfall(t, states):
    """Bottleneck migration — colored bands showing which variable constrains."""
    fig, ax = plt.subplots(figsize=(14, 4))
    bn = bottleneck_series(states)

    drawn_labels = set()
    for idx in range(6):
        mask = bn == idx
        if not np.any(mask):
            continue
        # find contiguous segments
        segments = []
        start = None
        for i in range(len(mask)):
            if mask[i] and start is None:
                start = i
            elif not mask[i] and start is not None:
                segments.append((start, i - 1))
                start = None
        if start is not None:
            segments.append((start, len(mask) - 1))

        for s, e in segments:
            label = SHORT[idx] if idx not in drawn_labels else ''
            ax.axvspan(t[s], t[e], alpha=0.6, color=COLORS[idx], label=label)
            drawn_labels.add(idx)

    ax.set_xlabel('Time (years)', fontsize=11)
    ax.set_title(
        'Bottleneck Migration — Which Variable Constrains the System?',
        fontsize=13, fontweight='bold',
    )
    ax.set_xlim(t[0], t[-1])
    ax.set_yticks([])

    handles = [
        mpatches.Patch(color=COLORS[i], label=f'{SHORT[i]}: {LABELS[i]}', alpha=0.7)
        for i in range(6)
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=9, ncol=2)

    fig.tight_layout()
    fig.savefig(OUT / '3_bottleneck_waterfall.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  [3/7] Bottleneck waterfall')


# ---------------------------------------------------------------------------
# Figure 4: Bifurcation Diagram
# ---------------------------------------------------------------------------
def fig4_bifurcation():
    """Bifurcation: vary mu (knowledge->verification reinforcement).

    This is the critical structural parameter — it determines whether
    the knowledge base can grow fast enough to protect verification
    from reward hacking pressure.
    """
    mus = np.linspace(0.0, 0.08, 80)
    final_states = []

    for m in mus:
        p = Params(mu=m)
        _, st = simulate(p, t_span=(0, 300), n_points=3000)
        final_states.append(st[-1])

    final_states = np.array(final_states)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        'Bifurcation Analysis — Knowledge-Verification Coupling (mu)',
        fontsize=14, fontweight='bold', y=0.98,
    )

    # Left: key variables vs mu
    ax = axes[0]
    for idx, label in [
        (F, 'Formalization'), (H, 'Interpretation'),
        (V, 'Verification'), (P, 'Autoproving'),
    ]:
        ax.plot(mus, final_states[:, idx], color=COLORS[idx], linewidth=2, label=label)
    ax.set_xlabel('Knowledge->Verification Coupling (mu)')
    ax.set_ylabel('Steady-State Value')
    ax.set_title('Can Knowledge Protect Verification?', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: bottleneck at each mu
    ax = axes[1]
    bns = np.array([np.argmin(s) for s in final_states])
    for idx in range(6):
        mask = bns == idx
        if np.any(mask):
            ax.scatter(
                mus[mask], [idx] * np.sum(mask),
                c=COLORS[idx], s=40, label=SHORT[idx], zorder=3,
            )
    ax.set_xlabel('Knowledge->Verification Coupling (mu)')
    ax.set_ylabel('Bottleneck Variable')
    ax.set_yticks(range(6))
    ax.set_yticklabels(SHORT)
    ax.set_title('Bottleneck Phase Diagram', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / '4_bifurcation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  [4/7] Bifurcation diagram')


# ---------------------------------------------------------------------------
# Figure 5: Game Theory
# ---------------------------------------------------------------------------
def fig5_game_theory():
    """Coordination game: payoffs, welfare, tipping point."""
    gp = GameParams()
    fracs = np.linspace(0, 1, 500)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        'Formalization Coordination Game',
        fontsize=14, fontweight='bold', y=0.98,
    )

    # --- Left: payoff curves ---
    ax = axes[0]
    pay_form = [payoff(True, f, gp) for f in fracs]
    pay_skip = [payoff(False, f, gp) for f in fracs]

    ax.plot(fracs, pay_form, color='#3fb950', linewidth=2, label='Formalize')
    ax.plot(fracs, pay_skip, color='#f85149', linewidth=2, label='Free-ride')
    ax.axhline(0, color='#8b949e', linewidth=0.5, linestyle='--')

    nash = find_nash_equilibria(gp)
    for eq in nash:
        marker = 'o' if eq['stable'] else 'x'
        color = '#58a6ff' if eq['stable'] else '#d2a8ff'
        ax.axvline(eq['fraction'], color=color, linewidth=1.5, linestyle='--', alpha=0.7)
        ax.plot(
            eq['fraction'], eq['payoff_formalize'], marker,
            color=color, markersize=12, zorder=5,
        )

    ax.set_xlabel('Fraction Formalizing')
    ax.set_ylabel('Individual Payoff')
    ax.set_title('Payoff Curves & Nash Equilibria', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Middle: social welfare ---
    ax = axes[1]
    welfare = [social_welfare(f, gp) for f in fracs]
    ax.plot(fracs, welfare, color='#79c0ff', linewidth=2)

    opt = find_social_optimum(gp)
    ax.axvline(
        opt['fraction'], color='#3fb950', linewidth=2, linestyle='--',
        label=f"Optimum: {opt['fraction']:.0%}",
    )

    for eq in nash:
        if eq['stable']:
            ax.axvline(
                eq['fraction'], color='#f85149', linewidth=2, linestyle='--',
                label=f"Nash: {eq['fraction']:.0%}",
            )

    gap = coordination_gap(gp)
    ax.set_xlabel('Fraction Formalizing')
    ax.set_ylabel('Total Welfare')
    ax.set_title(
        f'Coordination Gap: {gap["gap_fraction"]:.0%} welfare loss',
        fontweight='bold',
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Right: tipping point vs prover power ---
    ax = axes[2]
    tip = tipping_point_analysis(gp)
    ax.plot(tip['prover_levels'], tip['thresholds'], color='#f0883e', linewidth=2)
    ax.fill_between(
        tip['prover_levels'], tip['thresholds'], 1,
        alpha=0.15, color='#3fb950', label='Formalize dominant',
    )
    ax.fill_between(
        tip['prover_levels'], 0, tip['thresholds'],
        alpha=0.15, color='#f85149', label='Free-ride dominant',
    )
    ax.set_xlabel('Autoprover Benefit Level')
    ax.set_ylabel('Tipping Point (fraction)')
    ax.set_title(
        'How Prover Power Shifts the\nCoordination Threshold',
        fontweight='bold',
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / '5_game_theory.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  [5/7] Game theory')


# ---------------------------------------------------------------------------
# Figure 6: Scenario Comparison
# ---------------------------------------------------------------------------
def fig6_scenarios():
    """Five futures overlaid."""
    scenarios = scenario_params()
    scenario_colors = {
        'baseline': '#8b949e',
        'verification_crisis': '#f85149',
        'formalization_boom': '#3fb950',
        'rubber_stamping': '#d2a8ff',
        'knowledge_fortress': '#58a6ff',
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        'Five Futures for Autoformalization',
        fontsize=14, fontweight='bold', y=0.98,
    )

    for name, p in scenarios.items():
        t, st = simulate(p, t_span=(0, 200))
        color = scenario_colors[name]
        for idx, ax in enumerate(axes.flat):
            label = name.replace('_', ' ').title() if idx == 0 else ''
            ax.plot(t, st[:, idx], color=color, linewidth=1.5, alpha=0.85, label=label)

    for idx, ax in enumerate(axes.flat):
        ax.set_title(LABELS[idx], fontsize=11, fontweight='bold', color=COLORS[idx])
        ax.set_xlabel('Time (years)')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 200)
        ax.set_ylim(bottom=0)

    axes[0, 0].legend(fontsize=8, loc='upper left')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / '6_scenarios.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  [6/7] Scenario comparison')


# ---------------------------------------------------------------------------
# Figure 7: Dependency Graph
# ---------------------------------------------------------------------------
def fig7_dependency_graph():
    """Causal topology of the first principles."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1.5, 9.5)
    ax.axis('off')
    ax.set_title(
        'Causal Topology of Autoformalization Principles',
        fontsize=16, fontweight='bold', pad=20,
    )

    nodes = {
        'VERIFICATION\nAXIOM': (5, 8.5),
        'AUTOPROVING': (5, 6.5),
        'TRANSLATION\nGAP': (1.5, 4.5),
        'BOTTLENECK\nMIGRATION': (8.5, 4.5),
        'INCENTIVE\nDYNAMICS': (8.5, 2),
        'HUMAN ROLE\nSHIFT': (5, 0),
        'INFRASTRUCTURE': (1.5, 2),
        'AUTO-\nFORMALIZATION': (1.5, 6.5),
    }

    node_colors = {
        'VERIFICATION\nAXIOM': '#f85149',
        'AUTOPROVING': '#58a6ff',
        'TRANSLATION\nGAP': '#d2a8ff',
        'BOTTLENECK\nMIGRATION': '#f0883e',
        'INCENTIVE\nDYNAMICS': '#79c0ff',
        'HUMAN ROLE\nSHIFT': '#3fb950',
        'INFRASTRUCTURE': '#8b949e',
        'AUTO-\nFORMALIZATION': '#56d364',
    }

    # Load-bearing tags
    load_bearing = {'VERIFICATION\nAXIOM', 'TRANSLATION\nGAP', 'BOTTLENECK\nMIGRATION'}

    edges = [
        ('VERIFICATION\nAXIOM', 'AUTOPROVING'),
        ('AUTOPROVING', 'BOTTLENECK\nMIGRATION'),
        ('AUTOPROVING', 'AUTO-\nFORMALIZATION'),
        ('BOTTLENECK\nMIGRATION', 'INCENTIVE\nDYNAMICS'),
        ('INCENTIVE\nDYNAMICS', 'HUMAN ROLE\nSHIFT'),
        ('TRANSLATION\nGAP', 'AUTO-\nFORMALIZATION'),
        ('TRANSLATION\nGAP', 'INFRASTRUCTURE'),
        ('INFRASTRUCTURE', 'HUMAN ROLE\nSHIFT'),
        ('AUTO-\nFORMALIZATION', 'BOTTLENECK\nMIGRATION'),
    ]

    # Draw edges
    for src, dst in edges:
        x0, y0 = nodes[src]
        x1, y1 = nodes[dst]
        ax.annotate(
            '', xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle='->', color='#484f58', lw=2,
                connectionstyle='arc3,rad=0.15',
            ),
        )

    # Draw nodes
    for label, (x, y) in nodes.items():
        color = node_colors[label]
        lw = 3 if label in load_bearing else 1
        bbox = dict(
            boxstyle='round,pad=0.7', facecolor=color, alpha=0.2,
            edgecolor=color, linewidth=lw,
        )
        fontsize = 12 if label in load_bearing else 10
        ax.text(
            x, y, label, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=color, bbox=bbox,
        )

    # Contradiction annotations
    ax.text(
        5, 3.8,
        'CONTRADICTION 1\nMath Singularity vs.\nHuman Irreplaceability',
        ha='center', fontsize=10, color='#f85149', fontstyle='italic',
        bbox=dict(
            boxstyle='round,pad=0.5', facecolor='#f85149',
            alpha=0.1, edgecolor='#f85149', linestyle='--',
        ),
    )

    ax.text(
        5, -1,
        'CONTRADICTION 2\nChicken-and-Egg Coordination Problem\n'
        '(K needs F, P needs K, F needs I, I needs P)',
        ha='center', fontsize=10, color='#f0883e', fontstyle='italic',
        bbox=dict(
            boxstyle='round,pad=0.5', facecolor='#f0883e',
            alpha=0.1, edgecolor='#f0883e', linestyle='--',
        ),
    )

    # Legend for load-bearing
    ax.text(
        9.5, 8.5, 'Bold border =\nload-bearing\nprinciple',
        ha='center', fontsize=9, color='#8b949e', fontstyle='italic',
    )

    fig.tight_layout()
    fig.savefig(OUT / '7_dependency_graph.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  [7/7] Dependency graph')


# ---------------------------------------------------------------------------
# Quantitative Analysis (console)
# ---------------------------------------------------------------------------
def print_analysis(params):
    """Print quantitative analysis to console."""
    sep = '=' * 72
    print(f'\n{sep}')
    print('  AUTOFORMALIZATION ECOSYSTEM — QUANTITATIVE ANALYSIS')
    print(sep)

    # --- Equilibria ---
    print('\n  FIXED POINT ANALYSIS')
    print('  ' + '-' * 40)
    equilibria = find_equilibria(params)
    if not equilibria:
        print('    No equilibria found (system may be divergent)')
    for i, eq in enumerate(equilibria):
        s = eq['state']
        print(f'\n    Equilibrium {i + 1} [{eq["stability"].upper()}]')
        print(f'      F={s[0]:.3f}  P={s[1]:.3f}  K={s[2]:.3f}  '
              f'H={s[3]:.3f}  V={s[4]:.3f}  I={s[5]:.3f}')
        print(f'      Bottleneck: {eq["bottleneck"]}  |  '
              f'Max Re(lambda): {eq["max_real_eig"]:.6f}')

    # --- Phase transitions ---
    print('\n  BOTTLENECK PHASE TRANSITIONS')
    print('  ' + '-' * 40)
    t, states = simulate(params, t_span=(0, 300))
    transitions = detect_transitions(t, states)
    if not transitions:
        print('    No transitions detected')
    for tr in transitions:
        print(f'    t = {tr["time"]:6.1f} yr:  {tr["from"]} --> {tr["to"]}')

    # --- Game theory ---
    print('\n  COORDINATION GAME')
    print('  ' + '-' * 40)
    gp = GameParams()
    gap = coordination_gap(gp)
    opt = gap['social_optimum']
    print(f'    Social optimum: {opt["fraction"]:.0%} formalize '
          f'(welfare = {opt["welfare"]:.2f})')
    for eq in gap['nash_equilibria']:
        tag = 'STABLE' if eq['stable'] else 'unstable'
        print(f'    Nash [{tag:>8s}]: {eq["fraction"]:.0%} formalize')
    print(f'    Coordination gap: {gap["gap_fraction"]:.0%} welfare loss')

    # --- Key insight ---
    print(f'\n  {"=" * 40}')
    print('  THE DEEP INSIGHT')
    print(f'  {"=" * 40}')
    print('  Autoproving does not transform mathematics.')
    print('  It REVEALS what mathematics always was:')
    print('  an interpretive, meaning-making activity')
    print('  that happened to require proof construction')
    print('  as a byproduct. The proof was never the')
    print('  point. The understanding was.')
    print(f'\n{sep}\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUT.mkdir(exist_ok=True)
    params = Params()

    print('Autoformalization Dynamics')
    print('A computational model of the autoformalization ecosystem\n')

    print('Simulating baseline dynamics (200 years)...')
    t, states = simulate(params, t_span=(0, 200))

    print('\nGenerating visualizations:')
    fig1_dashboard(t, states, params)
    fig2_phase_portraits(t, states, params)
    fig3_bottleneck_waterfall(t, states)
    fig4_bifurcation()
    fig5_game_theory()
    fig6_scenarios()
    fig7_dependency_graph()

    print_analysis(params)

    print(f'All figures saved to: {OUT.resolve()}/')


if __name__ == '__main__':
    main()
