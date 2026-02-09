# Autoformalization Ecosystem Dynamics

A computational dynamical system modeling the co-evolution of formalization, autoproving, verification, and human interpretation in mathematics.

## The Model

Six coupled ODEs capture the feedback loops, race conditions, and phase transitions of the autoformalization ecosystem:

| Variable | Meaning | Key Dynamics |
|----------|---------|-------------|
| **F(t)** | Formalization rate | Saturating response to incentive, bounded by human capacity |
| **P(t)** | Autoproving power | Grows with knowledge base, saturates (diminishing returns) |
| **K(t)** | Formalized knowledge base | Accumulates from F, depreciates slowly |
| **H(t)** | Human interpretation capacity | Logistic growth, overwhelmed by unverified proofs |
| **V(t)** | Verification integrity | Restored naturally, degraded by reward hacking, reinforced by knowledge |
| **I(t)** | Incentive to formalize | Saturating response to prover power, natural decay |

Three load-bearing principles are encoded:

1. **Verification Axiom** — Trust in automated proofs reduces entirely to verification environment integrity
2. **Translation Gap** — The informal-to-formal mapping is irreducibly semantic and cannot be verified from within the formal system
3. **Bottleneck Migration** — As any pipeline stage is automated, the bottleneck migrates to the adjacent un-automated stage

## Key Findings

### 1. The Baseline Predicts a Golden Age Followed by Collapse

![Dashboard](output/1_dashboard.png)

The default trajectory shows a ~25-year formalization boom (F peaks at 3.2x), followed by a slow verification crisis that destroys human interpretation capacity.

### 2. Only One Strategy Survives: The Knowledge Fortress

![Scenarios](output/6_scenarios.png)

Five futures modeled. Only **Knowledge Fortress** (blue) — investing heavily in verification infrastructure that leverages the formalized knowledge base — produces sustainable dynamics. Verification stays above 0.95, human interpretation recovers, knowledge base grows to 600+.

### 3. The Critical Parameter is Knowledge-Verification Coupling

![Bifurcation](output/4_bifurcation.png)

The bifurcation parameter isn't verification resilience or formalization speed — it's **how strongly the formalized knowledge base reinforces verification** (mu). This creates the race condition: can K grow fast enough to offset hacking pressure from P?

### 4. The Bottleneck Migrates Through Four Phases

![Bottleneck](output/3_bottleneck_waterfall.png)

`F → P → I → H` — Human interpretation becomes the permanent binding constraint, confirming the Translation Gap principle.

### 5. Phase Portraits Reveal the Verification Spiral

![Phase Portraits](output/2_phase_portraits.png)

Flow fields in three 2D projections show the trajectory from present-day conditions (green dot) to steady state (red square).

### 6. The Coordination Game Has a Catastrophic Failure Mode

![Game Theory](output/5_game_theory.png)

Two stable Nash equilibria: 0% and 100% formalization, with a tipping threshold at 97%. The 100% welfare gap at the bad equilibrium quantifies the cost of coordination failure.

### 7. Causal Topology

![Dependency Graph](output/7_dependency_graph.png)

The three load-bearing principles (bold borders), their causal relationships, and two hidden contradictions identified through structural analysis.

## The Deep Insight

> Autoproving doesn't transform mathematics — it **reveals** what mathematics always was: an interpretive, meaning-making activity that happened to require proof construction as a byproduct. The proof was never the point. Understanding was.

The model proves this computationally: the permanent bottleneck is always H (human interpretation), regardless of how powerful autoproving becomes.

## Usage

```bash
pip install -r requirements.txt
python run.py
```

Outputs seven PNG figures to `output/` and prints quantitative analysis to console.

## Structure

```
model.py          # 6-variable coupled ODE system + fixed-point analysis
game.py           # Formalization coordination game + Nash equilibria
run.py            # Simulation engine + 7 visualizations
requirements.txt  # numpy, scipy, matplotlib
```

## Three Genuine First Principles

Everything in the model derives from three axioms:

1. **The Verification Axiom**: Trust in automated proofs reduces to trust in verification environments. If this fails, nothing works.

2. **The Translation Gap**: The informal→formal mapping is irreducibly semantic and cannot be verified from within the formal system. This is the permanent constraint.

3. **The Bottleneck Migration Theorem**: As any stage in the pipeline is automated, the bottleneck migrates to the adjacent un-automated stage. This is invariant regardless of which technology improves.

## License

MIT
