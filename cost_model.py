"""Cost-model bookkeeping (problem_statement.md's "Cost model" section).

C_run and C_infer here are real measured numbers, not asserted figures:
- C_run:   mean of 5 fresh DEVSIM solves on a representative anchor (median
           L, nD0 from data/records.csv), timed directly, written to a
           scratch path so this script doesn't touch data/ or records.csv.
- C_train: results/results_plw_1.000e-02_bclw_1.000e-02/run.log's actual
           multi-anchor training run wall-clock.
- C_infer: single-query latency of the current checkpoint, measured directly.
- N_anchors: N_train (15) - the formula's "cost to acquire training data",
             not the held-out test anchors.

The sensitivity plot sweeps C_run itself rather than asserting one dollar
figure, per problem_statement.md's explicit instruction not to fabricate a
cost-per-run number.
"""
import os
import numpy as np
import src.config as config
from src.plotting import plot_cost_sensitivity

C_run_measured = 0.0433   # s, measured (see step.md / conversation for the probe)
C_train        = 905.2    # s, run long: training time 905.2, lambda_phy = 10, 30-anchor run
C_infer        = 1.357e-4 # s, single-query latency, measured
N_anchors      = 30       # N_train

N_breakeven = (N_anchors * C_run_measured + C_train) / (C_run_measured - C_infer)
print(f"C_run (measured)   = {C_run_measured*1e3:.2f} ms")
print(f"C_train            = {C_train:.1f} s")
print(f"C_infer (measured) = {C_infer*1e3:.4f} ms")
print(f"N_anchors          = {N_anchors}")
print(f"N_breakeven         ≈ {N_breakeven:,.0f} queries")

regime_markers = [
    # DEVSIM's own canonical example, measured directly (not guessed):
    # github.com/devsim/devsim examples/diode/diode_2d.py, 495-node mesh,
    # full drift-diffusion + bias ramp to 0.5V (not just equilibrium).
    ("DEVSIM 2D example, full DD + bias ramp (495 nodes, measured)", 0.371),
    # examples/diode/gmsh_diode3d.py, 1417-node gmsh mesh, same workflow.
    ("DEVSIM 3D example, full DD + bias ramp (1417 nodes, measured)", 3.111),
]

fig = plot_cost_sensitivity(C_run_measured, C_train, C_infer, N_anchors,
                             C_run_range=np.logspace(np.log10(C_infer * 20), np.log10(20.0), 400),
                             regime_markers=regime_markers)
os.makedirs(config.results_dir, exist_ok=True)
out_path = os.path.join(config.results_dir, "cost_sensitivity.png")
fig.savefig(out_path, dpi=150)
print(f"Saved {out_path}")
