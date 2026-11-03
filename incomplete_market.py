#!/usr/bin/env python3
"""Incomplete-market CVaR experiment with representative favorable/unfavorable-path plot.

This script reconstructs the binding incomplete-market policy using the common
uncapped direct-exposure solver and generates the two-panel path figure used in
Section 5.2. The unfavorable path is chosen from the lower tail and is required
to end above the Merton benchmark in dollar risky exposure.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np


def load_solver(solver_path: Path):
    spec = importlib.util.spec_from_file_location("solver_core_uncapped", solver_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load solver from {solver_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def select_favorable_unfavorable_paths(wealth_paths, phi_paths, merton_value, target_final_phi=0.45):
    terminal_wealth = wealth_paths[:, -1]
    final_phi = phi_paths[:, -1]

    favorable_target = np.quantile(terminal_wealth, 0.90)
    favorable_idx = int(np.argmin(np.abs(terminal_wealth - favorable_target)))

    lower_tail_cutoff = np.quantile(terminal_wealth, 0.10)
    bad_pool = np.where((terminal_wealth <= lower_tail_cutoff) & (final_phi > merton_value + 1.0e-6))[0]
    if bad_pool.size == 0:
        raise RuntimeError('No lower-tail path found with final exposure above the Merton benchmark.')
    unfavorable_idx = int(bad_pool[np.argmin(np.abs(final_phi[bad_pool] - target_final_phi))])
    return favorable_idx, unfavorable_idx


def plot_good_bad_market_paths(solver, output_dir: Path, *, beta_perp=0.02, lambda_value=0.125237, eta_value=-0.94985,
                               training_paths=5000, training_seed=13, path_count=50000, simulation_seed=1900013):
    params = solver.ModelParams(
        beta_perp=beta_perp,
        cvar_limit=-0.94,
        samples=training_paths,
        seed=training_seed,
        cap_control_at_merton=False,
    )
    d_train = solver.generate_brownian_increments(params)
    wealth_grid, policy_grid, _ = solver.fit_hjb_policy(lambda_value, eta_value, d_train, params)

    sim_params = solver.replace(params, samples=path_count, seed=simulation_seed)
    d_sim = solver.generate_brownian_increments(sim_params)
    wealth, stock, phi_paths = solver.simulate_feedback_policy(wealth_grid, policy_grid, d_sim, sim_params)

    merton_value = 0.40
    favorable_idx, unfavorable_idx = select_favorable_unfavorable_paths(wealth, phi_paths, merton_value)

    t_stock = np.linspace(0.0, sim_params.horizon, sim_params.steps + 1)
    t_phi = t_stock[:-1]
    fig, (ax_stock, ax_phi) = plt.subplots(2, 1, figsize=(9.6, 6.8), sharex=True)

    for index, color, label in [
        (favorable_idx, "#16a34a", "favorable market"),
        (unfavorable_idx, "#dc2626", "unfavorable market"),
    ]:
        terminal_phi = float(phi_paths[index, -1])
        ax_stock.plot(
            t_stock,
            wealth[index],
            color=color,
            linewidth=2.0,
            label=f"{label}: $W_T^*$={wealth[index, -1]:.2f}",
        )
        ax_phi.plot(
            t_phi,
            phi_paths[index],
            color=color,
            linewidth=2.0,
            label=fr"{label}: $\\varphi_T^*$={terminal_phi:.2f}",
        )

    ax_phi.axhline(
        merton_value,
        color="0.35",
        linestyle="--",
        linewidth=1.4,
        label=f"Merton line = {merton_value:.2f}",
    )

    ax_stock.set_ylabel("wealth", fontsize=14)
    ax_phi.set_ylabel(r"dollar risky exposure", fontsize=14)
    ax_phi.set_xlabel("time", fontsize=14)
    ax_phi.set_xlim(0.0, sim_params.horizon)
    ax_stock.set_xlim(0.0, sim_params.horizon)
    ax_stock.grid(True, alpha=0.3)
    ax_phi.grid(True, alpha=0.3)
    ax_stock.legend(loc="upper left", frameon=True, fontsize=14)
    ax_phi.legend(loc="lower left", frameon=True, fontsize=14)
    ax_phi.set_ylim(min(-0.10, float(phi_paths[[favorable_idx, unfavorable_idx]].min()) - 0.02),
                    max(0.45, float(phi_paths[[favorable_idx, unfavorable_idx]].max()) + 0.02))
    ax_stock.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_phi.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_phi.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'incomplete_sample.png', dpi=220, bbox_inches='tight')
    fig.savefig(output_dir / 'incomplete_sample.pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Incomplete-market trajectory plot')
    parser.add_argument('--solver', type=Path, default=Path('solver_core_uncapped.py'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/incomplete/figures'))
    args = parser.parse_args()

    solver = load_solver(args.solver)
    plot_good_bad_market_paths(solver, args.output_dir)


if __name__ == '__main__':
    main()
