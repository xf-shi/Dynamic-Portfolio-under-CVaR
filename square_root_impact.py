#!/usr/bin/env python3
"""Square-root price-impact experiment.

Run: python square_root_impact.py
Optional: python square_root_impact.py --output-dir results/power

The complete parameter search precedes independent 50,000-path OOS evaluation.
The final OOS sample is evaluation only. No Merton exposure cap or final-OOS
feasibility adjustment is applied. Numerical state/action bounds are retained.
The experiment generates CSV/JSON results and PNG/PDF figures from scratch.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def run_experiment(output_dir=None, use_numba=True):
    """Solve, independently evaluate, and plot both CVaR-budget cases."""
    import matplotlib
    matplotlib.use('Agg')

    # Settings and output helpers
    MODEL_KIND = 'power'

    # Required packages: numpy and matplotlib. Numba is optional acceleration.
    # Install dependencies with: python -m pip install -r requirements.txt
    from pathlib import Path
    import csv
    import json
    import platform
    import sys
    import time

    USE_NUMBA = use_numba
    OUTPUT_DIR = (Path(output_dir).expanduser().resolve() if output_dir is not None
                  else Path.cwd() / 'results' / MODEL_KIND)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_STARTED = time.perf_counter()
    print('Python:', platform.python_version())
    print('Results directory:', OUTPUT_DIR.resolve())

    def _format_plot_precision(figure):
        """Use two decimal places for displayed numbers; preserve all plotted data."""
        import re as _plot_re
        from matplotlib.text import Text as _PlotText
        from matplotlib.ticker import FuncFormatter as _PlotFormatter

        def _number(value):
            rounded = round(float(value), 2)
            return f"{0.0 if rounded == 0.0 else rounded:.2f}"

        def _tick(value, position):
            return _number(value)

        for axis in figure.axes:
            axis.xaxis.set_major_formatter(_PlotFormatter(_tick))
            axis.yaxis.set_major_formatter(_PlotFormatter(_tick))
        pattern = _plot_re.compile(r"(?<![\w.])[-−]?\d+\.\d+(?![\w.])")
        for label in figure.findobj(match=_PlotText):
            label.set_text(pattern.sub(
                lambda match: _number(match.group().replace("−", "-")),
                label.get_text(),
            ))

    def save_figure(figure, filename, **kwargs):
        """Write matching PNG and vector PDF figures."""
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        _format_plot_precision(figure)
        _format_plot_precision(figure)
        figure.savefig(filename, **kwargs)
        if filename.suffix.lower() != '.pdf':
            _format_plot_precision(figure)
            figure.savefig(filename.with_suffix('.pdf'), **kwargs)


    # 1. Imports
    from dataclasses import replace
    import math
    from pathlib import Path

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 160,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
    })


    # 2. Numerical Solver
    IMPACT_Q = 1.5  # q = 3/2 in (Lambda/(q*S^(3/2))) |v|^(3/2)


    from dataclasses import dataclass, replace


    from html import escape


    from pathlib import Path


    import argparse


    import json


    import math


    import sys


    import time


    from typing import Optional


    import numpy as np


    @dataclass(frozen=True)
    class ModelParams:
        """Scalar model and numerical parameters."""

        horizon: float = 1.0
        steps: int = 200
        samples: int = 2_000
        seed: int = 7

        # Dollar-exposure linear-impact dynamics (legacy names: phi=x, u=v):
        # dS^0 = mu_t*S^0 dt + sigma_t*S^0 dB,
        # dx = v dt,
        # dW = (mu_t*x - (impact_lambda/q)*|v|^q)dt + sigma_t*x dB.
        w0: float = 1.0
        mu: float = 0.08
        sigma: float = 0.20
        phi0: float = 0.0
        impact_lambda: float = 0.10

        # Legacy deterministic market-shape parameters kept for CLI compatibility.
        # The current price-impact experiment uses constant mu_t = mu and
        # sigma_t = sigma, so these are set to zero and ignored below.
        alpha0: float = 0.00
        alpha_decay: float = 2.0
        drift_cycle_amp: float = 0.00
        drift_cycle_freq: float = 1.0
        drift_cycle_phase: float = 0.0
        vol_trend: float = 0.00
        vol_cycle_amp: float = 0.00
        vol_cycle_freq: float = 1.0
        vol_cycle_phase: float = 0.4
        vol_pulse_amp: float = 0.00
        vol_pulse_center: float = 0.65
        vol_pulse_width: float = 0.18
        vol_floor: float = 0.05

        # Kept for compatibility with older helper functions/CLI output.
        r: float = 0.00
        b: float = 0.00
        beta_perp: float = 0.02

        # Objective:
        # E int [0.5*gamma*(sigma*phi_t)^2 - mu*phi_t
        #        + (impact_lambda/q)*|u_t|^q] dt.
        # The terminal cost is g(W_T) = 0. q_w, q_phi, q_terminal, and
        # terminal_target are kept for CLI compatibility, but are not used by the
        # current f and g.
        gamma: float = 5.0
        q_w: float = 0.02
        q_phi: float = 0.05
        q_terminal: float = 1.0
        terminal_target: float = 1.25

        # Inventory state bounds phi in [phi_min, phi_max].
        phi_min: float = -2.0
        phi_max: float = 2.0
        # Trading-rate control bounds u = dot(phi) in [trade_min, trade_max].
        trade_min: float = -4.0
        trade_max: float = 4.0

        # CVaR_alpha(ell(W_T)) <= c.
        alpha: float = 0.95
        cvar_limit: float = -0.70

        # Optimization tolerances.
        control_max_iter: int = 90
        control_grad_tol: float = 2.0e-4
        eta_tol: float = 8.0e-4
        eta_subgrad_tol: float = 6.0e-4
        lambda_tol: float = 8.0e-4
        residual_tol: float = 1.0e-3
        eta_max_iter: int = 16
        lambda_max_iter: int = 18
        lambda_initial_high: float = 1.0
        lambda_high_cap: float = 1.0e5

        # Inner control solver. "value_iteration" gives a closed-loop Markov
        # feedback policy phi_n(W_n); "open_loop_pg" keeps the older projected
        # gradient solver over deterministic piecewise-constant controls.
        control_solver: str = "value_iteration"
        state_grid_size: int = 61
        inventory_grid_size: int = 31
        action_grid_size: int = 31
        action_dense_width: float = 0.75
        action_outer_count: int = 9
        bellman_shock_count: int = 3
        state_grid_padding: float = 0.25


    @dataclass
    class ControlResult:
        phi: np.ndarray
        eta: float
        objective: float
        lagrangian: float
        residual: float
        cvar: float
        var: float
        mean_terminal_wealth: float
        terminal_wealth_std: float
        loss_tail_probability: float
        iterations: int
        grad_norm: float
        policy_grid: Optional[np.ndarray] = None
        wealth_grid: Optional[np.ndarray] = None
        inventory_grid: Optional[np.ndarray] = None
        terminal_wealth: Optional[np.ndarray] = None
        loss: Optional[np.ndarray] = None
        phi_paths: Optional[np.ndarray] = None
        trade_rate: Optional[np.ndarray] = None
        trade_rate_paths: Optional[np.ndarray] = None


    def generate_brownian_increments(params: ModelParams) -> np.ndarray:
        rng = np.random.default_rng(params.seed)
        dt = params.horizon / params.steps
        return rng.normal(0.0, math.sqrt(dt), size=(params.samples, params.steps))


    def market_drift(step: int, params: ModelParams) -> float:
        """Constant stock drift mu_t = mu."""

        return params.mu


    def market_volatility(step: int, params: ModelParams) -> float:
        """Constant stock volatility sigma_t = sigma."""

        return max(params.vol_floor, params.sigma)


    def market_drift_grid(params: ModelParams) -> np.ndarray:
        return np.array([market_drift(n, params) for n in range(params.steps)])


    def market_volatility_grid(params: ModelParams) -> np.ndarray:
        return np.array([market_volatility(n, params) for n in range(params.steps)])


    def simulate_price_impact_paths(
        trade_rate: np.ndarray,
        d_brownian: np.ndarray,
        params: ModelParams,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Euler simulation for the price-impact model.

        ``trade_rate`` is the control u_n = dot(varphi_n). The risky position
        varphi_n is a state variable with initial value params.phi0.
        """

        samples, steps = d_brownian.shape
        dt = params.horizon / steps
        trade_rate = np.asarray(trade_rate, dtype=float)
        if trade_rate.ndim == 1:
            if trade_rate.shape[0] != steps:
                raise ValueError(f"Expected {steps} trade-rate steps, got {trade_rate.shape[0]}.")
            trade_rate_paths = np.broadcast_to(trade_rate, (samples, steps)).copy()
        elif trade_rate.shape == (samples, steps):
            trade_rate_paths = trade_rate.copy()
        else:
            raise ValueError(
                "trade_rate must have shape (steps,) or (samples, steps); "
                f"got {trade_rate.shape}."
            )
        trade_rate_paths = np.clip(trade_rate_paths, params.trade_min, params.trade_max)

        wealth = np.empty((samples, steps + 1))
        phi_paths = np.empty((samples, steps + 1))
        wealth[:, 0] = params.w0
        phi_paths[:, 0] = params.phi0

        for n in range(steps):
            d_b = d_brownian[:, n]
            phi_n = phi_paths[:, n]
            u_n = trade_rate_paths[:, n]
            mu_n = market_drift(n, params)
            sigma_n = market_volatility(n, params)
            wealth[:, n + 1] = (
                wealth[:, n]
                + (mu_n * phi_n - params.impact_lambda / IMPACT_Q * np.abs(u_n)**IMPACT_Q) * dt
                + sigma_n * phi_n * d_b
            )
            phi_paths[:, n + 1] = np.clip(
                phi_n + u_n * dt,
                params.phi_min,
                params.phi_max,
            )
        return wealth, phi_paths, trade_rate_paths


    def simulate_wealth(
        trade_rate: np.ndarray,
        d_brownian: np.ndarray,
        params: ModelParams,
    ) -> np.ndarray:
        """Return only the wealth paths for an open-loop trading-rate control."""

        wealth, _, _ = simulate_price_impact_paths(trade_rate, d_brownian, params)
        return wealth


    def cvar_from_loss(loss: np.ndarray, alpha: float) -> tuple[float, float]:
        """Return empirical Rockafellar-Uryasev CVaR and VaR."""

        var = float(np.quantile(loss, alpha))
        cvar = float(var + np.maximum(loss - var, 0.0).mean() / (1.0 - alpha))
        return cvar, var


    def objective_gradient_and_stats(
        phi: np.ndarray,
        eta: float,
        lambda_value: float,
        d_brownian: np.ndarray,
        params: ModelParams,
    ) -> tuple[float, float, float, float, np.ndarray, dict[str, float]]:
        """Evaluate L(phi, eta, lambda) and grad_phi L.

        This is the inner-kernel derivative.  It uses a backward adjoint recursion,
        not the full sensitivity tensor dW_t/dphi_k.  Cost per gradient call is
        O(samples * steps).
        """

        wealth = simulate_wealth(phi, d_brownian, params)
        dt = params.horizon / params.steps
        growth = 1.0 + params.r * dt

        terminal_wealth = wealth[:, -1]
        loss = -terminal_wealth
        hinge = np.maximum(loss - eta, 0.0)
        tail_indicator = (loss > eta).astype(float)

        running = dt * np.sum(
            0.5 * params.gamma * (params.sigma * phi) ** 2 - params.mu * phi
        )
        terminal = np.zeros_like(terminal_wealth)
        objective = float(np.mean(running + terminal))
        residual = float(eta + hinge.mean() / (1.0 - params.alpha) - params.cvar_limit)
        lagrangian = objective + lambda_value * residual

        adjoint = np.zeros_like(terminal_wealth)
        if lambda_value != 0.0:
            adjoint -= lambda_value * tail_indicator / (1.0 - params.alpha)

        grad = np.empty_like(phi)
        for n in range(params.steps - 1, -1, -1):
            d_w_next_d_phi = params.mu * dt + params.sigma * d_brownian[:, n]
            grad[n] = dt * (
                params.gamma * params.sigma**2 * phi[n] - params.mu
            ) + np.mean(adjoint * d_w_next_d_phi)
            adjoint = growth * adjoint

        cvar, var = cvar_from_loss(loss, params.alpha)
        stats = {
            "cvar": cvar,
            "var": var,
            "mean_terminal_wealth": float(np.mean(terminal_wealth)),
            "terminal_wealth_std": float(np.std(terminal_wealth)),
            "loss_tail_probability": float(np.mean(loss > eta)),
        }
        return objective, residual, lagrangian, float(np.linalg.norm(grad)), grad, stats


    def project_control(phi: np.ndarray, params: ModelParams) -> np.ndarray:
        return np.clip(phi, params.trade_min, params.trade_max)


    def projected_gradient_norm(phi: np.ndarray, grad: np.ndarray, params: ModelParams) -> float:
        return float(np.linalg.norm(phi - project_control(phi - grad, params)))


    def open_loop_control_kernel(
        lambda_value: float,
        eta: float,
        d_brownian: np.ndarray,
        params: ModelParams,
        *,
        initial_phi: Optional[np.ndarray] = None,
    ) -> ControlResult:
        """Older fixed-(lambda, eta) open-loop projected-gradient kernel.

        Solves min_{phi in A} L(phi, eta, lambda) for deterministic
        piecewise-constant open-loop controls.
        """

        raise NotImplementedError(
            "open_loop_pg was written for the older direct-position control model. "
            "Use control_solver='value_iteration' for the price-impact model where "
            "the control is u=dot(varphi) and varphi is a state."
        )

        if initial_phi is None:
            phi = np.zeros(params.steps)
        else:
            phi = project_control(initial_phi.astype(float).copy(), params)

        obj, residual, lag, grad_norm, grad, stats = objective_gradient_and_stats(
            phi, eta, lambda_value, d_brownian, params
        )
        stationarity = projected_gradient_norm(phi, grad, params)

        step_size = 0.5
        accepted_iter = 0
        for iteration in range(1, params.control_max_iter + 1):
            if stationarity <= params.control_grad_tol:
                accepted_iter = iteration - 1
                break

            accepted = False
            local_step = step_size
            for _ in range(12):
                candidate = project_control(phi - local_step * grad, params)
                cand_obj, cand_residual, cand_lag, cand_grad_norm, cand_grad, cand_stats = (
                    objective_gradient_and_stats(
                        candidate, eta, lambda_value, d_brownian, params
                    )
                )
                cand_stationarity = projected_gradient_norm(candidate, cand_grad, params)
                if cand_lag <= lag - 1.0e-4 * np.dot(grad, phi - candidate):
                    phi = candidate
                    obj = cand_obj
                    residual = cand_residual
                    lag = cand_lag
                    grad_norm = cand_grad_norm
                    grad = cand_grad
                    stationarity = cand_stationarity
                    stats = cand_stats
                    step_size = min(local_step * 1.15, 3.0)
                    accepted_iter = iteration
                    accepted = True
                    break
                local_step *= 0.5

            if not accepted:
                step_size *= 0.5
                if step_size < 1.0e-10:
                    accepted_iter = iteration
                    break

        terminal_wealth = simulate_wealth(phi, d_brownian, params)[:, -1]
        cvar, var = cvar_from_loss(-terminal_wealth, params.alpha)
        return ControlResult(
            phi=phi,
            eta=eta,
            objective=obj,
            lagrangian=lag,
            residual=residual,
            cvar=cvar,
            var=var,
            mean_terminal_wealth=stats["mean_terminal_wealth"],
            terminal_wealth_std=stats["terminal_wealth_std"],
            loss_tail_probability=stats["loss_tail_probability"],
            iterations=accepted_iter,
            grad_norm=grad_norm,
            terminal_wealth=terminal_wealth,
            loss=-terminal_wealth,
        )


    def running_cost(
        phi: np.ndarray | float,
        trade_rate: np.ndarray | float,
        params: ModelParams,
        step: int,
    ) -> np.ndarray | float:
        """Running cost f(varphi, u) for the price-impact model."""

        mu_t = market_drift(step, params)
        sigma_t = market_volatility(step, params)
        return (
            0.5 * params.gamma * (sigma_t * phi) ** 2
            - mu_t * phi
            + params.impact_lambda / IMPACT_Q * np.abs(trade_rate)**IMPACT_Q
        )


    def build_state_grids(d_brownian: np.ndarray, params: ModelParams) -> tuple[np.ndarray, np.ndarray]:
        """Choose wealth and inventory grids from exploratory trading-rate probes."""

        dt = params.horizon / params.steps
        mu_bar = float(np.mean(market_drift_grid(params)))
        sigma_bar = float(np.mean(market_volatility_grid(params)))
        merton_phi = mu_bar / max(params.gamma * sigma_bar**2, 1.0e-12)
        ramp_to_merton = float(np.clip((merton_phi - params.phi0) / params.horizon, params.trade_min, params.trade_max))
        probes = [
            np.zeros(params.steps),
            np.full(params.steps, params.trade_min),
            np.full(params.steps, params.trade_max),
            np.full(params.steps, ramp_to_merton),
        ]

        # Add two turn-around probes so the wealth grid sees both long and short
        # inventories without relying only on monotone trading.
        half = params.steps // 2
        buy_then_sell = np.full(params.steps, params.trade_min)
        buy_then_sell[:half] = params.trade_max
        sell_then_buy = -buy_then_sell
        probes.extend([buy_then_sell, sell_then_buy])

        wealth_paths = []
        phi_paths = []
        for trade_rate in probes:
            wealth, phi, _ = simulate_price_impact_paths(trade_rate, d_brownian, params)
            wealth_paths.append(wealth)
            phi_paths.append(phi)

        wealth = np.concatenate([path.reshape(-1) for path in wealth_paths])
        low = float(np.quantile(wealth, 0.002))
        high = float(np.quantile(wealth, 0.998))
        width = max(high - low, 1.0e-3)
        low -= params.state_grid_padding * width
        high += params.state_grid_padding * width

        phi_seen = np.concatenate([path.reshape(-1) for path in phi_paths])
        phi_low = min(params.phi_min, float(np.min(phi_seen)) - dt * abs(params.trade_min))
        phi_high = max(params.phi_max, float(np.max(phi_seen)) + dt * abs(params.trade_max))
        phi_low = max(phi_low, params.phi_min)
        phi_high = min(phi_high, params.phi_max)

        return (
            np.linspace(low, high, params.state_grid_size),
            np.linspace(phi_low, phi_high, params.inventory_grid_size),
        )


    def bellman_shocks(params: ModelParams) -> tuple[np.ndarray, np.ndarray]:
        """Gauss-Hermite quadrature nodes for dB over one time step."""

        nodes, weights = np.polynomial.hermite.hermgauss(params.bellman_shock_count)
        dt = params.horizon / params.steps
        d_b = math.sqrt(2.0 * dt) * nodes
        probs = weights / math.sqrt(math.pi)
        return d_b, probs


    def action_grid(params: ModelParams) -> np.ndarray:
        """Trading-rate grid, dense near zero and sparse in the far tails."""

        dense_width = min(
            max(params.action_dense_width, 0.0),
            max(abs(params.trade_min), abs(params.trade_max)),
        )
        dense_low = max(params.trade_min, -dense_width)
        dense_high = min(params.trade_max, dense_width)
        dense = np.linspace(dense_low, dense_high, params.action_grid_size)

        outer_count = max(0, params.action_outer_count)
        pieces = [dense, np.array([0.0, params.trade_min, params.trade_max])]
        if outer_count > 0 and params.trade_min < dense_low:
            pieces.append(np.linspace(params.trade_min, dense_low, outer_count))
        if outer_count > 0 and dense_high < params.trade_max:
            pieces.append(np.linspace(dense_high, params.trade_max, outer_count))

        grid = np.concatenate(pieces)
        grid = np.round(grid, 12)
        grid = np.sort(np.unique(grid))
        return grid[(grid >= params.trade_min - 1.0e-12) & (grid <= params.trade_max + 1.0e-12)]


    def interp_value_2d(
        wealth: np.ndarray,
        phi: np.ndarray,
        wealth_grid: np.ndarray,
        inventory_grid: np.ndarray,
        values: np.ndarray,
    ) -> np.ndarray:
        """Bilinear interpolation with boundary clamping."""

        wealth_clipped = np.clip(wealth, wealth_grid[0], wealth_grid[-1])
        phi_clipped = np.clip(phi, inventory_grid[0], inventory_grid[-1])

        wi = np.searchsorted(wealth_grid, wealth_clipped, side="right") - 1
        pi = np.searchsorted(inventory_grid, phi_clipped, side="right") - 1
        wi = np.clip(wi, 0, wealth_grid.size - 2)
        pi = np.clip(pi, 0, inventory_grid.size - 2)

        w0 = wealth_grid[wi]
        w1 = wealth_grid[wi + 1]
        p0 = inventory_grid[pi]
        p1 = inventory_grid[pi + 1]
        tw = np.divide(wealth_clipped - w0, w1 - w0, out=np.zeros_like(wealth_clipped, dtype=float), where=w1 != w0)
        tp = np.divide(phi_clipped - p0, p1 - p0, out=np.zeros_like(phi_clipped, dtype=float), where=p1 != p0)

        v00 = values[wi, pi]
        v10 = values[wi + 1, pi]
        v01 = values[wi, pi + 1]
        v11 = values[wi + 1, pi + 1]

        return (
            (1.0 - tw) * (1.0 - tp) * v00
            + tw * (1.0 - tp) * v10
            + (1.0 - tw) * tp * v01
            + tw * tp * v11
        )


    def fit_feedback_policy(
        lambda_value: float,
        eta: float,
        d_brownian: np.ndarray,
        params: ModelParams,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Finite-horizon value iteration for u_n = pi_n(W_n, varphi_n)."""

        wealth_grid, inventory_grid = build_state_grids(d_brownian, params)
        actions = action_grid(params)
        d_b_nodes, d_b_probs = bellman_shocks(params)
        dt = params.horizon / params.steps

        wealth_mesh, inventory_mesh = np.meshgrid(wealth_grid, inventory_grid, indexing="ij")
        terminal_value = lambda_value * np.maximum(-wealth_grid - eta, 0.0) / (1.0 - params.alpha)
        value_next = np.repeat(terminal_value[:, None], inventory_grid.size, axis=1)
        value_grid = np.empty((params.steps + 1, wealth_grid.size, inventory_grid.size))
        value_grid[-1] = value_next
        policy_grid = np.empty((params.steps, wealth_grid.size, inventory_grid.size))

        for n in range(params.steps - 1, -1, -1):
            mu_n = market_drift(n, params)
            sigma_n = market_volatility(n, params)
            q_values = np.empty((actions.size, wealth_grid.size, inventory_grid.size))
            for a_idx, trade_rate in enumerate(actions):
                next_phi = inventory_mesh + trade_rate * dt
                feasible = (
                    (next_phi >= params.phi_min - 1.0e-12)
                    & (next_phi <= params.phi_max + 1.0e-12)
                )
                expected_continuation = np.zeros_like(value_next)
                for d_b, prob in zip(d_b_nodes, d_b_probs):
                    next_w = (
                        wealth_mesh
                        + (mu_n * inventory_mesh - params.impact_lambda / IMPACT_Q * np.abs(trade_rate)**IMPACT_Q) * dt
                        + sigma_n * inventory_mesh * d_b
                    )
                    expected_continuation += prob * interp_value_2d(
                        next_w,
                        next_phi,
                        wealth_grid,
                        inventory_grid,
                        value_next,
                    )
                q = dt * running_cost(inventory_mesh, trade_rate, params, n) + expected_continuation
                q_values[a_idx] = np.where(feasible, q, np.inf)

            best_idx = np.argmin(q_values, axis=0)
            value_next = np.take_along_axis(q_values, best_idx[None, :, :], axis=0)[0]
            value_grid[n] = value_next
            policy_grid[n] = actions[best_idx]

        return wealth_grid, inventory_grid, policy_grid, value_grid


    def simulate_feedback_policy(
        wealth_grid: np.ndarray,
        inventory_grid: np.ndarray,
        policy_grid: np.ndarray,
        d_brownian: np.ndarray,
        params: ModelParams,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate W, varphi, and state-dependent trading rates u_n(W_n, varphi_n)."""

        samples, steps = d_brownian.shape
        dt = params.horizon / steps
        wealth = np.empty((samples, steps + 1))
        phi_paths = np.empty((samples, steps + 1))
        trade_rate_paths = np.empty((samples, steps))
        wealth[:, 0] = params.w0
        phi_paths[:, 0] = params.phi0

        for n in range(steps):
            phi_n = phi_paths[:, n]
            trade_rate_n = interp_value_2d(
                wealth[:, n],
                phi_n,
                wealth_grid,
                inventory_grid,
                policy_grid[n],
            )
            trade_rate_n = np.clip(trade_rate_n, params.trade_min, params.trade_max)
            trade_rate_paths[:, n] = trade_rate_n
            mu_n = market_drift(n, params)
            sigma_n = market_volatility(n, params)
            wealth[:, n + 1] = (
                wealth[:, n]
                + (mu_n * phi_n - params.impact_lambda / IMPACT_Q * np.abs(trade_rate_n)**IMPACT_Q) * dt
                + sigma_n * phi_n * d_brownian[:, n]
            )
            phi_paths[:, n + 1] = np.clip(
                phi_n + trade_rate_n * dt,
                params.phi_min,
                params.phi_max,
            )

        return wealth, phi_paths, trade_rate_paths


    def feedback_value_iteration_kernel(
        lambda_value: float,
        eta: float,
        d_brownian: np.ndarray,
        params: ModelParams,
    ) -> ControlResult:
        """Closed-loop inner kernel using finite-horizon value iteration.

        For fixed (lambda, eta), the Lagrangian subproblem is Markovian in
        (W_n, varphi_n). This routine approximates the Bellman recursion on a
        wealth/inventory grid and returns a feedback trading-rate policy
        u_n = pi_n(W_n, varphi_n), represented by policy_grid[n, w_index, phi_index].
        """

        wealth_grid, inventory_grid, policy_grid, _ = fit_feedback_policy(
            lambda_value, eta, d_brownian, params
        )
        wealth, phi_paths, trade_rate_paths = simulate_feedback_policy(
            wealth_grid, inventory_grid, policy_grid, d_brownian, params
        )

        dt = params.horizon / params.steps
        terminal_wealth = wealth[:, -1]
        loss = -terminal_wealth
        hinge = np.maximum(loss - eta, 0.0)
        phi_on_intervals = phi_paths[:, :-1]
        running = dt * np.sum(
            [
                running_cost(phi_on_intervals[:, n], trade_rate_paths[:, n], params, n)
                for n in range(params.steps)
            ],
            axis=0,
        )
        objective = float(np.mean(running))
        residual = float(eta + hinge.mean() / (1.0 - params.alpha) - params.cvar_limit)
        lagrangian = objective + lambda_value * residual
        cvar, var = cvar_from_loss(loss, params.alpha)

        return ControlResult(
            phi=np.mean(phi_on_intervals, axis=0),
            eta=eta,
            objective=objective,
            lagrangian=lagrangian,
            residual=residual,
            cvar=cvar,
            var=var,
            mean_terminal_wealth=float(np.mean(terminal_wealth)),
            terminal_wealth_std=float(np.std(terminal_wealth)),
            loss_tail_probability=float(np.mean(loss > eta)),
            iterations=params.steps,
            grad_norm=0.0,
            policy_grid=policy_grid,
            wealth_grid=wealth_grid,
            inventory_grid=inventory_grid,
            terminal_wealth=terminal_wealth,
            loss=loss,
            phi_paths=phi_paths,
            trade_rate=np.mean(trade_rate_paths, axis=0),
            trade_rate_paths=trade_rate_paths,
        )


    def inner_control_kernel(
        lambda_value: float,
        eta: float,
        d_brownian: np.ndarray,
        params: ModelParams,
        *,
        initial_phi: Optional[np.ndarray] = None,
    ) -> ControlResult:
        """Algorithm 0: fixed-(lambda, eta) control oracle.

        Default mode uses value iteration to produce a closed-loop Markov feedback
        trading-rate control u_n(W_n, varphi_n).
        """

        if params.control_solver == "open_loop_pg":
            return open_loop_control_kernel(lambda_value, eta, d_brownian, params, initial_phi=initial_phi)
        if params.control_solver != "value_iteration":
            raise ValueError(f"Unknown control_solver={params.control_solver!r}")
        return feedback_value_iteration_kernel(lambda_value, eta, d_brownian, params)


    def eta_bracket(params: ModelParams) -> tuple[float, float]:
        """Conservative compact eta interval for the sample problem."""

        max_phi = max(abs(params.phi_min), abs(params.phi_max), abs(params.phi0))
        max_trade = max(abs(params.trade_min), abs(params.trade_max))
        max_mu = float(np.max(np.abs(market_drift_grid(params))))
        max_sigma = float(np.max(market_volatility_grid(params)))
        drift_bound = (
            abs(params.w0)
            + params.horizon * (max_mu * max_phi + params.impact_lambda / IMPACT_Q * np.abs(max_trade)**IMPACT_Q)
        )
        diffusion_bound = 6.0 * math.sqrt(params.horizon) * max_sigma * max_phi
        bound = max(2.0, drift_bound + diffusion_bound)
        return -bound, bound


    def eta_bisection(
        lambda_value: float,
        d_brownian: np.ndarray,
        params: ModelParams,
        *,
        warm_start_phi: Optional[np.ndarray] = None,
    ) -> ControlResult:
        """Algorithm 1: bisection over eta for fixed lambda > 0."""

        eta_low, eta_high = eta_bracket(params)
        phi_guess = warm_start_phi
        best: Optional[ControlResult] = None

        for _ in range(params.eta_max_iter):
            eta_mid = 0.5 * (eta_low + eta_high)
            result = inner_control_kernel(
                lambda_value,
                eta_mid,
                d_brownian,
                params,
                initial_phi=phi_guess,
            )
            phi_guess = result.phi
            best = result

            subgrad = lambda_value * (
                1.0 - result.loss_tail_probability / (1.0 - params.alpha)
            )
            if abs(subgrad) <= params.eta_subgrad_tol or eta_high - eta_low <= params.eta_tol:
                return result
            if subgrad < 0.0:
                eta_low = eta_mid
            else:
                eta_high = eta_mid

        assert best is not None
        return best


    def unconstrained_solution(d_brownian: np.ndarray, params: ModelParams) -> ControlResult:
        result = inner_control_kernel(0.0, 0.0, d_brownian, params)
        if result.terminal_wealth is None:
            terminal_wealth = simulate_wealth(result.phi, d_brownian, params)[:, -1]
            loss = -terminal_wealth
        else:
            terminal_wealth = result.terminal_wealth
            loss = result.loss if result.loss is not None else -terminal_wealth
        cvar, var = cvar_from_loss(loss, params.alpha)
        residual = cvar - params.cvar_limit
        return replace(
            result,
            eta=var,
            residual=residual,
            cvar=cvar,
            var=var,
            loss_tail_probability=float(np.mean(loss > var)),
        )


    def outer_bisection(
        d_brownian: np.ndarray, params: ModelParams
    ) -> tuple[ControlResult, float, list[dict[str, float]]]:
        """Algorithm 2: outer bisection over lambda."""

        history: list[dict[str, float]] = []
        unconstrained = unconstrained_solution(d_brownian, params)
        history.append(
            {
                "lambda": 0.0,
                "eta": unconstrained.eta,
                "residual": unconstrained.residual,
                "objective": unconstrained.objective,
                "cvar": unconstrained.cvar,
            }
        )
        if unconstrained.residual <= params.residual_tol:
            return unconstrained, 0.0, history

        lambda_low = 0.0
        lambda_high = params.lambda_initial_high
        high_result = eta_bisection(
            lambda_high, d_brownian, params, warm_start_phi=unconstrained.phi
        )
        history.append(
            {
                "lambda": lambda_high,
                "eta": high_result.eta,
                "residual": high_result.residual,
                "objective": high_result.objective,
                "cvar": high_result.cvar,
            }
        )

        while high_result.residual > 0.0:
            lambda_high *= 2.0
            if lambda_high > params.lambda_high_cap:
                raise RuntimeError(
                    "Failed to bracket a feasible multiplier before lambda_high_cap. "
                    "The chosen CVaR limit may be infeasible for this model/control set."
                )
            high_result = eta_bisection(
                lambda_high, d_brownian, params, warm_start_phi=high_result.phi
            )
            history.append(
                {
                    "lambda": lambda_high,
                    "eta": high_result.eta,
                    "residual": high_result.residual,
                    "objective": high_result.objective,
                    "cvar": high_result.cvar,
                }
            )

        best_feasible = high_result
        best_lambda = lambda_high
        warm_phi = high_result.phi

        for _ in range(params.lambda_max_iter):
            lambda_mid = 0.5 * (lambda_low + lambda_high)
            mid_result = eta_bisection(
                lambda_mid, d_brownian, params, warm_start_phi=warm_phi
            )
            warm_phi = mid_result.phi
            history.append(
                {
                    "lambda": lambda_mid,
                    "eta": mid_result.eta,
                    "residual": mid_result.residual,
                    "objective": mid_result.objective,
                    "cvar": mid_result.cvar,
                }
            )

            if abs(mid_result.residual) <= params.residual_tol or (
                lambda_high - lambda_low <= params.lambda_tol
            ):
                return mid_result, lambda_mid, history
            if mid_result.residual > 0.0:
                lambda_low = lambda_mid
            else:
                lambda_high = lambda_mid
                best_feasible = mid_result
                best_lambda = lambda_mid

        return best_feasible, best_lambda, history


    CASES: dict[str, ModelParams] = {
        "loose": ModelParams(
            cvar_limit=-0.70,
            samples=300,
            steps=200,
            seed=11,
            control_max_iter=80,
            eta_max_iter=8,
            lambda_max_iter=6,
            state_grid_size=31,
            inventory_grid_size=17,
            action_grid_size=151,
            action_dense_width=0.75,
            action_outer_count=9,
            bellman_shock_count=3,
        ),
        "active": ModelParams(
            cvar_limit=-0.960,
            samples=300,
            steps=200,
            seed=13,
            control_max_iter=120,
            eta_max_iter=10,
            lambda_max_iter=8,
            residual_tol=1.0e-3,
            state_grid_size=31,
            inventory_grid_size=17,
            action_grid_size=151,
            action_dense_width=0.75,
            action_outer_count=9,
            bellman_shock_count=3,
        ),
        "active_very_tight": ModelParams(
            cvar_limit=-0.995,
            samples=300,
            steps=200,
            seed=13,
            control_max_iter=120,
            eta_max_iter=10,
            lambda_max_iter=8,
            residual_tol=1.0e-3,
            state_grid_size=31,
            inventory_grid_size=17,
            action_grid_size=151,
            action_dense_width=0.75,
            action_outer_count=9,
            bellman_shock_count=3,
        ),
        "active_high_gamma": ModelParams(
            cvar_limit=-0.985,
            samples=300,
            steps=200,
            seed=13,
            gamma=8.0,
            control_max_iter=120,
            eta_max_iter=10,
            lambda_max_iter=8,
            residual_tol=1.0e-3,
            state_grid_size=31,
            inventory_grid_size=17,
            action_grid_size=151,
            action_dense_width=0.75,
            action_outer_count=9,
            bellman_shock_count=3,
        ),
        "tight_control_bounds": ModelParams(
            cvar_limit=-0.995,
            samples=300,
            steps=200,
            seed=17,
            phi_min=-0.45,
            phi_max=0.45,
            control_max_iter=120,
            eta_max_iter=10,
            lambda_max_iter=8,
            residual_tol=1.0e-3,
            state_grid_size=31,
            inventory_grid_size=17,
            action_grid_size=151,
            action_dense_width=0.75,
            action_outer_count=9,
            bellman_shock_count=3,
        ),
    }


    def generate_perpendicular_brownian_increments(params: ModelParams) -> np.ndarray:
        rng = np.random.default_rng(params.seed + 1_000_003)
        dt = params.horizon / params.steps
        return rng.normal(0.0, math.sqrt(dt), size=(params.samples, params.steps))


    def simulate_price_impact_paths(
        trade_rate: np.ndarray,
        d_brownian: np.ndarray,
        params: ModelParams,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate wealth and dollar exposure with perpendicular endowment risk."""
        samples, steps = d_brownian.shape
        dt = params.horizon / steps
        trade_rate = np.asarray(trade_rate, dtype=float)
        if trade_rate.ndim == 1:
            if trade_rate.shape[0] != steps:
                raise ValueError(f"Expected {steps} exposure-rate steps, got {trade_rate.shape[0]}.")
            trade_rate_paths = np.broadcast_to(trade_rate, (samples, steps)).copy()
        elif trade_rate.shape == (samples, steps):
            trade_rate_paths = trade_rate.copy()
        else:
            raise ValueError(
                "trade_rate must have shape (steps,) or (samples, steps); "
                f"got {trade_rate.shape}."
            )
        trade_rate_paths = np.clip(trade_rate_paths, params.trade_min, params.trade_max)
        perp_params = replace(params, samples=samples, steps=steps)
        d_brownian_perp = generate_perpendicular_brownian_increments(perp_params)

        wealth = np.empty((samples, steps + 1))
        exposure_paths = np.empty((samples, steps + 1))
        wealth[:, 0] = params.w0
        exposure_paths[:, 0] = params.phi0
        for n in range(steps):
            x_n = exposure_paths[:, n]
            v_n = trade_rate_paths[:, n]
            mu_n = market_drift(n, params)
            sigma_n = market_volatility(n, params)
            wealth[:, n + 1] = (
                wealth[:, n]
                + (mu_n * x_n - params.impact_lambda / IMPACT_Q * np.abs(v_n)**IMPACT_Q) * dt
                + sigma_n * x_n * d_brownian[:, n]
                + params.beta_perp * d_brownian_perp[:, n]
            )
            exposure_paths[:, n + 1] = np.clip(
                x_n
                + (v_n + mu_n * x_n) * dt
                + sigma_n * x_n * d_brownian[:, n],
                params.phi_min,
                params.phi_max,
            )
        return wealth, exposure_paths, trade_rate_paths


    def bellman_shocks(params: ModelParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Product Gauss-Hermite quadrature for independent dB and dB_perp."""
        nodes, weights = np.polynomial.hermite.hermgauss(params.bellman_shock_count)
        dt = params.horizon / params.steps
        one_d = math.sqrt(2.0 * dt) * nodes
        one_p = weights / math.sqrt(math.pi)
        d_b, d_b_perp = np.meshgrid(one_d, one_d, indexing="ij")
        probs = np.outer(one_p, one_p)
        return d_b.ravel(), d_b_perp.ravel(), probs.ravel()


    def fit_feedback_policy(
        lambda_value: float,
        eta: float,
        d_brownian: np.ndarray,
        params: ModelParams,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Value iteration in (wealth, dollar exposure) with endowment risk."""
        wealth_grid, inventory_grid = build_state_grids(d_brownian, params)
        actions = action_grid(params)
        d_b_nodes, d_b_perp_nodes, probs = bellman_shocks(params)
        dt = params.horizon / params.steps
        wealth_mesh, exposure_mesh = np.meshgrid(wealth_grid, inventory_grid, indexing="ij")
        terminal_value = lambda_value * np.maximum(-wealth_grid - eta, 0.0) / (1.0 - params.alpha)
        value_next = np.repeat(terminal_value[:, None], inventory_grid.size, axis=1)
        value_grid = np.empty((params.steps + 1, wealth_grid.size, inventory_grid.size))
        value_grid[-1] = value_next
        policy_grid = np.empty((params.steps, wealth_grid.size, inventory_grid.size))

        for n in range(params.steps - 1, -1, -1):
            mu_n = market_drift(n, params)
            sigma_n = market_volatility(n, params)
            q_values = np.empty((actions.size, wealth_grid.size, inventory_grid.size))
            for a_idx, exposure_rate in enumerate(actions):
                next_x = exposure_mesh + exposure_rate * dt
                feasible = (
                    (next_x >= params.phi_min - 1.0e-12)
                    & (next_x <= params.phi_max + 1.0e-12)
                )
                expected = np.zeros_like(value_next)
                for d_b, d_b_perp, prob in zip(d_b_nodes, d_b_perp_nodes, probs):
                    next_w = (
                        wealth_mesh
                        + (mu_n * exposure_mesh - params.impact_lambda / IMPACT_Q * np.abs(exposure_rate)**IMPACT_Q) * dt
                        + sigma_n * exposure_mesh * d_b
                        + params.beta_perp * d_b_perp
                    )
                    expected += prob * interp_value_2d(
                        next_w, next_x, wealth_grid, inventory_grid, value_next
                    )
                q = dt * running_cost(exposure_mesh, exposure_rate, params, n) + expected
                q_values[a_idx] = np.where(feasible, q, np.inf)
            best_idx = np.argmin(q_values, axis=0)
            value_next = np.take_along_axis(q_values, best_idx[None, :, :], axis=0)[0]
            value_grid[n] = value_next
            policy_grid[n] = actions[best_idx]
        return wealth_grid, inventory_grid, policy_grid, value_grid


    def simulate_feedback_policy(
        wealth_grid: np.ndarray,
        inventory_grid: np.ndarray,
        policy_grid: np.ndarray,
        d_brownian: np.ndarray,
        params: ModelParams,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate feedback exposure control with nontraded endowment shocks."""
        samples, steps = d_brownian.shape
        dt = params.horizon / steps
        perp_params = replace(params, samples=samples, steps=steps)
        d_brownian_perp = generate_perpendicular_brownian_increments(perp_params)
        wealth = np.empty((samples, steps + 1))
        exposure_paths = np.empty((samples, steps + 1))
        exposure_rate_paths = np.empty((samples, steps))
        wealth[:, 0] = params.w0
        exposure_paths[:, 0] = params.phi0
        for n in range(steps):
            x_n = exposure_paths[:, n]
            v_n = interp_value_2d(
                wealth[:, n], x_n, wealth_grid, inventory_grid, policy_grid[n]
            )
            v_n = np.clip(v_n, params.trade_min, params.trade_max)
            exposure_rate_paths[:, n] = v_n
            mu_n = market_drift(n, params)
            sigma_n = market_volatility(n, params)
            wealth[:, n + 1] = (
                wealth[:, n]
                + (mu_n * x_n - params.impact_lambda / IMPACT_Q * np.abs(v_n)**IMPACT_Q) * dt
                + sigma_n * x_n * d_brownian[:, n]
                + params.beta_perp * d_brownian_perp[:, n]
            )
            exposure_paths[:, n + 1] = np.clip(
                x_n
                + (v_n + mu_n * x_n) * dt
                + sigma_n * x_n * d_brownian[:, n],
                params.phi_min,
                params.phi_max,
            )
        return wealth, exposure_paths, exposure_rate_paths


    def eta_bracket(params: ModelParams) -> tuple[float, float]:
        max_x = max(abs(params.phi_min), abs(params.phi_max), abs(params.phi0))
        max_rate = max(abs(params.trade_min), abs(params.trade_max))
        max_mu = float(np.max(np.abs(market_drift_grid(params))))
        max_sigma = float(np.max(market_volatility_grid(params)))
        drift_bound = (
            abs(params.w0)
            + params.horizon * (max_mu * max_x + params.impact_lambda / IMPACT_Q * np.abs(max_rate)**IMPACT_Q)
        )
        diffusion_bound = 6.0 * math.sqrt(params.horizon) * (
            max_sigma * max_x + abs(params.beta_perp)
        )
        bound = max(2.0, drift_bound + diffusion_bound)
        return -bound, bound


    def outer_bisection(d_brownian, params):
        """Calibrate lambda against realized empirical CVaR, with minimum slack.

        The original routine bisected using the auxiliary eta/hinge residual. On a
        finite Monte Carlo sample, an approximate eta can make that residual differ
        materially from the empirical CVaR displayed in the figures. This version
        uses cvar_from_loss(result.loss) for every bracket decision and returns the
        feasible iterate closest to the requested limit.
        """
        history = []

        def evaluate(lambda_value, warm_start=None):
            result = eta_bisection(
                lambda_value, d_brownian, params, warm_start_phi=warm_start
            ) if lambda_value > 0.0 else unconstrained_solution(d_brownian, params)
            empirical_cvar, empirical_var = cvar_from_loss(result.loss, params.alpha)
            empirical_residual = float(empirical_cvar - params.cvar_limit)
            result = replace(
                result,
                cvar=float(empirical_cvar),
                var=float(empirical_var),
                residual=empirical_residual,
            )
            history.append({
                "lambda": float(lambda_value), "eta": float(result.eta),
                "residual": empirical_residual, "objective": float(result.objective),
                "cvar": float(empirical_cvar),
            })
            return result

        low_lambda = 0.0
        low = evaluate(low_lambda)
        if low.residual <= 0.0:
            return low, low_lambda, history

        high_lambda = params.lambda_initial_high
        high = evaluate(high_lambda, low.phi)
        while high.residual > 0.0:
            low_lambda, low = high_lambda, high
            high_lambda *= 2.0
            if high_lambda > params.lambda_high_cap:
                raise RuntimeError("Could not bracket an empirically feasible CVaR policy.")
            high = evaluate(high_lambda, low.phi)

        best_feasible, best_lambda = high, high_lambda
        best_slack = -high.residual
        for _ in range(params.lambda_max_iter):
            mid_lambda = 0.5 * (low_lambda + high_lambda)
            mid = evaluate(mid_lambda, best_feasible.phi)
            if mid.residual <= 0.0:
                slack = -mid.residual
                if slack < best_slack:
                    best_feasible, best_lambda, best_slack = mid, mid_lambda, slack
                high_lambda, high = mid_lambda, mid
            else:
                low_lambda, low = mid_lambda, mid
            if best_slack <= params.residual_tol:
                break
            if high_lambda - low_lambda <= params.lambda_tol:
                break
        return best_feasible, best_lambda, history


    def fit_hjb_policy(
        lambda_value: float,
        eta: float,
        d_brownian: np.ndarray,
        params: ModelParams,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Solve the (wealth, dollar-exposure) HJB PDE for fixed (lambda, eta).

        Unlike the older Bellman kernel, this discretizes the continuous-time
        generator directly and does not use shock quadrature or an action grid.
        For q=3/2 the trading-rate Hamiltonian has an analytic interior minimizer.
        """
        wealth_grid, exposure_grid = build_state_grids(d_brownian, params)
        dt = params.horizon / params.steps
        wealth_size = wealth_grid.size
        exposure_size = exposure_grid.size

        terminal_penalty = (
            lambda_value
            * np.maximum(-wealth_grid - eta, 0.0)
            / (1.0 - params.alpha)
        )
        value_next = np.repeat(terminal_penalty[:, None], exposure_size, axis=1)
        value_grid = np.empty((params.steps + 1, wealth_size, exposure_size))
        value_grid[-1] = value_next
        policy_grid = np.empty((params.steps, wealth_size, exposure_size))

        exposure_states = exposure_grid[None, :]
        edge_order = 2 if min(wealth_size, exposure_size) >= 3 else 1

        def derivatives(value):
            value_w = np.gradient(value, wealth_grid, axis=0, edge_order=edge_order)
            value_x = np.gradient(value, exposure_grid, axis=1, edge_order=edge_order)
            value_ww = np.gradient(value_w, wealth_grid, axis=0, edge_order=edge_order)
            value_xx = np.gradient(value_x, exposure_grid, axis=1, edge_order=edge_order)
            value_wx = np.gradient(value_w, exposure_grid, axis=1, edge_order=edge_order)
            return value_w, value_x, value_ww, value_xx, value_wx

        def hamiltonian(
            rate, mu_n, sigma_n, value_w, value_x, value_ww, value_xx, value_wx
        ):
            execution_cost = impact_cost(rate, params)
            running = (
                0.5 * params.gamma * (sigma_n * exposure_states) ** 2
                - mu_n * exposure_states
                + execution_cost
            )
            wealth_drift = mu_n * exposure_states - execution_cost
            exposure_drift = rate + mu_n * exposure_states
            common_variance = (sigma_n * exposure_states) ** 2
            return (
                running
                + wealth_drift * value_w
                + exposure_drift * value_x
                + 0.5 * common_variance * value_ww
                + 0.5 * common_variance * value_xx
                + common_variance * value_wx
            )

        def minimize_hamiltonian(value, n):
            value_w, value_x, value_ww, value_xx, value_wx = derivatives(value)
            mu_n = float(market_drift(n, params))
            sigma_n = float(market_volatility(n, params))

            # d/dv [(Lambda/q)|v|^q (1-V_w) + v V_x] = 0.
            curvature = params.impact_lambda * (1.0 - value_w)
            convex = curvature > 1.0e-10
            magnitude = np.where(
                convex,
                (np.abs(value_x) / np.maximum(curvature, 1.0e-14))
                ** (1.0 / (IMPACT_Q - 1.0)),
                0.0,
            )
            interior = np.clip(
                -np.sign(value_x) * magnitude,
                params.trade_min,
                params.trade_max,
            )

            best_rate = interior
            best_value = hamiltonian(
                interior, mu_n, sigma_n, value_w, value_x, value_ww, value_xx, value_wx
            )
            lower = np.full_like(interior, params.trade_min)
            upper = np.full_like(interior, params.trade_max)
            lower_value = hamiltonian(
                lower, mu_n, sigma_n, value_w, value_x, value_ww, value_xx, value_wx
            )
            upper_value = hamiltonian(
                upper, mu_n, sigma_n, value_w, value_x, value_ww, value_xx, value_wx
            )
            use_lower = lower_value < best_value
            best_rate = np.where(use_lower, lower, best_rate)
            best_value = np.where(use_lower, lower_value, best_value)
            use_upper = upper_value < best_value
            best_rate = np.where(use_upper, upper, best_rate)
            best_value = np.where(use_upper, upper_value, best_value)
            return best_rate, best_value

        def stable_substeps(n):
            dw = float(np.min(np.diff(wealth_grid)))
            dx = float(np.min(np.diff(exposure_grid)))
            sigma_n = abs(float(market_volatility(n, params)))
            max_x = max(abs(params.phi_min), abs(params.phi_max))
            max_v = max(abs(params.trade_min), abs(params.trade_max))
            common_variance = (sigma_n * max_x) ** 2
            diffusion_rate = common_variance * (
                1.0 / max(dw**2, 1.0e-14)
                + 1.0 / max(dx**2, 1.0e-14)
                + 2.0 / max(dw * dx, 1.0e-14)
            )
            transport_rate = (
                max_v + abs(float(market_drift(n, params))) * max_x
            ) / max(dx, 1.0e-14)
            return max(1, int(math.ceil(dt * (diffusion_rate + transport_rate) / 0.45)))

        for n in range(params.steps - 1, -1, -1):
            value = value_next
            substeps = stable_substeps(n)
            sub_dt = dt / substeps
            best_rate = np.zeros((wealth_size, exposure_size))
            for _ in range(substeps):
                best_rate, minimized_generator = minimize_hamiltonian(value, n)
                value = value + sub_dt * minimized_generator
                if not np.all(np.isfinite(value)):
                    raise FloatingPointError(
                        "The explicit HJB step became unstable; refine the state grids."
                    )
            value_next = value
            value_grid[n] = value
            policy_grid[n] = best_rate

        return wealth_grid, exposure_grid, policy_grid, value_grid


    fit_feedback_policy = fit_hjb_policy


    def eta_bisection(
        lambda_value: float,
        d_brownian: np.ndarray,
        params: ModelParams,
        *,
        warm_start_phi: Optional[np.ndarray] = None,
    ) -> ControlResult:
        """Minimize the fixed-lambda Lagrangian over eta by golden-section search."""
        eta_low, eta_high = eta_bracket(params)
        ratio = 0.5 * (math.sqrt(5.0) - 1.0)
        cache = {}

        def evaluate(eta):
            key = float(eta)
            if key not in cache:
                cache[key] = inner_control_kernel(
                    lambda_value,
                    key,
                    d_brownian,
                    params,
                    initial_phi=warm_start_phi,
                )
            return cache[key]

        eta_left = eta_high - ratio * (eta_high - eta_low)
        eta_right = eta_low + ratio * (eta_high - eta_low)
        left_result = evaluate(eta_left)
        right_result = evaluate(eta_right)
        for _ in range(max(params.eta_max_iter, 32)):
            if eta_high - eta_low <= min(params.eta_tol, 1.0e-5):
                break
            if left_result.lagrangian <= right_result.lagrangian:
                eta_high = eta_right
                eta_right, right_result = eta_left, left_result
                eta_left = eta_high - ratio * (eta_high - eta_low)
                left_result = evaluate(eta_left)
            else:
                eta_low = eta_left
                eta_left, left_result = eta_right, right_result
                eta_right = eta_low + ratio * (eta_high - eta_low)
                right_result = evaluate(eta_right)
        return min(cache.values(), key=lambda result: result.lagrangian)


    def price_scaled_impact_cost(rate, stock, params):
        stock_safe = np.maximum(np.asarray(stock, dtype=float), 1.0e-8)
        return (
            params.impact_lambda
            / (IMPACT_Q * stock_safe**IMPACT_Q)
            * np.abs(rate) ** IMPACT_Q
        )


    def build_stock_state_grid(params):
        """Positive stock grid covering approximately five diffusion standard deviations."""
        stock_size = 11
        max_mu = float(np.max(np.abs(market_drift_grid(params))))
        max_sigma = float(np.max(market_volatility_grid(params)))
        log_width = 5.0 * max_sigma * math.sqrt(params.horizon)
        log_center = (max_mu - 0.5 * max_sigma**2) * params.horizon
        return np.exp(np.linspace(log_center - log_width, log_center + log_width, stock_size))


    def interp_value_3d(w, x, s, wealth_grid, exposure_grid, stock_grid, values):
        """Vectorized trilinear interpolation on the (W,x,S) HJB grid."""
        w = np.clip(np.asarray(w), wealth_grid[0], wealth_grid[-1])
        x = np.clip(np.asarray(x), exposure_grid[0], exposure_grid[-1])
        s = np.clip(np.asarray(s), stock_grid[0], stock_grid[-1])
        iw = np.clip(np.searchsorted(wealth_grid, w) - 1, 0, wealth_grid.size - 2)
        ix = np.clip(np.searchsorted(exposure_grid, x) - 1, 0, exposure_grid.size - 2)
        is_ = np.clip(np.searchsorted(stock_grid, s) - 1, 0, stock_grid.size - 2)
        aw = (w - wealth_grid[iw]) / np.maximum(wealth_grid[iw + 1] - wealth_grid[iw], 1e-14)
        ax = (x - exposure_grid[ix]) / np.maximum(exposure_grid[ix + 1] - exposure_grid[ix], 1e-14)
        ass = (s - stock_grid[is_]) / np.maximum(stock_grid[is_ + 1] - stock_grid[is_], 1e-14)
        out = np.zeros(np.broadcast(w, x, s).shape, dtype=float)
        for dw_i in (0, 1):
            for dx_i in (0, 1):
                for ds_i in (0, 1):
                    weight = (
                        (aw if dw_i else 1.0 - aw)
                        * (ax if dx_i else 1.0 - ax)
                        * (ass if ds_i else 1.0 - ass)
                    )
                    out += weight * values[iw + dw_i, ix + dx_i, is_ + ds_i]
        return out


    def fit_hjb_policy(lambda_value, eta, d_brownian, params):
        """Finite-difference HJB solver in (W,x,S) with common market noise."""
        wealth_grid, exposure_grid = build_state_grids(d_brownian, params)
        stock_grid = build_stock_state_grid(params)
        dt = params.horizon / params.steps
        nw, nx, ns = wealth_grid.size, exposure_grid.size, stock_grid.size
        terminal = lambda_value * np.maximum(-wealth_grid - eta, 0.0) / (1.0 - params.alpha)
        value_next = np.broadcast_to(terminal[:, None, None], (nw, nx, ns)).copy()
        policy_grid = np.empty((params.steps, nw, nx, ns), dtype=float)
        x_states = exposure_grid[None, :, None]
        s_states = stock_grid[None, None, :]
        edge_order = 2 if min(nw, nx, ns) >= 3 else 1

        def derivatives(value):
            vw = np.gradient(value, wealth_grid, axis=0, edge_order=edge_order)
            vx = np.gradient(value, exposure_grid, axis=1, edge_order=edge_order)
            vs = np.gradient(value, stock_grid, axis=2, edge_order=edge_order)
            vww = np.gradient(vw, wealth_grid, axis=0, edge_order=edge_order)
            vxx = np.gradient(vx, exposure_grid, axis=1, edge_order=edge_order)
            vss = np.gradient(vs, stock_grid, axis=2, edge_order=edge_order)
            vwx = np.gradient(vw, exposure_grid, axis=1, edge_order=edge_order)
            vws = np.gradient(vw, stock_grid, axis=2, edge_order=edge_order)
            vxs = np.gradient(vx, stock_grid, axis=2, edge_order=edge_order)
            return vw, vx, vs, vww, vxx, vss, vwx, vws, vxs

        def hamiltonian(rate, mu_n, sigma_n, derivs):
            vw, vx, vs, vww, vxx, vss, vwx, vws, vxs = derivs
            cost = price_scaled_impact_cost(rate, s_states, params)
            diffusion_wx = sigma_n * x_states
            diffusion_s = sigma_n * s_states
            running = 0.5 * params.gamma * diffusion_wx**2 - mu_n * x_states + cost
            return (
                running
                + (mu_n * x_states - cost) * vw
                + (rate + mu_n * x_states) * vx
                + mu_n * s_states * vs
                + 0.5 * diffusion_wx**2 * (vww + vxx + 2.0 * vwx)
                + 0.5 * diffusion_s**2 * vss
                + diffusion_wx * diffusion_s * (vws + vxs)
            )

        def minimize(value, n):
            derivs = derivatives(value)
            vw, vx = derivs[0], derivs[1]
            mu_n = float(market_drift(n, params))
            sigma_n = float(market_volatility(n, params))
            curvature = params.impact_lambda * (1.0 - vw) / np.maximum(
                s_states**IMPACT_Q, 1.0e-14
            )
            convex = curvature > 1.0e-10
            magnitude = np.where(
                convex,
                (np.abs(vx) / np.maximum(curvature, 1.0e-14))
                ** (1.0 / (IMPACT_Q - 1.0)),
                0.0,
            )
            interior = np.clip(-np.sign(vx) * magnitude, params.trade_min, params.trade_max)
            best_rate = interior
            best_value = hamiltonian(interior, mu_n, sigma_n, derivs)
            for boundary in (params.trade_min, params.trade_max):
                candidate = np.full_like(interior, boundary)
                candidate_value = hamiltonian(candidate, mu_n, sigma_n, derivs)
                use = candidate_value < best_value
                best_rate = np.where(use, candidate, best_rate)
                best_value = np.where(use, candidate_value, best_value)
            return best_rate, best_value

        def stable_substeps(n):
            dw = float(np.min(np.diff(wealth_grid)))
            dx = float(np.min(np.diff(exposure_grid)))
            ds = float(np.min(np.diff(stock_grid)))
            sigma_n = abs(float(market_volatility(n, params)))
            max_x = max(abs(params.phi_min), abs(params.phi_max))
            max_s = float(stock_grid[-1])
            max_v = max(abs(params.trade_min), abs(params.trade_max))
            a = sigma_n * max_x
            b = sigma_n * max_s
            diffusion_rate = (
                a*a / max(dw*dw, 1e-14)
                + a*a / max(dx*dx, 1e-14)
                + b*b / max(ds*ds, 1e-14)
                + 2*a*a / max(dw*dx, 1e-14)
                + 2*a*b / max(dw*ds, 1e-14)
                + 2*a*b / max(dx*ds, 1e-14)
            )
            transport_rate = (
                (max_v + abs(float(market_drift(n, params))) * max_x) / max(dx, 1e-14)
                + abs(float(market_drift(n, params))) * max_s / max(ds, 1e-14)
            )
            return max(1, int(math.ceil(dt * (diffusion_rate + transport_rate) / 0.35)))

        for n in range(params.steps - 1, -1, -1):
            value = value_next
            substeps = stable_substeps(n)
            sub_dt = dt / substeps
            best_rate = np.zeros_like(value)
            for _ in range(substeps):
                best_rate, generator = minimize(value, n)
                value = value + sub_dt * generator
                if not np.all(np.isfinite(value)):
                    raise FloatingPointError("The three-state explicit HJB step became unstable.")
            value_next = value
            policy_grid[n] = best_rate
        return wealth_grid, exposure_grid, policy_grid, None


    fit_feedback_policy = fit_hjb_policy


    def simulate_feedback_policy(
        wealth_grid, exposure_grid, policy_grid, d_brownian, params
    ):
        """Simulate the three-state feedback policy pi(t,W,x,S)."""
        samples, steps = d_brownian.shape
        dt = params.horizon / steps
        stock_grid = build_stock_state_grid(params)
        wealth = np.empty((samples, steps + 1))
        exposure = np.empty((samples, steps + 1))
        stock = np.empty((samples, steps + 1))
        rates = np.empty((samples, steps))
        wealth[:, 0] = params.w0
        exposure[:, 0] = params.phi0
        stock[:, 0] = 1.0
        for n in range(steps):
            x_n = exposure[:, n]
            s_n = stock[:, n]
            rate = interp_value_3d(
                wealth[:, n], x_n, s_n,
                wealth_grid, exposure_grid, stock_grid, policy_grid[n],
            )
            rate = np.clip(rate, params.trade_min, params.trade_max)
            rates[:, n] = rate
            mu_n = float(market_drift(n, params))
            sigma_n = float(market_volatility(n, params))
            d_b = d_brownian[:, n]
            cost = price_scaled_impact_cost(rate, s_n, params)
            wealth[:, n + 1] = (
                wealth[:, n] + (mu_n * x_n - cost) * dt + sigma_n * x_n * d_b
            )
            exposure[:, n + 1] = np.clip(
                x_n + (rate + mu_n * x_n) * dt + sigma_n * x_n * d_b,
                params.phi_min, params.phi_max,
            )
            stock[:, n + 1] = s_n * np.exp(
                (mu_n - 0.5 * sigma_n**2) * dt + sigma_n * d_b
            )
        return wealth, exposure, rates


    def fit_constant_impact_hjb_policy(lambda_value, eta, d_brownian, params):
        wealth_grid, exposure_grid = build_state_grids(d_brownian, params)
        dt = params.horizon / params.steps
        nw, nx = wealth_grid.size, exposure_grid.size
        terminal = lambda_value * np.maximum(-wealth_grid - eta, 0.0) / (1.0 - params.alpha)
        value_next = np.broadcast_to(terminal[:, None], (nw, nx)).copy()
        policy_grid = np.empty((params.steps, nw, nx), dtype=np.float32)
        x_states = exposure_grid[None, :]
        edge_order = 2 if min(nw, nx) >= 3 else 1

        def derivatives(value):
            vw = np.gradient(value, wealth_grid, axis=0, edge_order=edge_order)
            vx = np.gradient(value, exposure_grid, axis=1, edge_order=edge_order)
            vww = np.gradient(vw, wealth_grid, axis=0, edge_order=edge_order)
            vxx = np.gradient(vx, exposure_grid, axis=1, edge_order=edge_order)
            vwx = np.gradient(vw, exposure_grid, axis=1, edge_order=edge_order)
            return vw, vx, vww, vxx, vwx

        def hamiltonian(rate, mu_n, sigma_n, derivs):
            vw, vx, vww, vxx, vwx = derivs
            cost = params.impact_lambda / IMPACT_Q * np.abs(rate) ** IMPACT_Q
            variance = (sigma_n * x_states) ** 2
            return (
                0.5 * params.gamma * variance - mu_n * x_states + cost
                + (mu_n * x_states - cost) * vw
                + (rate + mu_n * x_states) * vx
                + 0.5 * variance * (vww + vxx + 2.0 * vwx)
            )

        def minimize(value, n):
            derivs = derivatives(value)
            vw, vx = derivs[0], derivs[1]
            mu_n = float(market_drift(n, params))
            sigma_n = float(market_volatility(n, params))
            curvature = params.impact_lambda * (1.0 - vw)
            magnitude = np.where(
                curvature > 1.0e-10,
                (np.abs(vx) / np.maximum(curvature, 1.0e-14))
                ** (1.0 / (IMPACT_Q - 1.0)),
                0.0,
            )
            interior = np.clip(-np.sign(vx) * magnitude, params.trade_min, params.trade_max)
            best_rate = interior
            best_value = hamiltonian(interior, mu_n, sigma_n, derivs)
            for boundary in (params.trade_min, params.trade_max):
                candidate = np.full_like(interior, boundary)
                candidate_value = hamiltonian(candidate, mu_n, sigma_n, derivs)
                use = candidate_value < best_value
                best_rate = np.where(use, candidate, best_rate)
                best_value = np.where(use, candidate_value, best_value)
            return best_rate, best_value

        def stable_substeps(n):
            dw = float(np.min(np.diff(wealth_grid)))
            dx = float(np.min(np.diff(exposure_grid)))
            sigma_n = abs(float(market_volatility(n, params)))
            max_x = max(abs(params.phi_min), abs(params.phi_max))
            max_v = max(abs(params.trade_min), abs(params.trade_max))
            variance = (sigma_n * max_x) ** 2
            diffusion = variance * (
                1.0 / max(dw*dw, 1e-14)
                + 1.0 / max(dx*dx, 1e-14)
                + 2.0 / max(dw*dx, 1e-14)
            )
            transport = (
                max_v + abs(float(market_drift(n, params))) * max_x
            ) / max(dx, 1e-14)
            return max(1, int(math.ceil(dt * (diffusion + transport) / 0.45)))

        for n in range(params.steps - 1, -1, -1):
            value = value_next
            substeps = stable_substeps(n)
            sub_dt = dt / substeps
            best_rate = np.zeros_like(value)
            for _ in range(substeps):
                best_rate, generator = minimize(value, n)
                value = value + sub_dt * generator
                if not np.all(np.isfinite(value)):
                    raise FloatingPointError("The constant-impact HJB step became unstable.")
            value_next = value
            policy_grid[n] = best_rate
        return wealth_grid, exposure_grid, policy_grid, None


    def simulate_constant_impact_feedback(
        wealth_grid, exposure_grid, policy_grid, d_brownian, params
    ):
        samples, steps = d_brownian.shape
        dt = params.horizon / steps
        wealth = np.empty((samples, steps + 1))
        exposure = np.empty((samples, steps + 1))
        rates = np.empty((samples, steps))
        wealth[:, 0] = params.w0
        exposure[:, 0] = params.phi0
        for n in range(steps):
            x_n = exposure[:, n]
            rate = interp_value_2d(
                wealth[:, n], x_n, wealth_grid, exposure_grid, policy_grid[n]
            )
            rate = np.clip(rate, params.trade_min, params.trade_max)
            rates[:, n] = rate
            mu_n = float(market_drift(n, params))
            sigma_n = float(market_volatility(n, params))
            d_b = d_brownian[:, n]
            cost = params.impact_lambda / IMPACT_Q * np.abs(rate) ** IMPACT_Q
            wealth[:, n + 1] = (
                wealth[:, n] + (mu_n * x_n - cost) * dt + sigma_n * x_n * d_b
            )
            exposure[:, n + 1] = np.clip(
                x_n + (rate + mu_n * x_n) * dt + sigma_n * x_n * d_b,
                params.phi_min, params.phi_max,
            )
        return wealth, exposure, rates


    fit_feedback_policy = fit_constant_impact_hjb_policy


    simulate_feedback_policy = simulate_constant_impact_feedback


    # Optional acceleration
    # Optional JIT execution of the same finite-difference formulas and grids.
    # Fall back to the original NumPy solver if Numba is unavailable.
    try:
        if not USE_NUMBA:
            raise ImportError('Numba acceleration disabled')
        from numba import njit
        import numba
    except ImportError:
        print('Using the original NumPy solver; full searches will take longer.')
    else:
        from types import SimpleNamespace
        _solver_namespace = SimpleNamespace(**locals())
        def gradient_stencil(grid):
            n=len(grid)
            matrix=np.gradient(np.eye(n),grid,axis=0,edge_order=2)
            starts=np.maximum(0,np.minimum(np.arange(n)-1,n-3)).astype(np.int64)
            coefficients=np.array([[matrix[i,starts[i]+j] for j in range(3)] for i in range(n)])
            return starts,coefficients


        @njit(cache=False)
        def grad0(v,out,starts,coef):
            for i in range(v.shape[0]):
                k=starts[i]
                for j in range(v.shape[1]):
                    out[i,j]=coef[i,0]*v[k,j]+coef[i,1]*v[k+1,j]+coef[i,2]*v[k+2,j]


        @njit(cache=False)
        def grad1(v,out,starts,coef):
            for i in range(v.shape[0]):
                for j in range(v.shape[1]):
                    k=starts[j]
                    out[i,j]=coef[j,0]*v[i,k]+coef[j,1]*v[i,k+1]+coef[j,2]*v[i,k+2]


        @njit(cache=False)
        def solve_kernel(w,x,lam,eta,alpha,steps,dt,mu_t,sigma_t,gamma,coefficient,rate_min,rate_max,substeps,power_cost,sw,cw,sx,cx):
            nw,nx=len(w),len(x)
            value=np.empty((nw,nx))
            for i in range(nw):
                for j in range(nx):value[i,j]=lam*max(-w[i]-eta,0.0)/(1.0-alpha)
            vw=np.empty_like(value);vx=np.empty_like(value)
            vww=np.empty_like(value);vxx=np.empty_like(value);vwx=np.empty_like(value)
            policy=np.empty((steps,nw,nx))
            for n in range(steps-1,-1,-1):
                mu=mu_t[n];sigma=sigma_t[n]
                step_dt=dt/substeps[n]
                for ss in range(substeps[n]):
                    grad0(value,vw,sw,cw);grad1(value,vx,sx,cx)
                    grad0(vw,vww,sw,cw);grad1(vx,vxx,sx,cx);grad1(vw,vwx,sx,cx)
                    for i in range(nw):
                        for j in range(nx):
                            xx=x[j];variance=(sigma*xx)**2
                            if power_cost:
                                curvature=coefficient*(1.0-vw[i,j])
                                magnitude=(abs(vx[i,j])/max(curvature,1.e-14))**2 if curvature>1.e-10 else 0.0
                                sign=1.0 if vx[i,j]>0 else (-1.0 if vx[i,j]<0 else 0.0)
                                rate=min(rate_max,max(rate_min,-sign*magnitude))
                                cost=coefficient/1.5*abs(rate)**1.5
                                generator=(.5*gamma*variance-mu*xx+cost+(mu*xx-cost)*vw[i,j]+(rate+mu*xx)*vx[i,j]+.5*variance*(vww[i,j]+vxx[i,j]+2.0*vwx[i,j]))
                                for boundary in (rate_min,rate_max):
                                    cost_b=coefficient/1.5*abs(boundary)**1.5
                                    candidate=(.5*gamma*variance-mu*xx+cost_b+(mu*xx-cost_b)*vw[i,j]+(boundary+mu*xx)*vx[i,j]+.5*variance*(vww[i,j]+vxx[i,j]+2.0*vwx[i,j]))
                                    if candidate<generator:rate=boundary;generator=candidate
                            else:
                                rate=min(rate_max,max(rate_min,-vx[i,j]/coefficient))
                                generator=(.5*gamma*variance+.5*coefficient*rate**2-mu*xx+mu*xx*vw[i,j]+(rate+mu*xx)*vx[i,j]+.5*variance*(vww[i,j]+2.0*vwx[i,j]+vxx[i,j]))
                            value[i,j]=value[i,j]+step_dt*generator
                            if not math.isfinite(value[i,j]):raise ValueError('Nonfinite finite-difference update')
                            if ss==substeps[n]-1:policy[n,i,j]=rate
            return policy


        def power_solver(module):
            grids={}
            def solve(lam,eta,paths,p):
                key=(id(paths),repr(p))
                if key not in grids:grids[key]=module.build_state_grids(paths,p)
                w,x=grids[key]
                dt=p.horizon/p.steps;dw=float(np.min(np.diff(w)));dx=float(np.min(np.diff(x)))
                maximum=max(abs(p.phi_min),abs(p.phi_max));max_rate=max(abs(p.trade_min),abs(p.trade_max))
                mu=np.array([module.market_drift(n,p) for n in range(p.steps)])
                sigma=np.array([module.market_volatility(n,p) for n in range(p.steps)])
                subs=[]
                for n in range(p.steps):
                    variance=(abs(sigma[n])*maximum)**2
                    diffusion=variance*(1/max(dw*dw,1e-14)+1/max(dx*dx,1e-14)+2/max(dw*dx,1e-14))
                    transport=(max_rate+abs(mu[n])*maximum)/max(dx,1e-14)
                    subs.append(max(1,int(math.ceil(dt*(diffusion+transport)/.45))))
                sw,cw=gradient_stencil(w);sx,cx=gradient_stencil(x)
                policy=solve_kernel(w,x,lam,eta,p.alpha,p.steps,dt,mu,sigma,p.gamma,p.impact_lambda,p.trade_min,p.trade_max,np.array(subs),True,sw,cw,sx,cx)
                return w,x,policy.astype(np.float32),None
            return solve


        fit_feedback_policy = power_solver(_solver_namespace)
        print('Numba:', numba.__version__, '(same grids and finite-difference updates)')


    # 3. Settings
    BASE_PARAMS = replace(
        CASES["active"],
        # Moderate settings for Colab.  Increase samples/grids after the plots work.
        samples=1_000,
        steps=200,
        seed=13,
        beta_perp=0.0,
        impact_lambda=0.004472135955,
        state_grid_size=80,
        inventory_grid_size=100,
        # Very fine local trading-rate grid: spacing 0.01 on [-0.75, 0.75].
        action_grid_size=151,
        action_dense_width=0.75,
        action_outer_count=9,
        # Choose the active target to match the calibrated active policy closely.
        cvar_limit=-0.940,
        residual_tol=1.0e-6,
        lambda_tol=1.0e-7,
        lambda_max_iter=32,
        eta_tol=1.0e-4,
        eta_max_iter=22,
    )

    RUN_FULL_OUTER_SEARCH = True
    FRICTIONLESS_MERTON_X = BASE_PARAMS.mu / (BASE_PARAMS.gamma * BASE_PARAMS.sigma**2)
    # Backward-compatible alias used by a few plotting helpers.
    FRICTIONLESS_MERTON_PHI = FRICTIONLESS_MERTON_X
    actions = action_grid(BASE_PARAMS)
    local_actions = actions[(actions >= -BASE_PARAMS.action_dense_width) & (actions <= BASE_PARAMS.action_dense_width)]

    print(f"steps                 : {BASE_PARAMS.steps}")
    print(f"samples               : {BASE_PARAMS.samples}")
    print(f"mu, sigma             : {BASE_PARAMS.mu:.4f}, {BASE_PARAMS.sigma:.4f}")
    print(f"impact Lambda         : {BASE_PARAMS.impact_lambda:.4f}")
    print("impact scaling        : constant Lambda")
    print("HJB state             : (W_t, x_t)")
    print(f"impact exponent q     : {IMPACT_Q:.1f}")
    print(f"perp endowment beta   : {BASE_PARAMS.beta_perp:.4f}")
    print(f"Merton dollar exposure: {FRICTIONLESS_MERTON_X:.4f}")
    print(f"number of u actions   : {len(actions)}")
    print(f"local u-grid spacing  : {np.min(np.diff(local_actions)):.4f}")
    print(f"CVaR limit c          : {BASE_PARAMS.cvar_limit:.4f}")

    OOS_SAMPLES = 50_000
    OOS_SEED_SHIFT = 900_000
    print(f"final-only OOS samples: {OOS_SAMPLES:,}")


    # 4. Run The Price-Impact Cases
    import sys as _sys_trace
    import time as _time_trace
    _search_counter = 0
    _search_started = _time_trace.perf_counter()
    def _log_raw_search(lam, eta, cvar):
        nonlocal _search_counter
        _search_counter += 1
        _message = f'RAW_SEARCH query={_search_counter}, lambda={lam:.10f}, eta={eta:.10f}, search_CVaR={cvar:.10f}, seconds={_time_trace.perf_counter()-_search_started:.1f}'
        print(_message, flush=True)


    _original_eta_search=eta_bisection
    def eta_bisection(lam,paths,params,*,warm_start_phi=None):
        result=_original_eta_search(lam,paths,params,warm_start_phi=warm_start_phi)
        _log_raw_search(lam,result.eta,result.cvar)
        return result
    def lower_tail_wealth_cvar(values, alpha):
        return -cvar_from_loss(-np.asarray(values, dtype=float), alpha)[0]


    def cvar_path_from_wealth(wealth, alpha):
        return np.array([lower_tail_wealth_cvar(wealth[:, i], alpha) for i in range(wealth.shape[1])])


    def stock_paths_from_brownian(d_brownian, params, s0=1.0):
        """Exact GBM paths for dS_t^0 = mu S_t^0 dt + sigma S_t^0 dB_t."""
        samples, steps = d_brownian.shape
        stock = np.empty((samples, steps + 1), dtype=float)
        stock[:, 0] = s0
        dt = params.horizon / steps
        for n in range(steps):
            mu_n = market_drift(n, params)
            sigma_n = market_volatility(n, params)
            stock[:, n + 1] = stock[:, n] * np.exp(
                (mu_n - 0.5 * sigma_n**2) * dt + sigma_n * d_brownian[:, n]
            )
        return stock


    def summarize_case(name, wealth, dollar_exposure_paths, exposure_rate_paths, stock_paths, lambda_star, eta_star, params):
        terminal = wealth[:, -1]
        loss_cvar, loss_var = cvar_from_loss(-terminal, params.alpha)
        share_paths = dollar_exposure_paths / np.maximum(stock_paths, 1.0e-12)
        return {
            "name": name,
            "wealth": wealth,
            "stock_paths": stock_paths,
            # Legacy solver names phi and u now represent x and v respectively.
            "phi_paths": dollar_exposure_paths,
            "dollar_exposure_paths": dollar_exposure_paths,
            "dollar_position_paths": dollar_exposure_paths,
            "share_paths": share_paths,
            "trade_rate_paths": exposure_rate_paths,
            "lambda_star": float(lambda_star),
            "eta_star": float(eta_star),
            "cvar_limit": float(params.cvar_limit),
            "loss_cvar": float(loss_cvar),
            "lower_tail_cvar": float(-loss_cvar),
            "loss_var": float(loss_var),
            "wealth_var": float(-loss_var),
            "mean_terminal_wealth": float(np.mean(terminal)),
            "terminal_wealth_std": float(np.std(terminal)),
            "mean_phi": float(np.mean(share_paths)),
            "max_phi": float(np.max(share_paths)),
            "mean_dollar_position": float(np.mean(dollar_exposure_paths)),
            "max_dollar_position": float(np.max(dollar_exposure_paths)),
            "mean_u": float(np.mean(exposure_rate_paths)),
            "max_abs_u": float(np.max(np.abs(exposure_rate_paths))),
            "cvar_path": cvar_path_from_wealth(wealth, params.alpha),
        }


    params = BASE_PARAMS
    d_brownian = generate_brownian_increments(params)
    stock_paths = stock_paths_from_brownian(d_brownian, params)

    # Inactive CVaR case: its c=-0.86 constraint is slack, so lambda*=0.
    inactive_params = replace(params, cvar_limit=-0.86)
    inactive_result = unconstrained_solution(d_brownian, inactive_params)
    inactive_wealth, inactive_phi_paths, inactive_u_paths = simulate_feedback_policy(
        inactive_result.wealth_grid,
        inactive_result.inventory_grid,
        inactive_result.policy_grid,
        d_brownian,
        inactive_params,
    )
    inactive_case = summarize_case(
        "inactive",
        inactive_wealth,
        inactive_phi_paths,
        inactive_u_paths,
        stock_paths,
        0.0,
        inactive_result.eta,
        inactive_params,
    )

    # Active CVaR feedback policy.
    active_result, lambda_star, outer_history = outer_bisection(d_brownian, params)
    eta_star = active_result.eta

    active_wealth, active_phi_paths, active_u_paths = simulate_feedback_policy(
        active_result.wealth_grid,
        active_result.inventory_grid,
        active_result.policy_grid,
        d_brownian,
        params,
    )
    active_case = summarize_case(
        "active",
        active_wealth,
        active_phi_paths,
        active_u_paths,
        stock_paths,
        lambda_star,
        eta_star,
        params,
    )

    cases = [inactive_case, active_case]

    print("case       lambda*     loss CVaR     wealth CVaR   E[W_T]    std[W_T]   mean $pos  max $pos   mean |u|")
    print("-" * 108)
    for row in cases:
        print(
            f"{row['name']:<10s} "
            f"{row['lambda_star']:>8.5f} "
            f"{row['loss_cvar']:>12.6f} "
            f"{row['lower_tail_cvar']:>12.6f} "
            f"{row['mean_terminal_wealth']:>9.6f} "
            f"{row['terminal_wealth_std']:>9.6f} "
            f"{row['mean_dollar_position']:>9.6f} "
            f"{row['max_dollar_position']:>9.6f} "
            f"{np.mean(np.abs(row['trade_rate_paths'])):>9.6f}"
        )

    print()
    print(f"active residual CVaR(loss)-c : {active_case['loss_cvar'] - params.cvar_limit:.6e}")



    search_metrics=[dict(case='inactive',lambda_value=0.0,eta=inactive_result.eta,cvar=inactive_result.cvar,residual=inactive_result.cvar-inactive_params.cvar_limit),dict(case='active',lambda_value=lambda_star,eta=active_result.eta,cvar=active_result.cvar,residual=active_result.cvar-params.cvar_limit)]
    for case_name,result,lam in [('inactive',inactive_result,0.0),('active',active_result,lambda_star)]:
        np.savez_compressed(OUTPUT_DIR / f'frozen_{case_name}_policy.npz',wealth_grid=result.wealth_grid,exposure_grid=result.inventory_grid,policy_grid=result.policy_grid,lambda_value=lam,eta=result.eta)

    # Final-only out-of-sample validation. These paths are never used by either
    # the inner golden eta search or the outer lambda bisection.
    oos_params = replace(
        params,
        samples=OOS_SAMPLES,
        seed=params.seed + OOS_SEED_SHIFT,
    )
    d_brownian_oos = generate_brownian_increments(oos_params)

    for case, result in ((inactive_case, inactive_result), (active_case, active_result)):
        wealth_oos, exposure_oos, control_oos = simulate_feedback_policy(
            result.wealth_grid,
            result.inventory_grid,
            result.policy_grid,
            d_brownian_oos,
            oos_params,
        )
        loss_cvar_oos, loss_var_oos = cvar_from_loss(-wealth_oos[:, -1], params.alpha)
        case["wealth_oos"] = wealth_oos
        case["exposure_oos"] = exposure_oos
        case["control_oos"] = control_oos
        case["loss_cvar_oos"] = float(loss_cvar_oos)
        case["wealth_cvar_oos"] = float(-loss_cvar_oos)
        case["loss_var_oos"] = float(loss_var_oos)
        case["mean_terminal_wealth_oos"] = float(np.mean(wealth_oos[:, -1]))
        case["terminal_wealth_std_oos"] = float(np.std(wealth_oos[:, -1]))
        case["mean_exposure_oos"] = float(np.mean(exposure_oos))
        case["mean_abs_control_oos"] = float(np.mean(np.abs(control_oos)))
        case["cvar_path_oos"] = cvar_path_from_wealth(wealth_oos, params.alpha)
        case["oos_residual"] = float(loss_cvar_oos - case["cvar_limit"])

    print()
    print("FINAL OUT-OF-SAMPLE VALIDATION (not used in optimization)")
    print("case       lambda*     limit c      loss CVaR    residual       E[W_T]    std[W_T]   mean x     mean |v|")
    print("-" * 116)
    for case in cases:
        print(
            f"{case['name']:<10s} "
            f"{case['lambda_star']:>8.5f} "
            f"{case['cvar_limit']:>12.6f} "
            f"{case['loss_cvar_oos']:>12.6f} "
            f"{case['oos_residual']:>12.6e} "
            f"{case['mean_terminal_wealth_oos']:>10.6f} "
            f"{case['terminal_wealth_std_oos']:>10.6f} "
            f"{case['mean_exposure_oos']:>10.6f} "
            f"{case['mean_abs_control_oos']:>10.6f}"
        )

    # Use the final OOS sample as the canonical data source for all tables
    # and distribution/exposure figures below. Preserve search metrics under
    # explicit *_search keys for diagnostics.
    for case in cases:
        case["wealth_search"] = case["wealth"]
        case["exposure_search"] = case["dollar_exposure_paths"]
        case["control_search"] = case["trade_rate_paths"]
        case["loss_cvar_search"] = case["loss_cvar"]
        case["mean_terminal_wealth_search"] = case["mean_terminal_wealth"]
        case["wealth"] = case["wealth_oos"]
        case["dollar_exposure_paths"] = case["exposure_oos"]
        case["dollar_position_paths"] = case["exposure_oos"]
        case["trade_rate_paths"] = case["control_oos"]
        case["loss_cvar"] = case["loss_cvar_oos"]
        case["lower_tail_cvar"] = case["wealth_cvar_oos"]
        case["loss_var"] = case["loss_var_oos"]
        case["wealth_var"] = -case["loss_var_oos"]
        case["mean_terminal_wealth"] = case["mean_terminal_wealth_oos"]
        case["terminal_wealth_std"] = case["terminal_wealth_std_oos"]
        case["mean_dollar_position"] = case["mean_exposure_oos"]
        case["cvar_path"] = case["cvar_path_oos"]


    # 5. Shared Plot Helpers
    CASE_COLORS = {
        "inactive": "#2563eb",
        "active": "#dc2626",
    }

    CASE_LABELS = {
        "inactive": "nonbinding",
        "active": "c=-0.94, raw search",
    }
    FIGURE_DIR=OUTPUT_DIR / 'figures'
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    MERTON_COLOR = "#16a34a"
    TARGET_COLOR = "#374151"


    def gaussian_kde_1d(samples, grid, bandwidth=None):
        samples = np.asarray(samples, dtype=float)
        grid = np.asarray(grid, dtype=float)
        n = samples.size
        std = float(np.std(samples, ddof=1))
        if bandwidth is None:
            bandwidth = 1.06 * std * n ** (-1.0 / 5.0)
        bandwidth = max(float(bandwidth), 1.0e-6)
        z = (grid[:, None] - samples[None, :]) / bandwidth
        return np.exp(-0.5 * z * z).mean(axis=1) / (bandwidth * math.sqrt(2.0 * math.pi))


    def terminal_axis(cases, pad_fraction=0.10):
        all_terminal = np.concatenate([case["wealth"][:, -1] for case in cases])
        lo = float(np.quantile(all_terminal, 0.002))
        hi = float(np.quantile(all_terminal, 0.998))
        width = max(hi - lo, 1.0e-4)
        return lo - pad_fraction * width, hi + pad_fraction * width


    def case_summary_label(case):
        return (
            f"{case['name']}: lambda*={case['lambda_star']:.2f}, "
            f"E[W_T]={case['mean_terminal_wealth']:.2f}, "
            f"wealth tail CVaR={case['lower_tail_cvar']:.2f}, "
            f"floor={-case['cvar_limit']:.2f}"
        )


    def legend_below(ax, ncol=2):
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=ncol, fontsize=8)


    PLOT_SMOOTH_WINDOW = 9


    def smooth_1d(values, window=PLOT_SMOOTH_WINDOW):
        """Centered moving-average smoother for display only."""
        values = np.asarray(values, dtype=float)
        window = int(window)
        if window <= 1 or values.size < 3:
            return values.copy()
        if window % 2 == 0:
            window += 1
        window = min(window, values.size if values.size % 2 == 1 else values.size - 1)
        if window <= 1:
            return values.copy()
        pad = window // 2
        padded = np.pad(values, (pad, pad), mode="edge")
        kernel = np.ones(window, dtype=float) / window
        smoothed = np.convolve(padded, kernel, mode="valid")
        smoothed[0] = values[0]
        smoothed[-1] = values[-1]
        return smoothed


    def smooth_paths_for_plot(paths, window=PLOT_SMOOTH_WINDOW):
        return np.apply_along_axis(smooth_1d, 1, np.asarray(paths, dtype=float), window)


    time_grid = np.linspace(0.0, params.horizon, params.steps + 1)
    control_time_grid = time_grid[:-1]
    merton_inventory = FRICTIONLESS_MERTON_PHI
    merton_dollar_line = np.full_like(time_grid, FRICTIONLESS_MERTON_PHI, dtype=float)


    # Plot 1. Terminal Wealth Distribution
    def plot_terminal_wealth_distribution(cases, bins_count=44, figsize=(9.6, 5.6), save_path=None):
        x_low, x_high = terminal_axis(cases)
        x_grid = np.linspace(x_low, x_high, 500)
        bins = np.linspace(x_low, x_high, bins_count + 1)

        fig, ax = plt.subplots(figsize=figsize)
        for case in cases:
            color = CASE_COLORS.get(case["name"], "black")
            label = CASE_LABELS.get(case["name"], case["name"])
            terminal = case["wealth"][:, -1]
            density = gaussian_kde_1d(terminal, x_grid)

            ax.hist(
                terminal,
                bins=bins,
                density=True,
                alpha=0.12,
                color=color,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.plot(x_grid, density, color=color, linewidth=2.2, label=f"{label} KDE")

            mean_w = float(np.mean(terminal))
            wealth_var = case["wealth_var"]
            wealth_tail_cvar = case["lower_tail_cvar"]
            floor = -case["cvar_limit"]
            ax.axvline(mean_w, color=color, linewidth=1.1, alpha=0.75)
            ax.axvline(wealth_var, color=color, linewidth=1.1, linestyle=(0, (1, 3)))
            ax.axvline(wealth_tail_cvar, color=color, linewidth=1.1, linestyle=(0, (4, 2, 1, 2)))
            ax.axvline(floor, color=color, linewidth=1.1, linestyle=(0, (5, 3)), alpha=0.8)

        legend_handles = [
            Line2D([0], [0], color=CASE_COLORS["inactive"], lw=2.2, label="nonbinding KDE"),
            Line2D([0], [0], color=CASE_COLORS["inactive"], lw=1.1, linestyle=(0, (4, 2, 1, 2)), label=f"nonbinding worst-tail mean wealth = {cases[0]['lower_tail_cvar']:.2f}"),
            Line2D([0], [0], color=CASE_COLORS["inactive"], lw=1.1, linestyle="--", label=f"nonbinding mean = {cases[0]['mean_terminal_wealth']:.2f}"),
            Line2D([0], [0], color=CASE_COLORS["active"], lw=2.2, label="binding KDE"),
            Line2D([0], [0], color=CASE_COLORS["active"], lw=1.1, linestyle=(0, (4, 2, 1, 2)), label=f"binding worst-tail mean wealth = {cases[1]['lower_tail_cvar']:.2f}"),
            Line2D([0], [0], color=CASE_COLORS["active"], lw=1.1, linestyle="--", label=f"binding mean = {cases[1]['mean_terminal_wealth']:.2f}"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", ncol=1, fontsize=10, frameon=True, framealpha=0.92, edgecolor="0.85")

        # ax.set_title("Terminal wealth distribution under corrected GBM dollar exposure control", loc="left", pad=42)
        ax.set_xlabel(r"terminal wealth $W_T^*$", fontsize=16)
        ax.set_ylabel("density", fontsize=16)
        ax.set_xlim(0.6,1.5)
        ax.set_ylim(0.0, 11.0)

        # summary = "    ".join(case_summary_label(case) for case in cases)
        # ax.text(0.0, -0.19, summary, transform=ax.transAxes, fontsize=8, va="top")
        fig.tight_layout()
        if save_path is not None:
            save_figure(fig, save_path, bbox_inches="tight")
        return fig, ax


    fig_terminal, ax_terminal = plot_terminal_wealth_distribution(
        cases, save_path=FIGURE_DIR / "power_w_dist.png"
    )


    # Plot 2. Dollar Risky Exposure Trajectory
    def plot_exposure_trajectory(cases, figsize=(9.6, 5.2), include_band=True, include_sample=False, save_path=None):
        from matplotlib.patches import Patch

        fig, ax = plt.subplots(figsize=figsize)
        t = np.linspace(0.0, BASE_PARAMS.horizon, BASE_PARAMS.steps + 1)

        for case in cases:
            color = CASE_COLORS.get(case["name"], "black")
            label = CASE_LABELS.get(case["name"], case["name"])
            phi_paths = case["dollar_exposure_paths"]
            mean_phi = np.mean(phi_paths, axis=0)
            q10, q90 = np.quantile(phi_paths, [0.10, 0.90], axis=0)

            if include_band:
                ax.fill_between(t, q10, q90, color=color, alpha=0.10, linewidth=0)
            if include_sample:
                ax.plot(t, phi_paths[0], color=color, linewidth=0.9, alpha=0.32)
            ax.plot(t, mean_phi, color=color, linewidth=2.2, label=f"{label} risky dollar amount ")

        # ax.plot(merton_t, merton_line, color="0.30", linewidth=1.6, linestyle="--", label="Merton dollar exposure")
        ax.axhline(FRICTIONLESS_MERTON_X, color="0.35", linewidth=1.4, linestyle="--", label=f"Merton line = {FRICTIONLESS_MERTON_X:.2f}")
        ax.axhline(0.0, color="0.45", linewidth=0.8, alpha=0.7)
        ax.set_xlim(0.0, BASE_PARAMS.horizon)
        ax.set_ylim(0.0, 0.6)
        ax.set_xlabel("time", fontsize=16)
        ax.set_ylabel(r"dollar risky exposure", fontsize=16)
        handles = ax.get_legend_handles_labels()[0]
        if include_band:
            handles = handles + [
                Patch(
                    facecolor=CASE_COLORS.get(case["name"], "black"),
                    alpha=0.10,
                    edgecolor="none",
                    label=f"{CASE_LABELS.get(case['name'], case['name'])} 10-90% range",
                )
                for case in cases
            ]
        ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.92, edgecolor="0.85", fontsize=13)
        fig.tight_layout()
        if save_path is not None:
            save_figure(fig, save_path, bbox_inches="tight")
        return fig, ax


    fig_exposure, ax_exposure = plot_exposure_trajectory(
        cases, save_path=FIGURE_DIR / "power_exposure.png"
    )


    # Plot 3. Lower-Tail CVaR Of Wealth Over Time
    def plot_wealth_cvar_over_time(cases, figsize=(9.6, 5.2), save_path=None):
        """Plot CVaR using only the final 50,000-path OOS validation sample."""
        fig, ax = plt.subplots(figsize=figsize)
        for case in cases:
            name = case["name"]
            color = CASE_COLORS.get(name, "black")
            label = CASE_LABELS.get(name, name)
            wealth_cvar = case["cvar_path_oos"]
            terminal_gap = abs(-wealth_cvar[-1] - case["loss_cvar"])
            if terminal_gap > 1.0e-12:
                raise AssertionError(
                    f"{name}: table/figure terminal CVaR mismatch "
                    f"({terminal_gap:.3e})"
                )
            t = np.linspace(0.0, params.horizon, wealth_cvar.size)
            ax.plot(t, -wealth_cvar, color=color, linewidth=2.2,
                    label=f"{label} CVaR over time")
            ax.axhline(case["cvar_limit"], color=color, linewidth=1.1,
                       linestyle=(0, (5, 3)), alpha=0.75,
                       label=f"{label} constraint level")
        ax.set_xlabel("time", fontsize=16)
        ax.set_ylabel(r"CVaR$(-W_t^*)$", fontsize=13)
        ax.set_xlim(0.0, BASE_PARAMS.horizon)
        ax.set_ylim(-1.0, -0.84)
        ax.legend(loc="upper left", frameon=True, framealpha=0.92,
                  edgecolor="0.85", fontsize=16)
        fig.tight_layout()
        if save_path is not None:
            save_figure(fig, save_path, bbox_inches="tight")
        return fig, ax


    fig_cvar_time, ax_cvar_time = plot_wealth_cvar_over_time(
        cases, save_path=FIGURE_DIR / "power_cvar.png"
    )


    # Plot 4. Good And Bad Stock Paths
    def select_good_bad_market_paths(stock_paths, exposure_paths, merton_value, maturity_window=10):
        terminal_stock = stock_paths[:, -1]
        late_exposure = exposure_paths[:, -maturity_window:].mean(axis=1)

        good_candidates = np.where(terminal_stock >= np.quantile(terminal_stock, 0.75))[0]
        if good_candidates.size == 0:
            good_candidates = np.arange(terminal_stock.size)
        good_idx = good_candidates[
            np.argmin(np.abs(late_exposure[good_candidates] - merton_value))
        ]

        bad_candidates = np.where(terminal_stock <= np.quantile(terminal_stock, 0.25))[0]
        if bad_candidates.size == 0:
            bad_candidates = np.arange(terminal_stock.size)
        bad_idx = bad_candidates[np.argmin(late_exposure[bad_candidates])]
        if bad_idx == good_idx:
            order = bad_candidates[np.argsort(late_exposure[bad_candidates])]
            bad_idx = order[1] if order.size > 1 else bad_idx
        return int(good_idx), int(bad_idx)


    def plot_good_bad_market_paths(
        active_case,
        path_count=8000,
        maturity_window=10,
        figsize=(9.6, 6.2),
        save_path=None,
    ):
        plot_params = replace(
            params,
            samples=path_count,
            seed=params.seed + 250000,
        )
        d_brownian_plot = generate_brownian_increments(plot_params)
        wealth_paths, exposure_paths, control_paths = simulate_feedback_policy(
            active_result.wealth_grid,
            active_result.inventory_grid,
            active_result.policy_grid,
            d_brownian_plot,
            plot_params,
        )
        stock_paths = stock_paths_from_brownian(d_brownian_plot, plot_params)
        merton_value = float(FRICTIONLESS_MERTON_X)
        good_idx, bad_idx = select_good_bad_market_paths(
            stock_paths, exposure_paths, merton_value, maturity_window=maturity_window
        )

        t = np.linspace(0.0, plot_params.horizon, plot_params.steps + 1)
        good_color = "#16a34a"
        bad_color = CASE_COLORS.get("active", "#dc2626")
        fig, (ax_stock, ax_exposure, ax_control) = plt.subplots(
            3, 1, figsize=(figsize[0], 8.2), sharex=True
        )
        for idx, color, label in [
            (good_idx, good_color, "favorable market"),
            (bad_idx, bad_color, "unfavorable market"),
        ]:
            late_exposure = float(exposure_paths[idx, -maturity_window:].mean())
            ax_stock.plot(
                t,
                wealth_paths[idx],
                color=color,
                linewidth=2.0,
                label=f"{label}: "+r"$W_T^*$"+f"={wealth_paths[idx, -1]:.2f}",
            )
            ax_exposure.plot(
                t,
                exposure_paths[idx],
                color=color,
                linewidth=2.0,
                label=f"{label}: "+r"$\varphi_T^*$"+f"={late_exposure:.2f}",
            )
            ax_control.plot(
                t[:-1],
                control_paths[idx],
                color=color,
                linewidth=2.0,
                label=f"{label}: "+r"$\dot\varphi_T^*$"+f"={control_paths[idx,-1]:.2f}",
            )

        ax_exposure.axhline(
            merton_value,
            color="0.30",
            linewidth=1.5,
            linestyle="--",
            label=f"Merton line = {merton_value:.2f}",
        )
        ax_stock.set_ylabel("wealth", fontsize=14)
        ax_exposure.set_ylabel("dollar risky amount", fontsize=14)
        ax_control.set_ylabel(r"trading speed", fontsize=14)
        ax_control.set_xlabel("time", fontsize=14)
        ax_exposure.set_ylim(-0.35, 0.5)
        ax_control.set_xlim(0.0, 1.0)
        ax_control.set_ylim(-3.0, 4.0)
        ax_control.axhline(0.0, color="0.45", linewidth=0.8, alpha=0.7)
        ax_stock.legend(
            loc="upper left", frameon=True, framealpha=0.92, edgecolor="0.85", fontsize=11
        )
        ax_exposure.legend(
            loc="lower left", frameon=True, framealpha=0.92, edgecolor="0.85", fontsize=11
        )
        ax_control.legend(
            loc="lower left", frameon=True, framealpha=0.92, edgecolor="0.85", fontsize=11
        )
        fig.tight_layout()

        print(
            f"good path index={good_idx}, S_T={stock_paths[good_idx, -1]:.2f}, "
            f"late phi={exposure_paths[good_idx, -maturity_window:].mean():.2f}, "
            f"Merton={merton_value:.2f}"
        )
        print(
            f"bad path index={bad_idx}, S_T={stock_paths[bad_idx, -1]:.2f}, "
            f"late phi={exposure_paths[bad_idx, -maturity_window:].mean():.2f}"
        )
        if save_path is not None:
            save_figure(fig, save_path, bbox_inches="tight")
        return fig, (ax_stock, ax_exposure, ax_control), (good_idx, bad_idx)


    active_case = next(case for case in cases if case["name"] == "active")
    fig_good_bad, axes_good_bad, selected_good_bad_paths = plot_good_bad_market_paths(
        active_case, save_path=FIGURE_DIR / "power_sample.png"
    )


    # Plot 5. Trading-Rate Control Trajectory
    def plot_control_trajectory(
        cases,
        figsize=(9.6, 5.2),
        include_band=True,
        include_sample=False,
        save_path=None,
    ):
        fig, ax = plt.subplots(figsize=figsize)
        t = np.linspace(0.0, BASE_PARAMS.horizon, BASE_PARAMS.steps)

        for case in cases:
            name = case["name"]
            color = CASE_COLORS.get(name, "black")
            label = CASE_LABELS.get(name, name)
            control_paths = case["trade_rate_paths"]
            mean_control = np.mean(control_paths, axis=0)
            q10, q90 = np.quantile(control_paths, [0.10, 0.90], axis=0)

            if include_band:
                ax.fill_between(t, q10, q90, color=color, alpha=0.10, linewidth=0)
            if include_sample:
                ax.plot(t, control_paths[0], color=color, linewidth=0.9, alpha=0.32)
            ax.plot(
                t,
                mean_control,
                color=color,
                linewidth=2.2,
                label=f"{label} mean control",
            )

        ax.axhline(0.0, color="0.45", linewidth=0.8, alpha=0.7)
        ax.set_xlim(0.0, BASE_PARAMS.horizon)
        ax.set_ylim(-0.5, 4.0)
        ax.set_xlabel("time", fontsize=16)
        ax.set_ylabel(r"trading speed $\dot\varphi_t$", fontsize=16)

        handles = ax.get_legend_handles_labels()[0]
        if include_band:
            handles += [
                Patch(
                    facecolor=CASE_COLORS.get(case["name"], "black"),
                    alpha=0.10,
                    edgecolor="none",
                    label=f"{CASE_LABELS.get(case['name'], case['name'])} 10-90% range",
                )
                for case in cases
            ]
        ax.legend(
            handles=handles,
            loc="upper right",
            frameon=True,
            framealpha=0.92,
            edgecolor="0.85",
            fontsize=13,
        )
        fig.tight_layout()
        if save_path is not None:
            save_figure(fig, save_path, bbox_inches="tight")
        return fig, ax


    fig_control, ax_control = plot_control_trajectory(
        cases,
        save_path=FIGURE_DIR / "power_control.png",
    )


    # Export results and compare with the reported run
    # Export full-precision independent-OOS results.
    rows = []
    for case in cases:
        if isinstance(case, dict):
            row = dict(model=MODEL_KIND, case=case['name'],
                lambda_value=case['lambda_star'], eta=case['eta_star'],
                cvar_limit=case['cvar_limit'], oos_cvar=case['loss_cvar'],
                oos_residual=case['loss_cvar']-case['cvar_limit'],
                mean_terminal_wealth=case['mean_terminal_wealth'],
                mean_exposure=case['mean_dollar_position'])
        else:
            row = dict(model=MODEL_KIND, case=case.name,
                lambda_value=case.lambda_star, eta=case.eta_star,
                cvar_limit=case.params.cvar_limit, oos_cvar=case.cvar,
                oos_residual=case.cvar-case.params.cvar_limit,
                mean_terminal_wealth=float(case.terminal_wealth.mean()),
                mean_exposure=float(case.phi_paths.mean()))
        rows.append(row)

    with (OUTPUT_DIR / 'independent_oos_results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata = dict(model=MODEL_KIND, from_scratch=True,
        loads_precomputed_policy=False, merton_cap=False,
        oos_feasibility_adjustment=False, oos_samples=50_000,
        oos_seed=900013 if MODEL_KIND=='power' else 1900013,
        search_metrics=search_metrics, oos_results=rows,
        elapsed_seconds=time.perf_counter()-EXPERIMENT_STARTED,
        python=platform.python_version(), numpy=np.__version__,
        matplotlib=__import__('matplotlib').__version__)
    if MODEL_KIND == 'incomplete':
        metadata['search_protocol'] = dict(training_paths=5000,
            training_seed=13, multiplier_selection_paths=SELECTION_SAMPLES,
            multiplier_selection_seed=SELECTION_SEED,
            final_evaluation_used_for_selection=False)
        metadata['tail_diagnostics'] = dict(
            max_exposure=float(active_case.phi_paths.max()),
            fraction_paths_ever_above_merton=float(np.mean(
                np.any(active_case.phi_paths > 0.400001, axis=1))))
    (OUTPUT_DIR / 'run_metadata.json').write_text(json.dumps(metadata, indent=2))

    print('Full precision independent-OOS results:')
    for row in rows: print(row)
    print('Results saved to:', OUTPUT_DIR.resolve())

    plt.close('all')
    return rows


def main():
    parser = argparse.ArgumentParser(description='Square-root price-impact experiment')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Output folder (default: results/power in the current directory).')
    parser.add_argument('--no-numba', action='store_true',
                        help='Use the original NumPy solver instead of optional Numba acceleration.')
    args = parser.parse_args()
    run_experiment(output_dir=args.output_dir, use_numba=not args.no_numba)


if __name__ == '__main__':
    main()
