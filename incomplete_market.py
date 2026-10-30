#!/usr/bin/env python3
"""Incomplete-market CVaR experiment.

Run: python incomplete_market.py
Optional: python incomplete_market.py --output-dir results/incomplete

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
    MODEL_KIND = 'incomplete'

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
    from dataclasses import dataclass, replace
    import math
    import time
    from pathlib import Path
    from typing import Optional

    import numpy as np
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 180,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
    })


    # 2. Wealth-Only Incomplete-Market HJB Solver
    @dataclass
    class ModelParams:
        horizon: float = 1.0
        steps: int = 100
        samples: int = 5000
        seed: int = 7

        w0: float = 1.0
        s0: float = 1.0
        r: float = 0.00
        mu: float = 0.08
        sigma: float = 0.20
        mu_t: Optional[tuple[float, ...]] = None
        sigma_t: Optional[tuple[float, ...]] = None
        b: float = 0.0
        beta_perp: float = 0.0
        price_model: str = "gbm"

        gamma: float = 5.0
        alpha: float = 0.95
        cvar_limit: float = -0.70

        phi_min: float = -2.0
        phi_max: float = 2.0

        eta_tol: float = 1.0e-6
        lambda_tol: float = 1.0e-6
        residual_tol: float = 1.0e-6
        eta_max_iter: int = 40
        lambda_max_iter: int = 28
        lambda_initial_high: float = 1.0
        lambda_high_cap: float = 1.0e5

        state_grid_size: int = 201
        state_grid_step: Optional[float] = None
        state_grid_focus_strength: float = 1.0
        state_grid_focus_width: float = 0.20
        hjb_pde_substeps: int = 8
        hjb_cfl_safety: float = 0.20
        hjb_curvature_floor: float = 1.0e-10
        state_grid_padding: float = 0.25
        cap_control_at_merton: bool = False


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
        policy_grid: Optional[np.ndarray] = None
        wealth_grid: Optional[np.ndarray] = None
        terminal_wealth: Optional[np.ndarray] = None
        loss: Optional[np.ndarray] = None
        phi_paths: Optional[np.ndarray] = None
        wealth_paths: Optional[np.ndarray] = None
        stock_paths: Optional[np.ndarray] = None


    @dataclass
    class CaseResult:
        name: str
        params: ModelParams
        lambda_star: float
        eta_star: float
        objective: float
        cvar: float
        var: float
        residual: float
        terminal_wealth: np.ndarray
        loss: np.ndarray
        phi: np.ndarray
        elapsed_seconds: float
        oracle_calls: int
        mean_terminal_wealth: float
        phi_paths: Optional[np.ndarray] = None
        wealth_paths: Optional[np.ndarray] = None
        stock_paths: Optional[np.ndarray] = None
        policy_grid: Optional[np.ndarray] = None
        wealth_grid: Optional[np.ndarray] = None


    def _time_schedule(
        schedule: Optional[tuple[float, ...]], fallback: float, steps: int, name: str
    ) -> np.ndarray:
        if schedule is None:
            return np.full(steps, float(fallback), dtype=float)
        values = np.asarray(schedule, dtype=float).reshape(-1)
        if values.size == steps + 1:
            values = values[:-1]
        if values.size != steps:
            raise ValueError(
                f"{name} must have length steps or steps + 1; got {values.size} for steps={steps}."
            )
        return values.astype(float, copy=True)


    def mu_schedule(params: ModelParams, steps: Optional[int] = None) -> np.ndarray:
        return _time_schedule(params.mu_t, params.mu, params.steps if steps is None else steps, "mu_t")


    def sigma_schedule(params: ModelParams, steps: Optional[int] = None) -> np.ndarray:
        return _time_schedule(
            params.sigma_t, params.sigma, params.steps if steps is None else steps, "sigma_t"
        )


    def merton_exposure(params: ModelParams) -> np.ndarray:
        mu_t = mu_schedule(params)
        sigma_t = sigma_schedule(params)
        return mu_t / np.maximum(params.gamma * sigma_t**2, 1.0e-12)


    def generate_brownian_increments(params: ModelParams) -> np.ndarray:
        rng = np.random.default_rng(params.seed)
        dt = params.horizon / params.steps
        return rng.normal(0.0, math.sqrt(dt), size=(params.samples, params.steps))


    def generate_perpendicular_brownian_increments(params: ModelParams) -> np.ndarray:
        """Independent increments for the nontraded wealth shock."""
        rng = np.random.default_rng(params.seed + 1_000_003)
        dt = params.horizon / params.steps
        return rng.normal(0.0, math.sqrt(dt), size=(params.samples, params.steps))


    def riskless_growth(params: ModelParams, dt: float) -> float:
        return math.exp(params.r * dt)


    def stock_gross_return(
        d_brownian: np.ndarray | float,
        mu_n: float,
        sigma_n: float,
        dt: float,
        params: ModelParams,
    ) -> np.ndarray | float:
        """One-period stock gross return under the selected price model."""
        d_b = np.asarray(d_brownian)
        if params.price_model == "gbm":
            return np.exp((params.r + mu_n - 0.5 * sigma_n**2) * dt + sigma_n * d_b)
        if params.price_model == "euler":
            return riskless_growth(params, dt) + mu_n * dt + sigma_n * d_b
        raise ValueError(f"Unknown price_model={params.price_model!r}")


    def stock_excess_return(
        d_brownian: np.ndarray | float,
        mu_n: float,
        sigma_n: float,
        dt: float,
        params: ModelParams,
    ) -> np.ndarray | float:
        return stock_gross_return(d_brownian, mu_n, sigma_n, dt, params) - riskless_growth(
            params, dt
        )


    def simulate_wealth(phi: np.ndarray, d_brownian: np.ndarray, params: ModelParams) -> np.ndarray:
        samples, steps = d_brownian.shape
        dt = params.horizon / steps
        growth = riskless_growth(params, dt)
        mu_t = mu_schedule(params, steps)
        sigma_t = sigma_schedule(params, steps)
        d_brownian_perp = generate_perpendicular_brownian_increments(
            replace(params, samples=samples, steps=steps)
        )

        wealth = np.empty((samples, steps + 1))
        wealth[:, 0] = params.w0
        for n in range(steps):
            d_b = d_brownian[:, n]
            risky_excess = stock_excess_return(d_b, mu_t[n], sigma_t[n], dt, params)
            wealth[:, n + 1] = (
                growth * wealth[:, n] + phi[n] * risky_excess + params.b * dt + params.beta_perp * d_brownian_perp[:, n]
            )
        return wealth


    def cvar_from_loss(loss: np.ndarray, alpha: float) -> tuple[float, float]:
        var = float(np.quantile(loss, alpha))
        cvar = float(var + np.maximum(loss - var, 0.0).mean() / (1.0 - alpha))
        return cvar, var


    def lower_tail_wealth_cvar_path(wealth_paths: np.ndarray, alpha: float) -> np.ndarray:
        return np.array([-cvar_from_loss(-wealth_paths[:, n], alpha)[0] for n in range(wealth_paths.shape[1])])


    def running_cost(action: np.ndarray | float, params: ModelParams, n: Optional[int] = None) -> np.ndarray | float:
        action_arr = np.asarray(action)
        if n is None:
            if action_arr.ndim > 0 and action_arr.shape[-1] == params.steps:
                shape = (1,) * (action_arr.ndim - 1) + (params.steps,)
                mu_t = mu_schedule(params).reshape(shape)
                sigma_t = sigma_schedule(params).reshape(shape)
                return 0.5 * params.gamma * (sigma_t * action_arr) ** 2 - mu_t * action_arr
            return 0.5 * params.gamma * (params.sigma * action_arr) ** 2 - params.mu * action_arr

        mu_n = mu_schedule(params)[n]
        sigma_n = sigma_schedule(params)[n]
        return 0.5 * params.gamma * (sigma_n * action_arr) ** 2 - mu_n * action_arr


    def build_state_grid(
        d_brownian: np.ndarray,
        params: ModelParams,
        focus: Optional[float] = None,
    ) -> np.ndarray:
        """Adaptive one-dimensional wealth grid, focused near the CVaR kink."""
        merton_probe = np.clip(merton_exposure(params), params.phi_min, params.phi_max)
        probes = [
            np.full(params.steps, params.phi_min),
            merton_probe,
            np.zeros(params.steps),
        ]
        wealth_paths = [simulate_wealth(phi, d_brownian, params) for phi in probes]
        wealth = np.concatenate([path.reshape(-1) for path in wealth_paths])
        low = float(np.quantile(wealth, 0.002))
        high = float(np.quantile(wealth, 0.998))
        width = max(high - low, 1.0e-3)
        low -= params.state_grid_padding * width
        high += params.state_grid_padding * width
        if params.state_grid_step is not None and params.state_grid_step > 0.0:
            step = float(params.state_grid_step)
            grid_low = step * math.floor(low / step)
            grid_high = step * math.ceil(high / step)
            return np.arange(grid_low, grid_high + 0.5 * step, step)

        if (
            focus is None
            or not low < float(focus) < high
            or params.state_grid_focus_strength <= 0.0
            or params.state_grid_focus_width <= 0.0
        ):
            return np.linspace(low, high, params.state_grid_size)

        dense_size = max(4001, 40 * params.state_grid_size + 1)
        dense_grid = np.linspace(low, high, dense_size)
        focus_value = float(focus)
        width_value = float(params.state_grid_focus_width)
        strength_value = float(params.state_grid_focus_strength)
        scaled_distance = (dense_grid - focus_value) / width_value
        density = 1.0 + strength_value * np.exp(-0.5 * scaled_distance**2)
        cell_mass = 0.5 * (density[1:] + density[:-1]) * np.diff(dense_grid)
        cumulative_mass = np.concatenate(([0.0], np.cumsum(cell_mass)))
        total_mass = float(cumulative_mass[-1])
        focus_mass = float(np.interp(focus_value, dense_grid, cumulative_mass))

        interval_count = params.state_grid_size - 1
        left_intervals = int(round(interval_count * focus_mass / total_mass))
        left_intervals = int(np.clip(left_intervals, 1, interval_count - 1))
        right_intervals = interval_count - left_intervals
        left_targets = np.linspace(0.0, focus_mass, left_intervals + 1)
        right_targets = np.linspace(focus_mass, total_mass, right_intervals + 1)[1:]
        target_mass = np.concatenate((left_targets, right_targets))
        return np.interp(target_mass, cumulative_mass, dense_grid)


    def fit_hjb_policy(
        lambda_value: float,
        eta: float,
        d_brownian: np.ndarray,
        params: ModelParams,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One-dimensional finite-difference HJB on wealth for fixed (lambda, eta)."""
        wealth_focus = -float(eta) if lambda_value > 0.0 else None
        wealth_grid = build_state_grid(d_brownian, params, focus=wealth_focus)

        dt = params.horizon / params.steps
        mu_t = mu_schedule(params)
        sigma_t = sigma_schedule(params)
        merton_t = merton_exposure(params)
        terminal_penalty = (
            lambda_value
            * np.maximum(-wealth_grid - eta, 0.0)
            / (1.0 - params.alpha)
        )
        value_next = terminal_penalty.copy()

        wealth_size = wealth_grid.size
        value_grid = np.empty((params.steps + 1, wealth_size))
        value_grid[-1] = value_next
        policy_grid = np.empty((params.steps, wealth_size))
        edge_order = 2 if wealth_size >= 3 else 1

        def action_upper(n: int) -> float:
            upper = float(params.phi_max)
            if params.cap_control_at_merton:
                upper = min(upper, float(merton_t[n]))
            return max(float(params.phi_min), upper)

        def derivatives(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            value_w = np.gradient(value, wealth_grid, edge_order=edge_order)
            value_ww = np.gradient(value_w, wealth_grid, edge_order=edge_order)
            return value_w, value_ww

        def hamiltonian(
            action: np.ndarray,
            mu_n: float,
            sigma_n: float,
            value_w: np.ndarray,
            value_ww: np.ndarray,
        ) -> np.ndarray:
            wealth_variance = (sigma_n * action) ** 2 + params.beta_perp**2
            return (
                0.5 * params.gamma * (sigma_n * action) ** 2
                - mu_n * action
                + (params.r * wealth_grid + params.b + mu_n * action) * value_w
                + 0.5 * wealth_variance * value_ww
            )

        def minimize_hamiltonian(value: np.ndarray, n: int):
            value_w, value_ww = derivatives(value)
            mu_n = float(mu_t[n])
            sigma_n = float(sigma_t[n])
            upper_n = action_upper(n)

            quadratic = sigma_n**2 * (params.gamma + value_ww)
            linear = mu_n * (value_w - 1.0)
            convex = quadratic > params.hjb_curvature_floor
            interior = np.where(convex, -linear / np.maximum(quadratic, 1.0e-14), 0.0)
            interior = np.clip(interior, params.phi_min, upper_n)

            best_action = interior
            best_value = hamiltonian(interior, mu_n, sigma_n, value_w, value_ww)
            lower_action = np.full_like(interior, params.phi_min)
            upper_action = np.full_like(interior, upper_n)
            lower_value = hamiltonian(lower_action, mu_n, sigma_n, value_w, value_ww)
            upper_value = hamiltonian(upper_action, mu_n, sigma_n, value_w, value_ww)
            use_lower = lower_value < best_value
            best_action = np.where(use_lower, lower_action, best_action)
            best_value = np.where(use_lower, lower_value, best_value)
            use_upper = upper_value < best_value
            best_action = np.where(use_upper, upper_action, best_action)
            best_value = np.where(use_upper, upper_value, best_value)
            return best_action, best_value

        def stable_substeps(n: int) -> int:
            wealth_spacing = float(np.min(np.diff(wealth_grid)))
            sigma_n = abs(float(sigma_t[n]))
            maximum_action = max(abs(params.phi_min), abs(action_upper(n)))
            wealth_variance = (sigma_n * maximum_action) ** 2 + params.beta_perp**2
            diffusion_rate = wealth_variance / max(wealth_spacing**2, 1.0e-14)
            cfl_substeps = int(
                math.ceil(dt * diffusion_rate / max(params.hjb_cfl_safety, 1.0e-6))
            )
            return max(params.hjb_pde_substeps, cfl_substeps, 1)

        for n in range(params.steps - 1, -1, -1):
            value = value_next
            substeps = stable_substeps(n)
            sub_dt = dt / substeps
            best_action = np.zeros(wealth_size)
            for _ in range(substeps):
                best_action, minimized_generator = minimize_hamiltonian(value, n)
                value = value + sub_dt * minimized_generator
                if not np.all(np.isfinite(value)):
                    raise FloatingPointError(
                        "The explicit one-dimensional HJB step became unstable. "
                        "Reduce hjb_cfl_safety or increase hjb_pde_substeps."
                    )

            value_next = value
            value_grid[n] = value_next
            policy_grid[n] = best_action

        return wealth_grid, policy_grid, value_grid


    def simulate_feedback_policy(
        wealth_grid: np.ndarray,
        policy_grid: np.ndarray,
        d_brownian: np.ndarray,
        params: ModelParams,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate the one-dimensional feedback policy phi_n = f_n(W_n)."""
        samples, steps = d_brownian.shape
        dt = params.horizon / steps
        growth = riskless_growth(params, dt)
        mu_t = mu_schedule(params, steps)
        sigma_t = sigma_schedule(params, steps)
        merton_t = merton_exposure(replace(params, steps=steps))
        d_brownian_perp = generate_perpendicular_brownian_increments(
            replace(params, samples=samples, steps=steps)
        )

        wealth = np.empty((samples, steps + 1))
        stock = np.empty((samples, steps + 1))
        phi_paths = np.empty((samples, steps))
        wealth[:, 0] = params.w0
        stock[:, 0] = params.s0

        for n in range(steps):
            phi_n = np.interp(
                wealth[:, n],
                wealth_grid,
                policy_grid[n],
                left=policy_grid[n, 0],
                right=policy_grid[n, -1],
            )
            upper_n = params.phi_max
            if params.cap_control_at_merton:
                upper_n = min(upper_n, float(merton_t[n]))
            phi_n = np.clip(phi_n, params.phi_min, upper_n)
            phi_paths[:, n] = phi_n

            d_b = d_brownian[:, n]
            risky_excess = stock_excess_return(d_b, mu_t[n], sigma_t[n], dt, params)
            gross_return = stock_gross_return(d_b, mu_t[n], sigma_t[n], dt, params)
            wealth[:, n + 1] = (
                growth * wealth[:, n]
                + phi_n * risky_excess
                + params.b * dt
                + params.beta_perp * d_brownian_perp[:, n]
            )
            stock[:, n + 1] = stock[:, n] * gross_return

        return wealth, stock, phi_paths


    def feedback_value_iteration_kernel(
        lambda_value: float,
        eta: float,
        d_brownian: np.ndarray,
        params: ModelParams,
    ) -> ControlResult:
        wealth_grid, policy_grid, _ = fit_hjb_policy(
            lambda_value, eta, d_brownian, params
        )
        wealth, stock, phi_paths = simulate_feedback_policy(
            wealth_grid, policy_grid, d_brownian, params
        )

        dt = params.horizon / params.steps
        terminal_wealth = wealth[:, -1]
        loss = -terminal_wealth
        hinge = np.maximum(loss - eta, 0.0)
        running = dt * np.sum(running_cost(phi_paths, params), axis=1)
        objective = float(np.mean(running))
        residual = float(
            eta + hinge.mean() / (1.0 - params.alpha) - params.cvar_limit
        )
        lagrangian = objective + lambda_value * residual
        cvar, var = cvar_from_loss(loss, params.alpha)

        return ControlResult(
            phi=np.mean(phi_paths, axis=0),
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
            policy_grid=policy_grid,
            wealth_grid=wealth_grid,
            terminal_wealth=terminal_wealth,
            loss=loss,
            phi_paths=phi_paths,
            wealth_paths=wealth,
            stock_paths=stock,
        )


    def unconstrained_solution(d_brownian: np.ndarray, params: ModelParams) -> ControlResult:
        result = feedback_value_iteration_kernel(0.0, 0.0, d_brownian, params)
        loss = result.loss if result.loss is not None else -result.terminal_wealth
        cvar, var = cvar_from_loss(loss, params.alpha)
        return replace(
            result,
            eta=var,
            residual=cvar - params.cvar_limit,
            cvar=cvar,
            var=var,
            loss_tail_probability=float(np.mean(loss > var)),
            loss=loss,
        )


    def eta_bracket(params: ModelParams) -> tuple[float, float]:
        mu_abs_max = float(np.max(np.abs(mu_schedule(params))))
        sigma_abs_max = float(np.max(np.abs(sigma_schedule(params))))
        max_exposure = max(abs(params.phi_min), abs(params.phi_max))
        drift_bound = abs(params.w0) + params.horizon * (
            abs(params.r) * (abs(params.w0) + 4.0) + mu_abs_max * max_exposure + abs(params.b)
        )
        diffusion_bound = 6.0 * math.sqrt(params.horizon) * (sigma_abs_max * max_exposure + abs(params.beta_perp))
        return -max(2.0, drift_bound + diffusion_bound), max(2.0, drift_bound + diffusion_bound)


    def eta_golden_search(lambda_value: float, d_brownian: np.ndarray, params: ModelParams) -> ControlResult:
        """Golden-section search for eta using |eta - empirical VaR_alpha(loss)|.

        The fixed-point form is numerically stable on the nonuniform HJB grid and
        is equivalent to the Rockafellar eta condition at the solution.
        """
        if lambda_value <= 0.0:
            return unconstrained_solution(d_brownian, params)

        eta_low, eta_high = eta_bracket(params)
        rho = (math.sqrt(5.0) - 1.0) / 2.0

        eta1 = eta_high - rho * (eta_high - eta_low)
        eta2 = eta_low + rho * (eta_high - eta_low)
        result1 = feedback_value_iteration_kernel(lambda_value, eta1, d_brownian, params)
        result2 = feedback_value_iteration_kernel(lambda_value, eta2, d_brownian, params)

        def eta_score(result: ControlResult) -> float:
            loss = result.loss if result.loss is not None else -result.terminal_wealth
            empirical_var = float(np.quantile(loss, params.alpha))
            return abs(float(result.eta) - empirical_var)

        best = min(
            (result1, result2),
            key=lambda result: (eta_score(result), result.lagrangian),
        )

        for _ in range(max(params.eta_max_iter - 1, 0)):
            if eta_score(result1) <= eta_score(result2):
                eta_high = eta2
                eta2 = eta1
                result2 = result1
                eta1 = eta_high - rho * (eta_high - eta_low)
                result1 = feedback_value_iteration_kernel(lambda_value, eta1, d_brownian, params)
            else:
                eta_low = eta1
                eta1 = eta2
                result1 = result2
                eta2 = eta_low + rho * (eta_high - eta_low)
                result2 = feedback_value_iteration_kernel(lambda_value, eta2, d_brownian, params)
            best = min(
                (best, result1, result2),
                key=lambda result: (eta_score(result), result.lagrangian),
            )
            if eta_high - eta_low <= params.eta_tol:
                break

        return best


    def _history_row(lambda_value: float, result: ControlResult) -> dict[str, float]:
        return {
            "lambda": float(lambda_value),
            "eta": float(result.eta),
            "residual": float(result.residual),
            "objective": float(result.objective),
            "cvar": float(result.cvar),
        }


    def evaluate_policy_on_holdout(
        trained_result: ControlResult,
        lambda_value: float,
        d_brownian_holdout: np.ndarray,
        holdout_params: ModelParams,
    ) -> ControlResult:
        """Evaluate a trained HJB policy on an independent holdout sample."""
        wealth, stock, phi_paths = simulate_feedback_policy(
            trained_result.wealth_grid,
            trained_result.policy_grid,
            d_brownian_holdout,
            holdout_params,
        )
        terminal_wealth = wealth[:, -1]
        loss = -terminal_wealth
        hinge = np.maximum(loss - trained_result.eta, 0.0)
        rockafellar_residual = float(
            trained_result.eta
            + hinge.mean() / (1.0 - holdout_params.alpha)
            - holdout_params.cvar_limit
        )
        cvar, var = cvar_from_loss(loss, holdout_params.alpha)
        return replace(
            trained_result,
            phi=np.mean(phi_paths, axis=0),
            lagrangian=(
                trained_result.objective
                + lambda_value * rockafellar_residual
            ),
            residual=rockafellar_residual,
            cvar=cvar,
            var=var,
            mean_terminal_wealth=float(np.mean(terminal_wealth)),
            terminal_wealth_std=float(np.std(terminal_wealth)),
            loss_tail_probability=float(
                np.mean(loss > trained_result.eta)
            ),
            terminal_wealth=terminal_wealth,
            loss=loss,
            phi_paths=phi_paths,
            wealth_paths=wealth,
            stock_paths=stock,
        )


    def constraint_residual(result: ControlResult, params: ModelParams) -> float:
        """Independent holdout loss-CVaR residual for lambda bisection."""
        return float(result.cvar - params.cvar_limit)


    def outer_bisection(
        d_brownian: np.ndarray,
        params: ModelParams,
        d_brownian_holdout: np.ndarray,
        holdout_params: ModelParams,
    ) -> tuple[ControlResult, float, list[dict[str, float]]]:
        """Fit on training paths and bisect lambda on holdout CVaR."""
        history: list[dict[str, float]] = []

        def fit_and_evaluate(lambda_value: float) -> ControlResult:
            trained = eta_golden_search(lambda_value, d_brownian, params)
            return evaluate_policy_on_holdout(
                trained,
                lambda_value,
                d_brownian_holdout,
                holdout_params,
            )

        unconstrained = fit_and_evaluate(0.0)
        history.append(_history_row(0.0, unconstrained))
        target_tol = params.residual_tol
        unconstrained_residual = constraint_residual(unconstrained, params)
        if unconstrained_residual <= 0.0:
            return unconstrained, 0.0, history

        lambda_low = 0.0
        lambda_high = params.lambda_initial_high
        high_result = fit_and_evaluate(lambda_high)
        history.append(_history_row(lambda_high, high_result))

        while constraint_residual(high_result, params) > 0.0:
            lambda_low = lambda_high
            lambda_high *= 2.0
            if lambda_high > params.lambda_high_cap:
                raise RuntimeError(
                    "Failed to bracket a feasible holdout multiplier."
                )
            high_result = fit_and_evaluate(lambda_high)
            history.append(_history_row(lambda_high, high_result))

        best_result = min(
            (unconstrained, high_result),
            key=lambda result: abs(constraint_residual(result, params)),
        )
        best_lambda = (
            0.0 if best_result is unconstrained else lambda_high
        )
        for _ in range(params.lambda_max_iter):
            lambda_mid = 0.5 * (lambda_low + lambda_high)
            mid_result = fit_and_evaluate(lambda_mid)
            history.append(_history_row(lambda_mid, mid_result))
            mid_residual = constraint_residual(mid_result, params)
            if abs(mid_residual) < abs(
                constraint_residual(best_result, params)
            ):
                best_result = mid_result
                best_lambda = lambda_mid
            if abs(mid_residual) <= target_tol:
                return mid_result, lambda_mid, history
            if mid_residual > 0.0:
                lambda_low = lambda_mid
            else:
                lambda_high = lambda_mid
            if lambda_high - lambda_low <= params.lambda_tol:
                break

        return best_result, best_lambda, history


    def summarize_case(case: CaseResult) -> dict[str, float | str]:
        phi_values = case.phi_paths if case.phi_paths is not None else case.phi[None, :]
        return {
            "case": case.name,
            "lambda*": case.lambda_star,
            "E[W_T]": float(np.mean(case.terminal_wealth)),
            "std(W_T)": float(np.std(case.terminal_wealth)),
            "worst-tail mean wealth": -case.cvar,
            "floor -c": -case.params.cvar_limit,
            "slack c-CVaR": case.params.cvar_limit - case.cvar,
            "mean phi=varphi*S": float(np.mean(phi_values)),
            "min phi": float(np.min(phi_values)),
            "max phi": float(np.max(phi_values)),
            "runtime sec": case.elapsed_seconds,
        }


    def print_summary_table(cases: list[CaseResult]) -> None:
        rows = [summarize_case(case) for case in cases]
        headers = ["case", "lambda*", "E[W_T]", "worst-tail mean wealth", "floor -c", "mean phi=varphi*S"]
        print(" | ".join(headers))
        print(" | ".join("-" * len(h) for h in headers))
        for row in rows:
            values = []
            for header in headers:
                value = row[header]
                values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
            print(" | ".join(values))


    # 3. Settings
    BASE_PARAMS = ModelParams(
        horizon=1.0,
        steps=100,
        samples=5000,
        seed=7,
        w0=1.0,
        s0=1.0,
        r=0.0,
        mu=0.08,
        sigma=0.20,
        b=0.0,
        beta_perp=0.02,
        price_model="gbm",
        gamma=5.0,
        alpha=0.95,
        phi_min=-2.0,
        phi_max=2.0,
        cvar_limit=-0.86,
        eta_tol=5.0e-4,
        lambda_tol=1.0e-6,
        residual_tol=1.0e-5,
        eta_max_iter=40,
        lambda_max_iter=20,
        lambda_initial_high=1.0,
        lambda_high_cap=1.0e5,
        state_grid_size=201,
        state_grid_step=None,
        state_grid_focus_strength=1.0,
        state_grid_focus_width=0.20,
        state_grid_padding=0.25,
        hjb_pde_substeps=8,
        hjb_cfl_safety=0.20,
        hjb_curvature_floor=1.0e-10,
        cap_control_at_merton=False,
    )

    CASE_PARAMS = {
        "inactive": replace(BASE_PARAMS, cvar_limit=-0.86, seed=11),
        "active": replace(BASE_PARAMS, cvar_limit=-0.94, seed=13),
    }

    # Independent multiplier-selection paths belong to calibration.
    SELECTION_SAMPLES = 20_000
    SELECTION_SEED = 686035552

    time_grid = np.linspace(
        0.0, BASE_PARAMS.horizon, BASE_PARAMS.steps + 1
    )
    merton_line = merton_exposure(BASE_PARAMS)

    print("searches: outer lambda bisection, inner eta golden section")
    print(
        "HJB state: wealth only, "
        f"{BASE_PARAMS.state_grid_size} nonuniform points"
    )
    print(
        "wealth-grid focus: W=-eta, "
        f"width={BASE_PARAMS.state_grid_focus_width:.2f}"
    )
    print(
        f"time steps: {BASE_PARAMS.steps}, paths: {BASE_PARAMS.samples}"
    )
    print(
        f"r={BASE_PARAMS.r:.4f}, beta_perp={BASE_PARAMS.beta_perp:.4f}, "
        f"Merton exposure={merton_line[0]:.4f}"
    )


    # 4. Run Active And Inactive Cases
    # All policy parameters are computed below. No saved policy is loaded.
    FINAL_OOS_SAMPLES = 50_000
    FINAL_OOS_SEED = 1_900_013
    active_params = CASE_PARAMS['active']
    assert active_params.cap_control_at_merton is False
    training_paths = generate_brownian_increments(active_params)
    selection_params = replace(active_params, samples=SELECTION_SAMPLES,
                               seed=SELECTION_SEED)
    selection_paths = generate_brownian_increments(selection_params)

    # Cache only the deterministic pilot-derived grid endpoints. The eta-focused
    # grid is rebuilt for every candidate with the original floating-point formula.
    _original_build_grid = build_state_grid
    _plain_grid = _original_build_grid(training_paths, active_params, focus=None)
    _grid_low, _grid_high = float(_plain_grid[0]), float(_plain_grid[-1])

    def build_state_grid(d_brownian, params, focus=None):
        if d_brownian is not training_paths or params is not active_params:
            return _original_build_grid(d_brownian, params, focus=focus)
        low, high = _grid_low, _grid_high
        if (focus is None or not low < float(focus) < high
                or params.state_grid_focus_strength <= 0
                or params.state_grid_focus_width <= 0):
            return np.linspace(low, high, params.state_grid_size)
        dense_size = max(4001, 40 * params.state_grid_size + 1)
        dense_grid = np.linspace(low, high, dense_size)
        scaled_distance = (dense_grid - float(focus)) / float(params.state_grid_focus_width)
        density = 1.0 + float(params.state_grid_focus_strength) * np.exp(-0.5 * scaled_distance**2)
        cell_mass = 0.5 * (density[1:] + density[:-1]) * np.diff(dense_grid)
        cumulative_mass = np.concatenate(([0.0], np.cumsum(cell_mass)))
        total_mass = float(cumulative_mass[-1])
        focus_mass = float(np.interp(float(focus), dense_grid, cumulative_mass))
        interval_count = params.state_grid_size - 1
        left_intervals = int(round(interval_count * focus_mass / total_mass))
        left_intervals = int(np.clip(left_intervals, 1, interval_count - 1))
        right_intervals = interval_count - left_intervals
        left_targets = np.linspace(0.0, focus_mass, left_intervals + 1)
        right_targets = np.linspace(focus_mass, total_mass, right_intervals + 1)[1:]
        return np.interp(np.concatenate((left_targets, right_targets)),
                         cumulative_mass, dense_grid)

    for _focus in [None, 0.95, 1.10]:
        assert np.array_equal(
            _original_build_grid(training_paths, active_params, focus=_focus),
            build_state_grid(training_paths, active_params, focus=_focus))

    _eta_solver = eta_golden_search
    _query_count = 0
    def eta_golden_search(lam, paths, params):
        nonlocal _query_count
        result = _eta_solver(lam, paths, params)
        _query_count += 1
        print(f'RAW_SEARCH query={_query_count}, lambda={lam:.10f}, '
              f'eta={result.eta:.10f}, fit_CVaR={result.cvar:.10f}, '
              f'seconds={time.perf_counter()-EXPERIMENT_STARTED:.1f}', flush=True)
        return result

    active_fit, active_lambda, active_history = outer_bisection(
        training_paths, active_params, selection_paths, selection_params)
    inactive_params = CASE_PARAMS['inactive']
    inactive_fit = unconstrained_solution(
        generate_brownian_increments(inactive_params), inactive_params)
    search_metrics = [
        dict(case='inactive', lambda_value=0.0, eta=inactive_fit.eta,
             cvar=inactive_fit.cvar, residual=inactive_fit.cvar-inactive_params.cvar_limit),
        dict(case='active', lambda_value=float(active_lambda), eta=active_fit.eta,
             cvar=active_fit.cvar, residual=active_fit.cvar-active_params.cvar_limit),
    ]
    for name, fit, lam in [('inactive', inactive_fit, 0.0),
                           ('active', active_fit, active_lambda)]:
        np.savez_compressed(OUTPUT_DIR / f'computed_{name}_policy.npz',
            wealth_grid=fit.wealth_grid, policy_grid=fit.policy_grid,
            lambda_value=lam, eta=fit.eta)
    (OUTPUT_DIR / 'search_history.json').write_text(
        json.dumps(active_history, indent=2))

    # Freeze the fitted policy before generating any final evaluation paths.
    def make_oos_case(name, params, lam, fit):
        assert not params.cap_control_at_merton
        evaluation_params = replace(params, samples=FINAL_OOS_SAMPLES,
                                    seed=FINAL_OOS_SEED)
        db = generate_brownian_increments(evaluation_params)
        wealth, stock, actions = simulate_feedback_policy(
            fit.wealth_grid, fit.policy_grid, db, evaluation_params)
        loss = -wealth[:, -1]
        cvar, var = cvar_from_loss(loss, params.alpha)
        running = params.horizon / params.steps * np.sum(
            running_cost(actions, params), axis=1)
        return CaseResult(name=name, params=params, lambda_star=float(lam),
            eta_star=float(fit.eta), objective=float(running.mean()),
            cvar=float(cvar), var=float(var),
            residual=float(cvar-params.cvar_limit),
            terminal_wealth=wealth[:, -1], loss=loss, phi=actions.mean(axis=0),
            elapsed_seconds=0.0, oracle_calls=0,
            mean_terminal_wealth=float(wealth[:, -1].mean()), phi_paths=actions,
            wealth_paths=wealth, stock_paths=stock,
            wealth_grid=fit.wealth_grid, policy_grid=fit.policy_grid)

    inactive_case = make_oos_case('inactive', inactive_params, 0.0, inactive_fit)
    active_case = make_oos_case('active', active_params, active_lambda, active_fit)
    cases = [inactive_case, active_case]
    print('Independent OOS evaluation; no adjustment follows this evaluation.')
    print_summary_table(cases)
    print(f'Active multiplier: {active_lambda:.10f}')
    print(f'Active OOS CVaR: {active_case.cvar:.10f}')
    print(f'Active maximum exposure: {active_case.phi_paths.max():.10f}')
    print('Fraction of paths ever above Merton:',
          np.mean(np.any(active_case.phi_paths > 0.400001, axis=1)))


    # 5. Shared Plot Helpers
    CASE_COLORS = {
        "inactive": "#2563eb",
        "active": "#dc2626",
    }
    CASE_LABELS = {
        "inactive": "nonbinding",
        "active": "c=-0.94, raw search",
    }
    FIGURE_DIR = OUTPUT_DIR / 'figures'
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


    def gaussian_kde_1d(samples, grid, bandwidth=None):
        samples = np.asarray(samples, dtype=float)
        if bandwidth is None:
            scale = max(float(np.std(samples, ddof=1)), 1.0e-8)
            bandwidth = max(
                1.06 * scale * samples.size ** (-0.2),
                1.0e-4,
            )
        z = (grid[:, None] - samples[None, :]) / bandwidth
        return np.exp(-0.5 * z * z).mean(axis=1) / (
            bandwidth * np.sqrt(2.0 * np.pi)
        )


    def simulate_case_paths(case, path_count, seed_shift):
        params = replace(
            case.params,
            samples=path_count,
            seed=case.params.seed + seed_shift,
        )
        d_brownian = generate_brownian_increments(params)
        return simulate_feedback_policy(
            case.wealth_grid,
            case.policy_grid,
            d_brownian,
            params,
        )


    # Plot 1. Terminal Wealth Distribution
    def plot_terminal_wealth_distribution(
        cases,
        bins_count=44,
        figsize=(9.6, 5.5),
        save_path=None,
    ):
        all_wealth = np.concatenate(
            [case.terminal_wealth for case in cases]
        )
        padding = 0.04 * max(float(np.ptp(all_wealth)), 1.0e-3)
        grid = np.linspace(
            all_wealth.min() - padding,
            all_wealth.max() + padding,
            320,
        )
        bins = np.linspace(grid[0], grid[-1], bins_count + 1)
        fig, ax = plt.subplots(figsize=figsize)

        for case in cases:
            color = CASE_COLORS[case.name]
            label = CASE_LABELS[case.name]
            ax.hist(
                case.terminal_wealth,
                bins=bins,
                density=True,
                color=color,
                alpha=0.12,
                edgecolor="none",
            )
            ax.plot(
                grid,
                gaussian_kde_1d(case.terminal_wealth, grid),
                color=color,
                linewidth=2.2,
                label=f"{label} KDE",
            )
            ax.axvline(
                -case.cvar,
                color=color,
                linestyle="-.",
                linewidth=1.4,
                label=f"{label} worst-tail mean wealth = {-case.cvar:.2f}",
            )
            terminal_mean = float(np.mean(case.terminal_wealth))
            ax.axvline(
                terminal_mean,
                color=color,
                linestyle="--",
                linewidth=1.4,
                label=f"{label} mean = {terminal_mean:.2f}",
            )

        ax.set_xlabel(r"terminal wealth $W_T^*$", fontsize=16)
        ax.set_xlim(0.6,1.5)
        ax.set_ylim(0.0, 14.0)
        ax.set_ylabel("density", fontsize=16)
        ax.legend(loc="upper right", frameon=True, fontsize=10)
        fig.tight_layout()
        if save_path is not None:
            save_figure(fig, save_path, bbox_inches="tight")
        return fig, ax


    fig_terminal, ax_terminal = plot_terminal_wealth_distribution(
        cases, save_path=FIGURE_DIR / "incomplete_w_dist.png"
    )


    # Plot 2. Dollar Risky Exposure
    def plot_exposure_trajectory(
        cases,
        figsize=(9.6, 5.2),
        save_path=None,
    ):
        t = np.linspace(
            0.0, BASE_PARAMS.horizon, BASE_PARAMS.steps
        )
        fig, ax = plt.subplots(figsize=figsize)

        for case in cases:
            color = CASE_COLORS[case.name]
            label = CASE_LABELS[case.name]
            mean_phi = np.mean(case.phi_paths, axis=0)
            ax.plot(
                t,
                mean_phi,
                color=color,
                linewidth=2.2,
                label=f"{label} mean",
            )
            if case.name == "active":
                q10, q90 = np.quantile(
                    case.phi_paths, [0.10, 0.90], axis=0
                )
                ax.fill_between(
                    t,
                    q10,
                    q90,
                    color=color,
                    alpha=0.13,
                    linewidth=0,
                    label="c=-0.94: 10-90% range",
                )

        ax.axhline(
            float(merton_line[0]),
            color="0.35",
            linestyle="--",
            linewidth=1.4,
            label=f"Merton line = {merton_line[0]:.2f}",
        )
        ax.set_xlim(0.0, BASE_PARAMS.horizon)
        ax.set_ylim(-0.10, 0.50)
        ax.set_xlabel("time", fontsize=16)
        ax.set_ylabel(r"dollar risky exposure", fontsize=16)
        ax.legend(loc="best", frameon=True, fontsize=16)
        fig.tight_layout()
        if save_path is not None:
            save_figure(fig, save_path, bbox_inches="tight")
        return fig, ax


    fig_control, ax_control = plot_exposure_trajectory(
        cases, save_path=FIGURE_DIR / "incomplete_control.png"
    )


    # Plot 3. Wealth CVaR Over Time
    def plot_wealth_cvar_over_time(
        cases,
        terminal_tolerance=1.0e-12,
        figsize=(9.6, 5.2),
        save_path=None,
    ):
        fig, ax = plt.subplots(figsize=figsize)
        for case in cases:
            color = CASE_COLORS.get(case.name, "black")
            label = CASE_LABELS.get(case.name, case.name)
            # The table, terminal distribution, and CVaR path must use
            # exactly the same final OOS paths.
            wealth_paths = case.wealth_paths

            wealth_cvar = lower_tail_wealth_cvar_path(
                wealth_paths,
                case.params.alpha,
            )
            plot_terminal_cvar = wealth_cvar[-1]
            solver_terminal_cvar = -case.cvar
            terminal_floor = -case.params.cvar_limit
            terminal_gap = abs(
                plot_terminal_cvar - solver_terminal_cvar
            )
            if terminal_gap > terminal_tolerance:
                raise AssertionError(
                    f"{case.name}: table/figure terminal CVaR mismatch "
                    f"({terminal_gap:.3e})"
                )

            t = np.linspace(
                0.0,
                case.params.horizon,
                wealth_cvar.size,
            )
            ax.plot(
                t,
                -wealth_cvar,
                color=color,
                linewidth=2.2,
                label=f"{label} CVaR over time",
            )
            ax.axhline(
                -terminal_floor,
                color=color,
                linewidth=1.1,
                linestyle=(0, (5, 3)),
                alpha=0.75,
                label=f"{label} constraint level",
            )

        ax.set_xlabel("time", fontsize=16)
        ax.set_ylabel(r"CVaR$_\alpha(-W_t^*)$", fontsize=13)
        ax.set_xlim(0.0, BASE_PARAMS.horizon)
        ax.set_ylim(-1.0, -0.84)
        ax.legend(
            loc="best",
            frameon=True,
            framealpha=0.92,
            edgecolor="0.85",
            fontsize=16,
        )
        fig.tight_layout()
        if save_path is not None:
            save_figure(fig, save_path, bbox_inches="tight")
        return fig, ax


    fig_cvar_time, ax_cvar_time = plot_wealth_cvar_over_time(
        cases, save_path=FIGURE_DIR / "incomplete_cvar.png"
    )


    # Plot 4. Selected Good And Bad Market Paths
    def select_good_bad_paths(
        stock_paths,
        phi_paths,
        merton_value,
        horizon,
        close_after=0.6,
        not_close_before=0.5,
        close_tolerance=0.01,
    ):
        """Select favorable/adverse markets using terminal stock quartiles.

        Within the favorable quartile, choose the path whose risky exposure is
        closest to the Merton line on [close_after, T], subject to staying
        at least close_tolerance away from Merton before not_close_before.
        """
        terminal_stock = stock_paths[:, -1]
        control_time = np.linspace(0.0, horizon, phi_paths.shape[1])
        after_cutoff = control_time >= close_after
        before_cutoff = control_time < not_close_before
        if not np.any(after_cutoff):
            raise ValueError("close_after must be smaller than or equal to T.")
        if not np.any(before_cutoff):
            raise ValueError("not_close_before must be greater than zero.")
        absolute_gap = np.abs(phi_paths - merton_value)
        post_merton_gap = np.mean(
            absolute_gap[:, after_cutoff],
            axis=1,
        )
        minimum_early_gap = np.min(
            absolute_gap[:, before_cutoff],
            axis=1,
        )
        post_cutoff_phi = phi_paths[:, after_cutoff].mean(axis=1)

        good_pool = np.where(
            terminal_stock >= np.quantile(terminal_stock, 0.75)
        )[0]
        bad_pool = np.where(
            terminal_stock <= np.quantile(terminal_stock, 0.25)
        )[0]
        delayed_good_pool = good_pool[
            minimum_early_gap[good_pool] >= close_tolerance
        ]
        if delayed_good_pool.size == 0:
            raise RuntimeError(
                "No favorable path stays sufficiently below Merton before "
                f"t={not_close_before:.2f}."
            )
        close_good_pool = delayed_good_pool[
            post_merton_gap[delayed_good_pool] <= close_tolerance
        ]
        if close_good_pool.size == 0:
            close_good_pool = delayed_good_pool
        good_idx = close_good_pool[
            np.argmin(post_merton_gap[close_good_pool])
        ]
        bad_idx = bad_pool[np.argmin(post_cutoff_phi[bad_pool])]
        return int(good_idx), int(bad_idx)


    def plot_good_bad_market_paths(
        active_case,
        path_count=8000,
        close_after=0.6,
        not_close_before=0.5,
        close_tolerance=0.01,
        figsize=(9.6, 6.2),
        save_path=None,
    ):
        wealth, stock, phi_paths = simulate_case_paths(
            active_case,
            path_count=path_count,
            seed_shift=250000,
        )
        merton_value = float(merton_exposure(active_case.params)[-1])
        good_idx, bad_idx = select_good_bad_paths(
            stock,
            phi_paths,
            merton_value,
            horizon=active_case.params.horizon,
            close_after=close_after,
            not_close_before=not_close_before,
            close_tolerance=close_tolerance,
        )
        t_wealth = np.linspace(
            0.0,
            active_case.params.horizon,
            active_case.params.steps + 1,
        )
        t_phi = np.linspace(
            0.0,
            active_case.params.horizon,
            active_case.params.steps,
        )
        after_cutoff = t_phi >= close_after
        fig, (ax_wealth, ax_phi) = plt.subplots(
            2, 1, figsize=figsize, sharex=True
        )

        for index, color, label in [
            (good_idx, "#16a34a", "favorable market"),
            (bad_idx, CASE_COLORS["active"], "unfavorable market"),
        ]:
            post_cutoff_phi = float(
                phi_paths[index, after_cutoff].mean()
            )
            ax_wealth.plot(
                t_wealth,
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
                label=(
                    fr"{label}: $\varphi_T^*$={phi_paths[index,-1]:.2f}"
                ),
            )

        ax_phi.axhline(
            merton_value,
            color="0.35",
            linestyle="--",
            linewidth=1.4,
            label=f"Merton line = {merton_value:.2f}",
        )
        ax_wealth.set_ylabel(r"wealth", fontsize=14)
        ax_phi.set_ylabel(r"dollar risky exposure", fontsize=14)
        ax_phi.set_xlabel("time", fontsize=14)
        ax_phi.set_xlim(0.0, active_case.params.horizon)
        selected_values = phi_paths[[good_idx, bad_idx]]
        ax_phi.set_ylim(min(-0.1, float(selected_values.min())-0.05), max(0.45, float(selected_values.max())+0.05))
        ax_wealth.legend(loc="upper left", frameon=True, fontsize=14)
        ax_phi.legend(loc="lower left", frameon=True, fontsize=14)
        fig.tight_layout()
        if save_path is not None:
            save_figure(fig, save_path, bbox_inches="tight")
        return fig, (ax_wealth, ax_phi), (good_idx, bad_idx)


    active_case = next(case for case in cases if case.name == "active")
    fig_paths, axes_paths, selected_paths = plot_good_bad_market_paths(
        active_case, save_path=FIGURE_DIR / "incomplete_sample.png"
    )


    # Diagnostic: full feedback policy
    # Additional diagnostic 1: the complete feedback policy, including deep losses.
    wealth_display=np.linspace(0.80, 1.12, 260)
    decision_times=np.arange(active_case.params.steps)*active_case.params.horizon/active_case.params.steps
    policy_display=np.array([np.interp(wealth_display, active_case.wealth_grid, row) for row in active_case.policy_grid])
    fig_policy, ax_policy=plt.subplots(figsize=(9.6,5.5))
    image=ax_policy.pcolormesh(decision_times,wealth_display,policy_display.T,shading='auto',cmap='viridis',vmin=0,vmax=float(active_case.policy_grid.max()))
    fig_policy.colorbar(image,ax=ax_policy,label='dollar risky exposure')
    ax_policy.axhline(-active_case.eta_star,color='white',linestyle='--',linewidth=1.2,label=r'CVaR threshold $-\eta$')
    ax_policy.contour(decision_times,wealth_display,policy_display.T,levels=[0.400001],colors=['#fb7185'],linewidths=1.2)
    ax_policy.plot([],[],color='#fb7185',label='Merton exposure contour: 0.4')
    ax_policy.set(xlabel='time',ylabel='wealth',title='Uncapped feedback policy: deep-loss region')
    ax_policy.legend(loc='upper left',fontsize=10,facecolor='0.2',labelcolor='white')
    fig_policy.tight_layout()
    save_figure(fig_policy, FIGURE_DIR/'uncapped_policy_map.png',bbox_inches='tight')


    # Diagnostic: an extreme loss path
    # Additional diagnostic 2: select the lowest terminal wealth in the final OOS sample.
    # This is an explicit extreme-path illustration, not a representative path.
    extreme_index=int(np.argmin(active_case.terminal_wealth))
    t_wealth=np.arange(active_case.params.steps+1)*active_case.params.horizon/active_case.params.steps
    fig_extreme,(ax_ew,ax_ep)=plt.subplots(2,1,figsize=(9.6,6.2),sharex=True)
    ax_ew.plot(t_wealth,active_case.wealth_paths[extreme_index],color='#dc2626',linewidth=2,label=f'lowest terminal wealth among 50,000 paths: {active_case.terminal_wealth[extreme_index]:.2f}')
    ax_ew.axhline(-active_case.eta_star,color='0.4',linestyle='--',label=r'threshold $-\eta$')
    ax_ep.plot(decision_times,active_case.phi_paths[extreme_index],color='#dc2626',linewidth=2,label='uncapped exposure')
    ax_ep.axhline(0.4,color='0.35',linestyle='--',label='Merton line = 0.4')
    ax_ew.set(ylabel='wealth',title='Extreme-path diagnostic: selected by lowest terminal wealth')
    ax_ep.set(xlabel='time',ylabel='dollar risky exposure',xlim=(0,1))
    ax_ew.legend(fontsize=10);ax_ep.legend(fontsize=10)
    fig_extreme.tight_layout()
    save_figure(fig_extreme, FIGURE_DIR/'uncapped_extreme_path.png',bbox_inches='tight')
    print('Extreme path index:',extreme_index)
    print('Exposure on extreme path: min=',active_case.phi_paths[extreme_index].min(),'max=',active_case.phi_paths[extreme_index].max())


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
    parser = argparse.ArgumentParser(description='Incomplete-market CVaR experiment')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Output folder (default: results/incomplete in the current directory).')
    parser.add_argument('--no-numba', action='store_true',
                        help='Use the original NumPy solver instead of optional Numba acceleration.')
    args = parser.parse_args()
    run_experiment(output_dir=args.output_dir, use_numba=not args.no_numba)


if __name__ == '__main__':
    main()
