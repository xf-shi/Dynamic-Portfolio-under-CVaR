#!/usr/bin/env python3
"""Complete-market CVaR experiment.

Run: python complete_market.py
Optional: python complete_market.py --output-dir results/complete

The complete-market benchmark uses the same numerical search protocol as the
incomplete-market comparative statics, specialized to beta_perp = 0: a 201-point
wealth grid concentrated near the CVaR kink, a 5,000-path training sample, and an
independent 20,000-path search holdout for selecting the raw dual root. The final
50,000-path OOS sample is evaluation only. No Merton exposure cap or OOS
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
    MODEL_KIND = 'complete'

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


    # 2. Wealth-Only HJB Solver
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
        beta: float = 0.0
        price_model: str = "gbm"

        gamma: float = 5.0
        alpha: float = 0.95
        cvar_limit: float = -0.70

        phi_min: float = -2.0
        phi_max: float = 2.0

        eta_tol: float = 5.0e-4
        eta_subgrad_tol: float = 1.0e-5
        lambda_tol: float = 1.0e-6
        residual_tol: float = 1.0e-5
        eta_max_iter: int = 40
        lambda_max_iter: int = 20
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
        rockafellar_residual: float
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

        wealth = np.empty((samples, steps + 1))
        wealth[:, 0] = params.w0
        for n in range(steps):
            d_b = d_brownian[:, n]
            risky_excess = stock_excess_return(d_b, mu_t[n], sigma_t[n], dt, params)
            wealth[:, n + 1] = (
                growth * wealth[:, n] + phi[n] * risky_excess + params.b * dt + params.beta * d_b
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
        """Nonuniform wealth grid aligned with the incomplete-market protocol.

        The grid is built from the same pilot policies used in the comparative-
        statics solver and, for lambda>0, is concentrated near the terminal CVaR
        kink W=-eta.
        """
        probes = [
            np.full(params.steps, params.phi_min),
            np.clip(merton_exposure(params), params.phi_min, params.phi_max),
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
        dense = np.linspace(low, high, dense_size)
        focus_value = float(focus)
        focus_width = float(params.state_grid_focus_width)
        focus_strength = float(params.state_grid_focus_strength)
        density = 1.0 + focus_strength * np.exp(
            -0.5 * ((dense - focus_value) / focus_width) ** 2
        )
        cell_mass = 0.5 * (density[1:] + density[:-1]) * np.diff(dense)
        cumulative = np.concatenate(([0.0], np.cumsum(cell_mass)))
        total_mass = float(cumulative[-1])
        focus_mass = float(np.interp(focus_value, dense, cumulative))
        intervals = params.state_grid_size - 1
        left_intervals = int(round(intervals * focus_mass / total_mass))
        left_intervals = int(np.clip(left_intervals, 1, intervals - 1))
        right_intervals = intervals - left_intervals
        targets = np.concatenate((
            np.linspace(0.0, focus_mass, left_intervals + 1),
            np.linspace(focus_mass, total_mass, right_intervals + 1)[1:],
        ))
        return np.interp(targets, cumulative, dense)


    def linear_interpolate(
        wealth_values: np.ndarray | float,
        wealth_grid: np.ndarray,
        grid_values: np.ndarray,
    ) -> np.ndarray:
        """Linearly interpolate a value or policy table on the wealth grid."""
        wealth_values = np.asarray(wealth_values, dtype=float)
        wealth_clipped = np.clip(wealth_values, wealth_grid[0], wealth_grid[-1])
        return np.interp(wealth_clipped, wealth_grid, grid_values)


    def fit_hjb_policy(
        lambda_value: float,
        eta: float,
        d_brownian: np.ndarray,
        params: ModelParams,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Finite-difference HJB PDE solver for fixed (lambda, eta).

        The HJB state is wealth only. With dollar exposure as the action and
        deterministic return coefficients, neither the wealth dynamics nor the
        objective depends explicitly on the stock-price level.
        """
        focus = -float(eta) if lambda_value > 0.0 else None
        wealth_grid = build_state_grid(d_brownian, params, focus=focus)

        dt = params.horizon / params.steps
        mu_t = mu_schedule(params)
        sigma_t = sigma_schedule(params)
        value_next = (
            lambda_value
            * np.maximum(-wealth_grid - eta, 0.0)
            / (1.0 - params.alpha)
        )

        wealth_size = wealth_grid.size
        value_grid = np.empty((params.steps + 1, wealth_size))
        value_grid[-1] = value_next
        policy_grid = np.empty((params.steps, wealth_size))
        edge_order = 2 if wealth_size >= 3 else 1

        def derivatives(value: np.ndarray):
            value_w = np.gradient(
                value, wealth_grid, edge_order=edge_order
            )
            value_ww = np.gradient(
                value_w, wealth_grid, edge_order=edge_order
            )
            return value_w, value_ww

        def hamiltonian(
            action: np.ndarray,
            mu_n: float,
            sigma_n: float,
            value_w: np.ndarray,
            value_ww: np.ndarray,
        ) -> np.ndarray:
            wealth_diffusion = sigma_n * action + params.beta
            return (
                0.5 * params.gamma * (sigma_n * action) ** 2
                - mu_n * action
                + (params.r * wealth_grid + params.b + mu_n * action) * value_w
                + 0.5 * wealth_diffusion**2 * value_ww
            )

        def minimize_hamiltonian(value: np.ndarray, n: int):
            value_w, value_ww = derivatives(value)
            mu_n = float(mu_t[n])
            sigma_n = float(sigma_t[n])

            quadratic = sigma_n**2 * (params.gamma + value_ww)
            linear = (
                mu_n * (value_w - 1.0)
                + sigma_n * params.beta * value_ww
            )
            convex = quadratic > params.hjb_curvature_floor
            interior = np.where(
                convex,
                -linear / np.maximum(quadratic, 1.0e-14),
                0.0,
            )
            interior = np.clip(interior, params.phi_min, params.phi_max)

            best_action = interior
            best_value = hamiltonian(
                interior, mu_n, sigma_n, value_w, value_ww
            )
            lower_action = np.full_like(interior, params.phi_min)
            upper_action = np.full_like(interior, params.phi_max)
            lower_value = hamiltonian(
                lower_action, mu_n, sigma_n, value_w, value_ww
            )
            upper_value = hamiltonian(
                upper_action, mu_n, sigma_n, value_w, value_ww
            )
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
            maximum_action = max(abs(params.phi_min), abs(params.phi_max))
            wealth_diffusion = sigma_n * maximum_action + abs(params.beta)
            diffusion_rate = wealth_diffusion**2 / max(
                wealth_spacing**2, 1.0e-14
            )
            cfl_substeps = int(
                math.ceil(
                    dt
                    * diffusion_rate
                    / max(params.hjb_cfl_safety, 1.0e-6)
                )
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
                        "The explicit HJB step became unstable. Reduce "
                        "hjb_cfl_safety or increase hjb_pde_substeps."
                    )

            value_next = value
            value_grid[n] = value_next
            policy_grid[n] = best_action

        return wealth_grid, policy_grid, value_grid


    fit_feedback_policy = fit_hjb_policy


    def simulate_feedback_policy(
        wealth_grid: np.ndarray,
        policy_grid: np.ndarray,
        d_brownian: np.ndarray,
        params: ModelParams,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate a policy phi_n = f_n(W_n) with linear interpolation."""
        samples, steps = d_brownian.shape
        dt = params.horizon / steps
        growth = riskless_growth(params, dt)
        mu_t = mu_schedule(params, steps)
        sigma_t = sigma_schedule(params, steps)

        wealth = np.empty((samples, steps + 1))
        stock = np.empty((samples, steps + 1))
        phi_paths = np.empty((samples, steps))
        wealth[:, 0] = params.w0
        stock[:, 0] = params.s0

        for n in range(steps):
            phi_n = linear_interpolate(
                wealth[:, n],
                wealth_grid,
                policy_grid[n],
            )
            phi_n = np.clip(phi_n, params.phi_min, params.phi_max)
            phi_paths[:, n] = phi_n

            d_b = d_brownian[:, n]
            risky_excess = stock_excess_return(
                d_b, mu_t[n], sigma_t[n], dt, params
            )
            gross_return = stock_gross_return(
                d_b, mu_t[n], sigma_t[n], dt, params
            )
            wealth[:, n + 1] = (
                growth * wealth[:, n]
                + phi_n * risky_excess
                + params.b * dt
                + params.beta * d_b
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
        diffusion_bound = 6.0 * math.sqrt(params.horizon) * (sigma_abs_max * max_exposure + abs(params.beta))
        return -max(2.0, drift_bound + diffusion_bound), max(2.0, drift_bound + diffusion_bound)


    def eta_golden_search(
        lambda_value: float,
        d_brownian: np.ndarray,
        params: ModelParams,
    ) -> ControlResult:
        """Golden-section-style search using the CVaR quantile condition.

        This matches the protocol used in the incomplete-market comparative
        statics. Among queried eta values, we retain the result whose eta is
        closest to the empirical alpha-quantile of the induced loss.
        """
        if lambda_value <= 0.0:
            return unconstrained_solution(d_brownian, params)

        eta_low, eta_high = eta_bracket(params)
        rho = (math.sqrt(5.0) - 1.0) / 2.0
        eta1 = eta_high - rho * (eta_high - eta_low)
        eta2 = eta_low + rho * (eta_high - eta_low)
        result1 = feedback_value_iteration_kernel(lambda_value, eta1, d_brownian, params)
        result2 = feedback_value_iteration_kernel(lambda_value, eta2, d_brownian, params)

        def quantile_score(result: ControlResult) -> float:
            return abs(
                float(result.eta)
                - float(np.quantile(result.loss, params.alpha))
            )

        best = min(
            (result1, result2),
            key=lambda result: (quantile_score(result), result.lagrangian),
        )
        for _ in range(max(params.eta_max_iter - 1, 0)):
            if quantile_score(result1) <= quantile_score(result2):
                eta_high = eta2
                eta2 = eta1
                result2 = result1
                eta1 = eta_high - rho * (eta_high - eta_low)
                result1 = feedback_value_iteration_kernel(
                    lambda_value, eta1, d_brownian, params
                )
            else:
                eta_low = eta1
                eta1 = eta2
                result1 = result2
                eta2 = eta_low + rho * (eta_high - eta_low)
                result2 = feedback_value_iteration_kernel(
                    lambda_value, eta2, d_brownian, params
                )
            best = min(
                (best, result1, result2),
                key=lambda result: (quantile_score(result), result.lagrangian),
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


    def constraint_residual(
        result: ControlResult,
        params: ModelParams,
    ) -> float:
        """Empirical terminal loss-CVaR residual used by the lambda search."""
        return float(result.cvar - params.cvar_limit)


    def evaluate_policy_on_search_holdout(
        trained: ControlResult,
        lambda_value: float,
        d_brownian_holdout: np.ndarray,
        holdout_params: ModelParams,
    ) -> ControlResult:
        """Evaluate a trained feedback policy on the independent search holdout."""
        wealth, stock, phi_paths = simulate_feedback_policy(
            trained.wealth_grid,
            trained.policy_grid,
            d_brownian_holdout,
            holdout_params,
        )
        terminal_wealth = wealth[:, -1]
        loss = -terminal_wealth
        hinge = np.maximum(loss - trained.eta, 0.0)
        residual = float(
            trained.eta
            + hinge.mean() / (1.0 - holdout_params.alpha)
            - holdout_params.cvar_limit
        )
        cvar, var = cvar_from_loss(loss, holdout_params.alpha)
        return replace(
            trained,
            phi=np.mean(phi_paths, axis=0),
            lagrangian=trained.objective + lambda_value * residual,
            residual=residual,
            cvar=cvar,
            var=var,
            mean_terminal_wealth=float(np.mean(terminal_wealth)),
            terminal_wealth_std=float(np.std(terminal_wealth)),
            loss_tail_probability=float(np.mean(loss > trained.eta)),
            terminal_wealth=terminal_wealth,
            loss=loss,
            phi_paths=phi_paths,
            wealth_paths=wealth,
            stock_paths=stock,
        )


    def outer_bisection(
        d_brownian: np.ndarray,
        params: ModelParams,
        d_brownian_holdout: np.ndarray,
        holdout_params: ModelParams,
    ) -> tuple[ControlResult, float, list[dict[str, float]]]:
        """Outer bisection using the independent search holdout residual."""
        history: list[dict[str, float]] = []

        def fit_and_evaluate(lambda_value: float) -> ControlResult:
            trained = eta_golden_search(lambda_value, d_brownian, params)
            return evaluate_policy_on_search_holdout(
                trained, lambda_value, d_brownian_holdout, holdout_params
            )

        unconstrained = fit_and_evaluate(0.0)
        history.append(_history_row(0.0, unconstrained))
        unconstrained_residual = unconstrained.cvar - params.cvar_limit
        if unconstrained_residual <= 0.0:
            return unconstrained, 0.0, history

        lambda_low = 0.0
        lambda_high = params.lambda_initial_high
        high_result = fit_and_evaluate(lambda_high)
        history.append(_history_row(lambda_high, high_result))

        while high_result.cvar - params.cvar_limit > 0.0:
            lambda_low = lambda_high
            lambda_high *= 2.0
            if lambda_high > params.lambda_high_cap:
                raise RuntimeError("Failed to bracket a feasible multiplier.")
            high_result = fit_and_evaluate(lambda_high)
            history.append(_history_row(lambda_high, high_result))

        best_result = min(
            (unconstrained, high_result),
            key=lambda result: abs(result.cvar - params.cvar_limit),
        )
        best_lambda = 0.0 if best_result is unconstrained else lambda_high

        for _ in range(params.lambda_max_iter):
            lambda_mid = 0.5 * (lambda_low + lambda_high)
            mid_result = fit_and_evaluate(lambda_mid)
            history.append(_history_row(lambda_mid, mid_result))
            mid_residual = mid_result.cvar - params.cvar_limit

            if abs(mid_residual) < abs(best_result.cvar - params.cvar_limit):
                best_result = mid_result
                best_lambda = lambda_mid
            if abs(mid_residual) <= params.residual_tol:
                return mid_result, lambda_mid, history
            if mid_residual > 0.0:
                lambda_low = lambda_mid
            else:
                lambda_high = lambda_mid
            if lambda_high - lambda_low <= params.lambda_tol:
                break

        return best_result, best_lambda, history


    def solve_case(name: str, params: ModelParams) -> CaseResult:
        started = time.perf_counter()
        d_brownian = generate_brownian_increments(params)
        holdout_params = replace(
            params, samples=SEARCH_HOLDOUT_SAMPLES, seed=SEARCH_HOLDOUT_SEED
        )
        d_brownian_holdout = generate_brownian_increments(holdout_params)
        result, lambda_star, history = outer_bisection(
            d_brownian, params, d_brownian_holdout, holdout_params
        )
        terminal_wealth = result.terminal_wealth
        loss = result.loss if result.loss is not None else -terminal_wealth
        cvar, var = cvar_from_loss(loss, params.alpha)
        return CaseResult(
            name=name,
            params=params,
            lambda_star=lambda_star,
            eta_star=result.eta,
            objective=result.objective,
            cvar=cvar,
            var=var,
            residual=cvar - params.cvar_limit,
            rockafellar_residual=result.residual,
            terminal_wealth=terminal_wealth,
            loss=loss,
            phi=result.phi,
            elapsed_seconds=time.perf_counter() - started,
            oracle_calls=len(history),
            mean_terminal_wealth=float(np.mean(terminal_wealth)),
            phi_paths=result.phi_paths,
            wealth_paths=result.wealth_paths,
            policy_grid=result.policy_grid,
            wealth_grid=result.wealth_grid,
            stock_paths=result.stock_paths,
        )


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
        def solve_1d_kernel(w,lam,eta,alpha,steps,dt,mu,sigma,gamma,r,b,beta,lower,upper,subs,starts,coeff,floor):
            ngrid=len(w)
            value=np.maximum(-w-eta,0.0)*lam/(1-alpha)
            vw=np.empty(ngrid);vww=np.empty(ngrid)
            policy=np.empty((steps,ngrid));values=np.empty((steps+1,ngrid));values[-1]=value
            for n in range(steps-1,-1,-1):
                step_dt=dt/subs[n]
                for ss in range(subs[n]):
                    for i in range(ngrid):
                        k=starts[i];vw[i]=coeff[i,0]*value[k]+coeff[i,1]*value[k+1]+coeff[i,2]*value[k+2]
                    for i in range(ngrid):
                        k=starts[i];vww[i]=coeff[i,0]*vw[k]+coeff[i,1]*vw[k+1]+coeff[i,2]*vw[k+2]
                    for i in range(ngrid):
                        quadratic=sigma[n]**2*(gamma+vww[i])
                        linear=mu[n]*(vw[i]-1)+sigma[n]*beta*vww[i]
                        a=-linear/max(quadratic,1e-14) if quadratic>floor else 0.0
                        a=min(upper,max(lower,a))
                        h=.5*gamma*(sigma[n]*a)**2-mu[n]*a+(r*w[i]+b+mu[n]*a)*vw[i]+.5*(sigma[n]*a+beta)**2*vww[i]
                        for boundary in (lower,upper):
                            hb=.5*gamma*(sigma[n]*boundary)**2-mu[n]*boundary+(r*w[i]+b+mu[n]*boundary)*vw[i]+.5*(sigma[n]*boundary+beta)**2*vww[i]
                            if hb<h:a=boundary;h=hb
                        value[i]+=step_dt*h
                        if not math.isfinite(value[i]):raise ValueError('Nonfinite 1D finite-difference update')
                        if ss==subs[n]-1:policy[n,i]=a
                values[n]=value
            return policy,values


        def complete_solver(module):
            grids={}
            original=module.build_state_grid
            def solve(lam,eta,paths,p):
                focus = -float(eta) if lam > 0.0 else None
                key=(id(paths),repr(p),None if focus is None else round(focus,10))
                if key not in grids:grids[key]=original(paths,p,focus=focus)
                w=grids[key]
                dt=p.horizon/p.steps;dw=float(np.min(np.diff(w)))
                mu=module.mu_schedule(p);sigma=module.sigma_schedule(p)
                maximum=max(abs(p.phi_min),abs(p.phi_max))
                subs=np.array([max(p.hjb_pde_substeps,int(math.ceil(dt*(abs(float(s))*maximum+abs(p.beta))**2/max(dw**2,1e-14)/max(p.hjb_cfl_safety,1e-6))),1) for s in sigma])
                starts,coeff=gradient_stencil(w)
                policy,values=solve_1d_kernel(w,lam,eta,p.alpha,p.steps,dt,mu,sigma,p.gamma,p.r,p.b,p.beta,p.phi_min,p.phi_max,subs,starts,coeff,p.hjb_curvature_floor)
                return w,policy,values
            return solve


        fit_hjb_policy = complete_solver(_solver_namespace)
        fit_feedback_policy = fit_hjb_policy
        print('Numba:', numba.__version__, '(same grids and finite-difference updates)')


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
        beta=0.0,
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
    )

    SEARCH_HOLDOUT_SAMPLES = 20_000
    SEARCH_HOLDOUT_SEED = 686_035_552

    CASE_PARAMS = {
        "inactive": replace(BASE_PARAMS, cvar_limit=-0.86, seed=11),
        "active": replace(BASE_PARAMS, cvar_limit=-0.94, seed=13),
    }

    # Training paths fit the HJB policy; an independent 20,000-path search
    # holdout selects the raw dual root. Final OOS remains evaluation only.
    RUN_FULL_SEARCH = True

    time_grid = np.linspace(
        0.0, BASE_PARAMS.horizon, BASE_PARAMS.steps + 1
    )
    merton_line = merton_exposure(BASE_PARAMS)

    print(f"searches: outer lambda bisection, inner eta golden section")
    print(f"HJB state: wealth only, {BASE_PARAMS.state_grid_size} points")
    print(f"time steps: {BASE_PARAMS.steps}, training paths: {BASE_PARAMS.samples}")
    print(f"search holdout: {SEARCH_HOLDOUT_SAMPLES} paths, seed={SEARCH_HOLDOUT_SEED}")
    print(f"r = {BASE_PARAMS.r:.4f}, Merton exposure = {merton_line[0]:.4f}")


    # 4. Run Active And Inactive Cases
    import sys as _sys_trace
    import time as _time_trace
    _search_counter = 0
    _search_started = _time_trace.perf_counter()
    def _log_raw_search(lam, eta, cvar):
        nonlocal _search_counter
        _search_counter += 1
        _message = f'RAW_SEARCH query={_search_counter}, lambda={lam:.10f}, eta={eta:.10f}, search_CVaR={cvar:.10f}, seconds={_time_trace.perf_counter()-_search_started:.1f}'
        print(_message, flush=True)


    _original_eta_search = eta_golden_search
    def eta_golden_search(lam, paths, params):
        result = _original_eta_search(lam, paths, params)
        _log_raw_search(lam, result.eta, result.cvar)
        return result

    # Full aligned search. The independent search holdout selects the raw root;
    # no final-OOS feasibility adjustment follows it.
    search_validation_cases = [solve_case('inactive',CASE_PARAMS['inactive']), solve_case('active',CASE_PARAMS['active'])]
    search_metrics=[dict(case=c.name,lambda_value=c.lambda_star,eta=c.eta_star,cvar=c.cvar,residual=c.cvar-c.params.cvar_limit) for c in search_validation_cases]
    for case in search_validation_cases:
        np.savez_compressed(OUTPUT_DIR / f'frozen_{case.name}_policy.npz',wealth_grid=case.wealth_grid,policy_grid=case.policy_grid,lambda_value=case.lambda_star,eta=case.eta_star)
    FINAL_OOS_SAMPLES=50_000
    FINAL_OOS_SEED=1_900_013
    def evaluate_case_oos(case, samples=FINAL_OOS_SAMPLES, seed=FINAL_OOS_SEED):
        oos_params = replace(case.params, samples=int(samples), seed=int(seed))
        d_brownian_oos = generate_brownian_increments(oos_params)
        wealth, stock, phi_paths = simulate_feedback_policy(
            case.wealth_grid, case.policy_grid, d_brownian_oos, oos_params
        )
        terminal_wealth = wealth[:, -1]
        loss = -terminal_wealth
        cvar, var = cvar_from_loss(loss, oos_params.alpha)
        hinge = np.maximum(loss - case.eta_star, 0.0)
        rockafellar_residual = float(
            case.eta_star
            + hinge.mean() / (1.0 - oos_params.alpha)
            - oos_params.cvar_limit
        )
        return replace(
            case,
            cvar=float(cvar),
            var=float(var),
            residual=float(cvar - oos_params.cvar_limit),
            rockafellar_residual=rockafellar_residual,
            terminal_wealth=terminal_wealth,
            loss=loss,
            phi=np.mean(phi_paths, axis=0),
            mean_terminal_wealth=float(np.mean(terminal_wealth)),
            phi_paths=phi_paths,
            wealth_paths=wealth,
            stock_paths=stock,
        )

    cases=[evaluate_case_oos(case) for case in search_validation_cases]
    print('Independent OOS evaluation only; no parameter changes after this evaluation.')
    print_summary_table(cases)


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

        ax.set_xlabel(r"terminal wealth $W_T^*$",fontsize=16)
        ax.set_ylabel("density",fontsize=16)
        ax.set_xlim(0.6,1.5)
        ax.set_ylim(0.0, 14.0)
        ax.legend(loc="upper right", frameon=True, fontsize=10)
        fig.tight_layout()
        if save_path is not None:
            save_figure(fig, save_path, bbox_inches="tight")
        return fig, ax


    fig_terminal, ax_terminal = plot_terminal_wealth_distribution(
        cases, save_path=FIGURE_DIR / "complete_w_dist.png"
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
        ax.set_ylim(-0.1, 0.5)
        ax.set_xlabel("time", fontsize=14)
        ax.set_ylabel("dollar risky exposure", fontsize=16)
        ax.legend(loc="best", frameon=True, fontsize=16)
        fig.tight_layout()
        if save_path is not None:
            save_figure(fig, save_path, bbox_inches="tight")
        return fig, ax


    fig_control, ax_control = plot_exposure_trajectory(
        cases, save_path=FIGURE_DIR / "complete_control.png"
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
            loc="upper left",
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
        cases, save_path=FIGURE_DIR / "complete_cvar.png"
    )


    # Plot 4. Selected Favorable And Unfavorable Market Paths
    def select_good_bad_paths(
        wealth_paths,
        phi_paths,
        merton_value,
        target_final_phi=0.45,
    ):
        terminal_wealth = wealth_paths[:, -1]
        final_phi = phi_paths[:, -1]

        favorable_target = np.quantile(terminal_wealth, 0.90)
        favorable_idx = int(
            np.argmin(np.abs(terminal_wealth - favorable_target))
        )

        lower_tail_cutoff = np.quantile(terminal_wealth, 0.10)
        bad_pool = np.where(
            (terminal_wealth <= lower_tail_cutoff)
            & (final_phi > merton_value + 1.0e-6)
        )[0]
        if bad_pool.size == 0:
            raise RuntimeError(
                'No lower-tail path found with final exposure above the Merton benchmark.'
            )
        bad_idx = int(
            bad_pool[
                np.argmin(
                    np.abs(final_phi[bad_pool] - target_final_phi)
                )
            ]
        )
        return favorable_idx, bad_idx


    def plot_good_bad_market_paths(
        active_case,
        path_count=50000,
        figsize=(9.6, 6.8),
        save_path=None,
    ):
        wealth, stock, phi_paths = simulate_case_paths(
            active_case,
            path_count=path_count,
            seed_shift=250000,
        )
        merton_value = float(merton_exposure(active_case.params)[-1])
        good_idx, bad_idx = select_good_bad_paths(
            wealth,
            phi_paths,
            merton_value,
        )
        t_stock = np.linspace(
            0.0,
            active_case.params.horizon,
            active_case.params.steps + 1,
        )
        t_phi = t_stock[:-1]
        fig, (ax_stock, ax_phi) = plt.subplots(
            2, 1, figsize=figsize, sharex=True
        )

        for index, color, label in [
            (good_idx, "#16a34a", "favorable market"),
            (bad_idx, CASE_COLORS["active"], "unfavorable market"),
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
                label=fr"{label}: $\varphi_T^*$={terminal_phi:.2f}",
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
        ax_phi.set_xlim(0.0, BASE_PARAMS.horizon)
        ax_stock.set_xlim(0.0, BASE_PARAMS.horizon)
        ax_stock.grid(True, alpha=0.3)
        ax_phi.grid(True, alpha=0.3)
        y_phi_min = min(-0.10, float(phi_paths[[good_idx, bad_idx]].min()) - 0.02)
        y_phi_max = max(0.45, float(phi_paths[[good_idx, bad_idx]].max()) + 0.02)
        ax_phi.set_ylim(y_phi_min, y_phi_max)
        ax_stock.legend(loc="upper left", frameon=True, fontsize=14)
        ax_phi.legend(loc="lower left", frameon=True, fontsize=14)
        fig.tight_layout()
        if save_path is not None:
            save_figure(fig, save_path, bbox_inches="tight")
        return fig, (ax_stock, ax_phi), (good_idx, bad_idx)


    active_case = next(case for case in cases if case.name == "active")
    fig_paths, axes_paths, selected_paths = plot_good_bad_market_paths(
        active_case, save_path=FIGURE_DIR / "complete_sample.png"
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
        search_protocol=dict(training_paths=5000, training_seed_active=13,
            training_seed_nonbinding=11, selection_paths=SEARCH_HOLDOUT_SAMPLES,
            selection_seed=SEARCH_HOLDOUT_SEED, common_with_incomplete_comparative_statics=True),
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
    parser = argparse.ArgumentParser(description='Complete-market CVaR experiment')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Output folder (default: results/complete in the current directory).')
    parser.add_argument('--no-numba', action='store_true',
                        help='Use the original NumPy solver instead of optional Numba acceleration.')
    args = parser.parse_args()
    run_experiment(output_dir=args.output_dir, use_numba=not args.no_numba)


if __name__ == '__main__':
    main()
