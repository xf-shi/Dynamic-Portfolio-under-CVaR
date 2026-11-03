#!/usr/bin/env python3
"""Quadratic trading-rate regularization experiment.

Run: python linear_control.py
Optional: python linear_control.py --output-dir results/linear

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
    MODEL_KIND = 'linear'

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
    })


    # 2. Minimal fine-grid HJB, golden-eta, and lambda-bisection solver
    from dataclasses import dataclass, replace


    import math


    from typing import Optional


    import numpy as np


    @dataclass(frozen=True)
    class ModelParams:
        horizon: float = 1.0
        steps: int = 320
        samples: int = 10000
        seed: int = 13
        w0: float = 1.0
        x0: float = 0.0
        mu: float = 0.08
        sigma: float = 0.20
        gamma: float = 5.0
        Lambda: float = 0.01
        alpha: float = 0.95
        cvar_limit: float = -0.94
        rate_min: float = -2.0
        rate_max: float = 2.0
        x_min: float = -2.0
        x_max: float = 2.0
        wealth_min: float = 0.007218743674123118
        wealth_max: float = 1.9927812563258778
        wealth_nodes: int = 161
        exposure_nodes: int = 193
        hjb_min_substeps: int = 8
        hjb_cfl_safety: float = 0.20
        eta_low: float = -0.97
        eta_high: float = -0.93
        eta_tol: float = 5.0e-4
        lambda_low: float = 0.0
        lambda_high: float = 0.10
        lambda_tol: float = 1.0e-4
        cvar_tol: float = 1.0e-4
        lambda_max_iter: int = 12

        def __post_init__(self):
            if self.horizon <= 0.0 or self.steps <= 0 or self.samples <= 0:
                raise ValueError("horizon, steps, and samples must be positive")
            if not 0.0 < self.alpha < 1.0:
                raise ValueError("alpha must lie strictly between 0 and 1")
            if self.wealth_nodes < 3 or self.exposure_nodes < 3:
                raise ValueError("both HJB grids require at least three nodes")
            if self.wealth_min >= self.wealth_max or self.x_min >= self.x_max:
                raise ValueError("grid lower bounds must be below upper bounds")
            if self.rate_min >= self.rate_max or self.Lambda <= 0.0:
                raise ValueError("invalid rate bounds or non-positive Lambda")
            if self.hjb_min_substeps < 1 or self.hjb_cfl_safety <= 0.0:
                raise ValueError("invalid HJB time-stepping controls")
            if self.eta_low >= self.eta_high or self.lambda_low >= self.lambda_high:
                raise ValueError("search intervals must have positive width")


    @dataclass
    class ControlResult:
        eta: float
        objective: float
        lagrangian: float
        residual: float
        cvar: float
        var: float
        mean_terminal_wealth: float
        terminal_wealth_std: float
        loss_tail_probability: float
        policy_grid: np.ndarray
        wealth_grid: np.ndarray
        exposure_grid: np.ndarray
        terminal_wealth: np.ndarray
        wealth_paths: np.ndarray
        exposure_paths: np.ndarray
        rate_paths: np.ndarray


    def generate_brownian_increments(params: ModelParams) -> np.ndarray:
        rng = np.random.default_rng(params.seed)
        dt = params.horizon / params.steps
        return rng.normal(0.0, math.sqrt(dt), size=(params.samples, params.steps))


    def stock_paths_from_brownian(
        d_brownian: np.ndarray, params: ModelParams, s0: float = 1.0
    ) -> np.ndarray:
        stock = np.empty((d_brownian.shape[0], params.steps + 1))
        stock[:, 0] = s0
        dt = params.horizon / params.steps
        growth = (params.mu - 0.5 * params.sigma**2) * dt
        for n in range(params.steps):
            stock[:, n + 1] = stock[:, n] * np.exp(
                growth + params.sigma * d_brownian[:, n]
            )
        return stock


    def cvar_from_loss(loss: np.ndarray, alpha: float) -> tuple[float, float]:
        loss = np.asarray(loss, dtype=float)
        if loss.size == 0 or not np.all(np.isfinite(loss)):
            raise ValueError("loss must be a non-empty finite array")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie strictly between 0 and 1")
        var = float(np.quantile(loss, alpha))
        cvar = float(var + np.maximum(loss - var, 0.0).mean() / (1.0 - alpha))
        return cvar, var


    def lower_tail_wealth_cvar_path(wealth: np.ndarray, alpha: float) -> np.ndarray:
        return np.array([-cvar_from_loss(-wealth[:, n], alpha)[0] for n in range(wealth.shape[1])])


    def interp_value_2d(
        w: np.ndarray,
        x: np.ndarray,
        wealth_grid: np.ndarray,
        exposure_grid: np.ndarray,
        values: np.ndarray,
    ) -> np.ndarray:
        w = np.clip(np.asarray(w), wealth_grid[0], wealth_grid[-1])
        x = np.clip(np.asarray(x), exposure_grid[0], exposure_grid[-1])
        iw = np.clip(np.searchsorted(wealth_grid, w) - 1, 0, wealth_grid.size - 2)
        ix = np.clip(np.searchsorted(exposure_grid, x) - 1, 0, exposure_grid.size - 2)
        aw = (w - wealth_grid[iw]) / (wealth_grid[iw + 1] - wealth_grid[iw])
        ax = (x - exposure_grid[ix]) / (exposure_grid[ix + 1] - exposure_grid[ix])
        return (
            (1.0 - aw) * (1.0 - ax) * values[iw, ix]
            + aw * (1.0 - ax) * values[iw + 1, ix]
            + (1.0 - aw) * ax * values[iw, ix + 1]
            + aw * ax * values[iw + 1, ix + 1]
        )


    def running_cost(x, v, params: ModelParams):
        return (
            0.5 * params.gamma * params.sigma**2 * x**2
            + 0.5 * params.Lambda * v**2
            - params.mu * x
        )


    def simulate_feedback_policy(
        wealth_grid: np.ndarray,
        exposure_grid: np.ndarray,
        policy_grid: np.ndarray,
        d_brownian: np.ndarray,
        params: ModelParams,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        wealth = np.empty((params.samples, params.steps + 1))
        exposure = np.empty_like(wealth)
        rates = np.empty((params.samples, params.steps))
        wealth[:, 0] = params.w0
        exposure[:, 0] = params.x0
        dt = params.horizon / params.steps
        for n in range(params.steps):
            x = exposure[:, n]
            v = interp_value_2d(
                wealth[:, n], x, wealth_grid, exposure_grid, policy_grid[n]
            )
            v = np.clip(v, params.rate_min, params.rate_max)
            rates[:, n] = v
            risky_return = params.mu * x * dt + params.sigma * x * d_brownian[:, n]
            wealth[:, n + 1] = wealth[:, n] + risky_return
            exposure[:, n + 1] = np.clip(
                x + v * dt + risky_return, params.x_min, params.x_max
            )
        return wealth, exposure, rates


    def solve_hjb(
        lambda_value: float,
        eta: float,
        d_brownian: np.ndarray,
        params: ModelParams,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backward finite-difference HJB solve on the (W,x) grid."""
        wealth_grid = np.linspace(params.wealth_min, params.wealth_max, params.wealth_nodes)
        exposure_grid = np.linspace(params.x_min, params.x_max, params.exposure_nodes)
        dt = params.horizon / params.steps
        terminal = (
            lambda_value
            * np.maximum(-wealth_grid - eta, 0.0)
            / (1.0 - params.alpha)
        )
        value = np.repeat(terminal[:, None], params.exposure_nodes, axis=1)
        policy = np.empty((params.steps, params.wealth_nodes, params.exposure_nodes))
        _, x_mesh = np.meshgrid(wealth_grid, exposure_grid, indexing="ij")

        dw = wealth_grid[1] - wealth_grid[0]
        dx = exposure_grid[1] - exposure_grid[0]
        max_x = max(abs(params.x_min), abs(params.x_max))
        transport = (
            abs(params.mu) * max_x / dw
            + (max(abs(params.rate_min), abs(params.rate_max)) + abs(params.mu) * max_x) / dx
        )
        diffusion = (params.sigma * max_x) ** 2 * (
            1.0 / dw**2 + 2.0 / (dw * dx) + 1.0 / dx**2
        )
        substeps = max(
            params.hjb_min_substeps,
            int(math.ceil(dt * (transport + diffusion) / params.hjb_cfl_safety)),
        )
        sub_dt = dt / substeps

        for n in range(params.steps - 1, -1, -1):
            for _ in range(substeps):
                value_w = np.gradient(value, wealth_grid, axis=0, edge_order=2)
                value_x = np.gradient(value, exposure_grid, axis=1, edge_order=2)
                value_ww = np.gradient(value_w, wealth_grid, axis=0, edge_order=2)
                value_xx = np.gradient(value_x, exposure_grid, axis=1, edge_order=2)
                value_wx = np.gradient(value_w, exposure_grid, axis=1, edge_order=2)
                rate = np.clip(
                    -value_x / params.Lambda, params.rate_min, params.rate_max
                )
                variance = (params.sigma * x_mesh) ** 2
                generator = (
                    running_cost(x_mesh, rate, params)
                    + params.mu * x_mesh * value_w
                    + (rate + params.mu * x_mesh) * value_x
                    + 0.5 * variance * (value_ww + 2.0 * value_wx + value_xx)
                )
                value = value + sub_dt * generator
            policy[n] = rate
        return wealth_grid, exposure_grid, policy


    def evaluate_control(
        lambda_value: float,
        eta: float,
        d_brownian: np.ndarray,
        params: ModelParams,
    ) -> ControlResult:
        wealth_grid, exposure_grid, policy = solve_hjb(
            lambda_value, eta, d_brownian, params
        )
        wealth, exposure, rates = simulate_feedback_policy(
            wealth_grid, exposure_grid, policy, d_brownian, params
        )
        loss = -wealth[:, -1]
        hinge = np.maximum(loss - eta, 0.0)
        running = sum(
            running_cost(exposure[:, n], rates[:, n], params)
            for n in range(params.steps)
        ) * (params.horizon / params.steps)
        objective = float(np.mean(running))
        residual = float(eta + hinge.mean() / (1.0 - params.alpha) - params.cvar_limit)
        cvar, var = cvar_from_loss(loss, params.alpha)
        return ControlResult(
            eta=eta,
            objective=objective,
            lagrangian=objective + lambda_value * residual,
            residual=residual,
            cvar=cvar,
            var=var,
            mean_terminal_wealth=float(np.mean(wealth[:, -1])),
            terminal_wealth_std=float(np.std(wealth[:, -1])),
            loss_tail_probability=float(np.mean(loss > eta)),
            policy_grid=policy,
            wealth_grid=wealth_grid,
            exposure_grid=exposure_grid,
            terminal_wealth=wealth[:, -1],
            wealth_paths=wealth,
            exposure_paths=exposure,
            rate_paths=rates,
        )


    def unconstrained_solution(
        d_brownian: np.ndarray, params: ModelParams
    ) -> ControlResult:
        result = evaluate_control(0.0, 0.0, d_brownian, params)
        cvar, var = cvar_from_loss(-result.terminal_wealth, params.alpha)
        result.eta = var
        result.cvar = cvar
        result.var = var
        result.residual = cvar - params.cvar_limit
        return result


    def golden_eta_search(
        lambda_value: float,
        d_brownian: np.ndarray,
        params: ModelParams,
    ) -> ControlResult:
        """Golden-section minimization of the fixed-lambda Lagrangian over eta."""
        if lambda_value <= 0.0:
            return unconstrained_solution(d_brownian, params)
        low, high = params.eta_low, params.eta_high
        rho = (math.sqrt(5.0) - 1.0) / 2.0
        eta1 = high - rho * (high - low)
        eta2 = low + rho * (high - low)
        result1 = evaluate_control(lambda_value, eta1, d_brownian, params)
        result2 = evaluate_control(lambda_value, eta2, d_brownian, params)
        while high - low > params.eta_tol:
            if result1.lagrangian <= result2.lagrangian:
                high, eta2, result2 = eta2, eta1, result1
                eta1 = high - rho * (high - low)
                result1 = evaluate_control(lambda_value, eta1, d_brownian, params)
            else:
                low, eta1, result1 = eta1, eta2, result2
                eta2 = low + rho * (high - low)
                result2 = evaluate_control(lambda_value, eta2, d_brownian, params)
        return result1 if result1.lagrangian <= result2.lagrangian else result2


    def outer_lambda_bisection(
        d_brownian: np.ndarray,
        params: ModelParams,
    ) -> tuple[ControlResult, float, list[dict[str, float]]]:
        """Bisection on lambda, with golden-section eta search at every step."""
        unconstrained = unconstrained_solution(d_brownian, params)
        history = [{"lambda": 0.0, "eta": unconstrained.eta, "cvar": unconstrained.cvar}]
        if unconstrained.cvar <= params.cvar_limit:
            return unconstrained, 0.0, history

        low, high = params.lambda_low, params.lambda_high
        high_result = golden_eta_search(high, d_brownian, params)
        if high_result.cvar > params.cvar_limit:
            raise RuntimeError("Increase lambda_high; the CVaR multiplier is not bracketed.")
        best, best_lambda = high_result, high

        for _ in range(params.lambda_max_iter):
            mid = 0.5 * (low + high)
            result = golden_eta_search(mid, d_brownian, params)
            history.append({"lambda": mid, "eta": result.eta, "cvar": result.cvar})
            residual = result.cvar - params.cvar_limit
            if residual > 0.0:
                low = mid
            else:
                high, best, best_lambda = mid, result, mid
            if abs(residual) <= params.cvar_tol:
                return result, mid, history
            if high - low <= params.lambda_tol:
                return best, best_lambda, history
        return best, best_lambda, history


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


        def solve_linear(lam,eta,paths,p):
            w=np.linspace(p.wealth_min,p.wealth_max,p.wealth_nodes)
            x=np.linspace(p.x_min,p.x_max,p.exposure_nodes)
            dt=p.horizon/p.steps;dw=w[1]-w[0];dx=x[1]-x[0]
            maximum=max(abs(p.x_min),abs(p.x_max))
            transport=abs(p.mu)*maximum/dw+(max(abs(p.rate_min),abs(p.rate_max))+abs(p.mu)*maximum)/dx
            diffusion=(p.sigma*maximum)**2*(1/dw**2+2/(dw*dx)+1/dx**2)
            substeps=max(p.hjb_min_substeps,int(math.ceil(dt*(transport+diffusion)/p.hjb_cfl_safety)))
            sw,cw=gradient_stencil(w);sx,cx=gradient_stencil(x)
            policy=solve_kernel(w,x,lam,eta,p.alpha,p.steps,dt,np.full(p.steps,p.mu),np.full(p.steps,p.sigma),p.gamma,p.Lambda,p.rate_min,p.rate_max,np.full(p.steps,substeps),False,sw,cw,sx,cx)
            return w,x,policy


        solve_hjb = solve_linear
        print('Numba:', numba.__version__, '(same grids and finite-difference updates)')


    # 3. Fine-grid settings
    BASE_PARAMS = ModelParams(
        wealth_nodes=161,
        exposure_nodes=193,
        steps=320,
        samples=10000,
        cvar_limit=-0.94,
    )
    INACTIVE_PARAMS = replace(BASE_PARAMS, cvar_limit=-0.86)
    ACTIVE_PARAMS = replace(BASE_PARAMS, cvar_limit=-0.94)

    # No OOS tuning panel, feasibility buffer, or post-search adjustment.
    RUN_FULL_SEARCH = True

    FRICTIONLESS_MERTON_X = (
        BASE_PARAMS.mu / (BASE_PARAMS.gamma * BASE_PARAMS.sigma**2)
    )
    params = ACTIVE_PARAMS

    print(f"grid (W,x,t)          : {BASE_PARAMS.wealth_nodes} x "
          f"{BASE_PARAMS.exposure_nodes} x {BASE_PARAMS.steps}")
    print(f"Merton dollar exposure: {FRICTIONLESS_MERTON_X:.4f}")
    print("eta search             : golden-section")
    print("lambda search          : outer bisection")


    # 4. Solve the inactive and active constrained cases
    import sys as _sys_trace
    import time as _time_trace
    _search_counter = 0
    _search_started = _time_trace.perf_counter()
    def _log_raw_search(lam, eta, cvar):
        nonlocal _search_counter
        _search_counter += 1
        _message = f'RAW_SEARCH query={_search_counter}, lambda={lam:.10f}, eta={eta:.10f}, search_CVaR={cvar:.10f}, seconds={_time_trace.perf_counter()-_search_started:.1f}'
        print(_message, flush=True)


    _original_eta_search = golden_eta_search
    def golden_eta_search(lam,paths,params):
        result=_original_eta_search(lam,paths,params)
        _log_raw_search(lam,result.eta,result.cvar)
        return result

    d_brownian=generate_brownian_increments(BASE_PARAMS)
    inactive_result,inactive_lambda,inactive_history=outer_lambda_bisection(d_brownian,INACTIVE_PARAMS)
    active_result,active_lambda,active_history=outer_lambda_bisection(d_brownian,ACTIVE_PARAMS)
    search_metrics=[dict(case='inactive',lambda_value=inactive_lambda,eta=inactive_result.eta,cvar=inactive_result.cvar,residual=inactive_result.cvar-INACTIVE_PARAMS.cvar_limit),dict(case='active',lambda_value=active_lambda,eta=active_result.eta,cvar=active_result.cvar,residual=active_result.cvar-ACTIVE_PARAMS.cvar_limit)]
    # Freeze both policies before generating any final OOS paths.
    for case_name,result,lam in [('inactive',inactive_result,inactive_lambda),('active',active_result,active_lambda)]:
        np.savez_compressed(OUTPUT_DIR / f'frozen_{case_name}_policy.npz',wealth_grid=result.wealth_grid,exposure_grid=result.exposure_grid,policy_grid=result.policy_grid,lambda_value=lam,eta=result.eta)
    FINAL_OOS_SAMPLES=50_000
    FINAL_OOS_SEED=1_900_013
    oos_params=replace(BASE_PARAMS,samples=FINAL_OOS_SAMPLES,seed=FINAL_OOS_SEED)
    d_brownian_oos=generate_brownian_increments(oos_params)
    stock_paths=stock_paths_from_brownian(d_brownian_oos,oos_params)
    def evaluate_fixed_policy_oos(result, lambda_value, case_params):
        case_oos_params = replace(
            case_params, samples=FINAL_OOS_SAMPLES, seed=FINAL_OOS_SEED
        )
        wealth, exposure, rates = simulate_feedback_policy(
            result.wealth_grid,
            result.exposure_grid,
            result.policy_grid,
            d_brownian_oos,
            case_oos_params,
        )
        terminal_wealth = wealth[:, -1]
        loss = -terminal_wealth
        cvar, var = cvar_from_loss(loss, case_oos_params.alpha)
        hinge = np.maximum(loss - result.eta, 0.0)
        rockafellar_residual = float(
            result.eta
            + hinge.mean() / (1.0 - case_oos_params.alpha)
            - case_oos_params.cvar_limit
        )
        return replace(
            result,
            lagrangian=result.objective + lambda_value * rockafellar_residual,
            residual=rockafellar_residual,
            cvar=float(cvar),
            var=float(var),
            mean_terminal_wealth=float(np.mean(terminal_wealth)),
            terminal_wealth_std=float(np.std(terminal_wealth)),
            loss_tail_probability=float(np.mean(loss > result.eta)),
            terminal_wealth=terminal_wealth,
            wealth_paths=wealth,
            exposure_paths=exposure,
            rate_paths=rates,
        )

    def summarize(name, result, lambda_value, case_params):
        wealth_cvar = lower_tail_wealth_cvar_path(
            result.wealth_paths, case_params.alpha
        )
        return {
            "name": name,
            "params": case_params,
            "horizon": float(case_params.horizon),
            "wealth_grid": result.wealth_grid,
            "exposure_grid": result.exposure_grid,
            "policy_grid": result.policy_grid,
            "wealth": result.wealth_paths,
            "stock_paths": stock_paths,
            "dollar_exposure_paths": result.exposure_paths,
            "trade_rate_paths": result.rate_paths,
            "lambda_star": float(lambda_value),
            "eta_star": float(result.eta),
            "loss_cvar": float(result.cvar),
            "lower_tail_cvar": float(-result.cvar),
            "loss_var": float(result.var),
            "wealth_var": float(-result.var),
            "mean_terminal_wealth": float(result.mean_terminal_wealth),
            "terminal_wealth_std": float(result.terminal_wealth_std),
            "mean_dollar_position": float(np.mean(result.exposure_paths)),
            "max_dollar_position": float(np.max(result.exposure_paths)),
            "cvar_limit": float(case_params.cvar_limit),
            "cvar_path": wealth_cvar,
        }

    inactive_oos_result=evaluate_fixed_policy_oos(inactive_result,inactive_lambda,INACTIVE_PARAMS)
    active_oos_result=evaluate_fixed_policy_oos(active_result,active_lambda,ACTIVE_PARAMS)
    inactive_case=summarize('inactive',inactive_oos_result,inactive_lambda,INACTIVE_PARAMS)
    active_case=summarize('active',active_oos_result,active_lambda,ACTIVE_PARAMS)
    cases=[inactive_case,active_case]
    print('Independent OOS evaluation only; no OOS feasibility tuning panel or adjustment.')
    for case in cases:print(case['name'],'lambda=',case['lambda_star'],'eta=',case['eta_star'],'OOS CVaR=',case['loss_cvar'],'residual=',case['loss_cvar']-case['cvar_limit'])


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
        case_params = case["params"]
        simulation_params = replace(
            case_params,
            samples=path_count,
            seed=case_params.seed + seed_shift,
        )
        d_brownian = generate_brownian_increments(simulation_params)
        return simulate_feedback_policy(
            case["wealth_grid"],
            case["exposure_grid"],
            case["policy_grid"],
            d_brownian,
            simulation_params,
        )


    # Plot 1. Terminal Wealth Distribution
    def plot_terminal_wealth_distribution(
        cases,
        bins_count=44,
        figsize=(9.6, 5.5),
        save_path=None,
    ):
        all_wealth = np.concatenate(
            [case["wealth"][:, -1] for case in cases]
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
            color = CASE_COLORS[case["name"]]
            label = CASE_LABELS[case["name"]]
            ax.hist(
                case["wealth"][:, -1],
                bins=bins,
                density=True,
                color=color,
                alpha=0.12,
                edgecolor="none",
            )
            ax.plot(
                grid,
                gaussian_kde_1d(case["wealth"][:, -1], grid),
                color=color,
                linewidth=2.2,
                label=f"{label} KDE",
            )
            ax.axvline(
                case["lower_tail_cvar"],
                color=color,
                linestyle="-.",
                linewidth=1.4,
                label=f"{label} worst-tail mean wealth = {case['lower_tail_cvar']:.2f}",
            )
            terminal_mean = float(np.mean(case["wealth"][:, -1]))
            ax.axvline(
                terminal_mean,
                color=color,
                linestyle="--",
                linewidth=1.4,
                label=f"{label} mean = {terminal_mean:.2f}",
            )

        ax.set_xlabel(r"terminal wealth $W_T^*$", fontsize=16)
        ax.set_xlim(0.6,1.5)
        ax.set_ylim(0.0, 11.0)
        ax.set_ylabel("density", fontsize=16)
        ax.legend(loc="upper right", frameon=True, fontsize=10)
        fig.tight_layout()
        if save_path is not None:
            save_figure(fig, save_path, bbox_inches="tight")
        return fig, ax


    fig_terminal, ax_terminal = plot_terminal_wealth_distribution(
        cases, save_path=FIGURE_DIR / "linear_w_dist.png"
    )


    # Plot 2. Dollar Risky Exposure
    def plot_exposure_trajectory(cases, figsize=(9.6, 5.2), include_band=True, save_path=None):
        fig, ax = plt.subplots(figsize=figsize)
        for case in cases:
            name = case["name"]
            color = CASE_COLORS.get(name, "black")
            label = CASE_LABELS.get(name, name)
            exposure_paths = case["dollar_exposure_paths"]
            t = np.linspace(0.0, case["horizon"], exposure_paths.shape[1])
            mean_exposure = np.mean(exposure_paths, axis=0)
            q10, q90 = np.quantile(exposure_paths, [0.10, 0.90], axis=0)
            if include_band:
                ax.fill_between(t, q10, q90, color=color, alpha=0.10, linewidth=0)
            ax.plot(
                t,
                mean_exposure,
                color=color,
                linewidth=2.2,
                label=f"{label} risky dollar amount",
            )

        ax.axhline(0.0, color="0.45", linewidth=0.8, alpha=0.7)
        ax.axhline(
            FRICTIONLESS_MERTON_X,
            color="0.35",
            linestyle="--",
            linewidth=1.5,
            label=f"Merton line = {FRICTIONLESS_MERTON_X:.2f}",
        )
        ax.set_xlim(0.0, max(case["horizon"] for case in cases))
        ax.set_ylim(0.0, 0.6)
        ax.set_xlabel("time", fontsize=16)
        ax.set_ylabel("dollar risky amount", fontsize=16)
        handles = ax.get_legend_handles_labels()[0]
        if include_band:
            handles += [
                Patch(
                    facecolor=CASE_COLORS.get(case["name"], "black"),
                    alpha=0.10,
                    edgecolor="none",
                    label=(
                        f"{CASE_LABELS.get(case['name'], case['name'])} 10-90% range"
                    ),
                )
                for case in cases
            ]
        ax.legend(
            handles=handles,
            loc="upper left",
            frameon=True,
            framealpha=0.92,
            edgecolor="0.85",
            fontsize=13,
        )
        fig.tight_layout()
        if save_path is not None:
            save_figure(fig, save_path, bbox_inches="tight")
        return fig, ax


    fig_exposure, ax_exposure = plot_exposure_trajectory(
        cases, save_path=FIGURE_DIR / "linear_exposure.png"
    )


    # Plot 3. Wealth CVaR Over Time
    def plot_wealth_cvar_over_time(cases, figsize=(9.6, 5.2), save_path=None):
        fig, ax = plt.subplots(figsize=figsize)
        for case in cases:
            name = case["name"]
            color = CASE_COLORS.get(name, "black")
            label = CASE_LABELS.get(name, name)
            wealth_cvar = case["cvar_path"]
            terminal_gap = abs(-wealth_cvar[-1] - case["loss_cvar"])
            if terminal_gap > 1.0e-12:
                raise AssertionError(
                    f"{name}: table/figure terminal CVaR mismatch "
                    f"({terminal_gap:.3e})"
                )
            t = np.linspace(0.0, case["horizon"], wealth_cvar.size)
            ax.plot(t, -wealth_cvar, color=color, linewidth=2.2, label=f"{label} CVaR over time")
            ax.axhline(
                case["cvar_limit"],
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
        ax.legend(loc="upper left", frameon=True, framealpha=0.92, edgecolor="0.85", fontsize=16)
        fig.tight_layout()
        if save_path is not None:
            save_figure(fig, save_path, bbox_inches="tight")
        return fig, ax


    fig_cvar_time, ax_cvar_time = plot_wealth_cvar_over_time(
        cases, save_path=FIGURE_DIR / "linear_cvar.png"
    )


    # Plot 4. Good And Bad Market Paths
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
        if path_count < 2 or not 1 <= maturity_window <= active_case["params"].steps + 1:
            raise ValueError("invalid path_count or maturity_window")
        plot_params = replace(
            active_case["params"],
            samples=path_count,
            seed=active_case["params"].seed + 252000,
        )
        d_brownian_plot = generate_brownian_increments(plot_params)
        wealth_paths, exposure_paths, control_paths = simulate_feedback_policy(
            active_case["wealth_grid"],
            active_case["exposure_grid"],
            active_case["policy_grid"],
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
                label=f"{label}: "+r"$W_T^*$="+f"{wealth_paths[idx, -1]:.2f}",
            )
            ax_exposure.plot(
                t,
                exposure_paths[idx],
                color=color,
                linewidth=2.0,
                label=f"{label}: "+r"$\varphi_T^*$="+f"{late_exposure:.2f}",
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
        ax_stock.set_ylabel("wealth", fontsize=16)
        ax_exposure.set_ylabel("dollar risky amount", fontsize=16)
        ax_control.set_ylabel(r"trading speed", fontsize=16)
        ax_control.set_xlabel("time", fontsize=16)
        displayed_wealth = wealth_paths[[good_idx, bad_idx]]
        ax_stock.set_ylim(min(0.85, float(displayed_wealth.min())-0.01), max(1.10, float(displayed_wealth.max())+0.01))
        ax_exposure.set_ylim(-0.5, 0.5)
        ax_control.set_ylim(-2.0, 1.70)
        ax_control.set_xlim(0.0, 1.0)
        ax_control.axhline(0.0, color="0.45", linewidth=0.8, alpha=0.7)
        ax_stock.legend(
            loc="lower left", frameon=True, framealpha=0.92, edgecolor="0.85", fontsize=14
        )
        ax_exposure.legend(
            loc="lower left", frameon=True, framealpha=0.92, edgecolor="0.85", fontsize=14
        )
        ax_control.legend(
            loc="lower left", frameon=True, framealpha=0.92, edgecolor="0.85", fontsize=14
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
        active_case, save_path=FIGURE_DIR / "linear_sample.png"
    )


    # Plot 5. Control Trajectory
    def plot_control_trajectory(cases, figsize=(9.6, 5.2), include_band=True, save_path=None):
        fig, ax = plt.subplots(figsize=figsize)
        for case in cases:
            name = case["name"]
            color = CASE_COLORS.get(name, "black")
            label = CASE_LABELS.get(name, name)
            control_paths = case["trade_rate_paths"]
            t = np.linspace(0.0, case["horizon"], control_paths.shape[1], endpoint=False)
            mean_control = np.mean(control_paths, axis=0)
            q10, q90 = np.quantile(control_paths, [0.10, 0.90], axis=0)

            if include_band:
                ax.fill_between(t, q10, q90, color=color, alpha=0.10, linewidth=0)
            ax.plot(t, mean_control, color=color, linewidth=2.2, label=f"{label} mean control")

        ax.axhline(0.0, color="0.45", linewidth=0.8, alpha=0.7)
        ax.set_xlim(0.0, max(case["horizon"] for case in cases))
        ax.set_xlabel("time", fontsize=16)
        ax.set_ylabel(r"trading speed $\dot\varphi_t^*$", fontsize=16)
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
        ax.legend(
            handles=handles,
            loc="upper right",
            frameon=True,
            framealpha=0.92,
            edgecolor="0.85",
            fontsize=16,
        )
        fig.tight_layout()
        if save_path is not None:
            save_figure(fig, save_path, bbox_inches="tight")
        return fig, ax


    fig_control, ax_control = plot_control_trajectory(
        cases, save_path=FIGURE_DIR / "linear_control.png"
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
    parser = argparse.ArgumentParser(description='Quadratic trading-rate regularization experiment')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Output folder (default: results/linear in the current directory).')
    parser.add_argument('--no-numba', action='store_true',
                        help='Use the original NumPy solver instead of optional Numba acceleration.')
    args = parser.parse_args()
    run_experiment(output_dir=args.output_dir, use_numba=not args.no_numba)


if __name__ == '__main__':
    main()
