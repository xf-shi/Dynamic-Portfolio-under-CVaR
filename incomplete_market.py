# Python 3.12; numpy==2.3.5 numba==0.62.1 llvmlite==0.45.1 matplotlib==3.11.1
# For beta>0, Milstein evaluates the equivalent scalar wealth diffusion.
import argparse
import csv
import hashlib
import inspect
import json
import math
from pathlib import Path
from dataclasses import dataclass, asdict
from math import exp, sqrt
from statistics import NormalDist
import numpy as np
import numba
from numba import njit, prange, get_num_threads, set_num_threads

ENTRY = 'incomplete'
IS_ASSET = True
CASES = [
    ('beta_0.020_c_-0.86', {'model': 'incomplete', 'beta': 0.02, 'steps': 1600, 'dz': 0.00015625, 'dx': 0.005}, -0.86, {'paths': 65536, 'seed': 82641279, 'subdivisions': 16, 'method': 'milstein', 'batch_size': 32768}),
    ('beta_0.020_c_-0.94', {'model': 'incomplete', 'beta': 0.02, 'steps': 1600, 'dz': 0.00015625, 'dx': 0.005, 'x_bound': 1.0}, -0.94, {'paths': 65536, 'seed': 82641279, 'subdivisions': 16, 'method': 'milstein', 'batch_size': 32768}),
    ('beta_0.000_c_-0.94', {'steps': 1600, 'dz': 0.00015625, 'dx': 0.005}, -0.94, {'paths': 65536, 'seed': 82641279, 'subdivisions': 8, 'method': 'milstein', 'batch_size': 32768}),
    ('beta_0.005_c_-0.94', {'model': 'incomplete', 'beta': 0.005, 'steps': 1600, 'dz': 0.00015625, 'dx': 0.005}, -0.94, {'paths': 65536, 'seed': 82641279, 'subdivisions': 16, 'method': 'milstein', 'batch_size': 32768}),
    ('beta_0.010_c_-0.94', {'model': 'incomplete', 'beta': 0.01, 'steps': 1600, 'dz': 0.00015625, 'dx': 0.005}, -0.94, {'paths': 65536, 'seed': 82641279, 'subdivisions': 16, 'method': 'milstein', 'batch_size': 32768}),
    ('beta_0.015_c_-0.94', {'model': 'incomplete', 'beta': 0.015, 'steps': 1600, 'dz': 0.00015625, 'dx': 0.005}, -0.94, {'paths': 65536, 'seed': 82641279, 'subdivisions': 16, 'method': 'milstein', 'batch_size': 32768}),
    ('beta_0.025_c_-0.94', {'model': 'incomplete', 'beta': 0.025, 'steps': 1600, 'dz': 0.00015625, 'dx': 0.005}, -0.94, {'paths': 65536, 'seed': 82641279, 'subdivisions': 16, 'method': 'milstein', 'batch_size': 32768}),
    ('beta_0.000_c_-0.86', {'steps': 1600, 'dz': 0.00015625, 'dx': 0.005}, -0.86, {'paths': 65536, 'seed': 82641279, 'subdivisions': 8, 'method': 'milstein', 'batch_size': 32768}),
    ('beta_0.000_c_-0.88', {'steps': 1600, 'dz': 0.00015625, 'dx': 0.005}, -0.88, {'paths': 65536, 'seed': 82641279, 'subdivisions': 8, 'method': 'milstein', 'batch_size': 32768}),
    ('beta_0.000_c_-0.90', {'steps': 1600, 'dz': 0.00015625, 'dx': 0.005}, -0.9, {'paths': 65536, 'seed': 82641279, 'subdivisions': 8, 'method': 'milstein', 'batch_size': 32768}),
    ('beta_0.000_c_-0.92', {'steps': 1600, 'dz': 0.00015625, 'dx': 0.005}, -0.92, {'paths': 65536, 'seed': 82641279, 'subdivisions': 8, 'method': 'milstein', 'batch_size': 32768}),
    ('beta_0.020_c_-0.88', {'model': 'incomplete', 'beta': 0.02, 'steps': 1600, 'dz': 0.00015625, 'dx': 0.005}, -0.88, {'paths': 65536, 'seed': 82641279, 'subdivisions': 16, 'method': 'milstein', 'batch_size': 32768}),
    ('beta_0.020_c_-0.90', {'model': 'incomplete', 'beta': 0.02, 'steps': 1600, 'dz': 0.00015625, 'dx': 0.005}, -0.9, {'paths': 65536, 'seed': 82641279, 'subdivisions': 16, 'method': 'milstein', 'batch_size': 32768}),
    ('beta_0.020_c_-0.92', {'model': 'incomplete', 'beta': 0.02, 'steps': 1600, 'dz': 0.00015625, 'dx': 0.005}, -0.92, {'paths': 65536, 'seed': 82641279, 'subdivisions': 16, 'method': 'milstein', 'batch_size': 32768}),
]

@dataclass(frozen=True)
class Config:
    model: str = 'complete'
    mu: float = 0.08
    sigma: float = 0.2
    gamma: float = 5.0
    alpha: float = 0.95
    w0: float = 1.0
    T: float = 1.0
    beta: float = 0.0
    coefficient: float = 0.01
    steps: int = 100
    dz: float = 0.00125
    z_bound: float = 1.5
    dx: float = 0.02
    x_bound: float = 2.0
    action_bound: float = 2.0
    cfl: float = 0.85
    n_eta: int = 64
    n_lambda: int = 20
    oos_paths: int = 50000
    oos_seed: int = 1900013

def bounds(p, budget):
    if p.model in ('complete', 'incomplete'):
        M = abs(p.w0) + p.mu * p.action_bound * p.T + sqrt(p.T * ((p.sigma * p.action_bound) ** 2 + p.beta ** 2))
    else:
        epsilon = 1 / p.T
        K = 2 * p.mu + p.sigma ** 2 + epsilon
        integral = p.action_bound ** 2 / (epsilon * K) * ((exp(K * p.T) - 1) / K - p.T)
        costmax = p.coefficient / 1.5 * p.action_bound ** 1.5 if p.model == 'square_root' else 0.0
        M = abs(p.w0) + p.mu * sqrt(p.T * integral) + p.sigma * sqrt(integral) + costmax * p.T
    normal = NormalDist()
    no_trade_cvar = -p.w0 + p.beta * sqrt(p.T) * normal.pdf(normal.inv_cdf(p.alpha)) / (1 - p.alpha)
    delta = 0.5 * (budget - no_trade_cvar)
    if delta <= 0:
        raise ValueError('This calibration has no certified zero-trade Slater point')
    lower_J = p.T * (0.5 * p.gamma * p.beta ** 2 - p.mu ** 2 / (2 * p.gamma * p.sigma ** 2))
    Jbar = 0.5 * p.gamma * p.beta ** 2 * p.T
    return dict(interval=[-M / sqrt(p.alpha), M / sqrt(1 - p.alpha)], upper=(Jbar - lower_J) / delta, delta=delta, no_trade_cvar=no_trade_cvar, lower_J=lower_J, moment_bound=M)

@dataclass(frozen=True)
class Result:
    lam: float
    eta: float
    J: float
    C: float
    value: float

def golden(control, lam, interval, iterations, risk_only=False):
    if iterations < 1 or not interval[0] < interval[1]:
        raise ValueError('N_eta >= 1 and a nonempty interval are required')
    low, high = interval
    rho = (sqrt(5.0) - 1.0) / 2.0
    e1 = high - rho * (high - low)
    e2 = low + rho * (high - low)
    p1, p2 = (control(lam, e1), control(lam, e2))
    score = (lambda p: p.C) if risk_only else lambda p: p.value
    for _ in range(iterations):
        if score(p1) <= score(p2):
            high, e2, p2 = (e2, e1, p1)
            e1 = high - rho * (high - low)
            p1 = control(lam, e1)
        else:
            low, e1, p1 = (e1, e2, p2)
            e2 = low + rho * (high - low)
            p2 = control(lam, e2)
    return p1 if score(p1) <= score(p2) else p2

def nested(control, interval, upper, n_eta=64, n_lambda=20):
    if upper <= 0 or n_lambda < 1:
        raise ValueError('Positive Slater bound and N_lambda >= 1 required')
    initial = golden(control, 0.0, interval, n_eta, risk_only=True)
    trace = [dict(iteration=0, low=0.0, high=upper, **asdict(initial))]
    if initial.C <= 0.0:
        return (initial, trace)
    low, high = (0.0, upper)
    for k in range(1, n_lambda + 1):
        mid = (low + high) / 2.0
        current = golden(control, mid, interval, n_eta)
        trace.append(dict(iteration=k, low=low, high=high, **asdict(current)))
        if current.C == 0.0:
            return (current, trace)
        if current.C > 0.0:
            low = mid
        else:
            high = mid
    return (current, trace)

def slater_check(oracle, b):
    p = oracle.p
    if oracle.asset:
        s = oracle.solve(0.0, forced_zero=True)
        eta = -p.w0 + p.beta * p.T ** 0.5 * NormalDist().inv_cdf(p.alpha)
        z = p.w0 + eta
        H = float(np.interp(z, oracle.z, s['H']))
        C = eta + H - oracle.budget
    else:
        C = -p.w0 - oracle.budget
    if C > -b['delta']:
        raise AssertionError('Discrete no-trade Slater check failed; refine the grid')
    return dict(discrete_no_trade_C=C, required_upper_C=-b['delta'])

@njit(cache=True)
def clip(x, a, b):
    return min(b, max(a, x))

@njit(cache=True)
def optimal(dm, dp, dd, mu, sigma, gamma, bound):
    best = 1e+300
    action = 0.0
    for sign in range(2):
        lo = -bound if sign == 0 else 0.0
        hi = 0.0 if sign == 0 else bound
        d = dm if sign == 0 else dp
        qa = sigma * sigma * (gamma + dd)
        qb = mu * (d - 1.0)
        candidate = clip(-qb / max(qa, 1e-16), lo, hi) if qa > 0 else lo
        for a in (candidate, lo, hi):
            h = 0.5 * qa * a * a + qb * a
            if h < best:
                best = h
                action = a
    return action

@njit(cache=True)
def thomas(lower, diagonal, upper, rhs):
    n = len(rhs)
    c = np.empty(n)
    d = np.empty(n)
    out = np.empty(n)
    c[0] = upper[0] / diagonal[0]
    d[0] = rhs[0] / diagonal[0]
    for i in range(1, n):
        den = diagonal[i] - lower[i] * c[i - 1]
        c[i] = upper[i] / den
        d[i] = (rhs[i] - lower[i] * d[i - 1]) / den
    out[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        out[i] = d[i] - c[i] * out[i + 1]
    return out

@njit(cache=True)
def solve_asset_saved(z, lam, alpha, mu, sigma, beta, gamma, bound, steps, T, storage, Hstore, stride, forced_zero=False):
    n = len(z)
    h = z[1] - z[0]
    dt = T / steps
    k = lam / (1 - alpha)
    J = np.zeros(n)
    H = np.maximum(-z, 0) / (1 - alpha)
    Hstore[-1, :] = H
    lefta = 0.0 if forced_zero else clip((1 + k) * mu / (gamma * sigma * sigma), -bound, bound)
    righta = 0.0 if forced_zero else clip(mu / (gamma * sigma * sigma), -bound, bound)
    fl = 0.5 * gamma * ((sigma * lefta) ** 2 + beta ** 2) - mu * lefta
    fr = 0.5 * gamma * ((sigma * righta) ** 2 + beta ** 2) - mu * righta
    worst = 0
    maxdefect = 0.0
    for t in range(steps - 1, -1, -1):
        tau = (steps - t) * dt
        trial = J + lam * H
        lower = np.zeros(n)
        diag = np.ones(n)
        upper = np.zeros(n)
        actions = np.empty(n)
        actions[0] = lefta
        actions[-1] = righta
        for iteration in range(100):
            rhsJ = J.copy()
            rhsH = H.copy()
            rhsJ[0] = tau * fl
            rhsJ[-1] = tau * fr
            rhsH[0] = (-z[0] - tau * mu * lefta) / (1 - alpha)
            rhsH[-1] = 0.0
            for i in range(1, n - 1):
                a = 0.0 if forced_zero else optimal((trial[i] - trial[i - 1]) / h, (trial[i + 1] - trial[i]) / h, (trial[i + 1] - 2 * trial[i] + trial[i - 1]) / (h * h), mu, sigma, gamma, bound)
                actions[i] = a
                variance = (sigma * a) ** 2 + beta ** 2
                qm = 0.5 * variance / (h * h) + max(-mu * a, 0) / h
                qp = 0.5 * variance / (h * h) + max(mu * a, 0) / h
                lower[i] = -dt * qm
                upper[i] = -dt * qp
                diag[i] = 1 + dt * (qm + qp)
                rhsJ[i] += dt * (0.5 * gamma * variance - mu * a)
            newJ = thomas(lower, diag, upper, rhsJ)
            newH = thomas(lower, diag, upper, rhsH)
            new = newJ + lam * newH
            difference = np.max(np.abs(new - trial))
            trial = new
            if difference < 2e-12:
                break
        worst = max(worst, iteration + 1)
        if iteration == 99:
            raise ValueError('Asset policy iteration did not converge')
        J = newJ
        H = newH
        if t % stride == 0:
            Hstore[t // stride, :] = H
        if storage.shape[0] > 1:
            storage[t, :] = actions
        for i in range(1, n - 1):
            dm = (trial[i] - trial[i - 1]) / h
            dp = (trial[i + 1] - trial[i]) / h
            dd = (trial[i + 1] - 2 * trial[i] + trial[i - 1]) / (h * h)
            a = actions[i]
            ao = 0.0 if forced_zero else optimal(dm, dp, dd, mu, sigma, gamma, bound)

            def ham(b):
                var = (sigma * b) ** 2 + beta ** 2
                return 0.5 * gamma * var - mu * b + mu * b * (dp if b >= 0 else dm) + 0.5 * var * dd
            maxdefect = max(maxdefect, abs(ham(a) - ham(ao)))
    return (J, H, worst, maxdefect)

@njit(cache=True, inline='always')
def position(y, start, h, n):
    pos = (y - start) / h
    i = max(0, min(n - 2, int(math.floor(pos))))
    a = min(1.0, max(0.0, pos - i))
    return (i, a, pos)

@njit(cache=True)
def step_asset_cv(W, M, cost, exposure, policy, H0, H1, tw, z0, dz, eta, db, dbp, mu, sigma, beta, gamma, dt, milstein):
    outside = 0
    for n in range(len(W)):
        i, a, pos = position(W[n] + eta, z0, dz, len(policy))
        if pos < 0 or pos > len(policy) - 1:
            outside += 1
        control = (1 - a) * policy[i] + a * policy[i + 1]
        da = (policy[i + 1] - policy[i]) / dz if 0 <= pos <= len(policy) - 1 else 0.0
        dh = ((1 - tw) * (float(H0[i + 1]) - float(H0[i])) + tw * (float(H1[i + 1]) - float(H1[i]))) / dz
        shock = sigma * control * db[n] + beta * dbp[n]
        M[n] += dh * shock
        cost[n] += dt * (0.5 * gamma * ((sigma * control) ** 2 + beta ** 2) - mu * control)
        exposure[n] += dt * control
        correction = 0.0
        if milstein:
            square = db[n] * db[n]
            if beta:
                square = shock * shock / ((sigma * control) ** 2 + beta ** 2)
            correction = 0.5 * sigma * sigma * control * da * (square - dt)
        W[n] += mu * control * dt + shock + correction
    return (outside, 0)

def empirical_cvar(loss, alpha):
    loss = np.asarray(loss, dtype=float).reshape(-1)
    if loss.size < 1 or not np.isfinite(loss).all() or (not 0 < alpha < 1):
        raise ValueError('Finite losses and 0<alpha<1 required')
    eta = float(np.quantile(loss, alpha, method='inverted_cdf'))
    tail = np.maximum(loss - eta, 0) / (1 - alpha)
    se = float(tail.std(ddof=1) / math.sqrt(len(loss))) if len(loss) > 1 else 0.0
    return (float(eta + tail.mean()), eta, se)

def summarize(W, M, cost, exposure, eta, alpha, w0, T, budget):
    raw = eta + np.maximum(-W - eta, 0) / (1 - alpha)
    corrected = raw - M
    empirical, q, se = empirical_cvar(-W, alpha)
    empirical_tail = q + np.maximum(-W - q, 0) / (1 - alpha) - M
    N = len(W)
    root = math.sqrt(N)
    out = dict(N=N, raw_empirical_cvar=empirical, raw_cvar_se=se, empirical_eta=q, raw_frozen_risk=float(raw.mean()), raw_frozen_se=float(raw.std(ddof=1) / root), cv_frozen_risk=float(corrected.mean()), cv_frozen_se=float(corrected.std(ddof=1) / root), cv_cvar=float(empirical - M.mean()), cv_cvar_se=float(empirical_tail.std(ddof=1) / root), eta_optimization_gap=float(raw.mean() - empirical), martingale_mean=float(M.mean()), martingale_se=float(M.std(ddof=1) / root), J=float(cost.mean()), mean_exposure=float(exposure.mean() / T), mean_return_pct=float(100 * (W.mean() / w0 - 1)), variance_reduction_factor=float(raw.var(ddof=1) / corrected.var(ddof=1)) if corrected.var() > 0 else 0.0, fixed_eta_C=float(corrected.mean() - budget))
    return (out, corrected)

class Oracle:

    def __init__(self, p, budget, directory, order=3):
        self.p, self.budget = (p, budget)
        self.z = np.linspace(-p.z_bound, p.z_bound, 2 * round(p.z_bound / p.dz) + 1)
        self.x = np.array([0.0]) if IS_ASSET else np.linspace(-p.x_bound, p.x_bound, 2 * round(p.x_bound / p.dx) + 1)
        self.current = self.lam = None
        self.calls = 0
        self.cache = Path(directory) / 'oracle_cache'
        self.cache.mkdir(parents=True, exist_ok=True)
        if not IS_ASSET:
            self.geometry = geometry(p, order)
            self.stencils = stencils(p, self.geometry)

    @property
    def asset(self):
        return IS_ASSET

    def solve(self, lam, store=None, forced_zero=False):
        p = self.p
        cached = self.cache / (float(lam).hex().replace('.', '_') + '.npz')
        if store is None and (not forced_zero) and cached.exists():
            with np.load(cached) as data:
                return dict(J=data['J'].copy(), H=data['H'].copy())
        print(f'lambda={lam:.12g}', flush=True)
        if IS_ASSET:
            policy = np.zeros((1, 1)) if store is None else np.lib.format.open_memmap(store, mode='w+', dtype=np.float64, shape=(p.steps, len(self.z)))
            Hsave = np.zeros((1, len(self.z)), np.float32) if store is None else np.lib.format.open_memmap(Path(store).parent / 'risk_snapshots.npy', mode='w+', dtype=np.float32, shape=(p.steps + 1, len(self.z)))
            J, H, _, _ = solve_asset_saved(self.z, lam, p.alpha, p.mu, p.sigma, p.beta, p.gamma, p.action_bound, p.steps, p.T, policy, Hsave, p.steps + 1 if store is None else 1, forced_zero)
        else:
            x, rr, w, br, count, c0, c1, c2, K = self.geometry
            policy = np.zeros((1, 1, 1)) if store is None else np.lib.format.open_memmap(store, mode='w+', dtype=np.float64, shape=(p.steps, len(self.z), len(x)))
            Hsave = np.zeros((1, 1, 1), np.float32) if store is None else np.lib.format.open_memmap(Path(store).parent / 'risk_snapshots.npy', mode='w+', dtype=np.float32, shape=(p.steps + 1, len(self.z), len(x)))
            J, H, error = solve_sl(self.z, x, lam, p.alpha, rr, w, br, count, c0, c1, c2, K, *self.stencils, p.steps, p.T, policy, Hsave, 1)
            if error > 1e-08:
                raise AssertionError('Bellman component consistency failed')
            if not np.isfinite(J).all() or not np.isfinite(H).all() or H.min() < -1e-07:
                raise FloatingPointError('Invalid Bellman components')
            J, H = (J[:, len(x) // 2], H[:, len(x) // 2])
        if not np.isfinite(J).all() or not np.isfinite(H).all() or H.min() < -1e-07:
            raise FloatingPointError('Invalid Bellman components')
        if store is not None:
            policy.flush()
            Hsave.flush()
        elif not forced_zero:
            temporary = cached.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, J=J, H=H)
            temporary.replace(cached)
        return dict(J=J, H=H)

    def set_lambda(self, lam):
        if self.lam != lam:
            self.current = self.solve(lam)
            self.lam = lam

    def at(self, lam, eta):
        self.calls += 1
        self.set_lambda(lam)
        p = self.p
        s = self.current
        z = p.w0 + eta
        if z < self.z[0]:
            J = float(s['J'][0])
            H = float(s['H'][0]) + (self.z[0] - z) / (1 - p.alpha)
        elif z > self.z[-1]:
            J = float(s['J'][-1])
            H = 0.0
        else:
            J = float(np.interp(z, self.z, s['J']))
            H = float(np.interp(z, self.z, s['H']))
        C = eta + H - self.budget
        return Result(float(lam), float(eta), J, C, J + lam * C)

    def materialize(self, result, path):
        self.set_lambda(result.lam)
        original = self.current
        replay = self.solve(result.lam, store=str(path))
        error = max((float(np.max(np.abs(original[k] - replay[k]))) for k in ('J', 'H')))
        if error > 1e-11:
            raise AssertionError(f'Final policy replay changed component values: {error}')
        self.current = replay
        checked = self.at(result.lam, result.eta)
        if checked != result:
            raise AssertionError('Replayed result differs')
        return (replay, error)

def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='ascii')
    temporary.replace(path)

def solve_case(p, budget, directory, order=3):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    plan = dict(config=asdict(p), budget=budget, order=order, source=SOURCE_ID, numpy=np.__version__, numba=numba.__version__)
    path = directory / 'solver_plan.json'
    if path.exists() and json.loads(path.read_text(encoding='ascii')) != plan:
        raise ValueError('Output belongs to another configuration or source version')
    if not path.exists() and any(directory.iterdir()):
        raise ValueError('Use a new output directory')
    write_json(path, plan)
    completed = directory / 'solution.json'
    if completed.exists():
        frozen = json.loads(completed.read_text(encoding='ascii'))
        policy = np.load(directory / 'policy.npy', mmap_mode='r')
        H = np.load(directory / 'risk_snapshots.npy', mmap_mode='r')
        if policy.shape[0] != p.steps or H.shape != (p.steps + 1,) + policy.shape[1:]:
            raise ValueError('Incomplete saved policy')
        return frozen
    oracle = Oracle(p, budget, directory, order)
    b = bounds(p, budget)
    slater = slater_check(oracle, b)
    result, trace = nested(oracle.at, b['interval'], b['upper'], p.n_eta, p.n_lambda)
    write_json(directory / 'search_trace.json', trace)
    oracle.set_lambda(result.lam)
    values = oracle.current['J'] + result.lam * (oracle.z - p.w0 + oracle.current['H'] - budget)
    if result.value - float(values.min()) > 1e-08:
        raise AssertionError('Golden search exceeded a grid candidate')
    info, error = oracle.materialize(result, directory / 'policy.npy')
    np.savez_compressed(directory / 'curves.npz', z=oracle.z, x=oracle.x, J=info['J'], H=info['H'])
    frozen = dict(config=asdict(p), budget=budget, solution=asdict(result), bounds=b, slater=slater, replay_error=error, strategy_recovery=False)
    write_json(completed, frozen)
    return frozen

def evaluation_folder(directory, settings):
    return Path(directory) / 'evaluation' / f"{settings['method']}_m{settings['subdivisions']}_N{settings['paths']}_seed{settings['seed']}"

def saved_statistics(directory, settings):
    directory = Path(directory)
    frozen = json.loads((directory / 'solution.json').read_text(encoding='ascii'))
    p, s = (Config(**frozen['config']), frozen['solution'])
    folder = evaluation_folder(directory, settings)
    arrays = {k: [] for k in ('W', 'M', 'cost', 'exposure')}
    outz = outx = 0
    batches = sorted(folder.glob('batch_*.npz'))
    if len(batches) != math.ceil(settings['paths'] / settings['batch_size']):
        raise ValueError('Incomplete evaluation')
    for saved in batches:
        with np.load(saved) as data:
            for key in arrays:
                arrays[key].append(data[key].copy())
            outz += int(data['outz'])
            outx += int(data['outx'])
    arrays = {k: np.concatenate(v) for k, v in arrays.items()}
    if len(arrays['W']) != settings['paths']:
        raise ValueError('Evaluation path count mismatch')
    stats, _ = summarize(arrays['W'], arrays['M'], arrays['cost'], arrays['exposure'], s['eta'], p.alpha, p.w0, p.T, frozen['budget'])
    total = settings['paths'] * p.steps * settings['subdivisions']
    stats.update(method=settings['method'], subdivisions=settings['subdivisions'], simulation_steps=p.steps * settings['subdivisions'], seed=settings['seed'], cv_frozen_CI95=[stats['cv_frozen_risk'] - 1.96 * stats['cv_frozen_se'], stats['cv_frozen_risk'] + 1.96 * stats['cv_frozen_se']], oracle_to_SDE_risk_difference=stats['cv_frozen_risk'] - (frozen['budget'] + s['C']), outz_fraction=outz / total, outx_fraction=outx / total, strategy_recovery=False)
    return (frozen, stats)

def evaluate(directory, settings):
    directory = Path(directory)
    frozen = json.loads((directory / 'solution.json').read_text(encoding='ascii'))
    p, s = (Config(**frozen['config']), frozen['solution'])
    paths, seed, m = (settings['paths'], settings['seed'], settings['subdivisions'])
    if settings['method'] not in ('euler', 'milstein'):
        raise ValueError('Unknown evaluation method')
    policy = np.load(directory / 'policy.npy', mmap_mode='r')
    H = np.load(directory / 'risk_snapshots.npy', mmap_mode='r')
    if H.shape != (p.steps + 1,) + policy.shape[1:]:
        raise ValueError('Policy and risk snapshots do not match')
    folder = evaluation_folder(directory, settings)
    folder.mkdir(parents=True, exist_ok=True)
    plan = dict(config=asdict(p), solution=s, settings=settings, source=SOURCE_ID, numpy=np.__version__, numba=numba.__version__)
    path = folder / 'frozen_evaluation.json'
    if path.exists() and json.loads(path.read_text(encoding='ascii')) != plan:
        raise ValueError('Evaluation settings changed')
    write_json(path, plan)
    dt = p.T / (p.steps * m)
    if not IS_ASSET:
        moments = cost_moments(p.mu, p.sigma, p.T / p.steps)
    for batch, offset in enumerate(range(0, paths, settings['batch_size'])):
        size = min(settings['batch_size'], paths - offset)
        saved = folder / f'batch_{batch:04d}.npz'
        if saved.exists():
            with np.load(saved) as data:
                if len(data['W']) != size:
                    raise ValueError('Incomplete batch')
            continue
        W, X, M = (np.full(size, p.w0), np.zeros(size), np.zeros(size))
        cost, exposure, zeros = (np.zeros(size), np.zeros(size), np.zeros(size))
        rng = np.random.default_rng(np.random.SeedSequence([seed, batch, 0]))
        rngp = np.random.default_rng(np.random.SeedSequence([seed, batch, 1]))
        outz = outx = 0
        for t in range(p.steps):
            if not IS_ASSET:
                held = lookup_rate(W, X, policy[t], -p.z_bound, p.dz, -p.x_bound, p.dx, s['eta'])
                conditional_cost(cost, exposure, X, held, moments, p.mu, p.sigma, p.gamma, p.coefficient, p.model == 'square_root', p.T / p.steps)
            for k in range(m):
                db = rng.standard_normal(size) * math.sqrt(dt)
                dbp = rngp.standard_normal(size) * math.sqrt(dt) if p.beta else zeros
                tw = k / m
                if IS_ASSET:
                    oz, ox = step_asset_cv(W, M, cost, exposure, policy[t], H[t], H[t + 1], tw, -p.z_bound, p.dz, s['eta'], db, dbp, p.mu, p.sigma, p.beta, p.gamma, dt, settings['method'] == 'milstein')
                else:
                    oz, ox = step_held(W, X, M, held, H[t], H[t + 1], tw, -p.z_bound, p.dz, -p.x_bound, p.dx, s['eta'], db, p.mu, p.sigma, p.coefficient, p.model == 'square_root', dt, settings['method'] == 'milstein')
                outz += oz
                outx += ox
        temporary = saved.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, W=W, M=M, cost=cost, exposure=exposure, outz=outz, outx=outx)
        temporary.replace(saved)
        print(f'{directory.name}: {offset + size}/{paths} paths', flush=True)
    _, stats = saved_statistics(directory, settings)
    write_json(folder / 'statistics.json', stats)
    return (frozen, stats)

def summary_row(key, frozen, stats):
    p, s = (Config(**frozen['config']), frozen['solution'])
    C = stats['cv_frozen_risk'] - frozen['budget']
    band = 1.96 * stats['cv_frozen_se']
    error = abs(C) + band if s['lam'] > 0 else C + band
    passed = error <= 0.0001 and abs(stats['oracle_to_SDE_risk_difference']) + band <= 0.0001
    passed = passed and (abs(s['C']) <= 0.0001 if s['lam'] > 0 else s['C'] <= 0)
    return dict(case=key, model=p.model, beta=p.beta, budget=frozen['budget'], lam=s['lam'], eta=s['eta'], J=s['J'], C_oracle=s['C'], C_SDE=C, CVaR=stats['cv_cvar'], CVaR_SE=stats['cv_cvar_se'], risk_fixed_eta=stats['cv_frozen_risk'], risk_SE=stats['cv_frozen_se'], return_pct=stats['mean_return_pct'], mean_exposure=stats['mean_exposure'], J_SDE=stats['J'], raw_CVaR=stats['raw_empirical_cvar'], paths=stats['N'], seed=stats['seed'], empirical_risk_check_passed=bool(passed))

def plot_summary(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fields = [('lam', 'lambda'), ('CVaR', 'CVaR'), ('return_pct', 'Mean return (%)'), ('mean_exposure', 'Mean exposure')]
    groups = [('cases', rows, None)]
    if ENTRY == 'incomplete':
        groups = [('beta_statics', sorted([r for r in rows if r['budget'] == -0.94], key=lambda r: r['beta']), 'beta')]
        groups += [(f'budget_statics_beta_{b:.3f}', sorted([r for r in rows if r['beta'] == b], key=lambda r: r['budget']), 'budget') for b in (0.0, 0.02)]
    for label, data, xfield in groups:
        if not data:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)
        for ax, (field, title) in zip(axes.flat, fields):
            x = [r[xfield] for r in data] if xfield else np.arange(len(data))
            ax.plot(x, [r[field] for r in data], 'o-')
            if xfield is None:
                ax.set_xticks(x, [r['case'].split('_')[-1] for r in data])
            ax.set_xlabel(xfield or 'Case')
            ax.set_ylabel(title)
            ax.grid(alpha=0.2)
        fig.savefig(output / f'{ENTRY}_{label}.png', dpi=180)
        fig.savefig(output / f'{ENTRY}_{label}.svg')
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'results')
    parser.add_argument('--case-key', action='append')
    parser.add_argument('--threads', type=int, default=min(4, get_num_threads()))
    parser.add_argument('--list', action='store_true')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--verify-only', action='store_true')
    modes.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.threads <= get_num_threads():
        parser.error('Invalid thread count')
    if args.case_key and (not set(args.case_key) <= {c[0] for c in CASES}):
        parser.error('Unknown case key; use --list')
    cases = [c for c in CASES if not args.case_key or c[0] in args.case_key]
    if args.list:
        print(json.dumps([dict(key=k, config=asdict(Config(**p)), budget=c, evaluation=e) for k, p, c, e in cases], indent=2))
        return
    set_num_threads(args.threads)
    rows = []
    for key, changes, budget, settings in cases:
        p, directory = (Config(**changes), args.output / key)
        if args.verify_only:
            frozen, stats = saved_statistics(directory, settings)
            if frozen['config'] != asdict(p) or frozen['budget'] != budget:
                raise ValueError('Saved experiment configuration differs')
        else:
            if not args.evaluate_only:
                solve_case(p, budget, directory)
            else:
                frozen = json.loads((directory / 'solution.json').read_text(encoding='ascii'))
                if frozen['config'] != asdict(p) or frozen['budget'] != budget:
                    raise ValueError('Saved experiment configuration differs')
            frozen, stats = evaluate(directory, settings)
        row = summary_row(key, frozen, stats)
        rows.append(row)
        print(json.dumps(row), flush=True)
    if not args.verify_only:
        with (args.output / f'{ENTRY}_summary.csv').open('w', newline='', encoding='ascii') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        if not args.no_plots:
            plot_summary(rows, args.output)
    if not all((r['empirical_risk_check_passed'] for r in rows)):
        raise SystemExit('Empirical risk checks failed; output preserved without policy modification')

SOURCE_ID = hashlib.sha256(''.join(inspect.getsource(v.py_func if hasattr(v, 'py_func') else v)
    for v in list(globals().values()) if getattr(v, '__module__', None) == __name__
    and (inspect.isfunction(v) or inspect.isclass(v) or hasattr(v, 'py_func'))).encode()).hexdigest()

if __name__ == '__main__':
    main()
