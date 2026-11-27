# Python 3.12; numpy==2.3.5 numba==0.62.1 llvmlite==0.45.1 matplotlib==3.11.1
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

ENTRY = 'square_root'
IS_ASSET = False
CASES = [
    ('square_root_nonbinding', {'model': 'square_root', 'coefficient': 0.0045, 'steps': 50, 'dz': 0.0003125, 'z_bound': 0.25, 'dx': 0.0025, 'x_bound': 1.0, 'action_bound': 4.0, 'oos_seed': 900013}, -0.86, {'paths': 1048576, 'seed': 82641282, 'subdivisions': 64, 'method': 'milstein', 'batch_size': 32768}),
    ('square_root_binding', {'model': 'square_root', 'coefficient': 0.0045, 'steps': 50, 'dz': 0.000625, 'z_bound': 0.25, 'dx': 0.0025, 'x_bound': 1.0, 'action_bound': 4.0, 'oos_seed': 900013}, -0.94, {'paths': 65536, 'seed': 82641282, 'subdivisions': 64, 'method': 'milstein', 'batch_size': 32768}),
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

def cost_moments(mu, sigma, dt):
    m1x = 1.0
    m1u = 0.0
    m2xx = 1.0
    m2xu = 0.0
    m2uu = 0.0
    A1 = B1 = A2 = B2 = C2 = 0.0
    power = dt
    b = 2 * mu + sigma * sigma
    for n in range(14):
        factor = power / (n + 1)
        A1 += m1x * factor
        B1 += m1u * factor
        A2 += m2xx * factor
        B2 += m2xu * factor
        C2 += m2uu * factor
        m2xx, m2xu, m2uu = (b * m2xx / (n + 1), (b * m2xu + 2 * m1x) / (n + 1), (b * m2uu + 2 * m1u) / (n + 1))
        m1x, m1u = (mu * m1x / (n + 1), (mu * m1u + (1.0 if n == 0 else 0.0)) / (n + 1))
        power *= dt
    return (A1, B1, A2, B2, C2)

def geometry(p, order=5):
    dt = p.T / p.steps
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    weights = weights / math.sqrt(math.pi)
    returns = np.exp((p.mu - 0.5 * p.sigma ** 2) * dt + p.sigma * math.sqrt(2 * dt) * nodes)
    x = np.linspace(-p.x_bound, p.x_bound, 2 * round(p.x_bound / p.dx) + 1)
    hz = p.dz
    hx = p.dx
    R = p.action_bound
    K = p.coefficient / 1.5 * dt if p.model == 'square_root' else 0.0
    A1, B1, A2, B2, C2 = cost_moments(p.mu, p.sigma, dt)
    c0 = 0.5 * p.gamma * p.sigma ** 2 * A2 * x * x - p.mu * A1 * x
    c1 = 0.5 * p.gamma * p.sigma ** 2 * B2 * x - p.mu * B1
    c2 = 0.5 * p.gamma * p.sigma ** 2 * C2 + (0.5 * p.coefficient * dt if not K else 0.0)
    allbreaks = []
    for xx in x:
        points = [-R, 0.0, R]
        for rr in returns:
            qx = 0.5 * dt * (rr + 1)
            qz = 0.5 * dt * (rr - 1)
            bx = xx * rr
            bz = xx * (rr - 1)
            xmin = bx - qx * R
            xmax = bx + qx * R
            jlo = max(0, int(math.ceil((xmin - x[0]) / hx - 1e-12)))
            jhi = min(len(x) - 1, int(math.floor((xmax - x[0]) / hx + 1e-12)))
            for j in range(jlo, jhi + 1):
                u = (x[j] - bx) / qx
                if -R < u < R:
                    points.append(u)
            extrema = [-R, 0.0, R]
            if K:
                v = 2 * abs(qz) / (3 * K)
                if v * v < R:
                    extrema.append(math.copysign(v * v, qz))
            vals = [bz + qz * u - K * abs(u) ** 1.5 for u in extrema]
            nlo = int(math.ceil(min(vals) / hz - 1e-12))
            nhi = int(math.floor(max(vals) / hz + 1e-12))
            for m in range(nlo, nhi + 1):
                target = m * hz
                if not K:
                    if abs(qz) > 1e-18:
                        u = (target - bz) / qz
                        if -R < u < R:
                            points.append(u)
                else:
                    for sign in (-1.0, 1.0):
                        for root in np.roots([-K, sign * qz, 0.0, bz - target]):
                            if abs(root.imag) < 1e-08 and -1e-10 <= root.real <= math.sqrt(R) + 1e-10:
                                u = sign * max(0.0, float(root.real)) ** 2
                                if -R < u < R:
                                    points.append(u)
        points.sort()
        unique = []
        for u in points:
            if not unique or u - unique[-1] > 1e-10:
                unique.append(u)
        allbreaks.append(np.array(unique))
    count = np.array([len(v) for v in allbreaks], np.int32)
    breaks = np.zeros((len(x), int(count.max())))
    for j, v in enumerate(allbreaks):
        breaks[j, :len(v)] = v
    return (x, returns, weights, breaks, count, c0, c1, float(c2), K)

def stencils(p, g):
    x, rr, w, br, count, *tail = g
    K = tail[-1]
    dt = p.T / p.steps
    shape = (len(x), br.shape[1] - 1, len(rr))
    iz = np.zeros(shape, np.int32)
    jx = np.zeros(shape, np.int32)
    coefficients = np.zeros(shape + (5,))
    for j, xx in enumerate(x):
        for interval in range(count[j] - 1):
            mid = 0.5 * (br[j, interval] + br[j, interval + 1])
            for k, r in enumerate(rr):
                qx = 0.5 * dt * (r + 1)
                qz = 0.5 * dt * (r - 1)
                zshift = xx * (r - 1) + qz * mid - K * abs(mid) ** 1.5
                off = int(math.floor(zshift / p.dz))
                xm = xx * r + qx * mid
                xp = (xm - x[0]) / p.dx
                cell = max(0, min(len(x) - 2, int(math.floor(xp))))
                if xp < 0:
                    b0 = 0.0
                    b1 = 0.0
                elif xp > len(x) - 1:
                    b0 = 1.0
                    b1 = 0.0
                else:
                    b0 = (xx * r - x[0]) / p.dx - cell
                    b1 = qx / p.dx
                iz[j, interval, k] = off
                jx[j, interval, k] = cell
                coefficients[j, interval, k, :] = [(xx * (r - 1) - off * p.dz) / p.dz, qz / p.dz, -K / p.dz, b0, b1]
    return (iz, jx, coefficients)

@njit(cache=True, inline='always')
def local(V, z, x, zmin, hz, xmin, hx, left_slope):
    nz, nx = V.shape
    xp = (x - xmin) / hx
    j = max(0, min(nx - 2, int(math.floor(xp))))
    b = min(1.0, max(0.0, xp - j))
    zp = (z - zmin) / hz
    if zp < 0.0 or zp > nz - 1:
        i = 0 if zp < 0 else nz - 1
        slope = left_slope if zp < 0 else 0.0
        edge = zmin if zp < 0 else zmin + (nz - 1) * hz
        value = (1 - b) * V[i, j] + b * V[i, j + 1] + slope * (z - edge)
        vx = (V[i, j + 1] - V[i, j]) / hx if 0 <= xp <= nx - 1 else 0.0
        return (value, slope, vx, 0.0)
    i = max(0, min(nz - 2, int(math.floor(zp))))
    a = min(1.0, max(0.0, zp - i))
    v00 = V[i, j]
    v10 = V[i + 1, j]
    v01 = V[i, j + 1]
    v11 = V[i + 1, j + 1]
    value = (1 - a) * ((1 - b) * v00 + b * v01) + a * ((1 - b) * v10 + b * v11)
    vz = ((1 - b) * (v10 - v00) + b * (v11 - v01)) / hz
    vx = ((1 - a) * (v01 - v00) + a * (v11 - v10)) / hx if 0 <= xp <= nx - 1 else 0.0
    cross = (v11 - v10 - v01 + v00) / (hz * hx) if 0 <= xp <= nx - 1 else 0.0
    return (value, vz, vx, cross)

@njit(cache=True, inline='always')
def polynomial(A, B, C, D, E, u):
    power = abs(u) * math.sqrt(abs(u))
    return A + B * u + C * u * u + D * power + E * u * power

@njit(cache=True)
def interval_min(A, B, C, D, E, lo, hi):
    best_u = lo
    best = polynomial(A, B, C, D, E, lo)
    value = polynomial(A, B, C, D, E, hi)
    if value < best:
        best = value
        best_u = hi
    if abs(D) + abs(E) < 1e-25:
        if C > 0.0:
            u = -B / (2 * C)
            if lo < u < hi:
                value = polynomial(A, B, C, D, E, u)
                if value < best:
                    best = value
                    best_u = u
        return (best_u, best)
    sign = 1.0 if 0.5 * (lo + hi) >= 0 else -1.0
    vl = math.sqrt(min(abs(lo), abs(hi)))
    vh = math.sqrt(max(abs(lo), abs(hi)))
    a = 2.5 * sign * E
    b = 2 * C
    c = 1.5 * D
    d = sign * B
    scale = max(abs(a), abs(b), abs(c), abs(d), 1e-300)
    a /= scale
    b /= scale
    c /= scale
    d /= scale
    cut1 = vh
    cut2 = vh
    if abs(a) > 1e-14:
        discr = b * b - 3 * a * c
        if discr >= 0:
            root = math.sqrt(discr)
            for v in ((-b - root) / (3 * a), (-b + root) / (3 * a)):
                if vl < v < vh:
                    if cut1 == vh:
                        cut1 = v
                    else:
                        cut2 = v
    elif abs(b) > 1e-14:
        v = -c / (2 * b)
        if vl < v < vh:
            cut1 = v
    if cut1 > cut2:
        cut1, cut2 = (cut2, cut1)
    for i in range(3):
        low = vl if i == 0 else cut1 if i == 1 else cut2
        high = cut1 if i == 0 else cut2 if i == 1 else vh
        if high - low < 1e-15:
            continue
        pl = ((a * low + b) * low + c) * low + d
        ph = ((a * high + b) * high + c) * high + d
        if abs(pl) < 1e-13:
            u = sign * low * low
            value = polynomial(A, B, C, D, E, u)
            if value < best:
                best = value
                best_u = u
        if abs(ph) < 1e-13:
            u = sign * high * high
            value = polynomial(A, B, C, D, E, u)
            if value < best:
                best = value
                best_u = u
        if pl * ph < 0.0:
            for _ in range(40):
                mid = 0.5 * (low + high)
                pm = ((a * mid + b) * mid + c) * mid + d
                if pl * pm <= 0.0:
                    high = mid
                else:
                    low = mid
                    pl = pm
            v = 0.5 * (low + high)
            u = sign * v * v
            value = polynomial(A, B, C, D, E, u)
            if value < best:
                best = value
                best_u = u
    return (best_u, best)

@njit(cache=True)
def choose_cached(V, i, j, z, x, returns, weights, breaks, count, offsets, xcells, coefficients, c0, c1, c2, K, dt, hz, left_slope):
    best = 1e+300
    best_u = 0.0
    nz = V.shape[0]
    for interval in range(count[j] - 1):
        A = c0[j]
        B = c1[j]
        C = c2
        D = K
        E = 0.0
        for k in range(len(returns)):
            iz = i + offsets[j, interval, k]
            jx = xcells[j, interval, k]
            a0, a1, a15, b0, b1 = coefficients[j, interval, k]
            w = weights[k]
            if iz < 0 or iz >= nz - 1:
                edge = 0 if iz < 0 else nz - 1
                slope = left_slope if iz < 0 else 0.0
                v0 = V[edge, jx]
                dv = V[edge, jx + 1] - v0
                A += w * (v0 + b0 * dv + slope * (z[i] + x[j] * (returns[k] - 1) - z[edge]))
                B += w * (b1 * dv + slope * 0.5 * dt * (returns[k] - 1))
                D += w * (-slope * K)
            else:
                v00 = V[iz, jx]
                v10 = V[iz + 1, jx]
                v01 = V[iz, jx + 1]
                v11 = V[iz + 1, jx + 1]
                dz = v10 - v00
                dx = v01 - v00
                cross = v11 - v10 - v01 + v00
                A += w * (v00 + a0 * dz + b0 * dx + a0 * b0 * cross)
                B += w * (a1 * dz + b1 * dx + (a0 * b1 + a1 * b0) * cross)
                C += w * a1 * b1 * cross
                D += w * (a15 * dz + a15 * b0 * cross)
                E += w * a15 * b1 * cross
        u, value = interval_min(A, B, C, D, E, breaks[j, interval], breaks[j, interval + 1])
        if value < best:
            best = value
            best_u = u
    return (best_u, best)

@njit(cache=True)
def continuation(A, z, x, u, returns, weights, K, dt, zmin, hz, xmin, hx, left_slope):
    answer = 0.0
    exec_cost = K * abs(u) * math.sqrt(abs(u))
    for k in range(len(returns)):
        rr = returns[k]
        zz = z + x * (rr - 1) + 0.5 * dt * (rr - 1) * u - exec_cost
        xx = x * rr + 0.5 * dt * (rr + 1) * u
        value, _, _, _ = local(A, zz, xx, zmin, hz, xmin, hx, left_slope)
        answer += weights[k] * value
    return answer

@njit(cache=True, parallel=True)
def solve_sl(z, x, lam, alpha, returns, weights, breaks, count, c0, c1, c2, K, offsets, xcells, coefficients, nt, T, policy, Hstore, stride):
    nz = len(z)
    nx = len(x)
    hz = z[1] - z[0]
    hx = x[1] - x[0]
    dt = T / nt
    J = np.zeros((nz, nx))
    H = np.empty_like(J)
    for i in range(nz):
        H[i, :] = max(-z[i], 0) / (1 - alpha)
    if Hstore.shape[0] > 1:
        Hstore[-1, :, :] = H
    worst = 0.0
    for t in range(nt - 1, -1, -1):
        V = J + lam * H
        newJ = np.empty_like(J)
        newH = np.empty_like(H)
        errors = np.zeros(nz)
        zero_u = np.empty(nx)
        zero_J = np.empty(nx)
        zero_error = 0.0
        if lam == 0.0:
            i = nz // 2
            for j in range(nx):
                u, value = choose_cached(V, i, j, z, x, returns, weights, breaks, count, offsets, xcells, coefficients, c0, c1, c2, K, dt, hz, 0.0)
                stage = c0[j] + c1[j] * u + c2 * u * u + K * abs(u) * math.sqrt(abs(u))
                zero_u[j] = u
                zero_J[j] = stage + continuation(J, z[i], x[j], u, returns, weights, K, dt, z[0], hz, x[0], hx, 0.0)
                zero_error = max(zero_error, abs(zero_J[j] - value))
        for i in prange(nz):
            rowerror = 0.0
            for j in range(nx):
                if lam == 0.0:
                    u = zero_u[j]
                    value = zero_J[j]
                    newJ[i, j] = zero_J[j]
                else:
                    u, value = choose_cached(V, i, j, z, x, returns, weights, breaks, count, offsets, xcells, coefficients, c0, c1, c2, K, dt, hz, -lam / (1 - alpha))
                    stage = c0[j] + c1[j] * u + c2 * u * u + K * abs(u) * math.sqrt(abs(u))
                    newJ[i, j] = stage + continuation(J, z[i], x[j], u, returns, weights, K, dt, z[0], hz, x[0], hx, 0.0)
                newH[i, j] = continuation(H, z[i], x[j], u, returns, weights, K, dt, z[0], hz, x[0], hx, -1 / (1 - alpha))
                rowerror = max(rowerror, abs(newJ[i, j] + lam * newH[i, j] - value))
                if policy.shape[0] > 1:
                    policy[t, i, j] = u
            errors[i] = rowerror
        worst = max(worst, float(np.max(errors)), zero_error)
        J = newJ
        H = newH
        if Hstore.shape[0] > 1 and t % stride == 0:
            Hstore[t // stride, :, :] = H
    return (J, H, worst)

@njit(cache=True, inline='always')
def position(y, start, h, n):
    pos = (y - start) / h
    i = max(0, min(n - 2, int(math.floor(pos))))
    a = min(1.0, max(0.0, pos - i))
    return (i, a, pos)

@njit(cache=True, inline='always')
def gradients2(H, i, j, a, b, dz, dx):
    wz = ((1 - b) * (float(H[i + 1, j]) - float(H[i, j])) + b * (float(H[i + 1, j + 1]) - float(H[i, j + 1]))) / dz
    wx = ((1 - a) * (float(H[i, j + 1]) - float(H[i, j])) + a * (float(H[i + 1, j + 1]) - float(H[i + 1, j]))) / dx
    return (wz, wx)

@njit(cache=True)
def lookup_rate(W, X, policy, z0, dz, x0, dx, eta):
    u = np.empty(len(W))
    for n in range(len(W)):
        i, a, _ = position(W[n] + eta, z0, dz, policy.shape[0])
        j, b, _ = position(X[n], x0, dx, policy.shape[1])
        u[n] = (1 - a) * ((1 - b) * policy[i, j] + b * policy[i, j + 1]) + a * ((1 - b) * policy[i + 1, j] + b * policy[i + 1, j + 1])
    return u

@njit(cache=True)
def conditional_cost(cost, exposure, X, u, moments, mu, sigma, gamma, coef, power, dt):
    A1, B1, A2, B2, C2 = moments
    for n in range(len(X)):
        I1 = A1 * X[n] + B1 * u[n]
        I2 = A2 * X[n] * X[n] + B2 * X[n] * u[n] + C2 * u[n] * u[n]
        fee = coef / 1.5 * abs(u[n]) ** 1.5 if power else 0.5 * coef * u[n] * u[n]
        cost[n] += 0.5 * gamma * sigma * sigma * I2 - mu * I1 + fee * dt
        exposure[n] += I1

@njit(cache=True)
def step_held(W, X, M, u, H0, H1, tw, z0, dz, x0, dx, eta, db, mu, sigma, coef, power, dt, milstein):
    outz = outx = 0
    for n in range(len(W)):
        i, a, pos = position(W[n] + eta, z0, dz, H0.shape[0])
        j, b, xp = position(X[n], x0, dx, H0.shape[1])
        if pos < 0 or pos > H0.shape[0] - 1:
            outz += 1
        if xp < 0 or xp > H0.shape[1] - 1:
            outx += 1
        gz0, gx0 = gradients2(H0, i, j, a, b, dz, dx)
        gz1, gx1 = gradients2(H1, i, j, a, b, dz, dx)
        derivative = (1 - tw) * (gz0 + gx0) + tw * (gz1 + gx1)
        shock = sigma * X[n] * db[n]
        M[n] += derivative * shock
        correction = 0.5 * sigma * sigma * X[n] * (db[n] * db[n] - dt) if milstein else 0.0
        increment = mu * X[n] * dt + shock + correction
        fee = coef / 1.5 * abs(u[n]) ** 1.5 if power else 0.0
        W[n] += increment - fee * dt
        X[n] += increment + u[n] * dt
    return (outz, outx)

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
    if p.beta and settings['method'] != 'euler':
        raise ValueError('The two-noise model requires Euler evaluation')
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
    parser.add_argument('--verify-only', action='store_true')
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
            solve_case(p, budget, directory)
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
