from __future__ import annotations
from dataclasses import dataclass, replace
import math
import csv
import json
import os
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class ModelParams:
    horizon: float = 1.0
    steps: int = 100
    samples: int = 5000
    seed: int = 7
    w0: float = 1.0
    s0: float = 1.0
    r: float = 0.0
    mu: float = 0.08
    sigma: float = 0.20
    mu_t: Optional[tuple[float,...]] = None
    sigma_t: Optional[tuple[float,...]] = None
    b: float = 0.0
    beta_perp: float = 0.0
    price_model: str = 'gbm'
    gamma: float = 5.0
    alpha: float = 0.95
    cvar_limit: float = -0.94
    phi_min: float = -2.0
    phi_max: float = 2.0
    eta_tol: float = 5e-4
    lambda_tol: float = 1e-6
    residual_tol: float = 1e-5
    eta_max_iter: int = 40
    lambda_max_iter: int = 20
    lambda_initial_high: float = 1.0
    lambda_high_cap: float = 1e5
    state_grid_size: int = 201
    state_grid_step: Optional[float] = None
    state_grid_focus_strength: float = 1.0
    state_grid_focus_width: float = 0.20
    hjb_pde_substeps: int = 8
    hjb_cfl_safety: float = 0.20
    hjb_curvature_floor: float = 1e-10
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
    policy_grid: Optional[np.ndarray]=None
    wealth_grid: Optional[np.ndarray]=None
    terminal_wealth: Optional[np.ndarray]=None
    loss: Optional[np.ndarray]=None
    phi_paths: Optional[np.ndarray]=None
    wealth_paths: Optional[np.ndarray]=None
    stock_paths: Optional[np.ndarray]=None

def _time_schedule(schedule, fallback, steps, name):
    if schedule is None:
        return np.full(steps,float(fallback))
    values=np.asarray(schedule,dtype=float).reshape(-1)
    if values.size==steps+1:
        values=values[:-1]
    if values.size!=steps:
        raise ValueError(name)
    return values.copy()

def mu_schedule(p, steps=None):
    return _time_schedule(p.mu_t,p.mu,p.steps if steps is None else steps,'mu_t')

def sigma_schedule(p, steps=None):
    return _time_schedule(p.sigma_t,p.sigma,p.steps if steps is None else steps,'sigma_t')

def merton_exposure(p):
    mu=mu_schedule(p); sig=sigma_schedule(p)
    return mu/np.maximum(p.gamma*sig**2,1e-12)

def generate_brownian_increments(p):
    rng=np.random.default_rng(p.seed)
    dt=p.horizon/p.steps
    return rng.normal(0,math.sqrt(dt),size=(p.samples,p.steps))

def generate_perpendicular_brownian_increments(p):
    rng=np.random.default_rng(p.seed+1_000_003)
    dt=p.horizon/p.steps
    return rng.normal(0,math.sqrt(dt),size=(p.samples,p.steps))

def riskless_growth(p,dt):
    return math.exp(p.r*dt)

def stock_gross_return(d_b,mu,sig,dt,p):
    d_b=np.asarray(d_b)
    if p.price_model=='gbm':
        return np.exp((p.r+mu-.5*sig**2)*dt+sig*d_b)
    return riskless_growth(p,dt)+mu*dt+sig*d_b

def stock_excess_return(d_b,mu,sig,dt,p):
    return stock_gross_return(d_b,mu,sig,dt,p)-riskless_growth(p,dt)

def simulate_wealth(phi,d_b,p):
    samples,steps=d_b.shape
    dt=p.horizon/steps
    growth=riskless_growth(p,dt)
    mu_t=mu_schedule(p,steps)
    sig_t=sigma_schedule(p,steps)
    dbp=generate_perpendicular_brownian_increments(replace(p,samples=samples,steps=steps))
    w=np.empty((samples,steps+1)); w[:,0]=p.w0
    for n in range(steps):
        re=stock_excess_return(d_b[:,n],mu_t[n],sig_t[n],dt,p)
        w[:,n+1]=growth*w[:,n]+phi[n]*re+p.b*dt+p.beta_perp*dbp[:,n]
    return w

def cvar_from_loss(loss,alpha):
    var=float(np.quantile(loss,alpha))
    cvar=float(var+np.maximum(loss-var,0).mean()/(1-alpha))
    return cvar,var

def running_cost(action,p,n=None):
    a=np.asarray(action)
    if n is None:
        if a.ndim>0 and a.shape[-1]==p.steps:
            shape=(1,)*(a.ndim-1)+(p.steps,)
            mu=mu_schedule(p).reshape(shape)
            sig=sigma_schedule(p).reshape(shape)
            return .5*p.gamma*(sig*a)**2-mu*a
        return .5*p.gamma*(p.sigma*a)**2-p.mu*a
    mu=mu_schedule(p)[n]
    sig=sigma_schedule(p)[n]
    return .5*p.gamma*(sig*a)**2-mu*a

def build_state_grid(d_b,p,focus=None):
    merton=np.clip(merton_exposure(p),p.phi_min,p.phi_max)
    probes=[np.full(p.steps,p.phi_min),merton,np.zeros(p.steps)]
    wealth=np.concatenate([simulate_wealth(phi,d_b,p).reshape(-1) for phi in probes])
    low=float(np.quantile(wealth,.002)); high=float(np.quantile(wealth,.998))
    width=max(high-low,1e-3)
    low-=p.state_grid_padding*width; high+=p.state_grid_padding*width
    if p.state_grid_step is not None and p.state_grid_step>0:
        step=float(p.state_grid_step)
        gl=step*math.floor(low/step); gh=step*math.ceil(high/step)
        return np.arange(gl,gh+.5*step,step)
    if focus is None or not low<float(focus)<high or p.state_grid_focus_strength<=0 or p.state_grid_focus_width<=0:
        return np.linspace(low,high,p.state_grid_size)
    dense_size=max(4001,40*p.state_grid_size+1)
    dense=np.linspace(low,high,dense_size)
    fv=float(focus); widthv=float(p.state_grid_focus_width); strength=float(p.state_grid_focus_strength)
    density=1+strength*np.exp(-.5*((dense-fv)/widthv)**2)
    cell=.5*(density[1:]+density[:-1])*np.diff(dense)
    cum=np.concatenate(([0.],np.cumsum(cell)))
    total=float(cum[-1]); fm=float(np.interp(fv,dense,cum)); intervals=p.state_grid_size-1
    left=int(round(intervals*fm/total)); left=int(np.clip(left,1,intervals-1)); right=intervals-left
    targets=np.concatenate((np.linspace(0,fm,left+1),np.linspace(fm,total,right+1)[1:]))
    return np.interp(targets,cum,dense)

def fit_hjb_policy(lambda_value,eta,d_b,p):
    focus=-float(eta) if lambda_value>0 else None
    wg=build_state_grid(d_b,p,focus=focus)
    dt=p.horizon/p.steps
    mu_t=mu_schedule(p); sig_t=sigma_schedule(p); merton=merton_exposure(p)
    value_next=lambda_value*np.maximum(-wg-eta,0)/(1-p.alpha)
    value_grid=np.empty((p.steps+1,wg.size)); value_grid[-1]=value_next
    policy_grid=np.empty((p.steps,wg.size))
    edge_order=2 if wg.size>=3 else 1

    def action_upper(n):
        upper=float(p.phi_max)
        if p.cap_control_at_merton:
            upper=min(upper,float(merton[n]))
        return max(float(p.phi_min),upper)

    def derivatives(value):
        vw=np.gradient(value,wg,edge_order=edge_order)
        vww=np.gradient(vw,wg,edge_order=edge_order)
        return vw,vww

    def hamiltonian(action,mu,sig,vw,vww):
        var=(sig*action)**2+p.beta_perp**2
        return .5*p.gamma*(sig*action)**2-mu*action+(p.r*wg+p.b+mu*action)*vw+.5*var*vww

    def minimize(value,n):
        vw,vww=derivatives(value); mu=float(mu_t[n]); sig=float(sig_t[n]); upper=action_upper(n)
        quadratic=sig**2*(p.gamma+vww); linear=mu*(vw-1)
        convex=quadratic>p.hjb_curvature_floor
        interior=np.where(convex,-linear/np.maximum(quadratic,1e-14),0.0)
        interior=np.clip(interior,p.phi_min,upper)
        best_a=interior; best_v=hamiltonian(interior,mu,sig,vw,vww)
        low=np.full_like(interior,p.phi_min); up=np.full_like(interior,upper)
        lv=hamiltonian(low,mu,sig,vw,vww); uv=hamiltonian(up,mu,sig,vw,vww)
        use=lv<best_v; best_a=np.where(use,low,best_a); best_v=np.where(use,lv,best_v)
        use=uv<best_v; best_a=np.where(use,up,best_a); best_v=np.where(use,uv,best_v)
        return best_a,best_v

    def stable_substeps(n):
        spacing=float(np.min(np.diff(wg))); sig=abs(float(sig_t[n])); maxa=max(abs(p.phi_min),abs(action_upper(n)))
        var=(sig*maxa)**2+p.beta_perp**2
        rate=var/max(spacing**2,1e-14)
        cfl=int(math.ceil(dt*rate/max(p.hjb_cfl_safety,1e-6)))
        return max(p.hjb_pde_substeps,cfl,1)

    for n in range(p.steps-1,-1,-1):
        value=value_next; substeps=stable_substeps(n); sub_dt=dt/substeps; best_a=np.zeros(wg.size)
        for _ in range(substeps):
            best_a,gen=minimize(value,n)
            value=value+sub_dt*gen
            if not np.all(np.isfinite(value)):
                raise FloatingPointError('HJB unstable')
        value_next=value; value_grid[n]=value_next; policy_grid[n]=best_a
    return wg,policy_grid,value_grid

def simulate_feedback_policy(wg,pg,d_b,p):
    samples,steps=d_b.shape
    dt=p.horizon/steps
    growth=riskless_growth(p,dt)
    mu_t=mu_schedule(p,steps); sig_t=sigma_schedule(p,steps)
    merton=merton_exposure(replace(p,steps=steps))
    dbp=generate_perpendicular_brownian_increments(replace(p,samples=samples,steps=steps))
    w=np.empty((samples,steps+1)); s=np.empty((samples,steps+1)); phi=np.empty((samples,steps))
    w[:,0]=p.w0; s[:,0]=p.s0
    for n in range(steps):
        ph=np.interp(w[:,n],wg,pg[n],left=pg[n,0],right=pg[n,-1])
        upper=p.phi_max
        if p.cap_control_at_merton:
            upper=min(upper,float(merton[n]))
        ph=np.clip(ph,p.phi_min,upper)
        phi[:,n]=ph
        db=d_b[:,n]
        re=stock_excess_return(db,mu_t[n],sig_t[n],dt,p)
        gr=stock_gross_return(db,mu_t[n],sig_t[n],dt,p)
        w[:,n+1]=growth*w[:,n]+ph*re+p.b*dt+p.beta_perp*dbp[:,n]
        s[:,n+1]=s[:,n]*gr
    return w,s,phi

def feedback_value_iteration_kernel(lambda_value,eta,d_b,p):
    wg,pg,_=fit_hjb_policy(lambda_value,eta,d_b,p)
    wealth,stock,phi=simulate_feedback_policy(wg,pg,d_b,p)
    dt=p.horizon/p.steps
    tw=wealth[:,-1]; loss=-tw; hinge=np.maximum(loss-eta,0)
    running=dt*np.sum(running_cost(phi,p),axis=1)
    obj=float(np.mean(running))
    residual=float(eta+hinge.mean()/(1-p.alpha)-p.cvar_limit)
    lag=obj+lambda_value*residual
    cvar,var=cvar_from_loss(loss,p.alpha)
    return ControlResult(
        np.mean(phi,axis=0),eta,obj,lag,residual,cvar,var,float(np.mean(tw)),float(np.std(tw)),
        float(np.mean(loss>eta)),p.steps,pg,wg,tw,loss,phi,wealth,stock
    )

def unconstrained_solution(d_b,p):
    r=feedback_value_iteration_kernel(0,0,d_b,p)
    loss=r.loss; cvar,var=cvar_from_loss(loss,p.alpha)
    return replace(r,eta=var,residual=cvar-p.cvar_limit,cvar=cvar,var=var,
                   loss_tail_probability=float(np.mean(loss>var)))

def eta_bracket(p):
    mu_abs=float(np.max(np.abs(mu_schedule(p)))); sig_abs=float(np.max(np.abs(sigma_schedule(p))))
    maxe=max(abs(p.phi_min),abs(p.phi_max))
    drift=abs(p.w0)+p.horizon*(abs(p.r)*(abs(p.w0)+4)+mu_abs*maxe+abs(p.b))
    diff=6*math.sqrt(p.horizon)*(sig_abs*maxe+abs(p.beta_perp))
    x=max(2.,drift+diff)
    return -x,x

def eta_golden_search(lambda_value,d_b,p):
    if lambda_value<=0:
        return unconstrained_solution(d_b,p)
    low,high=eta_bracket(p); rho=(math.sqrt(5)-1)/2
    e1=high-rho*(high-low); e2=low+rho*(high-low)
    r1=feedback_value_iteration_kernel(lambda_value,e1,d_b,p)
    r2=feedback_value_iteration_kernel(lambda_value,e2,d_b,p)

    def score(r):
        return abs(float(r.eta)-float(np.quantile(r.loss,p.alpha)))

    best=min((r1,r2),key=lambda r:(score(r),r.lagrangian))
    for _ in range(max(p.eta_max_iter-1,0)):
        if score(r1)<=score(r2):
            high=e2; e2=e1; r2=r1; e1=high-rho*(high-low)
            r1=feedback_value_iteration_kernel(lambda_value,e1,d_b,p)
        else:
            low=e1; e1=e2; r1=r2; e2=low+rho*(high-low)
            r2=feedback_value_iteration_kernel(lambda_value,e2,d_b,p)
        best=min((best,r1,r2),key=lambda r:(score(r),r.lagrangian))
        if high-low<=p.eta_tol:
            break
    return best

def evaluate_policy_on_holdout(trained,lam,d_bh,ph):
    w,s,phi=simulate_feedback_policy(trained.wealth_grid,trained.policy_grid,d_bh,ph)
    tw=w[:,-1]; loss=-tw; hinge=np.maximum(loss-trained.eta,0)
    res=float(trained.eta+hinge.mean()/(1-ph.alpha)-ph.cvar_limit)
    cvar,var=cvar_from_loss(loss,ph.alpha)
    return replace(
        trained,phi=np.mean(phi,axis=0),lagrangian=trained.objective+lam*res,residual=res,
        cvar=cvar,var=var,mean_terminal_wealth=float(np.mean(tw)),terminal_wealth_std=float(np.std(tw)),
        loss_tail_probability=float(np.mean(loss>trained.eta)),terminal_wealth=tw,loss=loss,
        phi_paths=phi,wealth_paths=w,stock_paths=s
    )

def outer_bisection(d_b,p,d_bh,ph):
    hist=[]
    def fit_eval(lam):
        return evaluate_policy_on_holdout(eta_golden_search(lam,d_b,p),lam,d_bh,ph)
    unc=fit_eval(0); hist.append((0,unc.cvar-p.cvar_limit))
    if unc.cvar-p.cvar_limit<=0:
        return unc,0,hist
    lo=0.; hi=p.lambda_initial_high
    rh=fit_eval(hi); hist.append((hi,rh.cvar-p.cvar_limit))
    while rh.cvar-p.cvar_limit>0:
        lo=hi; hi*=2
        if hi>p.lambda_high_cap:
            raise RuntimeError('no bracket')
        rh=fit_eval(hi); hist.append((hi,rh.cvar-p.cvar_limit))
    best=min((unc,rh),key=lambda r:abs(r.cvar-p.cvar_limit))
    bestlam=0 if best is unc else hi
    for _ in range(p.lambda_max_iter):
        mid=.5*(lo+hi); rm=fit_eval(mid); resid=rm.cvar-p.cvar_limit
        hist.append((mid,resid))
        if abs(resid)<abs(best.cvar-p.cvar_limit):
            best=rm; bestlam=mid
        if abs(resid)<=p.residual_tol:
            return rm,mid,hist
        if resid>0:
            lo=mid
        else:
            hi=mid
        if hi-lo<=p.lambda_tol:
            break
    return best,bestlam,hist


# -----------------------------------------------------------------------------
# Section 5.3 comparative statics
# -----------------------------------------------------------------------------
# Numerical protocol used for both sweeps.  The 20,000-path holdout is used to
# select the numerical dual root.  No Merton cap and no ex-post feasibility
# adjustment are applied.
SEARCH_SEED = 13
HOLDOUT_SEED = 686035552
HOLDOUT_SAMPLES = 20_000
BETA_GRID = [0.0, 0.005, 0.010, 0.015, 0.020, 0.025]
C_GRID = [-0.86, -0.88, -0.90, -0.92, -0.94]
BETA_SWEEP_C = -0.94
C_SWEEP_BETAS = [0.0, 0.02]

# Root-search tolerances.  These are deliberately tighter than the precision
# reported in the paper; the underlying HJB / Monte Carlo approximation remains
# the dominant numerical error.
ROOT_RESIDUAL_TOL = 1e-5
ROOT_LAMBDA_TOL = 1e-6
ROOT_MAX_ITERS = 14
ROOT_INITIAL_HIGH = 0.05
ROOT_HIGH_CAP = 2.0


class FixedLambdaEvaluator:
    """Cache fixed-lambda solutions for one beta_perp value.

    For fixed beta_perp and lambda, changing the CVaR limit c only adds the
    constant -lambda*c to the Lagrangian.  Therefore the optimal policy and eta
    are independent of c.  We solve each fixed-lambda problem once and reuse it
    across all c values in the c-sweep.
    """

    def __init__(self, beta_perp: float):
        self.beta_perp = float(beta_perp)
        self.p = ModelParams(
            beta_perp=self.beta_perp,
            cvar_limit=0.0,              # harmless constant shift for lambda>0
            seed=SEARCH_SEED,
            cap_control_at_merton=False,
        )
        self.d_b = generate_brownian_increments(self.p)
        self.ph = replace(
            self.p,
            samples=HOLDOUT_SAMPLES,
            seed=HOLDOUT_SEED,
            cvar_limit=0.0,
            cap_control_at_merton=False,
        )
        self.d_bh = generate_brownian_increments(self.ph)
        self.cache: dict[float, dict] = {}

    def evaluate(self, lambda_value: float) -> dict:
        lam = max(0.0, float(lambda_value))
        key = round(lam, 14)
        if key in self.cache:
            return self.cache[key]

        trained = eta_golden_search(lam, self.d_b, self.p)
        held = evaluate_policy_on_holdout(trained, lam, self.d_bh, self.ph)
        row = {
            "beta_perp": self.beta_perp,
            "lambda_star": lam,
            "eta": float(held.eta),
            "cvar": float(held.cvar),
            "mean_exposure": float(np.mean(held.phi_paths)),
            "mean_terminal_wealth": float(held.mean_terminal_wealth),
            "max_exposure": float(np.max(held.phi_paths)),
            "p99_exposure": float(np.quantile(held.phi_paths, 0.99)),
        }
        self.cache[key] = row
        return row


def _residual(row: dict, target_c: float) -> float:
    return float(row["cvar"] - target_c)


def solve_lambda_for_c(
    evaluator: FixedLambdaEvaluator,
    target_c: float,
    lo: float = 0.0,
    hi: float | None = None,
) -> dict:
    """Solve CVaR(lambda)=target_c by safeguarded secant/bisection.

    The holdout CVaR is monotone decreasing in lambda for the present numerical
    calibration.  We still use a bracketed method so the code is robust to mild
    nonlinearities in the estimated curve.
    """
    target_c = float(target_c)
    lo = max(0.0, float(lo))
    rlo = evaluator.evaluate(lo)
    ylo = _residual(rlo, target_c)

    # If the unconstrained (or supplied lower-bracket) policy is already
    # feasible, lambda=0 is the correct nonbinding solution only when lo=0.
    if lo == 0.0 and ylo <= 0.0:
        out = dict(rlo)
        out.update(cvar_limit=target_c, residual=ylo, root_iterations=0,
                   root_status="nonbinding")
        return out

    if hi is None:
        hi = max(ROOT_INITIAL_HIGH, lo + ROOT_INITIAL_HIGH)
    hi = float(hi)
    if hi <= lo:
        hi = lo + ROOT_INITIAL_HIGH

    rhi = evaluator.evaluate(hi)
    yhi = _residual(rhi, target_c)
    while yhi > 0.0:
        lo, rlo, ylo = hi, rhi, yhi
        hi = min(ROOT_HIGH_CAP, max(1.5 * hi, hi + ROOT_INITIAL_HIGH))
        if hi <= lo + 1e-14:
            raise RuntimeError(
                f"Could not bracket target c={target_c} for beta={evaluator.beta_perp}."
            )
        rhi = evaluator.evaluate(hi)
        yhi = _residual(rhi, target_c)
        if hi >= ROOT_HIGH_CAP and yhi > 0.0:
            raise RuntimeError(
                f"Target c={target_c} is not bracketed below lambda={ROOT_HIGH_CAP} "
                f"for beta={evaluator.beta_perp}."
            )

    # At this point ylo>0 and yhi<=0 for a binding target.
    best = min((rlo, rhi), key=lambda r: abs(_residual(r, target_c)))
    iterations = 0
    for iterations in range(1, ROOT_MAX_ITERS + 1):
        ylo = _residual(rlo, target_c)
        yhi = _residual(rhi, target_c)

        # Secant proposal inside the bracket; fall back to midpoint if the
        # proposal is too close to an endpoint.
        if abs(yhi - ylo) > 1e-14:
            mid = hi - yhi * (hi - lo) / (yhi - ylo)
        else:
            mid = 0.5 * (lo + hi)
        guard = 0.05 * (hi - lo)
        if not (lo + guard < mid < hi - guard):
            mid = 0.5 * (lo + hi)

        rm = evaluator.evaluate(mid)
        ym = _residual(rm, target_c)
        if abs(ym) < abs(_residual(best, target_c)):
            best = rm

        if abs(ym) <= ROOT_RESIDUAL_TOL or (hi - lo) <= ROOT_LAMBDA_TOL:
            best = rm
            break

        if ym > 0.0:
            lo, rlo = mid, rm
        else:
            hi, rhi = mid, rm

    out = dict(best)
    out.update(
        cvar_limit=target_c,
        residual=_residual(best, target_c),
        root_iterations=iterations,
        root_status="binding",
    )
    return out


def solve_one_beta(beta_perp: float) -> tuple[dict, list[dict], list[dict]]:
    """Run the beta-sweep point and, when needed, the full c-sweep."""
    evaluator = FixedLambdaEvaluator(beta_perp)

    # Every beta contributes one point to the beta_perp sweep at c=-0.94.
    beta_row = solve_lambda_for_c(evaluator, BETA_SWEEP_C)

    c_rows: list[dict] = []
    if any(abs(beta_perp - b) < 1e-12 for b in C_SWEEP_BETAS):
        # Reuse the c=-0.94 root as a guaranteed upper bracket for all looser
        # targets.  Process c from loose to tight and use the previous root as
        # the next lower bracket.
        tight_lambda = float(beta_row["lambda_star"])
        previous_lambda = 0.0
        for c in C_GRID:
            if abs(c - BETA_SWEEP_C) < 1e-12:
                row = dict(beta_row)
                row["cvar_limit"] = c
                row["residual"] = float(row["cvar"] - c)
            else:
                row = solve_lambda_for_c(
                    evaluator,
                    c,
                    lo=previous_lambda,
                    hi=tight_lambda,
                )
            c_rows.append(row)
            previous_lambda = float(row["lambda_star"])

    evaluations = sorted(evaluator.cache.values(), key=lambda r: r["lambda_star"])
    return beta_row, c_rows, evaluations


def _worker(beta: float):
    return solve_one_beta(float(beta))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_plots(beta_rows: list[dict], c_rows: list[dict], outdir: Path) -> None:
    beta_rows = sorted(beta_rows, key=lambda r: r["beta_perp"])
    beta = np.array([r["beta_perp"] for r in beta_rows])
    lam = np.array([r["lambda_star"] for r in beta_rows])
    exp = np.array([r["mean_exposure"] for r in beta_rows])

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.plot(beta, lam, marker="o")
    ax.set_xlabel(r"Nontraded risk $\beta^\perp$")
    ax.set_ylabel(r"Shadow price $\widehat\lambda$")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "lambda_vs_beta.pdf", bbox_inches="tight")
    fig.savefig(outdir / "lambda_vs_beta.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.plot(beta, exp, marker="o")
    ax.axhline(0.40, linestyle="--", label="Merton benchmark")
    ax.set_xlabel(r"Nontraded risk $\beta^\perp$")
    ax.set_ylabel("Average risky exposure")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "exposure_vs_beta.pdf", bbox_inches="tight")
    fig.savefig(outdir / "exposure_vs_beta.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for b in C_SWEEP_BETAS:
        rows = sorted(
            [r for r in c_rows if abs(r["beta_perp"] - b) < 1e-12],
            key=lambda r: r["cvar_limit"],
            reverse=True,
        )
        c = np.array([r["cvar_limit"] for r in rows])
        y = np.array([r["lambda_star"] for r in rows])
        ax.plot(c, y, marker="o", label=rf"$\beta^\perp={b:.2f}$")
    ax.set_xlabel(r"CVaR limit $c$")
    ax.set_ylabel(r"Shadow price $\widehat\lambda$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "lambda_vs_c.pdf", bbox_inches="tight")
    fig.savefig(outdir / "lambda_vs_c.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for b in C_SWEEP_BETAS:
        rows = sorted(
            [r for r in c_rows if abs(r["beta_perp"] - b) < 1e-12],
            key=lambda r: r["cvar_limit"],
            reverse=True,
        )
        c = np.array([r["cvar_limit"] for r in rows])
        y = np.array([r["mean_exposure"] for r in rows])
        ax.plot(c, y, marker="o", label=rf"$\beta^\perp={b:.2f}$")
    ax.axhline(0.40, linestyle="--", label="Merton benchmark")
    ax.set_xlabel(r"CVaR limit $c$")
    ax.set_ylabel("Average risky exposure")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "position_vs_c.pdf", bbox_inches="tight")
    fig.savefig(outdir / "position_vs_c.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce Section 5.3 comparative statics (beta_perp and c sweeps)."
    )
    parser.add_argument(
        "--output-dir",
        default="section5_3_outputs",
        help="Directory for CSV/JSON results and figures.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(5, os.cpu_count() or 1),
        help="Number of beta values solved in parallel (default: up to 5).",
    )
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    beta_rows: list[dict] = []
    c_rows: list[dict] = []
    evaluation_log: dict[str, list[dict]] = {}

    # Each worker handles one beta value and retains its own fixed-lambda cache.
    # For beta=0 and 0.02 the same cache is reused for the c-sweep.
    if args.workers <= 1:
        results = [solve_one_beta(b) for b in BETA_GRID]
        for b, result in zip(BETA_GRID, results):
            beta_row, local_c_rows, evals = result
            beta_rows.append(beta_row)
            c_rows.extend(local_c_rows)
            evaluation_log[f"{b:.3f}"] = evals
            print(
                f"beta={b:.3f}: lambda={beta_row['lambda_star']:.6f}, "
                f"CVaR={beta_row['cvar']:.6f}, exposure={beta_row['mean_exposure']:.6f}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_worker, b): b for b in BETA_GRID}
            for fut in as_completed(futures):
                b = futures[fut]
                beta_row, local_c_rows, evals = fut.result()
                beta_rows.append(beta_row)
                c_rows.extend(local_c_rows)
                evaluation_log[f"{b:.3f}"] = evals
                print(
                    f"beta={b:.3f}: lambda={beta_row['lambda_star']:.6f}, "
                    f"CVaR={beta_row['cvar']:.6f}, exposure={beta_row['mean_exposure']:.6f}",
                    flush=True,
                )

    beta_rows.sort(key=lambda r: r["beta_perp"])
    c_rows.sort(key=lambda r: (r["beta_perp"], r["cvar_limit"]), reverse=False)

    beta_fields = [
        "beta_perp", "cvar_limit", "lambda_star", "eta", "cvar", "residual",
        "mean_exposure", "mean_terminal_wealth", "max_exposure", "p99_exposure",
        "root_iterations", "root_status",
    ]
    c_fields = [
        "beta_perp", "cvar_limit", "lambda_star", "eta", "cvar", "residual",
        "mean_exposure", "mean_terminal_wealth", "max_exposure", "p99_exposure",
        "root_iterations", "root_status",
    ]
    write_csv(outdir / "beta_sweep_results.csv", beta_rows, beta_fields)
    write_csv(outdir / "c_sweep_results.csv", c_rows, c_fields)

    (outdir / "beta_sweep_results.json").write_text(json.dumps(beta_rows, indent=2))
    (outdir / "c_sweep_results.json").write_text(json.dumps(c_rows, indent=2))
    (outdir / "fixed_lambda_evaluations.json").write_text(
        json.dumps(evaluation_log, indent=2)
    )

    make_plots(beta_rows, c_rows, outdir)

    print(f"\nWrote Section 5.3 outputs to: {outdir.resolve()}")
    print("  beta_sweep_results.csv")
    print("  c_sweep_results.csv")
    print("  lambda_vs_beta.pdf / .png")
    print("  exposure_vs_beta.pdf / .png")
    print("  lambda_vs_c.pdf / .png")
    print("  position_vs_c.pdf / .png")


if __name__ == "__main__":
    main()
