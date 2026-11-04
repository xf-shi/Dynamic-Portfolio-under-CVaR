from __future__ import annotations
from dataclasses import dataclass, replace
import math, time, json
from typing import Optional
import numpy as np

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
# Section 5.2 incomplete-market experiment
# -----------------------------------------------------------------------------
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

BETA_PERP = 0.02
SEARCH_PATHS = 5_000
SEARCH_SEED = 13
SELECTION_PATHS = 20_000
SELECTION_SEED = 686_035_552
FINAL_OOS_PATHS = 50_000
FINAL_OOS_SEED = 1_900_013
NONBINDING_C = -0.86
BINDING_C = -0.94


def evaluate_frozen_policy(result, lambda_value, params, *, samples=FINAL_OOS_PATHS, seed=FINAL_OOS_SEED):
    """Evaluate a frozen policy on an independent OOS sample.

    The OOS sample is evaluation only: lambda, eta, wealth grid, and policy grid
    are not changed after the multiplier-selection stage.
    """
    p = replace(params, samples=int(samples), seed=int(seed))
    d_b = generate_brownian_increments(p)
    wealth, stock, phi_paths = simulate_feedback_policy(
        result.wealth_grid, result.policy_grid, d_b, p
    )
    terminal_wealth = wealth[:, -1]
    loss = -terminal_wealth
    cvar, var = cvar_from_loss(loss, p.alpha)
    hinge = np.maximum(loss - result.eta, 0.0)
    rockafellar_residual = float(
        result.eta + hinge.mean() / (1.0 - p.alpha) - p.cvar_limit
    )
    return replace(
        result,
        phi=np.mean(phi_paths, axis=0),
        lagrangian=float(result.objective + lambda_value * rockafellar_residual),
        residual=float(cvar - p.cvar_limit),
        cvar=float(cvar),
        var=float(var),
        mean_terminal_wealth=float(np.mean(terminal_wealth)),
        terminal_wealth_std=float(np.std(terminal_wealth)),
        loss_tail_probability=float(np.mean(loss > result.eta)),
        terminal_wealth=terminal_wealth,
        loss=loss,
        phi_paths=phi_paths,
        wealth_paths=wealth,
        stock_paths=stock,
    )


def solve_case(cvar_limit: float):
    """Search on 5k paths, select lambda on 20k independent paths, then freeze."""
    p = ModelParams(
        beta_perp=BETA_PERP,
        cvar_limit=float(cvar_limit),
        samples=SEARCH_PATHS,
        seed=SEARCH_SEED,
        cap_control_at_merton=False,
    )
    d_train = generate_brownian_increments(p)

    p_select = replace(
        p,
        samples=SELECTION_PATHS,
        seed=SELECTION_SEED,
    )
    d_select = generate_brownian_increments(p_select)

    selected, lambda_star, history = outer_bisection(
        d_train, p, d_select, p_select
    )
    final = evaluate_frozen_policy(selected, lambda_star, p)
    return p, final, float(lambda_star), history


def lower_tail_loss_cvar_path(wealth_paths: np.ndarray, alpha: float) -> np.ndarray:
    """Unconditional CVaR_alpha(-W_t) evaluated separately at each date."""
    out = np.empty(wealth_paths.shape[1], dtype=float)
    for j in range(wealth_paths.shape[1]):
        out[j] = cvar_from_loss(-wealth_paths[:, j], alpha)[0]
    return out


def gaussian_kde_1d(samples, grid, bandwidth=None):
    samples = np.asarray(samples, dtype=float)
    if bandwidth is None:
        scale = max(float(np.std(samples, ddof=1)), 1.0e-8)
        bandwidth = max(1.06 * scale * samples.size ** (-0.2), 1.0e-4)
    z = (grid[:, None] - samples[None, :]) / bandwidth
    return np.exp(-0.5 * z * z).mean(axis=1) / (
        bandwidth * np.sqrt(2.0 * np.pi)
    )


def save_figure(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_exposure(nonbinding, binding, output_dir: Path):
    t = np.linspace(0.0, 1.0, binding.phi_paths.shape[1])
    fig, ax = plt.subplots(figsize=(9.6, 5.2))

    ax.plot(t, np.mean(nonbinding.phi_paths, axis=0), lw=2.2, label="nonbinding mean")
    bind_mean = np.mean(binding.phi_paths, axis=0)
    q10, q90 = np.quantile(binding.phi_paths, [0.10, 0.90], axis=0)
    ax.plot(t, bind_mean, lw=2.2, label="binding mean")
    ax.fill_between(t, q10, q90, alpha=0.16, linewidth=0, label="binding 10--90% range")
    ax.axhline(0.40, linestyle="--", linewidth=1.4, label="Merton benchmark = 0.40")

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("time")
    ax.set_ylabel("dollar risky exposure")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    fig.tight_layout()
    save_figure(fig, output_dir / "incomplete_control.png")


def plot_terminal_wealth(nonbinding, binding, output_dir: Path):
    arrays = [nonbinding.terminal_wealth, binding.terminal_wealth]
    all_w = np.concatenate(arrays)
    padding = 0.04 * max(float(np.ptp(all_w)), 1e-3)
    grid = np.linspace(all_w.min() - padding, all_w.max() + padding, 320)

    fig, ax = plt.subplots(figsize=(9.6, 5.5))
    for result, label in [(nonbinding, "nonbinding"), (binding, "binding")]:
        ax.plot(grid, gaussian_kde_1d(result.terminal_wealth, grid), lw=2.2, label=label)
        ax.axvline(float(np.mean(result.terminal_wealth)), linestyle="--", linewidth=1.2,
                   label=f"{label} mean = {np.mean(result.terminal_wealth):.2f}")

    ax.set_xlabel(r"terminal wealth $W_T^*$")
    ax.set_ylabel("density")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", frameon=True)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    fig.tight_layout()
    save_figure(fig, output_dir / "incomplete_w_dist.png")


def plot_cvar_path(nonbinding, binding, output_dir: Path):
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    for result, c, label in [
        (nonbinding, NONBINDING_C, "nonbinding"),
        (binding, BINDING_C, "binding"),
    ]:
        path = lower_tail_loss_cvar_path(result.wealth_paths, 0.95)
        t = np.linspace(0.0, 1.0, path.size)
        ax.plot(t, path, lw=2.2, label=f"{label} CVaR")
        ax.axhline(c, linestyle="--", linewidth=1.1, alpha=0.8,
                   label=f"{label} terminal limit")

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("time")
    ax.set_ylabel(r"CVaR$_\alpha(-W_t^*)$")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", frameon=True)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    fig.tight_layout()
    save_figure(fig, output_dir / "incomplete_cvar.png")


def select_favorable_unfavorable_paths(
    wealth_paths, phi_paths, merton_value=0.40, target_final_phi=0.45
):
    """Match the Section 5.2 diagnostic-path selection rule.

    Favorable: terminal wealth near the 90th percentile.
    Unfavorable: bottom-decile terminal wealth, final exposure above Merton,
    then choose the path whose final exposure is closest to 0.45.
    """
    terminal_wealth = wealth_paths[:, -1]
    final_phi = phi_paths[:, -1]

    favorable_target = np.quantile(terminal_wealth, 0.90)
    favorable_idx = int(np.argmin(np.abs(terminal_wealth - favorable_target)))

    lower_tail_cutoff = np.quantile(terminal_wealth, 0.10)
    bad_pool = np.where(
        (terminal_wealth <= lower_tail_cutoff)
        & (final_phi > merton_value + 1.0e-6)
    )[0]
    if bad_pool.size == 0:
        raise RuntimeError(
            "No lower-tail OOS path has final exposure above the Merton benchmark."
        )
    unfavorable_idx = int(
        bad_pool[np.argmin(np.abs(final_phi[bad_pool] - target_final_phi))]
    )
    return favorable_idx, unfavorable_idx


def plot_selected_paths(binding, output_dir: Path):
    wealth = binding.wealth_paths
    phi = binding.phi_paths
    favorable_idx, unfavorable_idx = select_favorable_unfavorable_paths(wealth, phi)

    t_w = np.linspace(0.0, 1.0, wealth.shape[1])
    t_phi = t_w[:-1]
    fig, (ax_w, ax_phi) = plt.subplots(2, 1, figsize=(9.6, 6.8), sharex=True)

    for idx, label in [
        (favorable_idx, "favorable realization"),
        (unfavorable_idx, "lower-tail realization"),
    ]:
        ax_w.plot(t_w, wealth[idx], lw=2.0,
                  label=f"{label}: $W_T^*$={wealth[idx,-1]:.2f}")
        ax_phi.plot(t_phi, phi[idx], lw=2.0,
                    label=fr"{label}: final exposure={phi[idx,-1]:.2f}")

    ax_phi.axhline(0.40, linestyle="--", linewidth=1.4, label="Merton benchmark = 0.40")
    ax_w.set_ylabel("wealth")
    ax_phi.set_ylabel("dollar risky exposure")
    ax_phi.set_xlabel("time")
    ax_w.grid(True, alpha=0.30)
    ax_phi.grid(True, alpha=0.30)
    ax_w.legend(loc="upper left", frameon=True)
    ax_phi.legend(loc="lower left", frameon=True)
    ax_w.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_phi.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_phi.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    fig.tight_layout()
    save_figure(fig, output_dir / "incomplete_sample_paths.png")

    return {
        "favorable_index": int(favorable_idx),
        "favorable_terminal_wealth": float(wealth[favorable_idx, -1]),
        "favorable_final_exposure": float(phi[favorable_idx, -1]),
        "unfavorable_index": int(unfavorable_idx),
        "unfavorable_terminal_wealth": float(wealth[unfavorable_idx, -1]),
        "unfavorable_final_exposure": float(phi[unfavorable_idx, -1]),
    }


def result_row(name, p, result, lambda_star):
    return {
        "case": name,
        "beta_perp": p.beta_perp,
        "cvar_limit": p.cvar_limit,
        "lambda_star": lambda_star,
        "eta_star": result.eta,
        "oos_cvar": result.cvar,
        "oos_residual": result.cvar - p.cvar_limit,
        "mean_terminal_wealth": result.mean_terminal_wealth,
        "expected_terminal_return_pct": 100.0 * (result.mean_terminal_wealth / p.w0 - 1.0),
        "average_exposure": float(np.mean(result.phi_paths)),
        "max_exposure": float(np.max(result.phi_paths)),
    }


def run_experiment(output_dir: Path):
    output_dir = Path(output_dir)
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("Section 5.2 incomplete-market experiment")
    print(f"beta_perp = {BETA_PERP:.3f}")
    print("Merton exposure = 0.40 (benchmark only; not a cap)")
    print(f"search paths = {SEARCH_PATHS}, multiplier-selection paths = {SELECTION_PATHS}")
    print(f"final OOS paths = {FINAL_OOS_PATHS}; no ex-post feasibility adjustment")

    p_non, nonbinding, lam_non, hist_non = solve_case(NONBINDING_C)
    print(f"nonbinding: lambda={lam_non:.8f}, eta={nonbinding.eta:.8f}, "
          f"OOS CVaR={nonbinding.cvar:.8f}, avg exposure={np.mean(nonbinding.phi_paths):.8f}")

    p_bind, binding, lam_bind, hist_bind = solve_case(BINDING_C)
    print(f"binding: lambda={lam_bind:.8f}, eta={binding.eta:.8f}, "
          f"OOS CVaR={binding.cvar:.8f}, avg exposure={np.mean(binding.phi_paths):.8f}")

    plot_exposure(nonbinding, binding, figure_dir)
    plot_terminal_wealth(nonbinding, binding, figure_dir)
    plot_cvar_path(nonbinding, binding, figure_dir)
    selected = plot_selected_paths(binding, figure_dir)

    rows = [
        result_row("nonbinding", p_non, nonbinding, lam_non),
        result_row("binding", p_bind, binding, lam_bind),
    ]
    with (output_dir / "incomplete_market_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "model": "incomplete_market",
        "section": "5.2",
        "beta_perp": BETA_PERP,
        "merton_cap": False,
        "action_bounds": [-2.0, 2.0],
        "search_paths": SEARCH_PATHS,
        "search_seed": SEARCH_SEED,
        "multiplier_selection_paths": SELECTION_PATHS,
        "multiplier_selection_seed": SELECTION_SEED,
        "final_oos_paths": FINAL_OOS_PATHS,
        "final_oos_seed": FINAL_OOS_SEED,
        "oos_feasibility_adjustment": False,
        "selected_paths": selected,
        "search_history": {
            "nonbinding": [[float(a), float(b)] for a, b in hist_non],
            "binding": [[float(a), float(b)] for a, b in hist_bind],
        },
        "results": rows,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Section 5.2 incomplete-market CVaR experiment"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/incomplete_market"),
        help="Output directory (default: results/incomplete_market)",
    )
    args = parser.parse_args()
    run_experiment(args.output_dir)


if __name__ == "__main__":
    main()
