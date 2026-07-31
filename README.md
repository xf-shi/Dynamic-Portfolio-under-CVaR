# Dynamic Portfolio Optimization under CVaR Constraints

This repository contains the numerical experiments accompanying the manuscript **Dynamic Portfolio Optimization under CVaR Constraints**. The code studies continuous-time portfolio choice subject to a terminal Conditional Value-at-Risk (CVaR) constraint in complete and incomplete markets, with both frictionless and price-impact dynamics.

The four Jupyter notebooks are self-contained: each includes the model specification, Hamilton--Jacobi--Bellman (HJB) solver, CVaR calibration, Monte Carlo simulation, and plotting routines. Only NumPy and Matplotlib are required for the numerical calculations.


## Overview

We consider a continuous-time investor who minimizes a convex trading objective subject to a CVaR constraint on terminal loss. 

In the numerical experiments, the minimization target is the linear--quadratic loss

$$
J(\varphi)
=\mathbb{E}\Bigg[
\frac{\gamma}{2}\langle W^{\varphi}\rangle_T
-\bigl(W_T^{\varphi}-W_0\bigr)
\Bigg] - \text{total costs}
=\mathbb{E}\Bigg[
\int_0^T
\Bigg(
\frac{\gamma}{2}\bigl[(\sigma\varphi_t)^2+(\beta^{\perp})^2\bigr]
-\mu\varphi_t
\Bigg)dt
\Bigg]  - \text{total costs} ,
$$

where $$\langle W^{\varphi}\rangle_T$$ is the quadratic variation of wealth and $$\gamma>0$$ is the risk-aversion parameter. Thus, the objective balances a quadratic variance penalty against the linear expected-return reward. In the price-impact experiment, the running loss additionally contains the trading-cost term $$\Lambda |v_t|^q/q$$.

Writing terminal loss as $$\ell(W_T)=-W_T$$, the risk constraint is

$$
\text{CVaR}_{\alpha}(L_T) \leq c,
$$

where $$W_T$$ is terminal wealth and $$\alpha=0.95$$ in all reported experiments. The active cases use $$=-0.94$$, equivalently requiring lower-tail wealth $$\text{CVaR}(W_T)$$ to be at least $$0.94$$.

The auxiliary-threshold representation of CVaR converts the constrained problem into a family of standard stochastic-control problems indexed by the Rockafellar threshold $$R\eta$$ and Lagrange multiplier $$\lambda$$. Numerically, we combine:

- a finite-difference HJB solver for the feedback policy;
- golden-section minimization over $$\eta$$;
- outer bisection over $$\lambda$$;
- Monte Carlo evaluation on training, holdout, or out-of-sample paths.

When the CVaR constraint is inactive, the computed strategy recovers the Merton benchmark. When it binds, the policy reduces risky exposure after adverse outcomes while preserving more exposure in favorable states. Incomplete-market risk strengthens this conservative response, while price impact lowers both the desired position and adjustment speed.

## Repository Contents

| File | Description |
|---|---|
| [`Complete_Market_Plots.ipynb`](Complete_Market_Plots.ipynb) | complete-market benchmark. |
| [`Incomplete_Market_Plots.ipynb`](Incomplete_Market_Plots.ipynb) | incomplete-market model with an independent, nontraded endowment shock. |
| [`Linear_Control_Dynamics_Plots.ipynb`](Linear_Control_Dynamics_Plots.ipynb) | quadratic transaction costs model in which dollar exposure evolves through a controlled adjustment rate. |
| [`Square_Root_Impact_Plots.ipynb`](Square_Root_Impact_Plots.ipynb) | Square-root price-impact model. |
| [`index.html`](index.html) | Standalone GitHub Pages overview of the project and stored results. |


## Common Model Parameters

The numerical experiments share the following baseline calibration.

| Symbol / code | Meaning | Value |
|---|---|---:|
| `horizon` | Trading horizon $$T$$ | `1.0` |
| `w0` | Initial wealth $$W_0$$ | `1.0` |
| `mu` | Risky-asset drift $$\mu$$ | `0.08` |
| `sigma` | Risky-asset volatility $$\sigma$$ | `0.20` |
| `gamma` | Risk-aversion coefficient $$\gamma$$ | `5.0` |
| `alpha` | CVaR confidence level $$\alpha$$ | `0.95` |
| `cvar_limit` | Active terminal-loss CVaR limit $$c$$ | `-0.94` |
| `x_merton` | Frictionless Merton dollar exposure $$\mu/\gamma\sigma^2$$ | `0.40` |

The notebooks label the unconstrained comparison as **inactive** and the binding CVaR solution as **active**. Some printed tables use the loss convention and therefore report CVaR with a minus sign; the result tables below use the equivalent positive lower-tail wealth convention.

## Numerical Experiments

### 1. Complete-Market Benchmark

The complete-market notebook uses wealth as the only HJB state. Dollar risky exposure \(\phi_t=\varphi_tS_t\) is controlled directly, and the wealth dynamics are

$$
dW_t = \phi_t(\mu\,dt+\sigma\,dB_t).
$$

For fixed \((\lambda,\eta)\), the bounded quadratic Hamiltonian is minimized analytically. The stored experiment uses a 201-point wealth grid, 100 time steps, and 5,000 Monte Carlo paths.

| Result | Inactive | Active |
|---|---:|---:|
| Mean terminal wealth | 1.0323 | 1.0242 |
| Lower-tail wealth CVaR | 0.8690 | 0.9400 |
| Mean dollar risky exposure | 0.4000 | 0.3001 |
| CVaR multiplier \(\lambda^*\) | 0.0000 | 0.0666 |

<p align="center">
  <img src="figures/complete_market.png" width="850" alt="Complete-market terminal wealth distributions for inactive and active CVaR policies">
</p>

The active policy compresses the left tail around the required wealth floor. This improvement is obtained by lowering average risky exposure from the Merton level of \(0.40\) to approximately \(0.30\).

### 2. Incomplete Market with Nontraded Risk

The incomplete-market experiment introduces an independent endowment shock:

$$
dW_t = \phi_t(\mu\,dt+\sigma\,dB_t)
       +\beta^{\perp}dB_t^{\perp},
\qquad B\perp B^{\perp},
$$

with \(\beta^{\perp}=0.02\). The HJB state remains one-dimensional, but the wealth grid is concentrated near \(W=-\eta\), where the terminal CVaR penalty has its kink. The multiplier is calibrated using 20,000 independent holdout paths.

| Result | Inactive | Active |
|---|---:|---:|
| Mean terminal wealth | 1.0317 | 1.0205 |
| Lower-tail wealth CVaR | 0.8642 | 0.9400 |
| Mean dollar risky exposure | 0.4000 | 0.2601 |
| CVaR multiplier \(\lambda^*\) | 0.0000 | 0.1306 |

<p align="center">
  <img src="figures/incomplete_market.png" width="850" alt="Incomplete-market mean dollar risky exposure and active policy range">
</p>

Because the independent shock cannot be hedged through the stock, the active strategy is more conservative than in the complete market. Its mean dollar exposure falls to approximately \(0.26\).

### 3. Linear Control Dynamics

The third notebook treats dollar risky exposure \(x_t\) as a state variable and its adjustment rate \(v_t\) as the control:

$$
\begin{aligned}
dW_t &= x_t(\mu\,dt+\sigma\,dB_t),\\
dx_t &= v_t\,dt+x_t(\mu\,dt+\sigma\,dB_t).
\end{aligned}
$$

This produces a two-state HJB problem on a \(161\times193\) wealth--exposure grid with 320 time steps. The calibrated active policy is feasible on all 40 tuning and holdout seeds, with worst-panel CVaR slack below \(10^{-5}\).

| Result | Inactive | Active |
|---|---:|---:|
| Mean terminal wealth | 1.0251 | 1.0197 |
| Lower-tail wealth CVaR | 0.8958 | 0.9409 |
| Standard deviation of terminal wealth | 0.0671 | 0.0553 |
| Mean dollar risky exposure | 0.3105 | 0.2421 |

<p align="center">
  <img src="figures/linear_market.png" width="850" alt="Linear-control wealth, risky exposure, and trading speed in favorable and unfavorable markets">
</p>

The path comparison illustrates the state dependence of the solution. Exposure approaches the Merton level after favorable outcomes but is progressively reduced following an unfavorable market path.

### 4. Square-Root Price Impact

The final experiment adds a nonlinear trading penalty with exponent \(q=3/2\):

$$
\begin{aligned}
dW_t &= \left(\mu x_t-\frac{\Lambda}{q}|v_t|^q\right)dt
       +\sigma x_t\,dB_t,\\
dx_t &= v_t\,dt+x_t(\mu\,dt+\sigma\,dB_t),
\qquad q=\frac{3}{2}.
\end{aligned}
$$

The saved calibration uses \(\Lambda=0.004472135955\), a fine local action grid with spacing `0.01`, and no independent endowment shock. After optimization, both policies are evaluated on a final sample of 50,000 paths that is not used for calibration.

| Out-of-sample result | Inactive | Active |
|---|---:|---:|
| Mean terminal wealth | 1.0260 | 1.0180 |
| Lower-tail wealth CVaR | 0.8839 | 0.9409 |
| Standard deviation of terminal wealth | 0.0726 | 0.0544 |
| Mean dollar risky exposure | 0.3500 | 0.2422 |
| Mean absolute trading rate | 0.4049 | 0.3439 |

<p align="center">
  <img src="figures/power_market.png" width="850" alt="Mean trading-rate controls under inactive and active CVaR policies with square-root price impact">
</p>

Price impact slows the initial movement toward the desired position. The active policy trades less aggressively and maintains a lower average exposure, reducing both terminal dispersion and downside risk.

## Summary of Stored Results

| Model | Active tail wealth CVaR | Inactive → active mean wealth | Inactive → active exposure |
|---|---:|---:|---:|---:|
| Complete market | 0.9400 | 1.0323 → 1.0242 | 0.4000 → 0.3001 | 
| Incomplete market | 0.9400 | 1.0317 → 1.0205 | 0.4000 → 0.2601 | 
| Linear control | 0.9409 | 1.0251 → 1.0197 | 0.3105 → 0.2421 | 
| Square-root impact | 0.9409 | 1.0260 → 1.0180 | 0.3500 → 0.2422 | 

These numerical values are the outputs saved in the notebooks. Fresh Monte Carlo simulations have ordinary sampling variation, and the results may change when the grids, seeds, tolerances, or model parameters are modified.


## Citation

If you use this repository in academic work, please cite:

> Anran Hu, Silvana M. Pesenti, and Xiaofei Shi. *Dynamic Portfolio Optimization under CVaR Constraints*. 2026.

```bibtex
@unpublished{hu2026dynamic,
  title  = {Dynamic Portfolio Optimization under CVaR Constraints},
  author = {Hu, Anran and Pesenti, Silvana M. and Shi, Xiaofei},
  year   = {2026},
  note   = {Manuscript}
}
