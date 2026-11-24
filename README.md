# Dynamic Portfolio Optimization under CVaR Constraints

Companion code for **"Dynamic Portfolio Optimization under CVaR Constraints"** by Anran Hu, Silvana M. Pesenti, and Xiaofei Shi.

The scripts solve the portfolio problems in Sections 5-6 using an inner search over the CVaR threshold and an outer search over the dual multiplier. Under the paper's strong-duality assumptions, the multiplier is the shadow price of the CVaR budget.

## Repository structure

The four experiment scripts are standalone. `comparative_statics.py` is an optional entry point that calls `incomplete_market.py`; keep these two files in the same directory.

| File | Experiment |
| --- | --- |
| `complete_market.py` | Section 5.1: complete-market benchmark. |
| `incomplete_market.py` | Sections 5.2-5.3: incomplete market and all 14 direct-exposure comparative-statics cases. |
| `comparative_statics.py` | Optional entry point for the same cases and algorithm as `incomplete_market.py`. |
| `quadratic_regularization.py` | Section 6.1: quadratic trading-rate regularization. |
| `square_root_impact.py` | Section 6.2: square-root price impact. |

There are 18 distinct formal cases: 14 direct-exposure cases and four trading-rate cases.

## Model and algorithm

Terminal loss is $-W_T$, with constraint $\mathrm{CVaR}_{\alpha}(-W_T)\le c$. The baseline budgets are $c=-0.86$ (nonbinding) and $c=-0.94$ (binding).

| Parameter | Value |
| --- | ---: |
| Horizon $T$ | 1 |
| Initial wealth $W_0$ | 1 |
| Risk-free rate $r$ | 0 |
| Excess return $\mu$ | 0.08 |
| Volatility $\sigma$ | 0.20 |
| Risk aversion $\gamma$ | 5 |
| CVaR level $\alpha$ | 0.95 |

Dollar risky exposure is denoted by $\varphi_t$. The complete market has $\beta^\perp=0$; the baseline incomplete market has independent nontraded risk $\beta^\perp=0.02$. The unconstrained Merton exposure is $\mu/(\gamma\sigma^2)=0.40$, and the numerical exposure bounds are $[-2,2]$.

In the trading-rate models, the dollar trading rate $u_t$ controls the exposure state:

```math
d\varphi_t=u_t\,dt+\varphi_t(\mu\,dt+\sigma\,dB_t),\qquad \varphi_0=0.
```

- **Quadratic regularization:** $\Lambda u_t^2/2$, with $\Lambda=0.01$, enters the objective only.
- **Square-root impact:** $\Lambda_{3/2}|u_t|^{3/2}/(3/2)$, with $\Lambda_{3/2}=0.004472135955$, enters the objective and is deducted from wealth.

The search uses the same numerical oracle for

```math
C_h(\pi,\eta)
=\eta+\frac{1}{1-\alpha}\mathbb E_h[(-W_T^\pi-\eta)^+]-c,
\qquad
\mathcal L_h(\pi,\eta;\lambda)=J_h(\pi)+\lambda C_h(\pi,\eta).
```

For positive $\lambda$, the inner golden-section search performs 64 reductions and compares $\mathcal L_h$. The outer search uses the returned fixed-$\eta$ residual $C_h$, performs up to 20 bisection iterations with an exact-zero early exit, and returns the last tested midpoint. At $\lambda=0$, the unconstrained policy is fixed and $\eta$ minimizes its risk auxiliary function.

The multiplier, threshold, and policy are then frozen for independent Monte Carlo evaluation. Fixed-$\eta$ risk and empirical CVaR are reported separately, with martingale variance reduction. Evaluation results do not enter the search or trigger policy recovery or retuning.

The $10^{-4}$ empirical risk-check tolerance is not a total continuous-model error bound or a guarantee of multiplier uniqueness. Square-root impact is a numerical experiment outside the affine-state convergence setting of the paper.

## Numerical settings

The wealth coordinate is $z=W+\eta$. All state grids are uniform.

| Model | Control time steps | State grid | State domain | Control bounds |
| --- | ---: | --- | --- | --- |
| Direct exposure | 1,600 | 19,201 points in $z$ | $[-1.5,1.5]$ | $[-2,2]$ |
| Quadratic regularization | 100 | $3,201\times1,601$ in $(z,\varphi)$ | $[-0.25,0.25]\times[-1,1]$ | $[-2,2]$ |
| Square-root, nonbinding | 50 | $1,601\times801$ in $(z,\varphi)$ | $[-0.25,0.25]\times[-1,1]$ | $[-4,4]$ |
| Square-root, binding | 50 | $801\times801$ in $(z,\varphi)$ | $[-0.25,0.25]\times[-1,1]$ | $[-4,4]$ |

Formal evaluation settings are specified in each script's `CASES` list:

| Experiment | Paths | Seed | Integrator | Substeps per control interval |
| --- | ---: | ---: | --- | ---: |
| Complete market | 65,536 | 82641279 | Milstein | 8 |
| Incomplete market, $\beta^\perp>0$ | 65,536 | 82641279 | Euler | 16 |
| Quadratic, nonbinding | 1,048,576 | 82641281 | Milstein | 64 |
| Quadratic, binding | 65,536 | 82641281 | Milstein | 64 |
| Square-root, nonbinding | 1,048,576 | 82641282 | Milstein | 64 |
| Square-root, binding | 65,536 | 82641282 | Milstein | 64 |

## Running the experiments

Tested with Python 3.12.14 and the following dependencies:

```bash
python -m pip install numpy==2.3.5 numba==0.68.1 llvmlite==0.50.0 matplotlib==3.11.1

python complete_market.py --output results
python incomplete_market.py --output results
python quadratic_regularization.py --output results
python square_root_impact.py --output results
```

The optional `comparative_statics.py` can replace `incomplete_market.py` in these commands. Outputs include case solutions, simulation batches, summary CSVs, and summary plots. Completed cases are reused when their settings match. Full-resolution runs take several hours.

To inspect the case definitions or recompute statistics from saved formal samples:

```bash
python incomplete_market.py --list
python incomplete_market.py --output results --verify-only
```

## Comparative statics

For the risk-budget sweep, the final multipliers are:

| $c$ | $\widehat\lambda$, $\beta^\perp=0$ | $\widehat\lambda$, $\beta^\perp=0.02$ |
| ---: | ---: | ---: |
| -0.86 | 0.000000 | 0.000000 |
| -0.88 | 0.005306 | 0.008982 |
| -0.90 | 0.015926 | 0.025336 |
| -0.92 | 0.031743 | 0.057224 |
| -0.94 | 0.058146 | 0.138464 |

For the background-risk sweep at $c=-0.94$:

| $\beta^\perp$ | $\widehat\lambda$ | Average dollar risky exposure |
| ---: | ---: | ---: |
| 0.000 | 0.058146 | 0.317031 |
| 0.005 | 0.064182 | 0.311886 |
| 0.010 | 0.076353 | 0.302687 |
| 0.015 | 0.098101 | 0.288045 |
| 0.020 | 0.138464 | 0.266086 |
| 0.025 | 0.232862 | 0.231447 |

The multiplier increases as the risk budget tightens. Increasing background risk also raises the multiplier and lowers average exposure in this calibration; the finding in $\beta^\perp$ is numerical rather than a general monotonicity theorem. At $\beta^\perp=0.025$, some simulated paths reach the upper exposure bound of 2.
