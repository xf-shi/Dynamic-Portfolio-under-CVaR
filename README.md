# Dynamic Portfolio Optimization under CVaR Constraints

Companion code for **"Dynamic Portfolio Optimization under CVaR Constraints"** by Anran Hu, Silvana M. Pesenti, and Xiaofei Shi.

## Experiments

The four Python scripts are standalone. They cover 18 distinct formal cases: 14 direct-exposure cases and four trading-rate cases.

| File | Experiment |
| --- | --- |
| `complete_market.py` | Section 5.1: complete-market nonbinding and binding benchmarks. |
| `incomplete_market.py` | Sections 5.2-5.3: incomplete-market benchmarks and all 14 direct-exposure comparative-statics cases. |
| `quadratic_regularization.py` | Section 6.1: quadratic trading-rate regularization. |
| `square_root_impact.py` | Section 6.2: square-root price impact. |

## Model and algorithm

Terminal loss is $-W_T$, and the risk constraint is $\mathrm{CVaR}_{\alpha}(-W_T)\le c$. The baseline budgets are $c=-0.86$ (nonbinding) and $c=-0.94$ (binding).

| Parameter | Value |
| --- | ---: |
| Horizon $T$ | 1 |
| Initial wealth $w_0$ | 1 |
| Risk-free rate $r$ | 0 |
| Excess return $\mu$ | 0.08 |
| Volatility $\sigma$ | 0.20 |
| Risk aversion $\gamma$ | 5 |
| CVaR level $\alpha$ | 0.95 |

In the direct-exposure models, $\varphi_t$ is dollar risky exposure and $\beta^\perp$ is independent nontraded background risk. The Merton benchmark is $\mu/(\gamma\sigma^2)=0.40$; it is not imposed as a cap on the policy.

In the trading-rate models, the control $u_t$ is the signed dollar trading rate and the exposure state satisfies

```math
d\varphi_t=u_t\,dt+\varphi_t(\mu\,dt+\sigma\,dB_t),\qquad \varphi_0=0.
```

- **Quadratic regularization:** $\Lambda u_t^2/2$, with $\Lambda=0.01$, enters the objective only.
- **Square-root impact:** $\Lambda_{3/2}|u_t|^{3/2}/(3/2)$, with $\Lambda_{3/2}=0.004472135955$, enters the objective and is deducted from wealth.

The search evaluates the objective and risk residual with the same numerical oracle:

```math
C_h(\pi_h,\eta)=\eta+\frac{1}{1-\alpha}\mathbb E_h[(-W_T^{\pi_h}-\eta)^+]-c,
\qquad
L_h(\pi_h,\eta,\lambda)=J_h(\pi_h)+\lambda C_h(\pi_h,\eta).
```

For positive $\lambda$, the inner golden-section search performs 64 reductions and compares $L_h$. The outer search uses the returned policy and fixed returned threshold to evaluate $C_h$, performs up to 20 bisection iterations, and returns the last tested midpoint. An exactly zero residual permits an early exit. At $\lambda=0$, the unconstrained policy is fixed and the threshold minimizes its risk auxiliary function.

After the search, the multiplier, threshold, and policy are frozen for independent Monte Carlo evaluation. Evaluation samples do not enter the search, and no policy recovery or retuning is applied. Square-root impact is a numerical experiment outside the paper's affine-state convergence setting.

## Numerical settings

All state grids are uniform, with wealth coordinate $z=W+\eta$.

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
| Incomplete market, $\beta^\perp>0$ | 65,536 | 82641279 | Milstein | 16 |
| Quadratic, nonbinding | 1,048,576 | 82641281 | Milstein | 64 |
| Quadratic, binding | 65,536 | 82641281 | Milstein | 64 |
| Square-root, nonbinding | 1,048,576 | 82641282 | Milstein | 64 |
| Square-root, binding | 65,536 | 82641282 | Milstein | 64 |

## Running the experiments

Use Python 3.12 and the tested dependencies:

```bash
python -m pip install numpy==2.3.5 numba==0.68.1 llvmlite==0.50.0 matplotlib==3.11.1

python complete_market.py --output results
python incomplete_market.py --output results
python quadratic_regularization.py --output results
python square_root_impact.py --output results
```

Matching completed cases are reused. Outputs include case solutions, simulation batches, summary CSVs, and summary plots. Full-resolution runs take several hours.

To list the direct-exposure cases or recompute their statistics from saved formal samples:

```bash
python incomplete_market.py --list
python incomplete_market.py --output results --verify-only
```

## Final results

The table reports all 18 distinct formal cases. Return is the simulated mean terminal return, $100(\widehat{\mathbb E}[W_T]/w_0-1)$. Average exposure estimates $T^{-1}\mathbb E[\int_0^T\varphi_t\,dt]$.

| Experiment | $\beta^\perp$ | $c$ | $\widehat\lambda$ | Loss CVaR | Return | Average exposure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Complete market | 0.000 | -0.86 | 0.000000 | -0.866987 | 3.23% | 0.400000 |
| Complete market | 0.000 | -0.88 | 0.005306 | -0.880062 | 3.15% | 0.391200 |
| Complete market | 0.000 | -0.90 | 0.015926 | -0.900030 | 3.02% | 0.374049 |
| Complete market | 0.000 | -0.92 | 0.031743 | -0.919996 | 2.83% | 0.350602 |
| Complete market | 0.000 | -0.94 | 0.058146 | -0.939972 | 2.55% | 0.317031 |
| Incomplete market | 0.020 | -0.86 | 0.000000 | -0.861910 | 3.23% | 0.400000 |
| Incomplete market | 0.020 | -0.88 | 0.008982 | -0.880067 | 3.12% | 0.386019 |
| Incomplete market | 0.020 | -0.90 | 0.025336 | -0.900055 | 2.93% | 0.363258 |
| Incomplete market | 0.020 | -0.92 | 0.057224 | -0.920059 | 2.64% | 0.327378 |
| Incomplete market | 0.020 | -0.94 | 0.138464 | -0.940052 | 2.15% | 0.266086 |
| Incomplete market | 0.005 | -0.94 | 0.064182 | -0.940042 | 2.52% | 0.311886 |
| Incomplete market | 0.010 | -0.94 | 0.076353 | -0.939977 | 2.44% | 0.302687 |
| Incomplete market | 0.015 | -0.94 | 0.098101 | -0.940041 | 2.33% | 0.288045 |
| Incomplete market | 0.025 | -0.94 | 0.232862 | -0.939999 | 1.87% | 0.231447 |
| Quadratic regularization | 0.000 | -0.86 | 0.000000 | -0.897105 | 2.48% | 0.310104 |
| Quadratic regularization | 0.000 | -0.94 | 0.048178 | -0.939992 | 2.04% | 0.250354 |
| Square-root impact | 0.000 | -0.86 | 0.000000 | -0.886100 | 2.63% | 0.348473 |
| Square-root impact | 0.000 | -0.94 | 0.058561 | -0.939982 | 2.02% | 0.266647 |

CVaR estimates use martingale variance reduction. Raw empirical CVaR, fixed-threshold risk, and Monte Carlo standard errors are also saved by the scripts. The empirical risk-check tolerance is $10^{-4}$; this is not a total continuous-model error bound. Display rounding does not establish strict feasibility.

For the nonbinding direct-exposure benchmarks, the theoretical expected return is $3.20\%$ in both market settings. Differences in the reported Monte Carlo returns reflect sampling variation.

Comparative statics use $\beta^\perp\in\{0,0.005,0.010,0.015,0.020,0.025\}$ at $c=-0.94$, and $c\in\{-0.86,-0.88,-0.90,-0.92,-0.94\}$ at $\beta^\perp=0$ and $0.02$. The monotonic pattern in background risk is specific to this calibration. At $\beta^\perp=0.025$, some simulated paths reach the upper numerical exposure bound of 2.
