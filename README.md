# Dynamic Portfolio Optimization under CVaR Constraints

Companion code for **"Dynamic Portfolio Optimization under CVaR Constraints"** by Anran Hu, Silvana M. Pesenti, and Xiaofei Shi.

## Experiments

The four Python scripts are standalone and cover 18 distinct cases.

| File | Experiment |
| --- | --- |
| `complete_market.py` | Section 5.1: complete-market nonbinding and binding benchmarks. |
| `incomplete_market.py` | Sections 5.2-5.3: incomplete-market benchmarks and all 14 direct-exposure comparative-statics cases, including the complete-market cases. |
| `quadratic_regularization.py` | Section 6.1: quadratic trading-rate regularization. |
| `square_root_impact.py` | Section 6.2: square-root price impact. |

## Model and algorithm

The terminal loss is $-W_T$, with constraint $\mathrm{CVaR}_{\alpha}(-W_T)\le c$. The baseline calibration is $T=1$, $w_0=1$, $r=0$, $\mu=0.08$, $\sigma=0.20$, $\gamma=5$, and $\alpha=0.95$. The Merton dollar exposure is $\mu/(\gamma\sigma^2)=0.400$ and is used as a reference, not a control cap.

In the trading-rate models, the control $u_t$ is the signed dollar trading rate, and

```math
d\varphi_t=u_t\,dt+\varphi_t(\mu\,dt+\sigma\,dB_t),\qquad \varphi_0=0.
```

Quadratic regularization uses $\Lambda=0.01$ and enters the objective only. Square-root execution costs use $\Lambda_{3/2}=0.0045$ and enter both the objective and wealth dynamics as $\Lambda_{3/2}|u_t|^{3/2}/(3/2)$. Case configurations are specified in the scripts.

The objective and risk residual are evaluated using the same numerical oracle:

```math
C_h(\pi_h,\eta)=\eta+\frac{1}{1-\alpha}\mathbb E_h[(-W_T^{\pi_h}-\eta)^+]-c,
\qquad
L_h(\pi_h,\eta,\lambda)=J_h(\pi_h)+\lambda C_h(\pi_h,\eta).
```

For $\lambda>0$, the inner golden-section search performs 64 reductions and compares $L_h$. The outer search evaluates $C_h$ at the returned policy and fixed returned threshold, performs up to 20 bisection iterations, and returns the last tested midpoint. An exactly zero residual permits an early exit. At $\lambda=0$, the unconstrained policy is fixed and the threshold minimizes its risk auxiliary function.

After optimization, the multiplier, threshold, and policy are frozen for independent Milstein evaluation. For nonzero background risk, evaluation uses the equivalent scalar wealth diffusion with coefficient $s(t,w)=\sqrt{\sigma^2\varphi_h(t,w)^2+(\beta^\perp)^2}$; it does not reconstruct the joint paths of the original two Brownian drivers. Evaluation samples do not enter the search, and no policy recovery or retuning is applied.

The square-root-impact experiment is outside the paper's affine-state convergence setting.

## Numerical settings

State grids are uniform, with wealth coordinate $z=W+\eta$.

| Model | Control time steps | State grid | State domain | Control bounds |
| --- | ---: | --- | --- | --- |
| Direct exposure | 1,600 | 19,201 points in $z$ | $[-1.500,1.500]$ | $[-2.000,2.000]$ |
| Quadratic regularization | 100 | $3,201\times1,601$ in $(z,\varphi)$ | $[-0.250,0.250]\times[-1.000,1.000]$ | $[-2.000,2.000]$ |
| Square-root, nonbinding | 50 | $1,601\times801$ in $(z,\varphi)$ | $[-0.250,0.250]\times[-1.000,1.000]$ | $[-4.000,4.000]$ |
| Square-root, binding | 50 | $801\times801$ in $(z,\varphi)$ | $[-0.250,0.250]\times[-1.000,1.000]$ | $[-4.000,4.000]$ |

All independent evaluations use Milstein time stepping.

| Experiment | Paths | Seed | Substeps per control interval |
| --- | ---: | ---: | ---: |
| Complete market | 65,536 | 82641279 | 8 |
| Incomplete market, $\beta^\perp>0$ | 65,536 | 82641279 | 16 |
| Quadratic, nonbinding | 1,048,576 | 82641281 | 64 |
| Quadratic, binding | 65,536 | 82641281 | 64 |
| Square-root, nonbinding | 1,048,576 | 82641282 | 64 |
| Square-root, binding | 65,536 | 82641282 | 64 |

## Running the experiments

Use Python 3.12 and install the dependencies:

```bash
python -m pip install numpy==2.3.5 numba==0.62.1 llvmlite==0.45.1 matplotlib==3.11.1 Pillow==13.0.0

python complete_market.py --output results/complete
python incomplete_market.py --output results/direct_exposure
python quadratic_regularization.py --output results/quadratic
python square_root_impact.py --output results/square_root
```

Use fresh, separate output directories for a full rerun. Matching completed cases are reused. The incomplete-market script includes the complete-market benchmarks, so running `complete_market.py` is optional when reproducing all 18 distinct cases. Outputs include solutions, simulation batches, summary CSVs, and summary plots.

To list the direct-exposure cases or verify saved simulation results:

```bash
python incomplete_market.py --list
python incomplete_market.py --output results/direct_exposure --verify-only
```

## Results

The 14 direct-exposure cases were solved from scratch and independently evaluated using Milstein. Both square-root-impact cases were also reoptimized and independently evaluated with $\Lambda_{3/2}=0.0045$. The two quadratic results come from the earlier full rerun of the identical `quadratic_regularization.py` source. The source files associated with these runs match the four scripts currently published in this repository.

The recorded quadratic runs used Numba 0.68.1 and llvmlite 0.50.0; the direct-exposure and updated square-root runs used Numba 0.62.1 and llvmlite 0.45.1. All used Python 3.12 and NumPy 2.3.5.

All numerical results are rounded to three decimal places. Return is $100(\widehat{\mathbb E}[W_T]/w_0-1)$, and average exposure estimates $T^{-1}\mathbb E[\int_0^T\varphi_t\,dt]$.

| Experiment | $\beta^\perp$ | $c$ | $\widehat\lambda$ | Loss CVaR | Return | Average exposure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Complete market | 0.000 | -0.860 | 0.000 | -0.867 | 3.225% | 0.400 |
| Complete market | 0.000 | -0.880 | 0.005 | -0.880 | 3.154% | 0.391 |
| Complete market | 0.000 | -0.900 | 0.016 | -0.900 | 3.015% | 0.374 |
| Complete market | 0.000 | -0.920 | 0.032 | -0.920 | 2.826% | 0.351 |
| Complete market | 0.000 | -0.940 | 0.058 | -0.940 | 2.555% | 0.317 |
| Incomplete market | 0.020 | -0.860 | 0.000 | -0.862 | 3.233% | 0.400 |
| Incomplete market | 0.020 | -0.880 | 0.009 | -0.880 | 3.120% | 0.386 |
| Incomplete market | 0.020 | -0.900 | 0.025 | -0.900 | 2.934% | 0.363 |
| Incomplete market | 0.020 | -0.920 | 0.057 | -0.920 | 2.643% | 0.327 |
| Incomplete market | 0.020 | -0.940 | 0.138 | -0.940 | 2.148% | 0.266 |
| Incomplete market | 0.005 | -0.940 | 0.064 | -0.940 | 2.517% | 0.312 |
| Incomplete market | 0.010 | -0.940 | 0.076 | -0.940 | 2.443% | 0.303 |
| Incomplete market | 0.015 | -0.940 | 0.098 | -0.940 | 2.325% | 0.288 |
| Incomplete market | 0.025 | -0.940 | 0.233 | -0.940 | 1.872% | 0.231 |
| Quadratic regularization | 0.000 | -0.860 | 0.000 | -0.897 | 2.481% | 0.310 |
| Quadratic regularization | 0.000 | -0.940 | 0.048 | -0.940 | 2.040% | 0.250 |
| Square-root impact | 0.000 | -0.860 | 0.000 | -0.886 | 2.633% | 0.348 |
| Square-root impact | 0.000 | -0.940 | 0.059 | -0.940 | 2.019% | 0.266 |

CVaR estimates use martingale variance reduction. Raw empirical CVaR, fixed-threshold risk, and Monte Carlo standard errors are also saved. The empirical risk-check tolerance is $10^{-4}$; it is not a total continuous-model error bound, and rounded values do not establish strict feasibility.

For both nonbinding direct-exposure benchmarks, the theoretical expected return is $3.200\%$. Differences in simulated mean returns reflect sampling variation. The comparative-static pattern in background risk is specific to this calibration; at $\beta^\perp=0.025$, some simulated paths reach the upper numerical exposure bound of $2.000$.
