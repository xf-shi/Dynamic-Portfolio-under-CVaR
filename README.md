# Dynamic Portfolio Optimization under CVaR Constraints

Companion code for **“Dynamic Portfolio Optimization under CVaR Constraints”** by Anran Hu, Silvana M. Pesenti, and Xiaofei Shi.

The paper studies continuous-time portfolio optimization subject to a hard Conditional Value-at-Risk (CVaR) constraint on terminal loss. Using the Rockafellar–Uryasev auxiliary-threshold representation, the constrained problem is reformulated through a Lagrangian dual problem. Numerically, the method combines a stochastic-control solver with an inner search over the CVaR threshold and an outer search over the dual multiplier.

A key economic interpretation is that the optimal multiplier is the **shadow price of the CVaR budget**: it measures the marginal value of relaxing the tail-risk constraint.

## Repository structure

The repository is organized around four standalone Python scripts. Each script contains its own model specification, solver, parameter search, simulation, and plotting code; no project-specific helper module is required.

| File | Experiment |
| --- | --- |
| `complete_market.py` | Section 5.1: complete-market benchmark, including nonbinding/binding calibrations and path diagnostics. |
| `incomplete_market.py` | Sections 5.2–5.3: incomplete-market benchmark with nontraded risk and the comparative statics in $\beta^\perp$ and the CVaR limit $c$. |
| `quadratic_regularization.py` | Dynamic portfolio management with quadratic trading-rate regularization. |
| `square_root_impact.py` | Dynamic portfolio management with square-root execution-price impact. |

All four experiments are recomputed from scratch. No precomputed policy is required.

## Model and notation

Terminal loss is

$$
L_T=-W_T,

$$

and the risk constraint is

$$
\mathrm{CVaR}_{\alpha}(L_T)\le c,
\qquad \alpha=0.95.

$$

The two baseline risk budgets are

- **nonbinding:** $c=-0.86$;
- **binding:** $c=-0.94$.

The empirical feasibility residual reported in the numerical experiments is

$$
\widehat{\mathrm{CVaR}}_{\alpha}(-W_T)-c.

$$

A nonpositive residual is empirically feasible on that simulation sample. A small positive residual on an independent evaluation sample is reported as is; it is not used to retune the policy.

### Direct-exposure benchmarks

In the complete- and incomplete-market experiments, $\varphi_t$ denotes dollar risky exposure:

$$
dW_t
=
\varphi_t(\mu\,dt+\sigma\,dB_t)
+
\beta^\perp dB_t^\perp.

$$

The complete-market benchmark sets $\beta^\perp=0$. The baseline incomplete-market experiment uses $\beta^\perp=0.02$, where $B^\perp$ is independent of the traded Brownian motion $B$.

The running objective is

$$
J(\varphi)
=
\mathbb{E}\left[
\int_0^T
\left\{
\frac{\gamma}{2}
\left[
(\sigma\varphi_t)^2+(\beta^\perp)^2
\right]
-\mu\varphi_t
\right\}dt
\right].

$$

The common calibration is

| Parameter | Symbol | Value |
| --- | --- | ---: |
| Horizon | $T$ | 1 |
| Initial wealth | $W_0$ | 1.00 |
| Risk-free rate | $r$ | 0.00 |
| Risky-asset drift | $\mu$ | 0.08 |
| Risky-asset volatility | $\sigma$ | 0.20 |
| Risk aversion | $\gamma$ | 5 |
| CVaR level | $\alpha$ | 0.95 |

The unconstrained Merton dollar exposure is

$$
\bar\varphi
=
\frac{\mu}{\gamma\sigma^2}
=
0.40.

$$

This is a **benchmark only**, not a control cap. The direct-exposure experiments retain the numerical action bounds

$$
-2\le \varphi_t\le 2.

$$

### Trading-rate models

For the portfolio-management extensions, let $\phi_t$ denote the number of shares and define

$$
\varphi_t=\phi_tS_t,
\qquad
\dot\varphi_t=\dot\phi_tS_t.

$$

Here $\dot\varphi_t$ denotes the signed dollar trading-rate control. The exposure state evolves as

$$
d\varphi_t
=
\dot\varphi_t\,dt
+
\varphi_t(\mu\,dt+\sigma\,dB_t).

$$

The two specifications are:

- **Quadratic regularization:** $\Lambda\dot\varphi_t^2/2$, with $\Lambda=0.01$. This is an objective regularizer and is not deducted from wealth.
- **Square-root price impact:** $\Lambda_{3/2}|\dot\varphi_t|^{3/2}/(3/2)$, with $\Lambda_{3/2}=0.004472135955$. This cost enters the objective and is also deducted from wealth through the execution-price adjustment.

## Calibration and independent evaluation

Policy selection and final evaluation use separate Monte Carlo samples.

1. The HJB policy and CVaR threshold are computed on the search sample.
2. In the complete- and incomplete-market experiments, an additional independent 20,000-path sample is used to select the raw dual root.
3. The multiplier, threshold, and feedback policy are frozen.
4. The frozen policy is evaluated on 50,000 independent out-of-sample (OOS) paths.

The final OOS sample is **evaluation only**. No safety buffer, feasibility adjustment, or parameter retuning is applied after seeing the final OOS results.

| Experiment | Search paths | Additional multiplier-selection paths | Final OOS paths | Final OOS seed |
| --- | ---: | ---: | ---: | ---: |
| Complete market | 5,000 | 20,000 | 50,000 | `1900013` |
| Incomplete market | 5,000 | 20,000 | 50,000 | `1900013` |
| Quadratic regularization | 10,000 | None | 50,000 | `1900013` |
| Square-root impact | 1,000 | None | 50,000 | `900013` |

For the direct-exposure experiments, the multiplier-selection sample uses seed `686035552`.

## Numerical resolution

| Experiment | Time steps | HJB state grid | Control | $\eta$ tolerance | $\lambda$ tolerance | CVaR residual tolerance |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| Complete market | 100 | 201 nonuniform points in $W$, concentrated near $W=-\eta$ | Dollar exposure in $[-2,2]$; no Merton cap | $5\times10^{-4}$ | $10^{-6}$ | $10^{-5}$ |
| Incomplete market | 100 | 201 nonuniform points in $W$, concentrated near $W=-\eta$ | Dollar exposure in $[-2,2]$; no Merton cap | $5\times10^{-4}$ | $10^{-6}$ | $10^{-5}$ |
| Quadratic regularization | 320 | $161\times193$ in $(W,\varphi)$ | Trading rate in $[-2,2]$ | $5\times10^{-4}$ | $10^{-4}$ | $10^{-4}$ |
| Square-root impact | 200 | $80\times100$ in $(W,\varphi)$ | Trading rate in $[-4,4]$ | $10^{-5}$ | $10^{-7}$ | $10^{-6}$ |

The direct-exposure wealth grids are rebuilt around the candidate terminal-loss kink $W=-\eta$. Numba acceleration is used when available and implements the same finite-difference formulas as the NumPy version.

## Reference out-of-sample results

The following values are from frozen policies evaluated on independent OOS samples.

| Experiment | Case | $c$ | $\widehat\lambda$ | Loss CVaR | $\mathbb{E}[W_T]$ | Average exposure | OOS residual |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Complete market | Nonbinding | -0.86 | 0.00000 | -0.8671 | 1.0322 | 0.4000 | `-7.06e-03` |
| Complete market | Binding | -0.94 | 0.05475 | -0.9392 | 1.0248 | 0.3061 | `+8.22e-04` |
| Incomplete market | Nonbinding | -0.86 | 0.00000 | -0.8620 | 1.0322 | 0.4000 | `-1.97e-03` |
| Incomplete market | Binding | -0.94 | 0.1252 | -0.9382 | 1.0213 | 0.2637 | `+1.75e-03` |
| Quadratic regularization | Nonbinding | -0.86 | 0.0000 | -0.90 | 1.02 | 0.31 | `-3.71e-02` |
| Quadratic regularization | Binding | -0.94 | 0.05 | -0.94 | 1.02 | 0.24 | `-4.26e-04` |
| Square-root impact | Nonbinding | -0.86 | 0.0000 | -0.88 | 1.03 | 0.35 | `-2.39e-02` |
| Square-root impact | Binding | -0.94 | 0.08 | -0.94 | 1.02 | 0.24 | `-8.84e-04` |

Financial quantities shown in manuscript figures are rounded to two decimal places. The optimization and stored numerical outputs retain full precision.

## Sensitivity analysis and the shadow price

For a fixed market environment, define

$$
p(c)
=
\inf_{\varphi}
\left\{
J(\varphi):
\mathrm{CVaR}_{\alpha}(-W_T)\le c
\right\}.

$$

Under the strong-duality assumptions of the paper,

$$
-\lambda^\star(c)\in\partial p(c).

$$

Thus $\lambda^\star$ is the shadow price of the CVaR budget. When $p$ is differentiable,

$$
\lambda^\star(c)=-p'(c).

$$

For a fixed market environment, tightening the CVaR budget therefore yields a nondecreasing optimal shadow price. This monotonicity is a theoretical consequence of the value-function sensitivity relation.

### Sensitivity to the CVaR budget

The numerical experiment varies

$$
c\in\{-0.86,-0.88,-0.90,-0.92,-0.94\}.

$$

| $c$ | $\widehat\lambda$, complete market | $\widehat\lambda$, $\beta^\perp=0.02$ |
| ---: | ---: | ---: |
| -0.86 | 0.0000 | 0.0000 |
| -0.88 | 0.0050 | 0.0082 |
| -0.90 | 0.0167 | 0.0249 |
| -0.92 | 0.0322 | 0.0552 |
| -0.94 | 0.0548 | 0.1252 |

The shadow price rises as the CVaR limit is tightened. Once the constraint binds, the incomplete-market multiplier is substantially larger, indicating that the same tail-risk budget is more costly in the presence of unhedgeable background risk.

### Sensitivity to nontraded background risk

Holding $c=-0.94$ fixed, the experiment varies

$$
\beta^\perp
\in
\{0,0.005,0.010,0.015,0.020,0.025\}.

$$

| $\beta^\perp$ | $\widehat\lambda$ | Average risky exposure |
| ---: | ---: | ---: |
| 0.000 | 0.0548 | 0.3061 |
| 0.005 | 0.0578 | 0.3041 |
| 0.010 | 0.0714 | 0.2976 |
| 0.015 | 0.0943 | 0.2840 |
| 0.020 | 0.1252 | 0.2633 |
| 0.025 | 0.2012 | 0.2274 |

Across this numerical grid, larger nontraded risk is associated with a higher shadow price and lower average risky exposure. This monotonicity in $\beta^\perp$ is a **numerical finding**, not a general theoretical claim.

At $\beta^\perp=0.025$, the maximum simulated exposure reaches the numerical upper action bound $2$. The endpoint is therefore useful as a sensitivity diagnostic, but the main qualitative pattern is already visible well before that boundary is reached.

## Path diagnostics

The direct-exposure experiments illustrate that the CVaR constraint does not simply impose a constant proportional reduction in risky exposure.

The path figures include representative favorable and unfavorable realizations together with the Merton benchmark $\bar\varphi=0.40$. In particular, the uncapped feedback policy can exhibit a lower-tail **rebound** near maturity: along selected unfavorable paths, risky exposure can rise above the static Merton benchmark even though average exposure remains substantially below it.

The selected lower-tail path is a diagnostic realization rather than a claim about typical behavior. The code identifies such paths explicitly from an independent simulation sample.

## Running the experiments

Run each experiment directly:

```bash
python complete_market.py
python incomplete_market.py
python quadratic_regularization.py
python square_root_impact.py
```

Each script generates its own numerical outputs and figures.

## Figure conventions

- Numeric labels, axes, and legends in manuscript figures use two decimal places.
- Rounding affects display only; optimization and simulation retain full precision.
- The dashed horizontal line in direct-exposure path figures is the unconstrained Merton benchmark $0.40$.
- Interim CVaR curves are unconditional diagnostics. The optimization imposes the CVaR constraint only at maturity.

## Reproducibility notes

- The final OOS sample is never used to select or modify the reported policy.
- No Merton exposure cap is imposed.
- No final-OOS feasibility buffer is used.
- Small positive OOS residuals are Monte Carlo evaluation outcomes and are reported without adjustment.
- The complete- and incomplete-market benchmarks use the same direct-exposure numerical protocol, making the $\beta^\perp=0$ benchmark directly comparable with the nontraded-risk sensitivity experiment.
