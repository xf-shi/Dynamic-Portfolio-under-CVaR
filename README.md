# Dynamic Portfolio Optimization under CVaR Constraints

Four self-contained Jupyter notebooks explore how a 95% CVaR constraint changes portfolio feedback decisions as the model moves from a frictionless complete market to nonlinear trading costs.

The companion [project page](index.html) gives a visual, nontechnical overview of the models, methods, and saved results. When this folder is published with GitHub Pages, `index.html` becomes the site landing page.

## Main result

Across all four stored experiments, the active CVaR policy protects the lower tail of terminal wealth by reducing risky exposure. This improves the bad-outcome average to roughly `0.94`, with a modest reduction in expected terminal wealth.

Some notebook tables print **loss CVaR** as a negative number. This README reports the equivalent positive **lower-tail wealth CVaR**, so a larger value indicates a safer tail.

| Experiment | What changes? | Inactive tail wealth CVaR | Active tail wealth CVaR | Validation |
|---|---|---:|---:|---|
| [Complete market](Complete_Market_Plots.ipynb) | Wealth-only benchmark | 0.8690 | 0.9400 | 5,000 configured paths |
| [Incomplete market](Incomplete_Market_Plots.ipynb) | Adds independent unhedgeable risk | 0.8642 | 0.9400 | 20,000-path holdout |
| [Linear trading control](Linear_Control_Dynamics_Plots.ipynb) | Exposure becomes a controlled state | 0.8958 | 0.9409 | 40/40 validation seeds feasible |
| [Square-root price impact](Square_Root_Impact_Plots.ipynb) | Adds a nonlinear `q = 3/2` trading penalty | 0.8839 | 0.9409 | 50,000-path final OOS test |

## Choose a notebook

1. **Start here:** [Complete_Market_Plots.ipynb](Complete_Market_Plots.ipynb) isolates the CVaR effect in a one-state HJB.
2. [Incomplete_Market_Plots.ipynb](Incomplete_Market_Plots.ipynb) adds an independent endowment shock and holdout calibration.
3. [Linear_Control_Dynamics_Plots.ipynb](Linear_Control_Dynamics_Plots.ipynb) models gradual changes in dollar exposure.
4. [Square_Root_Impact_Plots.ipynb](Square_Root_Impact_Plots.ipynb) adds nonlinear price impact and final out-of-sample validation.

Each notebook contains its solver, calibration, simulation, and plotting code. No proprietary optimizer is required.

## Run locally

Requires Python 3.10 or newer.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Then open a notebook and choose **Run All**. NumPy and Matplotlib are the only computational libraries; JupyterLab supplies the notebook interface.

The saved outputs are already included. Full active-policy searches can be compute-intensive, particularly for the incomplete-market and two-state HJB cases.

## Numerical workflow

- A finite-difference HJB solver computes state-dependent feedback policies.
- Golden-section search selects the Rockafellar variable `eta`.
- Outer bisection calibrates the CVaR multiplier `lambda`.
- Monte Carlo simulation evaluates wealth, risky exposure, controls, and tail risk.

## Publish with GitHub Pages

1. Create a GitHub repository and upload everything in this folder to its root.
2. In the repository, open **Settings → Pages**.
3. Under **Build and deployment**, select **Deploy from a branch**.
4. Choose the `main` branch and `/ (root)`, then save.

GitHub Pages will serve `index.html` automatically. The `.nojekyll` file keeps the folder as a plain static site.

## Reproducibility note

The values above come from the outputs saved in the notebooks. Fresh Monte Carlo samples have ordinary sampling variation, and results can change with grids, seeds, tolerances, or model parameters.
