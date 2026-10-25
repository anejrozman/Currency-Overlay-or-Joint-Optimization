# Currency Overlay or Joint Optimization?

Code, results and LaTeX sources of the MSc thesis *Currency Overlay or Joint Optimization? An Out-of-Sample Comparison across Risk Measures* (MSc Quantitative Finance, ETH Zürich and University of Zürich).

## Overview

An international investor can manage currency risk in two ways: choose the asset weights first and add a currency overlay on top (sequential optimization), or choose asset and currency weights together in one program (joint optimization). The thesis compares the two out of sample under five risk measures: variance, semi-variance, mean absolute deviation, Expected Shortfall and second-order Expected Shortfall.

Each measure is implemented in an unregularized, a regularized, and a mean-risk version, alongside an equally weighted portfolio and the fully hedged and unhedged asset-only benchmarks. The portfolios are built from the 10, 50 or 100 largest stocks of 14 economies and are backtested walk-forward with quarterly rebalancing, a ten-year estimation window and transaction costs. A second analysis measures how much of the joint program's in-sample advantage survives out of sample.

The framework follows Ulrych & Burkhardt (2023), [*Sparse and stable international portfolio optimization and currency risk management*](https://www.sciencedirect.com/science/article/pii/S026156062300150X), and Lucescu & Ulrych (2026), *Regularized Multi-Currency Expected Shortfall Portfolios*, and extends it from mean-variance and Expected Shortfall to the other risk measures.

## Repository structure

- `code/`: the Python code, installed as the `portfolio_backtest` package.
  - `src/portfolio_backtest/`: 
    - `backtest/`: the walk-forward `Backtester` and the `DataHandler` that loads its inputs.
    - `optimizers/`: one file per optimization program.
    - `strategy_registry.py`: the strategies that can be run, each linking an optimizer to its return series and backtest configuration.
    - `backtest_config.py`: the estimation window, rebalancing frequency and hyperparameter grid of every strategy.
    - `data_pipeline/`: turns the raw Datastream data into the processed return files.
    - `utils/`: return formulas, estimators, and file helpers.
  - `scripts/`: `run_strategy.py` runs one backtest. `euler/` holds the scripts that submit backtests and analyses to the ETH Euler Cluster.
  - `analysis/`: the scripts that produce the figures and tables of the thesis from the backtest results, plus exploratory notebooks.
- `data/`: see [Data](#2-data).
- `results/`:
  - `runs/`: one directory per backtest, `runs/<optimizer>/run_<N>/`, and the `top_runs*.csv` files that record which run each analysis uses.
  - `figures/` the outputs of the analysis scripts.
- `logs/`: the Euler Cluster job logs of the backtests.
- `thesis/`: the LaTeX sources and the compiled `thesis.pdf`.

## Reproducing the results

### 1. Installation

The code needs Python 3.12 and a [Gurobi](https://www.gurobi.com) license (free for academic use). Without Gurobi, the optimizers fall back to the open-source Clarabel solver.

```bash
git clone https://github.com/anejrozman/Currency-Overlay-or-Joint-Optimization.git
cd Currency-Overlay-or-Joint-Optimization/code
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -e .
cd ..
```

The commands below run from the repository root. `requirements.txt` pins the environment the thesis results were produced with. Add `pip install jupyter` to open the notebooks.

### 2. Data

The `data/` folder is empty in this repository, because the data are licensed and cannot be redistributed. They come from two sources:

- **Refinitiv Datastream**: daily stock prices and local-currency returns, the constituents of the investable universes (the 10, 50 and 100 largest stocks of each economy) with the currency of each stock, and daily spot and 1-, 3- and 6-month forward exchange rates against the US dollar.
- **Kenneth R. French's [data library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)**: the daily one-month US Treasury bill rate, used as the risk-free rate.

Save the exports in `data/raw/`. The file names the code expects are set in `code/src/portfolio_backtest/config.py` and `code/src/portfolio_backtest/data_pipeline/format_raw_data.py`. Then build the interim and processed files:

```bash
mkdir -p data/interim data/processed
python -m portfolio_backtest.data_pipeline.format_raw_data
python -m portfolio_backtest.data_pipeline.preprocess_data
```

### 3. Backtests

Each backtest runs one strategy on one universe:

```bash
python code/scripts/run_strategy.py --strategy es_joint --universe 10
```

`--help` lists the strategies, which are the keys of `strategy_registry.py`. A run writes its configuration, out-of-sample returns, selected hyperparameters, weights and performance metrics (`metrics.json`) to a new directory `results/runs/<optimizer>/run_<N>/`. 

The analyses read the run numbers from the `top_runs*.csv` files in `results/runs/`:

- `top_runs.csv`: the unregularized and regularized strategies and the benchmarks.
- `top_runs_w_mean.csv`: the mean-risk strategies (`*_w_mean`).
- `top_demeaned_runs.csv`: the de-meaned ES and SOES programs. Set `DEMEAN = True` in `backtest_config.py`, rerun the ES and SOES overlay and joint strategies (unregularized and regularized).

On the Euler cluster, `code/scripts/euler/run_backtest.sh` submits one backtest and takes the same options as `run_strategy.py`; `run_all_backtests.sh` submits a batch of them and `run_levels.sh --es` (or `--soes`) the whole level sweep. Every job writes its log to `logs/`.

### 4. Figures and tables

The scripts are in `code/analysis/` and run as, for example, `python code/analysis/compute_excess_returns.py`. Each writes to its own folder in `results/`, which the thesis includes directly.

**Data (Chapter 4)**
- `return_statistics.py` prints the summary statistics of the returns by economy.
- `asset_currency_dependance.py` draws the asset-currency correlation heatmaps (`results/figures/`).

**Backtest performance (Chapter 5)**
- The volatility, drawdown, Sharpe and Sortino ratios and turnover in the performance tables are the `metrics.json` of the runs in `top_runs.csv` and `top_runs_w_mean.csv`.
- `compute_excess_returns.py` computes the annualized excess returns (`results/excess_returns/`).
- `sharpe_change_tables.py` writes the tables of the change in Sharpe ratio from regularization and from the mean term (`results/sharpe_change/`).
- `generate_cumulative_and_difference_plots.py --type unregularized`, `--type regularized` and `--type w_mean` draw the cumulative return figures (`results/figures/cumulative_and_difference/`).

**Joint versus sequential optimization (Chapter 6)**
1. `in_sample_analysis.py <measure>`, for each of `mv`, `es`, `mad`, `smv` and `soes`, scores the unregularized programs at every rebalancing date (`results/in_sample_analysis/`).
2. `in_sample_vs_out_of_sample.py` computes the in-sample and out-of-sample gaps and draws `gap_vs_currency_share.pdf` and `delta_epsilon.pdf` (`results/in_sample_vs_out_of_sample/`).
3. `in_sample_vs_out_of_sample_at_diff_levels.py --es` and `--soes` repeat the analysis across confidence levels (`gap_vs_level.pdf`).
4. `in_sample_vs_out_of_sample_demeaned.py` repeats it for the de-meaned programs (`gap_vs_universe.pdf`).

The gap analyses re-solve every program at every date. On Euler, the `run_gap_analysis*.sh` scripts submit them as one job per cell, after which `--plot-only` builds the tables and figures from the saved results.

**Appendices E and G**
- `generate_runtime_plots.py` draws the runtime per rebalancing date from the job logs in `logs/` (Appendix E).
- `hyperparameter_boundary_analysis.py mv` followed by `generate_boundary_plots.py mv` produces the penalty boundary figures.
- `sparsity_analysis.py --optimizer-family normal` draws the share of assets and currencies held.
- `generate_drawdown_plot.py --optimizer-family normal` and `--optimizer-family with-mean-term` draw the drawdowns.
- `generate_turnover_plot.py` draws the asset and currency turnover.
- `additional_performance_metrics.py` writes the tables of realized risk measures and ES ratios (`results/additional_performance_metrics/`).

The notebooks in `code/analysis/` are exploratory.

## Author

Anej Rozman, MSc Quantitative Finance, ETH Zürich and University of Zürich. Released under the MIT License.
