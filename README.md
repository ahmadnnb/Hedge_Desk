# The Hedge Desk

*One client. One option. Seven decisions.* Portfolio Theory practical project.

A desk has sold a three-month at-the-money call and can hedge it only with the stock, once a day, while volatility moves randomly and cannot be traded. The project calibrates a Heston model to 36 option quotes, then compares hedges on one scoreboard: the delta hedge, a closed-form quadratic hedge, a backward regression learned from simulated paths, and neural networks trained on a squared-error goal and on a tail-risk (CVaR) goal. It repeats the comparison with 50 bp trading costs, stresses the frozen hedges under changed markets, and ends with a recommendation for each mandate.

**Read first:** [`REPORT.md`](REPORT.md), the project report.

## Headline results

Market test paths (4,000, used once). Loss = −P&L; CVaR₉₅ is the average of the worst 5% of losses.

| Mandate | Recommendation | Test result |
|---|---|---|
| No costs, quadratic criterion | the closed-form quadratic hedge | RMSE 1.423 (Heston delta 1.761, unhedged 4.804) |
| No costs, tail criterion | CVaR network, a modest upgrade; formula as fallback | CVaR₉₅ 3.374 against 3.434 for the formula |
| 50 bp costs | the formula inside a no-trade band | CVaR₉₅ 4.962 against 5.612, with 39% lower costs |

The evidence is stated with paired bootstrap intervals, three training seeds and four stress scenarios; several of the differences are small or unresolved, and the report says which.

## Repository layout

```
hedge_desk_final.ipynb      completed notebook: code, checks, written answers, executed outputs
requirements.txt            course dependency list (NumPy, SciPy, Matplotlib, PyTorch, notebook)
REPORT.md                   project report: method, results, decision, limits
figures/                    the notebook's calibration, outcome, learning and holding-response plots
optional/                   an optional extra experiment (rebalancing-grid refinement), not part of the notebook
```

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r requirements.txt
# copy the course's market_data.npz into this folder first (it is not included in the repository)
jupyter notebook hedge_desk_final.ipynb
```

- **Course mode is the default** (three seeds, full training). The research cell took about six minutes as executed (the course notes say a laptop CPU is enough; a GPU is not needed). For a fast development run, set the environment variable `HEDGE_DESK_QUICK=1` before starting Jupyter.
- The notebook writes its results, the freeze file, the fitted policies and the plots to `hedge_desk_output/`, which is git-ignored.
- **Last-digit differences are expected on other library versions.** The reported numbers come from the executed notebook. A rerun of the calibration in a different environment gave κ = 1.3694 against 1.3693 in the executed notebook, and neural-network training depends on the PyTorch build.

## Optional experiment

[`optional/optional_refinement_check.py`](optional/optional_refinement_check.py) is one of the next experiments declared in [`REPORT.md`](REPORT.md) §12: refine the rebalancing grid on desk-model paths only (no test paths) and see how much of the quadratic hedge's residual risk is time-discretisation error. It is written to be pasted into a new last cell of the notebook; it is not part of the notebook. A sandbox run gave the following premium-free P&L variance (Monte Carlo standard error about 0.05): 1.983 with 63 dates, 1.896 with 126 and 1.870 with 252, so most of the residual risk survives a four-fold finer grid.
