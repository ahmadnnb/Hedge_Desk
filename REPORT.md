# The Hedge Desk — project report

Portfolio Theory practical project: *one client, one option, seven decisions.*

This report summarises what the notebook does and what it finds. The code, the checks and the written answers are in [`hedge_desk_final.ipynb`](hedge_desk_final.ipynb). Every number below comes from the notebook as executed in course mode (three training seeds: 11, 22, 33) and from the project's desk memo, which is not part of this repository; the price range and vega ratio in §3 and §14 were recomputed from the quote sheet.

## 1. The problem

A desk has sold a three-month at-the-money call (spot 100, strike 100, zero interest rates) and received the quoted premium **c₀ = 3.613181**, the Black–Scholes price at the quoted implied volatility of 18.12%. It can trade only the stock, once a day, on 63 dates. The stock's volatility is random and cannot be traded, so the market is incomplete and no hedge removes all the risk.

Every hedge in the project is a rule that outputs the number of shares held between two dates. The *ledger* (task A1) turns holdings and a price path into a P&L:

```
P&L = c0 + sum_k  phi_{k+1} (S_{k+1} - S_k)  -  (S_N - K)^+  -  trading costs
```

A holding chosen at date t_k may use only information available at t_k. Hedges are scored on the market's 4,000 test paths (64 dates each) with five numbers: the mean P&L, the RMSE (root of the mean squared P&L, the quadratic criterion), the standard deviation, the CVaR₉₅ of the loss (the average of the worst 5% of losses) and the mean trading cost.

## 2. Roadmap

| Brief | Question | What the notebook does |
|---|---|---|
| 0 | What does the market imply? | Calibrates a Heston model to 36 option quotes; this is the "desk model" and the simulator. |
| 1 | How do I score a hedge? | Builds the ledger; implements the delta hedge and the quadratic (variance-optimal) hedge. |
| 2 | Can paths teach a hedge? | Backward regression on simulated paths; an exact check on a finite tree. |
| 3 | How do I compare credibly? | Paired bootstrap intervals, a look-ahead check, and a freeze file written before the test. |
| 4 | Can a network do better? | Trains networks on a squared-error goal and on a tail-risk (CVaR) goal. |
| 5 | What if trading costs money? | Adds 50 bp costs; compares a no-trade band with two cost-trained networks. |
| 6 | What breaks, and what do I recommend? | Stresses the frozen hedges, locates where they lose, and decides. |

## 3. Brief 0 — the desk model

**Data.** 36 implied volatilities: four maturities (1, 3, 6, 12 months) × nine strikes (80 to 120), spot 100, today's variance v₀ = 0.04.

**Calibration.** Four Heston parameters (κ, v̄, ξ, ρ) are fitted by nonlinear least squares (`scipy.optimize.least_squares`, with bounds, from four starting points, keeping the best) on residuals measured in **implied volatility**, with v₀ fixed.

| κ | v̄ | ξ | ρ | RMSE over the 36 quotes |
|---|---|---|---|---|
| 1.3693 | 0.0449 | 0.6886 | −0.5911 | 0.32 vol points (half-spread 0.3; Lecture 5's parameters: 0.97) |

![Calibration: quotes against the fitted smile](figures/calibration.png)

**Why volatility, not price.** Across the sheet, option prices run from 0.00015 to 21.9 (about five orders of magnitude) and vegas differ by a factor of about 2,800. A price fit would be ruled by the long-dated, high-vega quotes, while the quote noise is naturally stated in vol points. The problem is non-convex with a flat κ–v̄ valley, hence four starting points.

**What the quotes can and cannot identify.**
- One smile cannot separate κ and v̄. A three-month-only fit goes to κ = 0.10 (the bound) and v̄ = 0.075 with RMSE 0.14, yet implies nearly the same average variance (0.0404 against 0.0408). A term structure helps because exp(−κT) is far from 1 at 12 months.
- A price *history* identifies ξ and ρ (a change of measure alters drifts, not volatilities or correlations) but not κ and v̄, which absorb the variance risk premium; those need option prices.

**Why the premium is a quote.** It is what the client pays. A model price would bury the calibration error in every P&L: one half-spread is worth about 0.06 at the money (vega about 0.20 per vol point).

## 4. Brief 1 — the ledger and two formula hedges

The ledger is checked by hand on a three-date path (100, 110, 105) with holdings (0.5, 0.25), premium 5, strike 100 and 1% costs: gains 3.75, payoff 5 and costs 1.0375 give a P&L of 5 + 3.75 − 5 − 1.0375 = 2.7125, which the notebook asserts (together with the gradient with respect to the holdings).

- **Heston delta** C_S: the sensitivity of the model price to the stock, with variance held fixed.
- **Quadratic hedge** h = C_S + ρ ξ C_v / S, where C_v is the derivative of the price with respect to variance. With dW^v = ρ dW^S + √(1−ρ²) dW^⊥, the price process has a part driven by the stock's own shock and a part driven by the orthogonal shock. Trading the stock can only neutralise the first, which gives the formula. What remains is vega risk orthogonal to the stock, a share 1−ρ² = 0.651 of the vega variance at ρ = −0.5911. For a call C_v > 0 and ρ < 0, so the correction is negative: when the stock falls, variance tends to rise and cushions the call.
- The daily formula is a benchmark for the 63-date problem, not an optimum: incomplete-market risk (independent of trading frequency) and time-discretisation error (gamma over a finite step) both remain.

## 5. Brief 2 — learning the hedge backwards

- **Tree certificate (B1).** On a finite tree every conditional expectation can be computed. The node slopes equal the global weighted projection to 10⁻¹⁰, with an orthogonal residual (cash 6.958658, MSE 3.540497). This is stronger evidence than a small training loss.
- **Backward regression on Heston paths (B2).** Starting from the payoff, the hedge at each date is the regression slope of the next date's residual on the stock move, with a level term as nuisance. The notebook compares a compact and a rich basis and keeps the one with the better validation score:

| | Compact basis | Rich basis | Formula |
|---|---|---|---|
| Validation RMSE | 1.5068 | 1.4866 | 1.4680 |

The rich basis is selected. On the test paths the regression (RMSE 1.440) does not beat the formula (1.423); its MSE is higher by +0.049 (95% interval [+0.024, +0.072]).

## 6. Brief 3 — what counts as evidence

- **Roles of the data.** Quotes are used only to calibrate; 8,000 training and 2,000 validation desk-model paths per seed are used to fit and to choose; the market's test paths are used once, with frozen choices.
- **Paired intervals.** Two policies are compared on the same path indices (600 bootstrap resamples), so the shared stock-path noise cancels. The intervals concern fresh-path sampling error for the two fitted policies; they exclude retraining, calibration and model error. They are pointwise and exploratory: with six comparisons, the chance that at least one interval misses is about 26%.
- **Look-ahead check (C2).** A frozen regression passes a prefix perturbation test and a terminal-price rule fails it. Predictability (holdings at t_k use information up to t_k) is different from statistical separation (nothing fitted on test paths).
- **Freeze file.** `freeze.json` (SHA-256 prefix `ad334b826c263805`) records the configuration, desk model, calibration, premium, selected regression basis, band, library versions and fit records; the fitted network weights and regression coefficients are saved next to it. It is written, and its hash checked, before the test cell.

## 7. Brief 4 — training networks

A network with two hidden layers of 32 tanh units, shared across dates, maps the state (time, spot, variance) to a holding. A second version also takes the previous holding. Training simulates desk-model paths, pushes the holdings through the ledger, evaluates the objective, back-propagates through all 63 decisions and takes an Adam step.

![Validation curves for the quadratic and holding-input networks](figures/learning.png)

- **Squared-error network.** Trained on the mean squared P&L (400 steps, batch 1,000, learning rate 0.002).
- **Tail network.** Trained on the Rockafellar–Uryasev form of CVaR, R(z) = z + E[(L−z)⁺]/α with α = 0.05, and a threshold z fitted alongside the weights (500 steps). The notebook checks the boundary case on a batch with 90 zero losses, seven losses of 10 and three of 100: the tail takes only 2/7 of the atom at 10, giving 64, whereas averaging only the losses above the quantile would give 100. A stale threshold (below the smallest loss) turns the gradient into 20 times the mean-loss gradient. CVaR is invariant to cash, so the premium is held fixed at the quote.
- CVaR training lowers CVaR while raising the standard deviation: CVaR ignores the body and the upside of the distribution.

## 8. Brief 5 — trading costs

Costs of 50 bp (`rate = 0.005`) are charged on every trade, which makes the current position part of the state. The **band** keeps the holding inside [h − a, h + b] around the formula hedge h and otherwise trades to the nearer edge; (a, b) = (0.04, 0.08) is chosen on validation from a grid. Two cost-trained networks are compared with it: one using the market state only, and one that also sees the previous holding.

![Learned holding response at half the time to expiry](figures/holding_response.png)

Each line in the figure (memory network, seed 11, variance at its long-run level) has slope about 0.24: roughly three quarters of the gap to a stock-dependent target is closed per date. No segment lies on the 45° line, so what is visible is *partial adjustment, not a learned no-trade band*. One date, one variance level and one seed cannot show more, and the memory network's gain over the market-only network shows that the holding input helps, not that a band was learned.

## 9. Results on the market's test paths (4,000 paths, used once)

| Policy | Mean | RMSE | SD | CVaR₉₅ | Mean cost | CVaR over 3 seeds |
|---|---:|---:|---:|---:|---:|---|
| Unhedged | −0.101 | 4.804 | 4.804 | 13.589 | 0.000 | – |
| Heston delta (C_S) | −0.055 | 1.761 | 1.761 | 4.051 | 0.000 | – |
| Quadratic formula | −0.059 | 1.423 | 1.422 | 3.434 | 0.000 | – |
| Backward regression (rich) | −0.057 | 1.440 | 1.439 | 3.469 | 0.000 | – |
| MSE network (seed 11) | −0.063 | 1.458 | 1.457 | 3.576 | 0.000 | 3.555–3.587 |
| CVaR network (seed 11) | −0.071 | 1.519 | 1.517 | 3.374 | 0.000 | 3.320–3.374 |
| Formula, 50 bp | −1.717 | 2.385 | 1.656 | 5.612 | 1.657 | – |
| Band [h−0.04, h+0.08], 50 bp | −1.071 | 1.929 | 1.604 | 4.962 | 1.005 | – |
| Network, market state (seed 11), 50 bp | −1.255 | 2.161 | 1.760 | 4.969 | 1.178 | 4.969–4.985 |
| Network + previous holding (seed 11), 50 bp | −1.143 | 2.088 | 1.747 | 4.869 | 1.066 | 4.869–4.874 |

![Empirical CDF of terminal P&L and test CVaR by training seed](figures/outcomes.png)

Six paired comparisons (first minus second; negative favours the first; interval for seed 11; "other seeds" are the point differences for seeds 11, 22, 33 on the same paths):

| Comparison | Metric | Estimate | 95% interval | Other seeds | Reading |
|---|---|---:|---|---|---|
| Regression − formula | MSE | +0.049 | [+0.024, +0.072] | – | formula better |
| MSE network − regression | MSE | +0.052 | [+0.014, +0.096] | +0.05, +0.05, +0.04 | regression better (small) |
| CVaR network − formula | CVaR | −0.059 | [−0.107, −0.009] | −0.06, −0.11, −0.07 | lower; borderline |
| Band − formula (50 bp) | CVaR | −0.650 | [−0.786, −0.514] | – | band much lower |
| Memory network − band (50 bp) | CVaR | −0.093 | [−0.195, +0.008] | −0.09, −0.09, −0.09 | **unresolved** |
| Memory network − market-only network | CVaR | −0.100 | [−0.132, −0.069] | −0.10, −0.10, −0.11 | memory lower |

Reading the table:
- **No costs.** The formula has the lowest RMSE. The correction ρξC_v/S removes 35% of the delta hedge's MSE. A larger regression basis lowers each step's in-sample error but does not beat the formula on the test paths.
- **Tail.** The CVaR network's gain (−0.06, −0.11, −0.07 over seeds) is of the order of its seed spread (3.320–3.374).
- **Costs.** The formula pays 1.657 per path (46% of the premium); the band pays 1.005 and has the lowest RMSE and SD of the cost-aware policies. The memory network has the lowest CVaR (4.869) but pays more (1.066). A policy that pays less need not hedge better. An interval containing zero is not proof of equivalence.

## 10. Brief 6 — stress, and where the hedges lose

The frozen hedges are scored on 4,000 freshly simulated paths in each of four scenarios, with nothing refitted and the premium kept. CVaR₉₅ of the loss; change from the desk-model row in brackets:

| Scenario | Unhedged | Formula | CVaR net | Band, 50 bp | Memory net | CVaR net − formula | Memory − band |
|---|---:|---|---|---|---|---|---|
| Desk model | 14.388 | 3.437 | 3.338 | 4.869 | 4.831 | −0.099 [−0.151, −0.050] | −0.039 [−0.144, +0.075] |
| Vol-of-vol +30% | 14.085 | 4.088 (+0.65) | 4.016 (+0.68) | 5.510 (+0.64) | 5.517 (+0.69) | −0.071 [−0.124, −0.021] | +0.007 [−0.111, +0.138] |
| Correlation weaker by 40% | 17.628 | 4.007 (+0.57) | 3.982 (+0.64) | 5.445 (+0.58) | 5.553 (+0.72) | −0.025 [−0.087, +0.036] | +0.108 [−0.012, +0.240] |
| Long-run variance +40% | 15.077 | 3.569 (+0.13) | 3.459 (+0.12) | 5.016 (+0.15) | 4.965 (+0.13) | −0.110 [−0.163, −0.062] | −0.051 [−0.156, +0.062] |

- **What survives every scenario.** Every hedge beats no hedge (14.1–17.6 against at most 5.6). The CVaR network stays below the formula, but with weaker correlation its interval contains zero. Memory against band changes sign twice and all four intervals contain zero.
- **What degrades.** Higher vol-of-vol and weaker correlation add 0.57–0.72 to CVaR (17–20% without costs, 12–15% with costs); higher long-run variance adds only 0.12–0.15.
- **Simulator inheritance.** A network has absorbed the whole joint law of its training paths, so it inherits distribution error as well as parameter error. Its edge over the formula is larger on fresh desk-model paths (−0.099) than on the market's paths (−0.059), as would happen if part of the edge is specific to the desk's simulator (the intervals overlap). A scenario is a what-if, not a probability.

**Where the worst 5% come from** (market paths; HV = above-median realised volatility, "crossed" = crossed the strike in the last month):

| Policy | Share of worst 5%: HV crossed | HV not crossed | LV (both) | Mean P&L: HV crossed | HV not | LV crossed | LV not |
|---|---:|---:|---:|---:|---:|---:|---:|
| Formula | 45% | 56% | 0% | −1.26 | −0.83 | +1.15 | +0.78 |
| CVaR network | 35% | 65% | 0% | −0.80 | −1.01 | +1.73 | +0.59 |
| Band, 50 bp | 47% | 54% | 0% | −2.42 | −1.81 | +0.03 | −0.17 |
| Memory network, 50 bp | 32% | 64% | 4% | −1.87 | −2.15 | +0.87 | −0.47 |

The groups hold 16%, 34%, 9% and 41% of the paths (HV crossed, HV not, LV crossed, LV not). Late crossers hold 45% of the formula's tail with 16% of the paths. This fits a **gamma** explanation: a short call has the largest gamma at the money and close to expiry, so a stale daily hedge is hurt most by a late move toward the strike in a high-volatility path. It does not prove it: the groups are ex-post features and any higher-SD policy would also reshuffle the tail.

## 11. Decision

| Mandate | Recommendation |
|---|---|
| No costs, quadratic criterion | **Deploy the closed-form quadratic hedge.** Test RMSE 1.423; nothing learned beats it. |
| No costs, tail criterion | **The CVaR network is a modest upgrade** (CVaR 3.374 against 3.434, about 2%, of the order of the seed spread, unresolved when correlation weakens). Keep the formula as the fallback. |
| 50 bp costs | **The formula inside a no-trade band** [h − 0.04, h + 0.08]: CVaR 4.962 against 5.612 and 39% lower costs. The memory network is a further 0.093 lower, but its interval contains zero and its edge flips when correlation weakens. |

## 12. Limits, errors and the pricing question

**Five sources of error.**
1. *Approximation:* the finite regression basis and the restricted 2×32 network family; the continuous formula in a 63-date market.
2. *Optimisation:* the MSE networks stop at RMSE 1.454–1.458 after 400 noisy steps; CVaR training uses a biased batch gradient (seeds end at 3.320–3.374).
3. *Sampling:* CVaR rests on 200 tail paths out of 4,000; the memory-against-band interval cannot decide.
4. *Calibration:* a 0.32 vol-point misfit; κ and v̄ are poorly identified; the premium carries the quote noise.
5. *Model:* the market is not the calibrated Heston model and volatility cannot be traded.

**Certificate versus benchmarks.** Only the finite tree is exact. The Heston policies have estimated coefficients, a continuous-time benchmark, non-convex training and sampling evidence. There is no replication guarantee and no global optimality certificate.

**Pricing.** The scores are risk levels at the quoted premium for restricted, frozen policies, not a price. With CVaR the no-claim baseline is 0, so each score only bounds the risk-indifference price p* ≤ c₀ + score (7.05 for the formula, 6.93 for the best network), while p* ≥ E^Q[H] ≈ 3.7. The bracket is wide.

**Limits.** One calibrated Heston model; 4,000 market paths used once; three seeds, which do not estimate training variability; pointwise intervals; scenarios without probabilities; networks without accumulated gains in the state; ex-post diagnosis groups; everything under Q. *Declared next experiments (not run in the notebook):* compare network and formula holdings in late-life near-strike states; refine the rebalancing grid on desk-model paths to separate discretisation error from incompleteness (a snippet is in [`optional/`](optional/)); add accumulated net gains to the CVaR state.

## 13. Research record

Quotes only for calibration; 8,000 training and 2,000 validation desk-model paths per seed; test paths opened once; reference seed 11; band and basis chosen on validation; `freeze.json` (SHA-256 prefix `ad334b826c263805`) written and hash-checked before the test cell.

*Timing.* The desk memo and the written predictions were produced after the course-mode run, so the expected ranking (formula ≲ regression < delta ≪ unhedged; band < formula under costs) rests on theory and validation scores, not on pre-registration. The notebook records no change of configuration after the test results appeared (**authors to confirm**).

## 14. Corrections since the notebook was written

- The notebook's Brief 0 answer (M1) says that prices "differ by two orders of magnitude" (the same sentence is in the desk memo, which is not part of this repository). Across the whole 36-quote sheet they span 0.00015 to 21.9, about five orders of magnitude (the vegas differ by a factor of about 2,800). The conclusion is unchanged: fit in implied-volatility space.
