# Theoretical Foundations for a Horse-Racing Betting Portfolio Model: A Specification for Royal Ascot, Saturday 20 June 2026

## TL;DR
- A complete, implementable system has three layers: (1) a **conditional/multinomial logit fundamental model** (Bolton & Chapman 1986) that converts handicapping features into normalized win probabilities; (2) a **Benter-style second-stage logit** that blends those probabilities with market-implied probabilities (after de-vigging with Shin's method); and (3) a **Kelly log-growth bankroll allocator** that, via the Smoczynski–Tomkins closed form, distributes stakes across multiple horses in each race and across races, used at a **fractional (¼–½) Kelly** setting to control estimation error and drawdown.
- The two single most important theoretical results to implement correctly are the **Smoczynski–Tomkins explicit Kelly solution** for multiple mutually exclusive outcomes (the reserve-rate active-set rule with f_k* = p_k − b·β_k) and **Benter's combination step** (a second logit with log-fundamental-probability and log-public-probability as the only two regressors), because Benter's own data shows the fundamental model is biased toward the public estimate and unusable for staking until corrected.
- Royal Ascot 2026's final day (Saturday 20 June) is a seven-race card headed by the Group 1 QEII Jubilee Stakes (3:40pm, 6f, 4yo+), whose £1,000,000 prize money dwarfs the £175,000 Wokingham and £120,000 Golden Gates; the most model-tractable races for a portfolio approach are the large-field handicaps (Wokingham, Golden Gates) and Group races, with UK data sourceable from Betfair historical/Betfair SP, Timeform (via the Betfair API), the Racing API, and community Kaggle datasets.

## Key Findings

**1. Kelly is the correct objective, not mean-variance.** Kelly (1956) and Breiman (1961) establish that maximizing expected log-wealth (equivalently, acting each period to maximize E[log W]) asymptotically maximizes the growth rate and minimizes expected time to a wealth goal, dominating any "essentially different" strategy. For a single binary bet the optimum is f* = (bp − q)/b = edge/odds. Mean-variance (Markowitz 1952) cannot answer the "how much" question without an exogenous risk-appetite parameter; Kelly answers it directly.

**2. Betting several horses in one race is a solved convex problem with a closed form.** The Smoczynski–Tomkins (2010) solution gives an explicit active-set algorithm: rank horses by expected revenue rate, admit them while that rate exceeds the "reserve rate," [ResearchGate](https://researchgate.net/publication/268615950_An_explicit_solution_to_the_problem_of_optimizing_the_allocations_of_a_bettor's_wealth_when_wagering_on_horse_races) and stake f_k* = p_k − b·β_k. This is the exact engine for multi-horse, same-race allocation.

**3. Fractional Kelly is mandatory in practice.** Because win probabilities are estimated with error, full Kelly badly overbets; half-Kelly produces ≈75% of full Kelly's growth rate while cutting volatility in half (MacLean, Ziemba & Blazenko, "Growth versus security in dynamic investment analysis," *Management Science* 38(11):1562–1585, 1992), and practitioners (Benter included) recommend betting a fraction (≈¼–½) of the full Kelly amount.

**4. Win probabilities come from a conditional logit; place/show from Harville plus bias corrections.** The Bolton & Chapman (1986) multinomial/conditional logit is the seminal fundamental model. [INFORMS](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.32.8.1040) The Harville (1973) formula extends win probabilities to ordered finishes but is biased; the Henery (1981)/Stern (1990) models and the Lo–Bacon-Shone discounted-Harville approximation correct it.

**5. The market must be respected.** Public odds are a highly efficient probability estimate. Benter's central practical lesson: combine the fundamental model with the public's implied probabilities [Medium](https://medium.com/parimutuel-racetrack-analysis/bill-benter-1994-computer-based-horse-race-handicapping-and-wagering-systems-a-report-db747c250e77) via a second-stage logit, or you will lose.

**6. De-vigging matters.** Raw inverse odds over-sum to the overround; Shin's (1993) insider-trading model and the power method outperform naive normalization, especially for the favourite–longshot bias. [Springer](https://link.springer.com/article/10.1007/s10479-022-04722-3)

## Details

### Part 1 — Kelly Criterion for Horse Racing

**1.1 Origins.** John L. Kelly Jr., "A New Interpretation of Information Rate," *Bell System Technical Journal* 35(4), pp. 917–926 (1956), [arxiv](https://arxiv.org/pdf/2003.02743) framed a gambler with a private wire and showed that under fair odds the maximum bankroll growth rate equals the channel's information rate. [ENKR's Blog](https://blog.enkr1.com/kelly-criterion/) The bridge to gambling practice was made by Edward O. Thorp (Beat the Dealer, 1962; "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market," 2006), who used it to size blackjack and market bets. Earlier, Isaacs (1953) addressed optimal bet sizing in a related context. Breiman (1961) proved the asymptotic dominance and shortest-time-to-goal properties.

**1.2 The single-bet formula.** For a bet winning with probability p at net decimal odds b (profit b per unit staked), losing probability q = 1 − p:

  f* = (bp − q) / b = p − q/b.

For even money (b = 1), f* = 2p − 1 = p − q. The expected log growth per bet is g(f) = p·ln(1 + bf) + q·ln(1 − f).

**1.3 Multiple mutually exclusive outcomes — the core math.** In a single race with n horses, only one wins, so Win bets define n mutually exclusive outcomes. [De Gruyter Brill](https://www.degruyterbrill.com/document/doi/10.1515/jqas-2020-0122/html?lang=en) Let:
- p_k = (model) probability horse k wins, Σ p_k = 1;
- β_k = fraction of the betting pool on horse k (parimutuel), or the market-implied probability; Σ β_k = 1;
- D = 1 − tt, the dividend rate, where tt = track take (or, for fixed-odds, the normalization that removes the overround);
- O_k = D/β_k, the post-take decimal return per unit if k wins;
- f_k = fraction of wealth staked on horse k, f_k ≥ 0;
- b = reserve rate (fraction of wealth left unbet).

The optimization (maximize expected log-growth, no short-selling, no borrowing):

  maximize  L(f) = Σ_{k=1}^{n} p_k · log( (1 − Σ_i f_i) + f_k·(D/β_k) )
  subject to  Σ_k f_k ≤ 1,  f_k ≥ 0.

This is a concave program. **Smoczynski & Tomkins (2010)** ("An explicit solution to the problem of optimizing the allocations of a bettor's wealth when wagering on horse races," *The Mathematical Scientist* 35( [ResearchGate](https://www.researchgate.net/profile/Peter-Smoczynski/publication/268615950_An_explicit_solution_to_the_problem_of_optimizing_the_allocations_of_a_bettor's_wealth_when_wagering_on_horse_races/links/5c526fba299bf12be3efebba/An-explicit-solution-to-the-problem-of-optimizing-the-allocations-of-a-bettors-wealth-when-wagering-on-horse-races.pdf) 1):10–17) solve it in closed form via KKT conditions:

1. **Expected revenue rate** for each horse: e_k = p_k · D / β_k = p_k · O_k (model probability ÷ market-implied probability, after take).
2. Sort horses by e_k in non-increasing order (e_1 = best bet).
3. **Reserve rate** for a candidate bet-set S: b = ( Σ_{k∉S} p_k ) / ( D − Σ_{k∈S} β_k ). With S = ∅, b = 1/D.
4. **Inclusion rule:** horse k ∈ S ⇔ e_k > b. [Infogalactic](https://infogalactic.com/info/Kelly_criterion) Greedily add horses in sorted order, recomputing b after each addition, until e_k ≤ b. If S ends empty, do not bet.
5. **Optimal fraction:** f_k* = p_k − β_k·b for k ∈ S, else 0. [Wikipedia](https://en.wikipedia.org/wiki/Kelly_criterion) Equivalently f_k* = β_k(e_k − b)/D.
6. **Maximal growth rate:** L(f_opt) = Σ_{k∈S} p_k·ln(p_k·D/β_k) + (Σ_{k∉S} p_k)·ln(b). Optimal terminal wealth on each horse is W_k* = max{b, e_k}. [arxiv](https://arxiv.org/html/2603.13581)

Note the structural result (Whelan 2025, *Bulletin of Economic Research*): for genuinely mutually exclusive outcomes the true utility-maximizing solution [Wiley Online Library](https://onlinelibrary.wiley.com/doi/full/10.1111/boer.12474) can be more aggressive than naively applying two-outcome Kelly per horse, and can even assign positive stakes to negative-EV outcomes as hedges. [Karl Whelan](https://www.karlwhelan.com/sports-betting-kelly-criterion-multiple-outcomes/) The Smoczynski–Tomkins solution above is the correct full-Kelly allocation for win bets; the more general "hedging" behaviour appears under general concave utilities.

**1.4 Fractional Kelly, drawdown, and estimation error.** Full Kelly produces large drawdowns — MacLean, Ziemba & Blazenko (1992) report full Kelly carries a 33% probability of halving the bankroll before doubling it, and only a 91.6% chance of not losing more than half the wealth versus 99% for half-Kelly. It is also extremely sensitive to parameter error: Chopra & Ziemba (*Journal of Portfolio Management* 19(2):6–11, 1993) find "errors in the means are about 20 times errors in covariances" (a 20:2:1 mean:variance:covariance ratio), and for low-risk-aversion log (Kelly) investors these compound to roughly 100:3:1, so "log investors must estimate means well if they are to survive." Overestimating an edge therefore causes severe overbetting. Remedies:
- **Fractional Kelly:** stake α·f_k* with α ≈ 0.25–0.5. MacLean, Ziemba & Blazenko (1992) [Medium](https://medium.com/@tmapendembe_28659/the-dangers-of-full-kelly-criterion-why-most-traders-should-use-fractional-kelly-criterion-instead-0338e3bcc705) and MacLean, Sanegre, Zhao & Ziemba (2004) show half-Kelly retains ≈75% of the growth rate while roughly halving volatility. Benter (1994) explicitly recommends betting a fraction ≤ ½ of full Kelly. [De Gruyter Brill](https://www.degruyterbrill.com/document/doi/10.1515/jqas-2020-0122/html?lang=en)
- **Negative-power / iso-elastic utility** δW^δ/δ (δ<0): half-Kelly ≈ δ = −1, quarter-Kelly ≈ δ = −3 for lognormal returns. [UC Berkeley Statistics](https://www.stat.berkeley.edu/~aldous/157/Papers/Good_Bad_Kelly.pdf)
- **Risk-constrained Kelly** (Busseti, Ryu & Boyd 2016): add an explicit drawdown-probability constraint (a tractable convex problem); [arxiv](https://arxiv.org/pdf/1603.06183) a quadratic approximation reduces to a Markowitz-like mean-variance trade-off.
- **Bayesian / parameter-uncertainty Kelly** (Baker & McHale 2013; Metel et al. 2017, arXiv:1701.02814): shrink stakes to account for the sampling distribution of the estimated probabilities.

**1.5 Correlation and multi-race portfolios.** Within one race the Win outcomes are mutually exclusive (perfectly negatively dependent), exactly handled by §1.3. Across simultaneous races the win outcomes are approximately independent, so the multi-race Kelly problem is the separable sum of per-race log-growth terms maximized jointly under the single shared budget Σ_all f ≤ 1 — solvable as one convex program (Whitrow 2007 gives approximations for many simultaneous events). True correlation enters with exotic/multi-leg bets and when the same factor (e.g., going, draw bias) drives several races; there the Kelly objective should be evaluated on the joint outcome distribution rather than assuming independence. The Markowitz mean-variance framing trades off mean and variance of single-period return; the Kelly framing trades off mean and variance of the long-run growth rate (Busseti–Ryu–Boyd show the formal connection via a quadratic approximation). For a log-growth maximizer the prescription is "bet Kelly or less"; [UC Berkeley Statistics](https://www.stat.berkeley.edu/~aldous/157/Papers/Good_Bad_Kelly.pdf) the mean-variance investor "cannot eat his Sharpe ratio."

**1.6 Overround, de-vigging, and the favourite–longshot bias.** Bookmaker decimal odds o_i imply raw probabilities r_i = 1/o_i that sum to the booksum (1 + overround) > 1. [Springer](https://link.springer.com/article/10.1007/s10479-022-04722-3) Methods to recover "fair" probabilities π_i:
- **Basic normalization (multiplicative):** π_i = r_i / Σ_j r_j. [CRAN](https://cran.r-project.org/web/packages/implied/vignettes/introduction.html) Simple but ignores favourite–longshot bias.
- **Additive:** subtract an equal margin from each; can give negative values.
- **Power method** (Vovk & Zhadanov 2009; Clarke 2016): π_i ∝ r_i^(1/τ), τ solved so Σπ_i = 1; never leaves [0,1] and accommodates the favourite–longshot bias. [Science Publishing Group](https://www.sciencepublishinggroup.com/article/10.11648/j.ajss.20170506.12) Clarke, Kovalchik & Ingram (2017) find the power method universally outperforms the multiplicative method across three sports datasets. [Science Publishing Group](https://www.sciencepublishinggroup.com/article/10.11648/j.ajss.20170506.12)
- **Shin's model** (Shin 1992, 1993; implemented in the R `implied` package and Python `mberk/shin`): assumes a proportion z of insider traders [GitHub](https://github.com/mberk/shin) and inverts the resulting state-contingent prices. π_i found by solving for z (e.g., Jullien–Salanié nonlinear least squares, argmin_z (Σπ_i(z) − 1)²). [arxiv](https://arxiv.org/pdf/1802.08848) Shown by Štrumbelj (2014), Clarke et al. (2017) and Koning & Boot (2020) to be more accurate than basic normalization and to provide unbiased win-probability estimates while modelling the favourite–longshot bias; Cain, Law & Peel (2003) confirm Shin's prediction [Springer](https://link.springer.com/article/10.1007/s10479-022-04722-3) that margins rise with field size.

The favourite–longshot bias — longshots systematically over-bet (lose more), favourites under-bet — means raw odds are not unbiased probabilities; this is why de-vigging with a bias-aware method matters before computing edges.

### Part 2 — Predicting Win Probabilities from Historical Data

**2.1 The conditional/multinomial logit (Bolton & Chapman 1986).** "Searching for Positive Returns at the Track: A Multinomial Logit Model for Handicapping Horse Races," *Management Science* 32(8):1040–1060. Each horse h in race r has a feature vector v_h; a linear "performance index" V_h = β^T v_h is formed; the win probability is the softmax over the runners in that race: [arXiv](https://ar5iv.labs.arxiv.org/html/1701.02814)

  π_h = exp(β^T v_h) / Σ_{i=1}^{n_r} exp(β^T v_i).

β is fit by maximum likelihood over R races; with w^r the winner of race r,

  ln L(β) = Σ_{r=1}^{R} [ β^T v_{w^r} − ln Σ_{i=1}^{n_r} exp(β^T v_i) ]. [arxiv](https://arxiv.org/pdf/1701.02814)

This log-likelihood is concave, so β̂ is found by [arxiv](https://arxiv.org/pdf/1701.02814) Newton/IRLS; [arXiv](https://ar5iv.labs.arxiv.org/html/1701.02814) the gradient (score) and negative Hessian (Fisher information) have closed forms, and Cov(β̂) ≈ I(β̂)^{-1}. Bolton & Chapman used 200 races [ACM Digital Library](https://dl.acm.org/doi/abs/10.5555/2772602.2772613) and exploited rank-ordered finishes via the Chapman–Staelin "explosion" of each finishing order into independent choice sets to gain estimation efficiency. [Nycdatascience](https://blog.nycdatascience.com/blog/student-works/capstone/predicting-horse-racing-outcomes) They found an unobtrusive-bets strategy with a longshot side-constraint could yield positive expected returns despite the track take. [IDEAS/RePEc](https://ideas.repec.org/a/inm/ormnsc/v32y1986i8p1040-1060.html) Their model restricted to 1–1.25 mile races [gwern](https://gwern.net/doc/statistics/decision/1994-benter.pdf) on good/fast tracks because race-level variables that don't vary across runners drop out of the conditional logit. [Gwern](https://gwern.net/doc/statistics/decision/1986-bolton.pdf)

**2.2 Harville and the place/show problem.** Harville (1973, *JASA*) computes ordered-finish probabilities from win probabilities [SciSpace](https://scispace.com/pdf/probability-and-statistical-models-for-racing-1ycr6yyxlf.pdf) by sequential renormalization. Probability that i, j, k finish 1-2-3:

  P(i,j,k) = π_i · [π_j/(1−π_i)] · [π_k/(1−π_i−π_j)]. [uspto](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12562033)

Generally the probability of a full ordering multiplies π's divided by the residual unallocated probability at each rank. The place probability of a horse is the sum over orderings where it finishes in the top m. This is implied by independent exponential (extreme-value) running times. [SciSpace](https://scispace.com/pdf/probability-and-statistical-models-for-racing-1ycr6yyxlf.pdf)

**2.3 Known Harville bias and corrections.** Empirically (Benter 1994; Lo & Bacon-Shone), low-probability horses finish 2nd/3rd more often than Harville predicts and favourites less often, with the effect stronger for 3rd than 2nd. [Paceadvantage](http://www.paceadvantage.com/forum/archive/index.php/t-35264.html) Corrections:
- **Henery (1981)** (normal running times) and **Stern (1990)** (gamma running times) fit better but have no closed form [ResearchGate](https://www.researchgate.net/publication/4748916_Probability_and_Statistical_Models_for_Racing) and are numerically expensive.
- **Lo–Bacon-Shone–Busche discounted Harville:** raise the win probabilities to a power before renormalizing at each stage. The ordering probability becomes
  π_{ijkl} = π_i · [π_j^{λ1}/Σ_{s≠i} π_s^{λ1}] · [π_k^{λ2}/Σ_{s≠i,j} π_s^{λ2}] · [π_l^{λ3}/Σ_{s≠i,j,k} π_s^{λ3}],
  with λ1, λ2, λ3 calibrated to data (λ decreasing for later places). A common practitioner rule of thumb: powers of ≈0.81 for 2nd and ≈0.65 for 3rd when using model probabilities. [Paceadvantage](http://www.paceadvantage.com/forum/archive/index.php/t-35264.html) Benter reports MLE values [Gwern](https://gwern.net/doc/statistics/decision/1994-benter.pdf) γ ≈ 0.81 and δ ≈ 0.65 for his Hong Kong data. [Scribd](https://www.scribd.com/doc/166556276/Benter) Lo (1994) found the Stern model (r=4) outperformed both Harville and Henery on Japanese data. [Nessis](https://www.nessis.org/nessis07/Victor_Lo.pdf)
- The **Henery softmax / `ohenery`** generalization: μ_i = exp(η_i); when all γ = 1 it reduces to the Harville softmax.

**2.4 Modern machine learning.** Gradient-boosted trees (XGBoost, LightGBM, CatBoost) dominate tabular racing data because they handle nonlinear interactions, missing values, and heterogeneous features; random forests and neural networks are also used. CatBoost has been reported to give the lowest error in finish-time prediction studies. Key feature families (Benter 1994; modern practice):
- **Speed/time:** standardized past race times, speed figures, sectional/in-running times.
- **Form:** recent finishing positions, lengths behind winner, win/place strike rates, momentum.
- **Class & ratings:** official rating, Timeform rating, class of past races, prize-money level.
- **Conditions fit:** going/ground preference, distance preference (Benter's residual-based DPGA factor), course/track preference, surface.
- **Weight & setup:** weight carried, weight-for-age, headgear (blinkers/hood/tongue-tie), draw/barrier/post position (critical in large-field sprints like the Wokingham), days since last run, age, number of past races (a surprisingly strong factor in Benter's model). [gwern](https://gwern.net/doc/statistics/decision/1994-benter.pdf)
- **Connections:** jockey and trainer strike rates and combinations.

For ranking within a race, the ML scores should be passed through a race-wise softmax (or a learning-to-rank objective) so they normalize to win probabilities. Critically, a tree model's class-probability output is not race-normalized and is usually mis-calibrated; calibrate then normalize.

**2.5 Benter's combination model (the crucial second stage).** Benter (1994), "Computer Based Horse Race Handicapping and Wagering Systems: A Report" (in Hausch, Lo & Ziemba, Efficiency of Racetrack Betting Markets), shows the fundamental model alone is biased toward the public estimate and unusable for staking. [Gwern](https://gwern.net/doc/statistics/decision/1994-benter.pdf) The fix: fit a *second* conditional logit using only two regressors — the log of the out-of-sample fundamental probability and the log of the public's implied probability: [Semantic Scholar](https://www.semanticscholar.org/paper/Computer-Based-Horse-Race-Handicapping-and-Wagering-Benter/2ea3ed4fa5ea9645614d76dd0a79201740949566)

  c_i = exp(α·f_i + γ·π_i) / Σ_j exp(α·f_j + γ·π_j), [gwern](https://gwern.net/doc/statistics/decision/1994-benter.pdf)

where f_i = ln(out-of-sample fundamental model probability), π_i = ln(public implied probability), c_i = combined probability. α and γ are estimated by maximizing the combined-model log-likelihood over a held-out race set. Using out-of-sample fundamental estimates is essential to avoid overstating the model's significance. The relative magnitudes of α and γ measure how much the model adds over the public.

**2.6 Model assessment.** Benter adopts a pseudo-R² (the "R̄f" or explanatory-power measure from Bolton & Chapman) and a heuristic ΔR² = R²_combined − R²_public as a profitability proxy. [gwern](https://gwern.net/doc/statistics/decision/1994-benter.pdf) In his data R²_public = .1218, R²_fundamental = .1245, R²_combined = .1396, so ΔR² = .0178 — small but, he reports, enough for significant profit. The lesson illustrated by the "tipster" counterexample: a stand-alone model with a high R² adds nothing if it merely reproduces the public estimate; [Gwern](https://gwern.net/doc/statistics/decision/1994-benter.pdf) what matters is the *gain* over the public when combined.

**2.7 Calibration and overfitting.** After producing probabilities:
- **Calibrate** with Platt scaling [Train in Data](https://www.blog.trainindata.com/complete-guide-to-platt-scaling/) (sigmoid, few parameters, safer on small data) or isotonic regression (PAVA, more flexible, [APXML](https://apxml.com/courses/mastering-gradient-boosting-algorithms/chapter-7-boosting-advanced-topics-customization/probability-calibration-boosting) needs more data, can overfit) on a held-out fold; assess with Brier score, log loss, and reliability diagrams. Temperature scaling is the neural-net analogue.
- **Avoid overfitting** via strict data partitioning / hold-out races [INFORMS](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.32.8.1040) (Benter's central methodological warning), time-series cross-validation (train on past, test on future — never shuffle across time), regularization (L1/L2), and limiting feature count relative to the ~500–1000 races Benter cites as the practical minimum for development plus testing.

### Part 3 — Royal Ascot, Saturday 20 June 2026

**3.1 The meeting.** Royal Ascot 2026 runs Tuesday 16 June to Saturday 20 June 2026, [William Hill](https://news.williamhill.com/horse-racing/royal-ascot/race-schedule-2026/) 35 races (seven per day), a record £10.65 million in prize money across the meeting (up from £10.05m in 2025), with all eight Group 1 races now worth a minimum of £700,000 (Ascot Racecourse/William Hill News); total 2026 Ascot prize money for the year reaches a record £19.4 million. Saturday 20 June is the final day, a seven-race card. The published 2026 race-by-race schedule for the final day (times, grades, distances):

| Time | Race | Grade/Class | Distance | Eligibility | Prize |
|---|---|---|---|---|---|
| 2:30pm | Norfolk Stakes | Group 2 | 5f | 2yo | £175,000 |
| 3:05pm | Hardwicke Stakes | Group 2 (Class 1) | 1m4f (≈1m3f211y) | 4yo+ | £250,000 |
| 3:40pm | QEII Jubilee Stakes | Group 1 (Class 1) | 6f | 4yo+ | £1,000,000 |
| 4:20pm | Jersey Stakes | Group 3 (Class 1) | 7f | 3yo | — |
| 5:00pm | Wokingham Stakes (Heritage Handicap) | Class 2 | 6f | 3yo+ | £175,000 |
| 5:35pm | Golden Gates Stakes (Handicap) | Class 2 | 1m2f | 3yo | £120,000 |
| 6:10pm | Queen Alexandra Stakes (Conditions) | Class 2 | 2m5½f | 4yo+ | — |

Note on the card order: Ascot Racecourse officially confirmed that the Norfolk and Chesham Stakes "have swapped positions, with the Norfolk Stakes now the first race on the Saturday in order to maximise television exposure for that race in the USA" (via Racing Ahead, 2026); the **Chesham Stakes** (Listed, 7f, 2yo) therefore moves to Thursday 18 June (2:30pm), not Saturday. Some early-season listings still show the Chesham on Saturday — the user should verify the final declared card the week of the meeting, as Ascot occasionally reshuffles.

**3.2 Headline race and early prices.** The QEII Jubilee Stakes (formerly the Golden/Diamond/Platinum Jubilee; Group 1 since 2002) is the marquee Saturday event and the meeting's premier sprint. Its £1,000,000 prize fund is joint-richest of the meeting (with the Prince of Wales's Stakes), with the winner taking home £567,100 (Sportsnaut/Yahoo Sports' 2026 prize-money breakdown). Ante-post indications (per Racing TV's 2026 racecard snapshot) for the day's races included, in the Jubilee, Joliestar (~9/4), Satono Reve (~9/2) and Overpass (~17/2); in the Hardwicke, Kalpana (~5/2), Goliath (~6/1), Jan Brueghel (~9/2); [Racing TV](https://www.racingtv.com/racecourses/uk/ascot/racecards) in the Wokingham, Binhareer and Double Rush (~9/2). **These are early/indicative ante-post prices and field sizes that will change; treat them as illustrative, not declared.** The large-field handicaps (Wokingham, ~28 runners; Golden Gates, ~16) [Racing TV](https://www.racingtv.com/racecourses/uk/ascot/racecards) and the competitive Group races are the best candidates for a portfolio/multi-horse Kelly approach because mispricing and field size create more positive-edge opportunities; the Queen Alexandra (marathon conditions race, small specialist field) and short-field Group 1 are lower-opportunity for staking.

**3.3 Why this matters for the model.** UK flat handicaps like the Wokingham are exactly the "many high-advantage opportunities" environment Benter describes — large fields, draw/going effects, and a deep betting market. But note Benter's key data caveat: a *closed* racing population (as in Hong Kong) is far easier to model than UK racing, where horses ship between many tracks and past-performance normalization and opponent-strength estimation are harder. Expect the UK to be a harder, lower-edge environment than the Hong Kong results in the literature suggest.

### Data Sources

- **Betfair Historical Data Service** (historicdata.betfair.com): timestamped exchange price/volume data; **Betfair Starting Price (BSP)** files at promo.betfair.com/betfairsp give per-runner BSP plus in-play min/max — ideal "market probability" inputs (back-and-lay, lower effective margin than bookmakers). Python wrapper: `betfairlightweight`.
- **Timeform** (via the Betfair-owned **Timeform API** / Global Sports API): ratings, going, sectionals (spreadsheet), horse-by-horse comments, smart stats; commercial terms; coverage from the early 1990s.
- **Racing Post:** racecards, results, form, going; the de facto UK form source (scraping restricted by ToS).
- **The Racing API** (theracingapi.com): 500,000+ results and racecards plus bookmaker odds, covering UK/Ireland/Australia/USA — a clean programmatic option.
- **British Horseracing Authority** (britishhorseracing.com): going, non-runners, official results/ratings.
- **Hong Kong Jockey Club:** the closed-population gold standard used by Benter; HKJC publishes detailed results/sectionals.
- **Kaggle / community datasets:** e.g., "Horse Racing" 1990–2020 [Kaggle](https://www.kaggle.com/datasets/hwaitt/horse-racing) (hwaitt), "Horse Racing Data" with Betfair SP and engineered fields (anushamishra), "One week of Betfair data: horses" [Kaggle](https://www.kaggle.com/datasets/zygmunt/betfair-horses) (zygmunt). Provenance and completeness vary; verify before use.
- **PA Media racing feed** supplies most licensed operators/news outlets. The market is fragmented; serious users combine multiple feeds (results from one, odds/exchange from another).

## Recommendations

**Stage 1 — Build and validate the fundamental model first.** Assemble ≥1,000 UK flat races with full past-performance data per runner (Benter's practical minimum). Fit a conditional logit (Bolton–Chapman softmax) as the interpretable baseline; in parallel train a gradient-boosted ranker (LightGBM/XGBoost with a race-grouped objective) and compare out-of-sample log-loss and pseudo-R². Use strictly time-ordered train/validation/test splits. **Benchmark:** the model only matters if, when combined with the de-vigged market (Stage 2), ΔR² > 0 on held-out races. If ΔR² ≤ 0, do not bet.

**Stage 2 — De-vig the market and run Benter's combination.** Convert Betfair SP / best bookmaker odds to fair probabilities with Shin's method (or the power method) — not naive normalization. Fit the two-regressor second-stage logit (log-fundamental, log-public) on out-of-sample fundamental probabilities. The output c_i is your only staking-grade probability.

**Stage 3 — Calibrate, then size with fractional Kelly.** Calibrate c_i (Platt/isotonic on a held-out fold; check Brier/log-loss/reliability). For each Saturday race, run the Smoczynski–Tomkins active-set algorithm to get f_k*, then stake **α·f_k* with α = 0.25 initially** (move toward 0.5 only after a verified positive-edge track record). Apply across the seven races as one budget-constrained convex program. **Thresholds that change the action:** (a) if calibration reliability drifts (Brier worsens out-of-sample), cut α; (b) if estimated edge e_k = p_k·O_k is below ~1.05–1.10 after de-vigging, skip the bet (estimation noise dominates thin edges); (c) if realized drawdown exceeds a preset cap (e.g., 20%), drop to a fixed small fractional stake until equity recovers (risk-constrained Kelly logic).

**Stage 4 — Race selection.** Prioritize the large-field handicaps (Wokingham, Golden Gates) and competitive Group races where the model can find multiple positive-edge runners; de-prioritize the small-field Group 1 and the specialist Queen Alexandra. Confirm the declared card, runners, draw and going on the morning of 20 June 2026.

**Hand-off note to Claude Code.** Implement five modules: (1) odds→fair-probability (Shin + power), (2) fundamental model (conditional logit + GBM ranker with race-softmax), (3) Benter second-stage logit, (4) calibration (CalibratedClassifierCV), (5) Smoczynski–Tomkins Kelly allocator with a fractional multiplier and drawdown guard. Place/show/exotics use discounted-Harville with calibrated λ's.

## Caveats
- **Early/illustrative Royal Ascot data.** Field sizes, runners and ante-post prices cited for 20 June 2026 are pre-declaration and will change; the Norfolk/Chesham swap is officially confirmed (Norfolk opens Saturday, Chesham moves to Thursday), but the full declared card must still be confirmed the week of the meeting.
- **The literature's profits are mostly Hong Kong, a closed population.** Benter's edge was built in an unusually tractable, high-liquidity, closed market. UK racing is harder; do not assume comparable returns.
- **Estimation error dominates.** Errors in estimated win probability are far more damaging than covariance errors (Chopra–Ziemba's ~20:1 ratio, compounding to ~100:1 for Kelly investors) and cause Kelly overbetting; fractional Kelly and conservative edge thresholds are not optional.
- **Market efficiency and adverse selection.** Public odds already embed most fundamental information plus genuine inside information (stable intentions, workouts) you cannot observe; your model must beat that, net of margin and after your own bets move the price.
- **Some restated formulas are second-hand.** The Smoczynski–Tomkins closed form here is reconstructed from multiple restatements (Whelan 2025; Metel et al. 2017; de Gruyter 2020) that cite the original; verify the D-placement convention (f_k = p_k − b·β_k vs. the state-price form) against your stake-normalization before going live. The maximal-growth-rate expression is structurally derived rather than quoted verbatim.
- **No betting system guarantees profit**, and this document is a theoretical specification, not financial advice.