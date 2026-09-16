# Lee–Mykland implementation

Lee, S. S. and Mykland, P. A. (2008), "Jumps in Financial Markets: A New
Nonparametric Test and Jump Dynamics", *Review of Financial Studies* 21(6),
2535–2563. Implemented in `research/lee_mykland/detect_lee_mykland_jumps.py`
and applied to this study's panel by `research/makinen/label_jumps.py`. There
is exactly one implementation in the repository; the second file imports from
the first rather than restating it.

No substitute (return threshold, rolling z-score, standard-deviation cutoff,
percentage move) is used anywhere.

## Return definition

$$r_i = m_i - m_{i-1}, \qquad m_i = \frac{\text{best bid}_i + \text{best ask}_i}{2}$$

on the 1-minute grid, computed **within a single contract only**.

*Departure from the paper:* returns are arithmetic (changes in probability
points), not log. These are prediction-market contracts bounded in $(0,1)$; in
log space one half-tick is a $0.0055$ return at $p=0.90$ but $0.41$ at
$p=0.01$, so a log-return test spends its power on ticks in near-resolved
contracts. In probability points a half-tick is $0.005$ everywhere. Two checks
support the change: positive and negative jump counts become far more
symmetric, and the YES and NO legs of the same market produce exactly mirrored
statistics (identical $|L|$, opposite sign), which is arithmetically forced
when $p_{\text{YES}} + p_{\text{NO}} = 1$ and is simply untrue of log returns.

## Local volatility — bipower variation

$$\hat{\sigma}^2(i) = \frac{1}{K-2}\sum_{j=i-K+2}^{i-1} |r_j|\,|r_{j-1}|$$

Adjacent absolute returns are multiplied, which is what makes the estimator
robust to jumps: a lone jump enters only two of the $K-2$ products and each
time is multiplied by an ordinary-sized neighbour, so its influence is
$O(1/K)$ rather than dominating. Realised variance $\sum r^2$ has no such
property — a jump inflates its own denominator and masks itself. Measured on
this data, swapping bipower for realised variance loses **85%** of the
detections, because $\hat{\sigma}$ at those points comes out 2.4× too large.

**Window:** every term is indexed strictly below $i$. The window is
**trailing, never centred**; the statistic at minute $i$ uses no observation at
or after $i+1$.

## Test statistic

$$L(i) = \frac{r_i}{\hat{\sigma}(i)}$$

## Critical threshold — extreme-value normalisation

Under the null of no jump, $|L|$ behaves like $|N(0,1)|/c$ with
$c=\sqrt{2/\pi}\approx 0.7979$, because bipower estimates $c^2\sigma^2$ rather
than $\sigma^2$. LM Lemma 1 gives the limiting distribution of the maximum over
$n$ tested observations:

$$C_n = \frac{(2\log n)^{1/2}}{c} - \frac{\log\pi + \log\log n}{2c(2\log n)^{1/2}}, \qquad S_n = \frac{1}{c(2\log n)^{1/2}}$$

and $(\max|L| - C_n)/S_n \to \xi$ with $P(\xi \le x) = \exp(-e^{-x})$ (standard
Gumbel). Observation $i$ is a jump when

$$|L(i)| > C_n + S_n\,\beta^*, \qquad \beta^* = -\log(-\log(1-\alpha))$$

This is a family-wise threshold over the $n$ observations actually tested, so
multiple testing is already accounted for and the cutoff rescales automatically
with sample size.

## Settings used

| parameter | value | note |
|---|---|---|
| $\alpha$ | **0.01** | as in the paper; never tuned |
| $\beta^*$ | 4.6001 | $-\log(-\log 0.99)$ |
| $K$ | **30** minutes | see below |
| $n$ | 81,458 | tested observations |
| $C_n$ | 5.4898 | |
| $S_n$ | 0.2635 | |
| **threshold** | **6.7021** | vs 2.576 for a naive normal cutoff |

**Why $K=30$ and not the paper's 600.** The paper studies continuously traded
equities, where 600 minutes is roughly a day and a half of trailing history. A
Polymarket game contract is created hours before first pitch, trades for one
game and resolves; on the in-game grid it lives a median of ~161 minutes. At
$K=600$ almost no observation is testable. $K$ was chosen from a sweep
(`results/lee_mykland/K_sweep_in_game.csv`) on a volatility-tracking
diagnostic — the ratio of forward realised volatility to the trailing bipower
estimate — where $K=30$ gave the tightest agreement (IQR 1.30–1.67, versus
0.99–1.86 at $K=120$) while still testing 72% of in-game returns. This is a
window-length choice forced by contract lifetime, not an adjustment of the
test's significance level.

## Treatment of the start of each series

The first observations of a contract have no complete bipower window and
receive **no statistic**. They are recorded as warm-up (`tested = False`) and
are never reported as "no jump". 12,900 of 94,358 panel minutes are warm-up or
otherwise untestable.

## Treatment of missing observations

A return is formed only between two **adjacent** minutes that both carry a
quote. Where the minute grid has a hole the return is undefined, the window
simply contains fewer products, and nothing is interpolated or forward-filled
across the gap. A statistic additionally requires at least 7 **non-zero**
products in the window: 31% of in-game minute returns are exactly zero
(discrete tick prices), and a window of near-all-zero products drives
$\hat{\sigma}\to 0$ and $L\to\infty$. Without this guard the first run produced
statistics of order $10^8$ off $\hat{\sigma}=6\times10^{-11}$. LM's asymptotics
assume $\sigma>0$; this guard enforces that and is documented rather than
silently applied.

## Market boundaries

There is no overnight gap to handle: these contracts trade continuously and
each one exists only within a single session. Every contract is therefore its
own series, and returns, bipower windows and statistics never cross a contract
or session boundary. The panel is additionally restricted to **in-game**
minutes (first pitch to final out, from `data/game_windows.json`), because the
pre-game period is near-motionless and would otherwise fill the volatility
window with a quieter regime than the one being tested — measured at 3.08×
too low before trimming, 1.42× after.

## Result and sanity check

81,458 tested minutes, **1,516 jumps (873 positive, 643 negative) = 1.86% of
tested minutes**, median absolute jump return 0.18 in probability points.
Stage 1A verdict: **PLAUSIBLE**. Figures in `results/makinen/jumps/`.
