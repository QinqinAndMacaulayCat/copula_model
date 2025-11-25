
# Copula-Based Portfolio Risk Modeling Using S&P 500 Constituents

Team members: Yue Yang, Ziran Guo, Yuxi Geng, Qinqin Huang

## 1. Introduction and Objective

Traditional Value-at-Risk (VaR) estimation methods often rely on simplifying assumptions such as normally distributed returns or independence among asset returns. However, in real financial markets, asset returns exhibit non-linear dependence and tail co-movements that such models fail to capture.

This project proposes to develop a portfolio VaR estimation framework based on copula models using daily return data of S&P 500 constituent stocks. The key objective is to model the joint dependence structure accurately, particularly in the tails, and to evaluate whether copula-based VaR improves risk estimation over the assumption of independence. The performance of the model will be assessed via traffic light backtesting methods.

The specific steps include:

- Data collection and preprocessing of daily returns for S&P 500 stocks.
- Fitting marginal distributions to individual stock returns.
- Constructing elliptical copula models (Gaussian, Student-t) to capture dependencies.
- Simulating portfolio returns using the fitted copula and marginal distributions.
- Estimating VaR at different confidence levels from the simulated returns. 
- Backtesting how many times actual losses exceed predicted VaR and evaluating model performance by comparing with benchmarks assuming independence among assets.

## 2. Methodology

### 2.1 Data Preparation

We downloaded stock prices from yahoo finance and transformed to lognormal return rates. Data ranges from 2010-01-01 to 2025-11-03. Missing log return values are filled with 0.

### 2.2 Marginal Distribution Modeling

For each stock's daily log-returns, fit candidate univariate distributions (Normal / Student‑t / Empirical), select the best one with information criteria and goodness-of-fit tests, and transform series using the Probability Integral Transform (PIT) to obtain Uniform(0,1) margins.

#### 2.2.1 Univariate Normal Distribution

We treat the Normal (Gaussian) distribution as a parametric candidate with location parameter $\mu$ and scale parameter $\sigma>0$. The probability density function (pdf) is

$$
f(x;\mu,\sigma)=\frac{1}{\sigma\sqrt{2\pi}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right),
$$

and the cumulative distribution function (CDF) is the standard Normal CDF shifted and scaled:

$$
F(x;\mu,\sigma)=\Phi\left(\frac{x-\mu}{\sigma}\right).
$$

For independent observations $x_1,\dots,x_n$ the log-likelihood is

$$
\ell(\mu,\sigma)=\sum_{i=1}^n \ln f(x_i;\mu,\sigma) = -\frac{n}{2}\ln(2\pi)-n\ln\sigma -\frac{1}{2\sigma^2}\sum_{i=1}^n (x_i-\mu)^2.
$$

The MLEs are closed-form:

$$
\hat{\mu}=\bar{x}=\frac{1}{n}\sum_{i=1}^n x_i,\qquad \hat{\sigma}^2=\frac{1}{n}\sum_{i=1}^n (x_i-\bar{x})^2.
$$

We transform observed returns to PIT (Uniform(0,1)) via the fitted CDF:

$$
U_i=F(x_i;\hat{\mu},\hat{\sigma})=\Phi\left(\frac{x_i-\hat{\mu}}{\hat{\sigma}}\right).
$$

In practice we clip PIT values to $(\varepsilon,1-\varepsilon)$ with a small $\varepsilon$ (here $10^{-6}$) to avoid exact 0 or 1 that would cause numerical issues when inverting marginals during simulation.

#### 2.2.2 Univariate t-Distribution

The univariate Student‑t distribution with degrees of freedom $\nu>0$, location $\mu$ and scale $\sigma>0$ has pdf

$$
f(x;\nu,\mu,\sigma)=\frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right)\sqrt{\nu\pi}\,\sigma}\left(1+\frac{1}{\nu}\left(\frac{x-\mu}{\sigma}\right)^2\right)^{-\frac{\nu+1}{2}}.
$$

Its heavier tails (relative to the Normal) are controlled by $\nu$; as $\nu\to\infty$ the t distribution converges to the Normal. There is no simple closed-form MLE for all parameters simultaneously; numerical MLE (e.g. `scipy.stats.t.fit`) is used to obtain $\hat{\nu},\hat{\mu},\hat{\sigma}$. The log-likelihood is

$$
\ell(\nu,\mu,\sigma)=\sum_{i=1}^n \ln f(x_i;\nu,\mu,\sigma).
$$

We obtain PIT values via the fitted t CDF:

$$
U_i=F_{t,\hat{\nu}}\left(\frac{x_i-\hat{\mu}}{\hat{\sigma}}\right)
$$

and apply the same clipping $(\varepsilon,1-\varepsilon)$ as above. In model selection the t candidate is preferred when information criteria (AIC/BIC) favor it or when diagnostics indicate heavy tails (small estimated $\hat{\nu}$) that materially improve fit.

#### 2.2.3 Univariate Empirical Distribution

The empirical (nonparametric) CDF for a sample $\{x_1,\dots,x_n\}$ is the empirical distribution function (ECDF)

$$
\hat{F}_n(x)=\frac{1}{n}\sum_{i=1}^n \mathbf{1}\{x_i\le x\}.
$$

To construct PIT values that avoid exact 0 and 1 we use a plotting-position formula. If $r_i$ is the rank of $x_i$ (1 = smallest), the plotting-position PIT is

$$
U_i=\frac{r_i-\tfrac{1}{2}}{n}.
$$

This rank-based transform preserves the empirical tail behavior without imposing a parametric shape and is used when both parametric candidates are deemed inadequate by GOF diagnostics. As above, values are clipped to $(\varepsilon,1-\varepsilon)$ prior to copula estimation and simulation.

#### 2.2.4 Selecting the Best Fit

Selecting the best marginal per ticker combines likelihood‑based information criteria, distributional goodness-of-fit tests, and practical heuristics. The formal elements are:

- Information criteria
  - Akaike Information Criterion (AIC):

  $$
  \mathrm{AIC}=2k-2\ell(\hat{\theta}),
  $$

  where $k$ is the number of estimated parameters and $\ell(\hat{\theta})$ is the maximized log-likelihood. AIC estimates out-of-sample KL divergence (up to an additive constant) and penalizes model complexity linearly in $k$.

  - Bayesian Information Criterion (BIC):

  $$
  \mathrm{BIC}=k\ln n - 2\ell(\hat{\theta}),
  $$

  where $n$ is the sample size. BIC penalizes complexity more strongly for larger samples and is consistent (selects the true model with probability →1 when the true model is among candidates and $n \to \infty$).

  We compute both criteria for each parametric candidate and prefer models with lower AIC and/or BIC. In implementation we compare combined evidence (for robustness) but either criterion may be decisive depending on sample size.

- Goodness-of-fit (GOF) tests
  - Kolmogorov–Smirnov (KS) statistic: for fitted CDF $F$ and empirical CDF $\hat{F}_n$,

  $$
  D_n=\sup_x |\hat{F}_n(x)-F(x)|.
  $$

  The KS test assesses maximum deviation; small p-values indicate the parametric CDF is unlikely to have generated the data.

  - Cramér–von Mises (CvM) statistic: an L2 measure of discrepancy,

  $$
  W_n= n\int (\hat{F}_n(x)-F(x))^2\,dF(x),
  $$

  which is more sensitive to deviations across the whole support (including tails) than the KS statistic.

  We compute p-values for both tests under the fitted null. If both parametric candidates are strongly rejected (p-values below a chosen threshold such as 0.05), we prefer the empirical marginal.

- Practical heuristics and safeguards
  - Minimum sample size: extremely short series are skipped because parameter estimates and test statistics are unreliable.
  - Tail bias toward Student‑t: when the estimated degrees-of-freedom $\hat{\nu}$ for the t-distribution is small (indicating heavy tails) we may prefer t even when information criteria differences are modest.
  - Numerical guards: enforce lower bounds on scale parameters and on $\nu$ to avoid degenerate fits and ensure finite variance where required.

Operationally, the routine `run_fitting` computes log-likelihoods, AIC, BIC, KS and CvM p-values for the Normal and t candidates, and falls back to the empirical PIT when both parametric fits are rejected. The chosen model and parameter estimates are recorded in `best_table`; the per-ticker PITs (Uniform(0,1)) are collected into `pit_df`, and these objects are used as inputs for copula parameter estimation and simulation.

### 2.3 Fitting Copulas to Data

#### 2.3.1 Gaussian Copula

Gaussian copula assumes that the correlation structure of multivariates is linear and can be captured by a multivariate normal distribution. The key parameter to estimate is the correlation matrix $\Sigma$.

We can either use the Pearson correlation matrix or the Kendall's tau matrix to estimate the correlation matrix for the Gaussian copula. However, the Kendall's tau matrix is more robust to outliers so we will use it in this project. 

Assume we have $n$ assets with historical log returns data. Let $F_i$ be the empirical CDF of the log returns for asset $i$.

The steps to fit a Gaussian copula are as follows:

1. Compute the empirical Kendall's tau matrix from the empirical df of the log returns.

$$
\tau_{ij} = P((F_i - F_i')(F_j - F_j') > 0) - P((F_i - F_i')(F_j - F_j') < 0)
$$

Where $(F_i', F_j')$ is an independent copy of $(F_i, F_j)$.


2. Convert the Kendall's tau matrix to the correlation matrix using the relation:

    $$
    \rho_{ij} = \sin\left( \frac{\pi}{2} \tau_{ij} \right)
    $$

  $\rho_{ij}$ is the $(i,j)$-th entry of the correlation matrix $\Sigma$.

#### 2.3.2 t Copula

The Gaussian copula is simple but may not capture tail dependencies well. The t copula can model stronger symmetric tail dependence. In other words, t copula can capture heavier tails in both lower and upper tails.

The correlation matrix for the t copula is estimated in the same way as the Gaussian copula using the Kendall's tau matrix.
The degrees of freedom parameter $\nu$ can be estimated using maximum likelihood estimation (MLE) or method of moments based on the empirical data.

The log-likelihood function for the t copula is given by:

$$
ln L(\nu; \Sigma, \hat{U}_1, \hat{U}_2, \dots, \hat{U}_n) = \sum_{t=1}^{n} \ln g_{\nu, \Sigma}(t_{\nu}^{-1}(\hat{U}_{t, 1}), t_{\nu}^{-1}(\hat{U}_{t, 2}), \dots, t_{\nu}^{-1}(\hat{U}_{t, d})) - \sum_{i=1}^{d} \sum_{t=1}^{n} \ln g_{\nu}(t_{\nu}^{-1}(\hat{U}_{t, i}))
$$

Where,

- $\hat{U}_{t} = (U_{t, 1}, U_{t, 2}, \dots, U_{t, d}) = (\hat{F}_1(X_{t, 1}), \hat{F}_2(X_{t, 2}), \dots, \hat{F}_d(X_{t, d}))$ are the pseudo-observations.
- $t_{\nu}^{-1}$ is the inverse CDF of the univariate t distribution with $\nu$ degrees of freedom.
- $g_{\nu, \Sigma}$ is the joint density of a random vector with $t_d(\nu, 0, \Sigma)$ distribution.


### 2.4 Simulate Paths

To estimate VaR using copula models, we will simulate multiple paths of portfolio returns based on the fitted copula and marginal distributions.

Assume we have $n$ assets in the portfolio and we want to simulate $N$ scenarios of portfolio returns over $m$ days.

#### 2.4.1 Gaussian Copula

Steps to simulate portfolio returns using Gaussian copula:

1. Generate $n times m$ matrix of independent standard normal variables $Z$.
2. Apply Cholesky decomposition to the correlation matrix $\Sigma$ obtained from the historical returns to get a lower triangular matrix $L$ such that $\Sigma = LL^T$.
3. Transform the independent standard normal variables to correlated normal variables: $Y = ZL^T$.
4. Convert the correlated normal variables to uniform variables using the standard normal CDF: $U_{ij} = \Phi(Y_{ij})$ for $i = 1, ..., n$ and $j = 1, ..., m$.
5. Invert the uniform variables to the original marginal distributions using the inverse CDF (quantile function) of each asset's fitted marginal distribution: $X_{ij} = F_i^{-1}(U_{ij})$.
6. Repeat steps 1-5 for $N$ simulations to obtain $N$ scenarios of portfolio returns.


#### 2.4.2 Student-t Copula

Steps to simulate portfolio returns using Student-t copula:

1. Same as steps 1-3 in Gaussian copula.
2. Simulate $n \times m$ matrix of independent chi-squared variables $W$ with $\nu$ degrees of freedom.
3. Scale the correlated normal variables to obtain Student-t variables: $T_{ij} = Y_{ij} / \sqrt{W_{ij}/\nu}$.
4. Convert the Student-t variables to uniform variables using the Student-t CDF: $U_{ij} = t_{\nu}(T_{ij})$ for $i = 1, ..., n$ and $j = 1, ..., m$.
5. Invert the uniform variables to the original marginal distributions using the inverse CDF (quantile function) of each asset's fitted marginal distribution: $X_{ij} = F_i^{-1}(U_{ij})$.
6. Repeat steps 1-5 for $N$ simulations to obtain $N$ scenarios of portfolio returns.


### 2.5 VaR estimation

We work with an equally-weighted portfolio of all S&P 500 constituents. For each refit date we have a 10-day simulation horizon and 1,000 simulated paths. On a given horizon day *h* and simulation path *i*, we first compute the portfolio log return as the simple average of the simulated log-returns of all assets in that path and day. Collecting these portfolio returns across the 1,000 paths gives an empirical distribution of portfolio returns for that horizon day.

The 1-day 99% VaR for horizon day *h* is then defined as the 1st percentile of that empirical distribution (i.e. the 1% quantile of simulated portfolio returns for that day). Applying the same procedure to the copula-based simulations and to the independent-returns simulations gives two VaR series:

- VaR_copula,t : 1-day 99% VaR from the copula model  
- VaR_indep,t : 1-day 99% VaR from the independent-returns benchmark  

By construction both VaR series are negative numbers and can be interpreted as 1-day portfolio losses at the 99% confidence level.

### 2.6 Backtesting and Basel traffic-light framework

To assess the accuracy of the VaR forecasts, we backtest both models against realised portfolio returns. For each trading day *t*, we align the VaR forecast from day *t-1* with the realised equal-weighted portfolio return on day *t*. A VaR exception (or “hit”) occurs when the realised loss exceeds the predicted VaR level. In other words:

- hit_copula,t = 1 if realised return_t < VaR_copula,t-1, otherwise 0  
- hit_indep,t  = 1 if realised return_t < VaR_indep,t-1, otherwise 0  

We then evaluate the models using the Basel traffic-light framework for 99% VaR. Over a 250-day window, we count the number of exceptions

N = sum of hits over the last 250 days

and map N into a colour zone:

- **Green** if N $\leq$ 4  
- **Yellow** if 5 $\leq$ N $\leq$ 9  
- **Red** if N $\geq$ 10  

This procedure is applied to both the copula-based VaR and the independent-returns VaR so that we can compare their backtesting performance under the same Basel traffic-light rule.


## 3. Results

Since the program involves intensive computation and takes a long time to run, we only refit asset marginal distributions and copula models every year and simulate daily returns for the next year based on the fitted models. However, the VaR estimation and backtesting are done on a daily basis, in which we use a rolling window to calculate the VaR for the next few days (e.g., 10 days) and compare with the actual losses then count the number of exceedances over the testing period. Since the main focus is comparing the copula-based VaR with the independence assumption-based VaR rather than precise VaR values, this approximation is acceptable using yearly refitting.

In each refitting, we use the past 5 years of daily returns to fit the marginal distributions and copula models. The results below show the fitting results, simulated portfolio returns, VaR estimates, and backtesting results.

### 3.1 Distribution Fitting Results

The marginal-distribution fitting was run over the available return history and produced per-ticker model choices and PIT series.

**Model selection (counts)**

| Best Model | Count | Percent |
| ---------- | -----:| -------:|
| Student‑t  | 498   | 99.2%   |
| Empirical  | 4     | 0.8%    |
| Normal     | 0     | 0.0%    |

The vast majority of tickers (498 of 502) were best fit by a Student‑t marginal, with only four tickers requiring an empirical fit. No ticker in this run selected a Gaussian marginal under the AIC/BIC + GOF procedure.

**Parameter summary (Student‑t)**

- Degrees of freedom ($\nu$): mean = 3.3591, std = 0.3779, min = 2.01, median = 3.3491, max = 5.1234.
- Location (t_loc): mean $\approx$ 0.0007941 (daily), typical values near zero.
- Scale (t_scale): mean $\approx$ 0.01257 (daily volatility scale).

The small estimated degrees-of-freedom (median $\approx$ 3.35) indicate pronounced heavy tails in marginal return distributions, which explains the systematic preference for Student‑t marginals over Gaussian.

**PIT diagnostics (pooled across tickers)**

- Number of non-missing PIT observations: 1,888,093
- Mean(PIT) = 0.49854 (ideal = 0.5)
- Var(PIT) = 0.083285 (ideal = 1/12 $\approx$ 0.083333)
- Median(PIT) = 0.49972
- Proportion of PIT in (0.49, 0.51) $\approx$ 2.04%

These pooled diagnostics suggest the PITs are close to Uniform(0,1) on aggregate: mean and variance are very near the theoretical values, and the Q–Q diagnostics in Figure 1 confirm only modest deviations.



![Figure 1 — Aggregate PIT histogram](figures/pit_hist.png){width=600px}

*Figure 1: pooled PIT histogram over all tickers; the dashed line shows the uniform density.*

![Figure 2 — PIT Q–Q plot vs Uniform](figures/pit_qq.png){width=600px}

*Figure 2: PIT Q–Q plot (pooled) vs theoretical Uniform(0,1); empirical quantiles close to the 45° line.*

![Figure 3 — Example ticker with Student‑t fit (PODD)](figures/PODD_fit_t.png){width=600px}

*Figure 3: example ticker panel (returns histogram with fitted PDFs, Q–Q plots, and PIT histogram).* 

![Figure 4 — Example ticker with Normal/t comparison (PODD)](figures/PODD_fit_normal.png){width=600px}

*Figure 4: alternate example fit for comparison.*

![Figure 5 — Example ticker with empirical fallback (GS)](figures/GS_fit_empirical.png){width=600px}

*Figure 5: ticker where empirical (rank) marginal was used as fallback; shows empirical PIT behavior.*


- The near-universal selection of Student‑t marginals and the low estimated degrees-of-freedom motivate the use of a t‑copula (or other tail-dependent copulas) when modeling joint behavior since heavy marginal tails increase the likelihood of joint extremes.
- The small number of empirical fallbacks indicates the parametric candidate set (Normal, t) is adequate for almost all tickers; where empirical fits occur, they should be inspected individually and may reflect micro-capillars, short histories, or data issues.
- PIT diagnostics are satisfactory in aggregate but remain useful at the ticker level: non-uniform PITs for individual tickers can bias copula estimation and should be handled (e.g., longer windows, smoothed CDFs, or explicit nonparametric marginals) if problematic.


### 3.2 Copula Fitting Results

The following are the copula fitting results at each refitting date:

```plaintext
fitting date: 2020-01-08 00:00:00, copula: t, nu: 12.417756424535327
fitting date: 2021-01-08 00:00:00, copula: t, nu: 9.195719667646452
fitting date: 2022-01-10 00:00:00, copula: t, nu: 9.117256117881958
fitting date: 2023-01-12 00:00:00, copula: t, nu: 10.180435891793604
fitting date: 2024-01-17 00:00:00, copula: t, nu: 10.288289682819688
fitting date: 2025-01-21 00:00:00, copula: t, nu: 11.462445359219407
```

From the results, we can see that the t copula is consistently selected as the best-fitting copula model at each refitting date, indicating the presence of tail dependence among the asset returns. The estimated degrees of freedom parameter $\nu$ ranges from approximately 7.96 to 9.82, suggesting moderate tail heaviness in the joint distribution of asset returns.

The correlation matrices are too large to show here in full, but here are the snippets of the correlation matrices at each refitting date. We can see that the correlations among assets do not change significantly over time.

```plaintext
fitting date: 2020-01-08 00:00:00, correlation matrix:
[[1.         0.25135828 0.22668861 ... 0.2755526  0.36510618 0.18096865]
 [0.25135828 1.         0.19294253 ... 0.22807553 0.32770435 0.11335533]
 [0.22668861 0.19294253 1.         ... 0.21777834 0.31088217 0.45800343]
 ...
 [0.2755526  0.22807553 0.21777834 ... 1.         0.3645142  0.15124268]
 [0.36510618 0.32770435 0.31088217 ... 0.3645142  1.         0.28515786]
 [0.18096865 0.11335533 0.45800343 ... 0.15124268 0.28515786 1.        ]]

fitting date: 2021-01-08 00:00:00, correlation matrix:
[[1.         0.26263353 0.21670038 ... 0.26932472 0.33672786 0.16073783]
 [0.26263353 1.         0.23304742 ... 0.29520966 0.41469945 0.16328013]
 [0.21670038 0.23304742 1.         ... 0.2171963  0.30843546 0.47078713]
 ...
 [0.26932472 0.29520966 0.2171963  ... 1.         0.37279705 0.15226454]
 [0.33672786 0.41469945 0.30843546 ... 0.37279705 1.         0.26804008]
 [0.16073783 0.16328013 0.47078713 ... 0.15226454 0.26804008 1.        ]]

fitting date: 2022-01-10 00:00:00, correlation matrix:
[[1.         0.26520473 0.16530195 ... 0.2467958  0.26536886 0.12163457]
 [0.26520473 1.         0.24837493 ... 0.32711292 0.44485892 0.17722045]
 [0.16530195 0.24837493 1.         ... 0.17684218 0.25438244 0.45194714]
 ...
 [0.2467958  0.32711292 0.17684218 ... 1.         0.36129222 0.1092786 ]
 [0.26536886 0.44485892 0.25438244 ... 0.36129222 1.         0.21916796]
 [0.12163457 0.17722045 0.45194714 ... 0.1092786  0.21916796 1.        ]]

fitting date: 2023-01-12 00:00:00, correlation matrix:
[[1.         0.32303348 0.16946741 ... 0.29402492 0.2826605  0.15276508]
 [0.32303348 1.         0.30050247 ... 0.40269805 0.48868892 0.23283978]
 [0.16946741 0.30050247 1.         ... 0.20642493 0.27499455 0.47781056]
 ...
 [0.29402492 0.40269805 0.20642493 ... 1.         0.39341918 0.1329136 ]
 [0.2826605  0.48868892 0.27499455 ... 0.39341918 1.         0.24694012]
 [0.15276508 0.23283978 0.47781056 ... 0.1329136  0.24694012 1.        ]]

fitting date: 2024-01-17 00:00:00, correlation matrix:
[[1.         0.33219196 0.15426941 ... 0.27997159 0.27046316 0.13835175]
 [0.33219196 1.         0.30707339 ... 0.44573637 0.51498306 0.24933645]
 [0.15426941 0.30707339 1.         ... 0.18496381 0.26589756 0.46847979]
 ...
 [0.27997159 0.44573637 0.18496381 ... 1.         0.40843087 0.147931  ]
 [0.27046316 0.51498306 0.26589756 ... 0.40843087 1.         0.24424112]
 [0.13835175 0.24933645 0.46847979 ... 0.147931   0.24424112 1.        ]]

fitting date: 2025-01-21 00:00:00, correlation matrix:
[[1.         0.32144924 0.15336631 ... 0.27221015 0.25305857 0.14491005]
 [0.32144924 1.         0.30775105 ... 0.45122881 0.5131643  0.26961728]
 [0.15336631 0.30775105 1.         ... 0.17882451 0.25748707 0.47164354]
 ...
 [0.27221015 0.45122881 0.17882451 ... 1.         0.40739566 0.15282154]
 [0.25305857 0.5131643  0.25748707 ... 0.40739566 1.         0.24952815]
 [0.14491005 0.26961728 0.47164354 ... 0.15282154 0.24952815 1.        ]]
```


### 3.3 Simulated Portfolio Returns


Assume an equally weighted portfolio of all S&P 500 constituent stocks and calculate the 1% and 99% quantiles of the simulated portfolio returns at each refitting date. We can see that the copula-based simulations produce more extreme quantiles compared to the independence assumption, reflecting the impact of dependencies among asset returns.

```plaintext
fitting date: 2020-01-08 00:00:00, 
 independent 1% quantile: -0.00037, copula 1% quantile: -0.01014, 
 independent 99% quantile: 0.00180, copula 99% quantile: 0.01154
fitting date: 2021-01-08 00:00:00, 
 independent 1% quantile: -0.00065, copula 1% quantile: -0.01148, 
 independent 99% quantile: 0.00219, copula 99% quantile: 0.01308
fitting date: 2022-01-10 00:00:00, 
 independent 1% quantile: -0.00051, copula 1% quantile: -0.01067, 
 independent 99% quantile: 0.00214, copula 99% quantile: 0.01234
fitting date: 2023-01-12 00:00:00, 
 independent 1% quantile: -0.00061, copula 1% quantile: -0.01175, 
 independent 99% quantile: 0.00219, copula 99% quantile: 0.01335
fitting date: 2024-01-17 00:00:00, 
 independent 1% quantile: -0.00067, copula 1% quantile: -0.01213, 
 independent 99% quantile: 0.00214, copula 99% quantile: 0.01363
fitting date: 2025-01-21 00:00:00, 
 independent 1% quantile: -0.00063, copula 1% quantile: -0.01224, 
 independent 99% quantile: 0.00218, copula 99% quantile: 0.01378
```


### 3.4 VaR Estimates

We estimate 1-day 99% portfolio VaR for both the copula-based model and an independent-returns benchmark over 1,462 trading days from 2020-01-08 to 2025-01-21. For each refit date, we simulate multi-asset return paths, form an equally-weighted portfolio of the S&P 500 constituents, and take the 1% quantile of simulated portfolio returns as the VaR estimate. This produces two daily VaR series, denoted $\mathrm{VaR}_t^{\text{copula}}$ and $\mathrm{VaR}_t^{\text{indep}}$.

Over this sample, the level and distribution of the two VaR series differ sharply. The copula-based VaR is much more conservative, with an average 1-day VaR of about $-3.16\%$ and a median of $-3.15\%$; the 5th and 95th percentiles are roughly $-3.98\%$ and $-2.40\%$, and the most extreme values range from $-5.21\%$ to $-1.98\%$. By contrast, the independent-returns VaR is very tight around zero, with an average of about $-0.20\%$, a median of $-0.20\%$, and a 5–95% range of approximately $-0.25\%$ to $-0.13\%$. The difference $\mathrm{VaR}_t^{\text{copula}} - \mathrm{VaR}_t^{\text{indep}}$ is always negative in our sample (mean $\approx$ $-2.96\%$), implying that on every day the copula model assigns a larger potential loss than the independent benchmark.

These differences in VaR levels are consistent with the exception statistics discussed in the backtesting section. Using 99% VaR, the copula model generates 33 exceptions over 1,462 days (about 2.3%), which is only slightly above the nominal 1% rate. In contrast, the independent-returns VaR produces 549 exceptions (about 37.6%), indicating a severe underestimation of tail risk. Together, the VaR estimates and exception rates suggest that the copula-based model captures joint downside risk much more realistically than the independent-returns benchmark.


### 3.5 Backtesting Results
To evaluate the out-of-sample performance of our 1-day 99% portfolio VaR, we apply the Basel traffic-light framework to simulated and historical returns. For each day we compare the realized equally-weighted S&P 500 portfolio return to two VaR models: a copula-based model that preserves the cross-sectional dependence across 503 assets, and a benchmark model that assumes assets are independent with the same marginal distributions. We then count “exceptions” – days when the realized loss is larger than the predicted 99% VaR – and classify 250-day windows into green ($\leq$ 4 exceptions), yellow (5–9) and red ($\geq$ 10) zones.

Over the five non-overlapping 250-day blocks from January 2020 to January 2024, the independent model performs very poorly: each block records between roughly 75 and 118 exceptions, far above the Basel upper bound of 9, so the independent VaR is always in the red zone. This indicates that ignoring dependence leads to severe underestimation of portfolio tail risk. The copula model performs substantially better. In three of the five 250-day blocks it produces zero exceptions, placing those windows firmly in the green zone and implying that the VaR is conservative in “normal” periods. However, in two stress-heavy blocks the copula model still generates 20 and 10 exceptions respectively, which pushes those windows into the red zone and reveals that the fitted dependence structure does not fully capture extreme joint moves observed during turbulent markets.

| Block | Period (start–end)      | days | Copula: exceptions | Copula TL | Indep.: exceptions | Indep. TL |
|:-----:|:------------------------|----:|------------------:|:---------:|------------------:|:---------:|
| 1     | 2020-01-08 – 2021-01-08 | 250  | 20                 | **Red**   | 97                 | **Red**   |
| 2     | 2021-01-08 – 2022-01-10 | 250  | 0                  | **Green** | 75                 | **Red**   |
| 3     | 2022-01-10 – 2023-01-12 | 250  | 10                 | **Red**   | 118                | **Red**   |
| 4     | 2023-01-12 – 2024-01-17 | 250  | 0                  | **Green** | 94                 | **Red**   |
| 5     | 2024-01-17 – 2025-01-21 | 250  | 0                  | **Green** | 83                 | **Red**   |


Aggregating over the entire backtest horizon (about 1,460 trading days from 2020-01-08 onward), the copula VaR records 33 breaches while the independent model records 549. Both models therefore fall into the red zone when viewed over the full sample, but the magnitude of improvement is clear: modeling dependence via a copula dramatically reduces, though does not eliminate, VaR violations. Overall, the backtest suggests that the copula-based approach is meaningfully more realistic than the independence benchmark and can provide useful risk estimates in typical market conditions, yet it remains vulnerable during extreme episodes, where additional features such as time-varying volatility, regime shifts or more flexible tail dependence would likely be needed to achieve regulatory-grade performance.

# Bibliography

McNeil, A. J., Frey, R., & Embrechts, P. (2015). Quantitative risk management: Concepts, techniques and tools (Revised edition). Princeton University Press.


# Appendix: Code Implementation
## Import Libraries and Data Preprocessing

```python

#import necessary libraries
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import argparse
import time
import multiprocessing
import matplotlib.pyplot as plt
from io import StringIO
from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF
from pathlib import Path
from typing import Dict, Tuple
from scipy.stats import kendalltau, t, multivariate_t, multivariate_normal, norm
from scipy.special import gammaln
from scipy.optimize import minimize

#data download and preprocessing
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text
    tbl = pd.read_html(StringIO(html))[0]
    syms = (tbl["Symbol"].astype(str)
                    .str.replace(".", "-", regex=False)
                    .str.upper())
    return sorted(syms.unique())

tickers = get_sp500_tickers()

raw = yf.download(
    tickers=" ".join(tickers),
    start=START_DATE,
    end=END_DATE,
    interval="1d",
    auto_adjust=True,
    group_by="ticker",
    threads=True,
    progress=False
)

if isinstance(raw.columns, pd.MultiIndex):
    prices = raw.xs("Close", axis=1, level=-1)
else:
    col = "Close" if "Close" in raw.columns else raw.columns[0]
    prices = raw[[col]].rename(columns={col: tickers[0]})

prices = prices.sort_index().dropna(how="all")
prices = prices.loc[:, prices.notna().sum() > 0]
prices.to_csv("sp500_prices.csv", float_format="%.6f")

logret = np.log(prices / prices.shift(1)).dropna(how="all")
logret.to_csv("sp500_log_returns.csv", float_format="%.8f")
```


## Fitting Marginal Distributions

```python
def fit_normal(x: np.ndarray) -> Tuple[float, float]:
    # MLE for normal equals sample mean/std (unbiased std is close; we use MLE style with ddof=0)
    mu = np.mean(x)
    sigma = np.std(x, ddof=0)
    # guard against zero sigma
    sigma = sigma if sigma > 1e-12 else 1e-12
    return mu, sigma

def fit_t(x: np.ndarray) -> Tuple[float, float, float]:
    # scipy.stats.t.fit returns (df, loc, scale)
    df, loc, scale = stats.t.fit(x)
    # guard
    if scale <= 1e-12:
        scale = 1e-12
    if df < 2.01:
        df = 2.01  # ensure finite variance
    return df, loc, scale

def loglik_normal(x: np.ndarray, mu: float, sigma: float) -> float:
    return np.sum(stats.norm.logpdf(x, loc=mu, scale=sigma))

def loglik_t(x: np.ndarray, df: float, loc: float, scale: float) -> float:
    return np.sum(stats.t.logpdf(x, df=df, loc=loc, scale=scale))

def aic(loglik: float, k: int) -> float:
    return 2*k - 2*loglik

def bic(loglik: float, k: int, n: int) -> float:
    return k*np.log(n) - 2*loglik

def gof_pvalues(x: np.ndarray, cdf_callable) -> Tuple[float, float]:
    # KS test with fitted CDF
    ks_stat, ks_p = stats.kstest(x, cdf_callable)
    # Cramer–von Mises test
    cvm_res = stats.cramervonmises(x, cdf_callable)
    cvm_p = getattr(cvm_res, 'pvalue', np.nan)
    return ks_p, cvm_p

def empirical_pit(x: np.ndarray) -> np.ndarray:
    # Plotting position (rank - 0.5)/n to avoid 0/1
    order = np.argsort(x)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(x)+1)
    u = (ranks - 0.5)/len(x)
    return u

def run_fitting(returns: pd.DataFrame):
    """Fit marginals for each column in returns and produce PIT DataFrame and best_table.

    Returns:
        pit_df (pd.DataFrame): uniform PIT series per ticker (aligned to returns.index)
        best_table (pd.DataFrame): per-ticker model info and parameters
    """
    best_rows = []
    # collect per-column PIT Series and concat at the end to avoid frame fragmentation
    col_series = []
    for col in returns.columns:
        x = returns[col].dropna().values
        n = len(x)
        if n < 20:
            continue  # skip very short series

        # Fit Normal
        mu, sigma = fit_normal(x)
        ll_norm = loglik_normal(x, mu, sigma)
        aic_norm = aic(ll_norm, k=2)
        bic_norm = bic(ll_norm, k=2, n=n)
        norm_cdf = lambda s: stats.norm.cdf(s, loc=mu, scale=sigma)
        ks_norm, cvm_norm = gof_pvalues(x, norm_cdf)

        # Fit Student‑t
        df_t, loc_t, scale_t = fit_t(x)
        ll_t = loglik_t(x, df_t, loc_t, scale_t)
        aic_t = aic(ll_t, k=3)
        bic_t = bic(ll_t, k=3, n=n)
        t_cdf = lambda s: stats.t.cdf(s, df=df_t, loc=loc_t, scale=scale_t)
        ks_t, cvm_t = gof_pvalues(x, t_cdf)

        # Select between Normal and t using information criteria first
        choose_t = (aic_t + bic_t) < (aic_norm + bic_norm)
        # bias towards t if very heavy tails
        if df_t < 10 and (abs(aic_t - aic_norm) + abs(bic_t - bic_norm)) < 10:
            choose_t = True

        # If both tests reject badly for both, use Empirical
        both_bad = ( (ks_norm < 0.05 and cvm_norm < 0.05) and (ks_t < 0.05 and cvm_t < 0.05) )

        if both_bad:
            best = 'empirical'
            params = {}
            # PIT via empirical ranks (aligned to original index)
            series = returns[col].dropna()
            u = pd.Series(empirical_pit(series.values), index=series.index)
        else:
            if choose_t:
                best = 't'
                params = {'df': df_t, 'loc': loc_t, 'scale': scale_t}
                series = returns[col].dropna()
                u = pd.Series(stats.t.cdf(series.values, df=df_t, loc=loc_t, scale=scale_t), index=series.index)
            else:
                best = 'normal'
                params = {'mu': mu, 'sigma': sigma}
                series = returns[col].dropna()
                u = pd.Series(stats.norm.cdf(series.values, loc=mu, scale=sigma), index=series.index)

        # Clip to (1e-6, 1-1e-6) to avoid exact 0/1
        u = u.clip(1e-6, 1-1e-6)
        # append the Series (keeps its index); will concat later
        col_series.append(u.rename(col))

        best_rows.append({
            'ticker': col,
            'n': n,
            'best_model': best,
            'mu': params.get('mu', np.nan),
            'sigma': params.get('sigma', np.nan),
            't_df': params.get('df', np.nan),
            't_loc': params.get('loc', np.nan),
            't_scale': params.get('scale', np.nan),
            'll_norm': ll_norm,
            'aic_norm': aic_norm,
            'bic_norm': bic_norm,
            'ks_p_norm': ks_norm,
            'cvm_p_norm': cvm_norm,
            'll_t': ll_t,
            'aic_t': aic_t,
            'bic_t': bic_t,
            'ks_p_t': ks_t,
            'cvm_p_t': cvm_t
        })

    # build pit_df by concatenating per-column Series (this is fast and avoids fragmentation)
    if col_series:
        pit_df = pd.concat(col_series, axis=1)
        # ensure we have the original full index (missing values remain NaN)
        pit_df = pit_df.reindex(index=returns.index)
    else:
        pit_df = pd.DataFrame(index=returns.index)

    best_table = pd.DataFrame(best_rows).set_index('ticker').sort_values(['best_model','n'], ascending=[True, False])

    return pit_df, best_table

def main_distr_fitting(returns, save: bool = True, out_dir: Path = None, use_parquet: bool = False, best_output: Path = None):
    """Load returns from path, run fitting, and optionally save/print PIT DataFrame and best_table.

    Args:
        returns: pd.DataFrame of returns data.
        save: whether to save outputs to disk.
        out_dir: directory to save outputs (defaults to input parent).
        use_parquet: if True try to save PIT as parquet (pyarrow required).
        best_output: explicit path for best_table CSV (overrides out_dir).

    Returns:
        pit_df, best_table
    """
    pit_df, best_table = run_fitting(returns)

    if save:
        out_dir = Path(out_dir) if out_dir is not None else path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        # save PIT
        pit_path = out_dir / 'pit_uniform.parquet' if use_parquet else out_dir / 'pit_uniform.csv.gz'
        try:
            if use_parquet:
                pit_df.to_parquet(pit_path)
            else:
                pit_df.to_csv(pit_path, compression='gzip')
        except Exception as e:
            # fallback to csv if parquet failed
            if use_parquet:
                pit_path = out_dir / 'pit_uniform.csv.gz'
                pit_df.to_csv(pit_path, compression='gzip')
            else:
                raise

        # save best_table
        if best_output is None:
            best_path = out_dir / 'best_marginals.csv'
        else:
            best_path = Path(best_output)
            best_path.parent.mkdir(parents=True, exist_ok=True)
        best_table.to_csv(best_path)

    return pit_df, best_table

def inverse_gaussian(u: np.array, mu: float, sigma: float):
    return norm.ppf(u, loc=mu, scale=sigma)

def inverse_t(u: np.array, nu: float, mu: float = 0, sigma: float = 1):
    return t.ppf(u, df=nu, loc=mu, scale=sigma)

def inverse_empirical(u: np.array, data: np.array):
    """
    Inverse empirical CDF using linear interpolation.
    Parameters
    ----------
    u : np.array
        Uniform variables in (0,1).
    data : np.array
        Original data to build empirical CDF.
    Returns
    -------
    np.array
        Transformed variables on original scale.
    """
    sorted_data = np.sort(data)
    n = len(sorted_data)
    ecdf_values = np.arange(1, n + 1) / n
    return np.interp(u, ecdf_values, sorted_data)
```

## Fitting Copula Models

```python

def nearest_positive_definite_corr(matrix):
    matrix = (matrix + matrix.T) / 2
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals_clipped = np.clip(eigvals, 1e-6, None)
    clipped_matrix = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    diag = np.sqrt(np.diag(clipped_matrix))
    corr = clipped_matrix / np.outer(diag, diag)
    np.fill_diagonal(corr, 1.0)
    return corr

def correlation_matrix(data: np.array) -> np.array:
    """
    Computes the correlation matrix from the given data.
    Parameters
    ----------
    data : np.array
        Input data of shape (n_samples, n_assets). Each column represents cdf values for an asset returns
    Returns
    -------
    np.array
        Correlation matrix of shape (n_assets, n_assets).
    """
    # Kendall's tau correlation
    n_assets = data.shape[1]
    corr_matrix = np.zeros((n_assets, n_assets))
    for i in range(n_assets):
        for j in range(n_assets):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                tau, _ = kendalltau(data[:, i], data[:, j])
                corr_matrix[i, j] = np.sin(np.pi / 2 * tau)

    # If needed, ensure the correlation matrix is positive definite
    corr_matrix = nearest_positive_definite_corr(corr_matrix)
    return corr_matrix


def log_likelihood_gaussian(corr_matrix: np.array,
                            data: np.array
                            ) -> float:
    """
    Computes the log-likelihood for the Gaussian copula.
    Parameters
    ----------
    corr_matrix : np.array
        Correlation matrix of shape (n_assets, n_assets).
    data : np.array
        Pseudo-observations of shape (n_samples, n_assets).
    Returns
    -------
    float
        Log-likelihood value.
    """
    x = norm.ppf(data)
    mvn_logpdf = multivariate_normal.logpdf(x, mean=np.zeros(corr_matrix.shape[0]), cov=corr_matrix)
    marginal_logpdf = norm.logpdf(x)
    ll = np.sum(mvn_logpdf) - np.sum(marginal_logpdf)
    return ll


def log_likelihood_t(nu: float,
                     corr_matrix: np.array,
                     data: np.array
                     ) -> float:
    """
    Computes the log-likelihood for the t copula.
    Parameters
    ----------
    nu : float
        Degrees of freedom for the t copula.
    corr_matrix : np.array
        Correlation matrix of shape (n_assets, n_assets).
    data : np.array
        Pseudo-observations of shape (n_samples, n_assets).
    Returns
    -------
    float
        Log-likelihood value.
    """
    x = t.ppf(data, df=nu)

    # multivariate t density
    mvn_logpdf = multivariate_t.logpdf(x, df=nu, shape=corr_matrix)

    # marginal t densities
    marginal_logpdf = t.logpdf(x, df=nu)

    ll = np.sum(mvn_logpdf) - np.sum(marginal_logpdf)

    return ll


def fit_t_copula(data: np.array, 
                 corr_matrix: np.array,
                 initial_nu: float = 3.0
                 ) -> float:    
    """
    Fits the t copula to the given data by estimating the degrees of freedom.
    Parameters
    ----------
    data : np.array
        Pseudo-observations of shape (n_samples, n_assets).
    initial_nu : float
        Initial guess for the degrees of freedom.
    Returns
    -------
    float
        Estimated degrees of freedom for the t copula.
    """     
    def neg_log_likelihood(nu):
        return -log_likelihood_t(nu, corr_matrix, data)    
    result = minimize(neg_log_likelihood, 
                      x0=np.array([initial_nu]), 
                      bounds=[(2.01, 30.0)],
                      method='L-BFGS-B')
    fitted_nu = result.x[0]
    return fitted_nu


def model_selection(data: np.array) -> tuple:
    """
    Selects the best copula model (Gaussian or t) based on AIC and BIC.
    Parameters
    ----------
    data : np.array
        Pseudo-observations of shape (n_samples, n_assets).
    Returns
    -------
    tuple
        (best_copula_name, correlation_matrix, fitted_nu)
    """ 

    t = time.time()
    corr = correlation_matrix(data)
    print(f"Correlation matrix computed in {time.time() - t} seconds")
    t = time.time()
    ll_gaussian = log_likelihood_gaussian(corr, data)
    n_params_gaussian = data.shape[1] * (data.shape[1] - 1) // 2 + data.shape[1]
    aic_gaussian = aic(ll_gaussian, n_params_gaussian)
    bic_gaussian = bic(ll_gaussian, n_params_gaussian)
    print(f"Gaussian log-likelihood computed in {time.time() - t} seconds")
    t = time.time()

    fitted_nu = fit_t_copula(data, corr)
    print(f"t copula fitting computed in {time.time() - t} seconds")
    t = time.time()
    ll_t = log_likelihood_t(fitted_nu, corr, data)
    n_params_t = n_params_gaussian + 1
    aic_t = aic(ll_t, n_params_t)
    bic_t = bic(ll_t, n_params_t)
    print(f"t copula log-likelihood computed in {time.time() - t} seconds")

    if aic_gaussian < aic_t and bic_gaussian < bic_t:
        return ('gaussian', corr, None)
    else:
        return ('t', corr, fitted_nu)


def generator_archimedean(theta: float,
                          data: np.array,
                          copula: str) -> np.array:
    """
    Computes the generator function for Archimedean copulas (Clayton, Gumbel, Frank).

    Parameters
    ----------
    theta : float
        Parameter for the Archimedean copula.       
    data : np.array
        Pseudo-observations of shape (n_samples, n_assets).
    copula : str
        Type of Archimedean copula ('clayton', 'gumbel', 'frank').
    Returns
    -------
    np.array
        The transformed data using the generator function. (n_samples, n_assets)

    """
    if copula == 'clayton':
        return (data ** (-theta) - 1) / theta
    elif copula == 'gumbel':
        return (-np.log(data)) ** theta
    elif copula == 'frank':
        return -np.log((np.exp(-theta * data) - 1) / (np.exp(-theta) - 1))
    else:
        raise ValueError("Unsupported copula type. Choose from 'clayton', 'gumbel', 'frank'.")

def Inverse_generator_archimedean(theta: float,
                              t_values: np.array,
                              copula: str) -> np.array:
    """
    Computes the inverse generator function for Archimedean copulas (Clayton, Gumbel, Frank).

    Parameters
    ----------
    theta : float
        Parameter for the Archimedean copula.
    t_values : np.array
        Transformed data using the generator function. (n_samples, n_assets).
    copula : str
        Type of Archimedean copula ('clayton', 'gumbel', 'frank').
    Returns 
    -------
    np.array
        The pseudo-observations of shape (n_samples, n_assets).
    """
    if copula == 'clayton':
        return (1 + theta * t_values) ** (-1 / theta)
    elif copula == 'gumbel':
        return np.exp(-t_values ** (1 / theta))
    elif copula == 'frank':
        return - (1 / theta) * np.log(1 + np.exp(-t_values) * (np.exp(-theta) - 1))
    else:
        raise ValueError("Unsupported copula type. Choose from 'clayton', 'gumbel', 'frank'.")

def diff_generator_archimedean(theta: float,
                                    data: np.array,
                                    copula: str
                                    ) -> np.array:
    """
    Computes the differential of the generator function for Archimedean copulas (Clayton, Gumbel, Frank).
    Parameters
    ----------
    theta : float
        Parameter for the Archimedean copula.
    data : np.array
        Pseudo-observations of shape (n_samples, n_assets).
    copula : str
        Type of Archimedean copula ('clayton', 'gumbel', 'frank').
    Returns
    -------
    np.array
        The differential values of shape (n_samples, n_assets).
    """
    eps = 1e-6 
    if copula == 'clayton':
        return -(data ** (-theta - 1))
    elif copula == 'gumbel':
        log_data = np.clip(-np.log(data), eps, None)
        return theta * log_data ** (theta - 1) / data
    elif copula == 'frank':
        exp_neg_theta = np.exp(-theta)
        exp_neg_theta_data = np.exp(-theta * data)
        return (theta * exp_neg_theta_data) / ( (exp_neg_theta_data - 1) * (exp_neg_theta - 1) )
    else:
        raise ValueError("Unsupported copula type. Choose from 'clayton', 'gumbel', 'frank'.")

def log_likelihood_archimedean(theta: float, data: np.array, copula: str) -> float:
    """
    Log-likelihood for Archimedean Copulas using analytical forms.
    """
    n_samples, d = data.shape
    phi_u = generator_archimedean(theta, data, copula)      # shape (n_samples, d)
    sum_phi = np.sum(phi_u, axis=1)                         # shape (n_samples,)
    log_phi_prime = np.sum(np.log(diff_generator_archimedean(theta, data, copula)), axis=1)

    if copula == 'clayton':
        
        term1 = np.sum(np.log(theta + np.arange(1, d)))
        term2 = -(1 / theta + d) * np.log(sum_phi)
        log_c = term1 + term2 + log_phi_prime

    elif copula == 'gumbel':
        A = (-1) ** (d - 1)
        term1 = np.log(A * theta ** (-d + 1)) + gammaln(d + 1)
        log_sum_phi = np.log(sum_phi)
        log_phi_d = term1 + (d - 1) * np.log(log_sum_phi) - sum_phi ** (1 / theta)
        log_c = log_phi_d + log_phi_prime

    elif copula == 'frank':
        exp_neg_theta = np.exp(-theta)
        exp_neg_sum_phi = np.exp(-sum_phi)
        numerator = theta * exp_neg_sum_phi
        denominator = (exp_neg_sum_phi - 1) * (exp_neg_theta - 1)
        log_phi_d = np.log(numerator) - np.log(denominator)
        log_c = log_phi_d + log_phi_prime

    else:
        raise ValueError("Unsupported copula.")

    return np.sum(log_c)

def main_copula(returns: pd.DataFrame,
         train_days: int,
         sim_days: int,
         refit_freq: int,
         n_paths: int,
         random_state: int = None,
         ) -> (dict, dict, dict):
    
    """
    main function to fit copula model and simulate paths.
    Parameters
    ----------
    returns : np.array
        Historical return data for the assets with index as time and columns as assets.
    fitted_distributions : dict
        Dictionary of fitted marginal distributions for each asset. with keys as asset names and values as distribution objects.
    train_days : int
        Number of days to use for training the copula model.
    sim_days : int
        Number of days to simulate.
    refit_freq : int
        Number of days between refitting the copula model.
    n_paths : int
        Number of simulation paths.
    random_state : int, optional
        Random seed for reproducibility.    
    Returns
    -------
    (all_simulated_paths, model_info, all_independent_paths) : (dict, dict, dict)
        all_simulated_paths : dict
            Dictionary with keys as fit date and values as simulated return paths (np.array) of shape (n_paths, sim_days, n_assets).
        model_info : dict
            Dictionary with keys as fit date and values as another dict containing:
                'best_distributions' : pd.DataFrame
                    DataFrame of best fitted distributions for each asset.
                'copula_name' : str
                    Name of the selected copula model.
                'corr_matrix' : np.array
                    Correlation matrix used in the copula model.
                'nu' : float or None
                    Degrees of freedom for t-copula, None for Gaussian copula.  

        all_independent_paths : dict
            Dictionary with keys as fit date and values as simulated return paths (np.array) from independent model of shape (n_paths, sim_days, n_assets).

    """

    model_info = {}
    all_simulated_paths = {}
    all_independent_paths = {}

    total_iterations = (len(returns) - train_days) // refit_freq + 1

    # Prepare arguments for parallel processing
    args_list = []
    for i in range(total_iterations):
        start_idx = i * refit_freq
        end_idx = start_idx + train_days
        train_data = returns.iloc[start_idx:end_idx]
        args_list.append((train_data, random_state, n_paths, returns.shape[1], sim_days))
    # Use multiprocessing to parallelize the fitting and simulation
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(one_load, args_list)
    for i, (simulated_returns, independent_returns, best_table, copula_name, corr_mat, nu) in enumerate(results):
        fit_date = returns.index[i * refit_freq + train_days - 1]
        all_simulated_paths[fit_date] = simulated_returns
        all_independent_paths[fit_date] = independent_returns
        model_info[fit_date] = {
            'best_distributions': best_table,
            'copula_name': copula_name,
            'corr_matrix': corr_mat,
            'nu': nu
        }

    return all_simulated_paths, model_info, all_independent_paths


def one_load(train_data: pd.DataFrame,
             random_state: int,
             n_paths: int,
             n_assets: int,
             sim_days: int):
    time_start = time.time()
    print("starting distribution fitting...")
    # Fit distribution to each asset's returns
    pit_df, best_table = main_distr_fitting(returns=train_data, 
                    save=False,                   
                    use_parquet=False,
                    best_output=True)

    print(f"Time elapsed: {time.time() - time_start} seconds")
    time_start = time.time()

    print("starting copula fitting...")
    # select copula model and fit
    empirical_cdf = pit_df.values
    copula_name, corr_mat, copula_nu = model_selection(empirical_cdf)

    print(f"Selected copula: {copula_name}")
    print(f"time elapsed: {time.time() - time_start} seconds")
    time_start = time.time()
    
    print("starting simulation...")

    # Simulate paths
    if copula_name == 'gaussian':
        simulated_paths = simulate_gaussian_copula(n_paths=n_paths,
                                                    corr_matrix=corr_mat,
                                                    n_assets=n_assets,
                                                    n_steps=sim_days,
                                                    random_state=random_state)
    elif copula_name == 't':
        simulated_paths = simulate_t_copula(n_paths=n_paths,
                                            df=copula_nu,
                                            corr_matrix=corr_mat,
                                            n_assets=n_assets,
                                            n_steps=sim_days,
                                            random_state=random_state)
    else:
        raise NotImplementedError(f"Copula model '{copula_name}' not implemented in simulation.")

    # Simulate independent paths as a benchmark
    independent_paths = simulate_independent(n_paths=n_paths,
                                                n_assets=n_assets,
                                                n_steps=sim_days,
                                                random_state=random_state)

    
    print(f"time elapsed: {time.time() - time_start} seconds")
    time_start = time.time()

    print("starting inverse CDF transformation...")
    # Transform simulated uniform variables back to original scale using inverse CDFs
    simulated_returns = np.zeros_like(simulated_paths)
    independent_returns = np.zeros_like(independent_paths)
    for asset_idx, asset_name in enumerate(train_data.columns):

        distr = best_table.loc[asset_name, 'best_model']

        if distr == 'normal':
            mu = best_table.loc[asset_name, 'mu']
            sigma = best_table.loc[asset_name, 'sigma']
            simulated_returns[:, :, asset_idx] = inverse_gaussian(simulated_paths[:, :, asset_idx], mu, sigma)
            independent_returns[:, :, asset_idx] = inverse_gaussian(independent_paths[:, :, asset_idx], mu, sigma)
        elif distr == 't':
            nu = int(best_table.loc[asset_name, 't_df'])
            mu = best_table.loc[asset_name, 't_loc']
            sigma = best_table.loc[asset_name, 't_scale']
            simulated_returns[:, :, asset_idx] = inverse_t(simulated_paths[:, :, asset_idx], nu, mu, sigma)
            independent_returns[:, :, asset_idx] = inverse_t(independent_paths[:, :, asset_idx], nu, mu, sigma)

        elif distr == 'empirical':
            simulated_returns[:, :, asset_idx] = inverse_empirical(simulated_paths[:, :, asset_idx], train_data[asset_name].values)
            independent_returns[:, :, asset_idx] = inverse_empirical(independent_paths[:, :, asset_idx], train_data[asset_name].values)
        else:
            raise NotImplementedError(f"Distribution '{distr}' not implemented in inverse CDF transformation.")


    print(f"time elapsed: {time.time() - time_start} seconds")
    time_start = time.time()

    return simulated_returns, independent_returns, best_table, copula_name, corr_mat, copula_nu
```

## Monte Carlo Simulation Using Copulas

```python
def simulate_independent(n_paths: int,
                         n_assets: int,
                         n_steps: int,
                         random_state: int = None
                         ) -> np.array:
    """
    Simulates paths assuming independence among assets.
    Parameters
    ----------
    n_paths : int
        Number of simulation paths.
    n_assets : int
        Number of assets.
    n_steps : int
        Number of time steps.
    Returns
    -------
    np.array
        Simulated paths.
    """
    if random_state is not None:
        np.random.seed(random_state)

    U = np.random.uniform(size=(n_paths, n_steps, n_assets))

    return U

def simulate_multivariate_normal(n_paths: int,
                                    corr_matrix: np.array,
                                    n_assets: int,
                                    n_steps: int,
                                    random_state: int = None
                                    ) -> np.array:
        """
        Simulates paths using a multivariate normal distribution.
        Parameters
        ----------
        n_paths : int
            Number of simulation paths.
        corr_matrix : np.array
            Correlation matrix for the assets. Should be positive definite.
        n_assets : int
            Number of assets.
        n_steps : int
            Number of time steps.
        Returns
        -------
        np.array
            Simulated paths.
        """
        if random_state is not None:
            np.random.seed(random_state)
    
        # Simulate independent standard normal variables
        Z = np.random.normal(size=(n_paths, n_steps, n_assets))
        
        # Cholesky decomposition
    
        if np.all(np.linalg.eigvals(corr_matrix) > 0) is False:
            raise ValueError("Correlation matrix must be positive definite.")
    
        L = np.linalg.cholesky(corr_matrix)
    
        # Multiply by the Cholesky factor to introduce correlation
        X = np.einsum('mjk, kl -> mjl', Z, L.T) 
    
        return X


def simulate_gaussian_copula(n_paths: int,
                             corr_matrix: np.array,
                             n_assets: int,
                             n_steps: int,
                             random_state: int = None
                             ) -> np.array:
    """
    Simulates paths using a Gaussian copula.
    Parameters
    ----------
    n_paths : int
        Number of simulation paths.
    delta_t : float
        Time increment for each step.
    corr_matrix : np.array
        Correlation matrix for the assets. Should be positive definite.
    n_assets : int
        Number of assets.
    n_steps : int
        Number of time steps.
    Returns
    -------
    np.array
        Simulated paths.
    """
    X = simulate_multivariate_normal(n_paths,
                                     corr_matrix,
                                     n_assets,
                                     n_steps,
                                     random_state) 

    # CDF transformation to uniform

    U = norm.cdf(X)

    return U


def simulate_t_copula(n_paths: int,
                     corr_matrix: np.array,
                     n_assets: int,
                     n_steps: int,
                     df: int,
                     random_state: int = None
                     ) -> np.array:
    """
    Simulates paths using a t-copula.
    Parameters
    ----------
    n_paths : int
        Number of simulation paths.
    delta_t : float
        Time increment for each step.
    corr_matrix : np.array
        Correlation matrix for the assets. Should be positive definite.
    n_assets : int
        Number of assets.
    n_steps : int
        Number of time steps.
    df : int
        Degrees of freedom for the t-distribution.
    Returns
    -------
    np.array
        Simulated paths.
    """

    X = simulate_multivariate_normal(n_paths,
                                     corr_matrix,
                                     n_assets,
                                     n_steps,
                                     random_state)
    # Scale by chi-squared distribution to get t-distribution
    chi2_samples = np.random.chisquare(df, size=(n_paths, n_steps, 1))
    X_t = X / np.sqrt(chi2_samples / df)

    # CDF transformation to uniform
    U = t.cdf(X_t, df=df)

    return U


def simulate_clayton_copula(n_paths: int,
                            theta: float,
                            n_assets: int,
                            n_steps: int,
                            random_state: int = None
                            ) -> np.array:
    """
    Simulates paths using a Clayton copula.
    Parameters
    ----------
    n_paths : int
        Number of simulation paths.
    theta : float
        Parameter for the Clayton copula (theta > 0).
    n_assets : int
        Number of assets.
    n_steps : int
        Number of time steps.
    Returns
    -------
    np.array
        Simulated paths.
    """
    if random_state is not None:
        np.random.seed(random_state)

    V = np.random.gamma(1/theta, 1, size=(n_paths, n_steps, 1))
    X = np.random.uniform(size=(n_paths, n_steps, n_assets))
    U = (1 - np.log(X) / V) ** (-1/theta)

    return U


def simulate_gumbel_copula(n_paths: int,
                           theta: float,
                           n_assets: int,
                           n_steps: int,
                           random_state: int = None
                           ) -> np.array:
    """
    Simulates paths using a Gumbel copula.
    Parameters
    ----------
    n_paths : int
        Number of simulation paths.
    theta : float
        Parameter for the Gumbel copula (theta >= 1).
    n_assets : int
        Number of assets.
    n_steps : int
        Number of time steps.
    Returns
    -------
    np.array
        Simulated paths.
    """
    if random_state is not None:
        np.random.seed(random_state)

    alpha = 1 / theta
    beta = 1
    gamma = (np.cos(np.pi / (2 * theta))) ** theta

    # Simulate stable variables
    V = levy_stable.rvs(alpha=alpha, beta=beta, scale=gamma, loc=0,
                        size=(n_paths, n_steps, 1),
                        random_state=random_state)
    X = np.random.uniform(size=(n_paths, n_steps, n_assets))

    U = np.exp(-(-np.log(X) / V) ** (1 / theta))

    return U


def simulate_frank_copula(n_paths: int,
                          theta: float,
                          n_assets: int,
                          n_steps: int,
                          random_state: int = None
                          ) -> np.array:
    """
    Simulates paths using a Frank copula.
    Parameters
    ----------
    n_paths : int
        Number of simulation paths.
    theta : float
        Parameter for the Frank copula (theta != 0).
    n_assets : int
        Number of assets.
    n_steps : int
        Number of time steps.
    Returns
    -------
    np.array
        Simulated paths.
    """
    if random_state is not None:
        np.random.seed(random_state)

    k = np.arange(1, 1000)
    p = (1 - np.exp(-theta))**k/(theta * k)
    p /= p.sum()
    V = np.random.choice(k, size=(n_paths, n_steps, 1), p=p.flatten())

    X = np.random.uniform(size=(n_paths, n_steps, n_assets))

    # Frank inverse generator
    exp_neg_theta = np.exp(-theta)
    numerator = (np.exp(-V * theta) - 1) * (1 - np.exp(-theta * X))
    denominator = 1 - exp_neg_theta
    denominator = np.where(denominator == 0, 1e-10, denominator)  # Avoid division by zero
    term = np.clip(1 + numerator / denominator, a_min=1e-10, a_max=None)
    U = - (1 / theta) * np.log(term)
    U = np.clip(U, 0, 1)
    return U

```

## VaR and Backtesting

```python
def portfolio_var(sim_matrix, alpha=0.01):
    """
    sim_matrix: (n_paths, horizon, n_assets)
    return: VaR series for each horizon day, shape (horizon,)
    """
    portfolio_paths = sim_matrix.mean(axis=2)
    var_series = np.quantile(portfolio_paths, alpha, axis=0)
    return var_series

def tl_color(n_exceptions, window=250):
    """
    Basel traffic-light thresholds for 250 days, 99% VaR.
    If window != 250, we still use the same cutoffs for illustration.
    """
    if n_exceptions <= 4:
        return "Green"
    elif n_exceptions <= 9:
        return "Yellow"
    else:
        return "Red"


```


## Main Functions
``` python

sim_days = 253
n_paths = 1000
train_days = 252 * 10
refit_days = 253
all_simulated_returns, model_info, all_independent_returns = main_copula(data, n_paths=n_paths, sim_days=sim_days, 
                                                                         random_state=42, train_days=train_days, refit_freq=refit_days)

# Analyze the simulated returns
for key in model_info.keys():
    # print(f"fitting date: {key}, copula: {model_info[key]['copula_name']}, nu: {model_info[key].get('nu', 'N/A')}")
    simulated_matrix = all_independent_returns[key]
    simulated_matrix_copula = all_simulated_returns[key]

    portfolio_returns = np.sum(simulated_matrix, axis=2) / simulated_matrix.shape[2]
    portfolio_returns_copula = np.sum(simulated_matrix_copula, axis=2) / simulated_matrix_copula.shape[2]

    low_quantile = 0.1
    high_quantile = 0.9
    independent_var_low = np.quantile(portfolio_returns, low_quantile)
    independent_var_high = np.quantile(portfolio_returns, high_quantile)
    copula_var_low = np.quantile(portfolio_returns_copula, low_quantile)
    copula_var_high = np.quantile(portfolio_returns_copula, high_quantile)

    print(f"fitting date: {key}, \n independent 1% quantile: {independent_var_low:.5f}, copula 1% quantile: {copula_var_low:.5f}, \n independent 99% quantile: {independent_var_high:.5f}, copula 99% quantile: {copula_var_high:.5f}")

# Analyze the model info
for key in model_info.keys():
    print(f"fitting date: {key}, copula: {model_info[key]['copula_name']}, nu: {model_info[key].get('nu', 'N/A')}")

for key in model_info.keys():
    corr_matrix = model_info[key]['corr_matrix']
    print(f"fitting date: {key}, correlation matrix:\n{corr_matrix}\n")

records = []
sorted_dates = sorted(model_info.keys())
for refit_date in sorted_dates:
    sim_cop = copula_returns[refit_date]
    sim_ind = indep_returns[refit_date]

    n_paths, horizon, n_assets = sim_cop.shape

    var_cop_series = portfolio_var(sim_cop, alpha=0.01)
    var_ind_series = portfolio_var(sim_ind, alpha=0.01)

    if refit_date not in hist_idx:
        continue
    start_pos = hist_idx.get_loc(refit_date)

    for h in range(horizon):
        pos = start_pos + 1 + h
        if pos >= len(hist_idx):
            break

        realized_date = hist_idx[pos]
        realized_ret = hist.iloc[pos].mean()

        records.append({
            "VaR_date": refit_date,
            "Realized_date": realized_date,
            "Realized_return": realized_ret,
            "VaR_copula": var_cop_series[h],
            "VaR_indep": var_ind_series[h]
        })
var_df = pd.DataFrame(records).sort_values("Realized_date").reset_index(drop=True)

var_df["hit_copula"] = (var_df["Realized_return"] < var_df["VaR_copula"]).astype(int)
var_df["hit_indep"]  = (var_df["Realized_return"] < var_df["VaR_indep"]).astype(int)

# last 250 days
window = 250
n_exc_copula_250 = int(var_df["hit_copula"].tail(window).sum())
n_exc_indep_250  = int(var_df["hit_indep"].tail(window).sum())

TL_copula_250 = tl_color(n_exc_copula_250, window)
TL_indep_250  = tl_color(n_exc_indep_250, window)

print("=== Traffic-Light Backtest (last 250 days) ===")
print(f"Copula model:      exceptions = {n_exc_copula_250:3d}, TL = {TL_copula_250}")
print(f"Independent model: exceptions = {n_exc_indep_250:3d}, TL = {TL_indep_250}")

# from 2020-01-08
start_date = pd.Timestamp("2020-01-08")
sub = var_df[var_df["VaR_date"] >= start_date].copy()

n_exc_copula_all = int(sub["hit_copula"].sum())
n_exc_indep_all  = int(sub["hit_indep"].sum())
window_all = len(sub)

TL_copula_all = tl_color(n_exc_copula_all, window_all)
TL_indep_all  = tl_color(n_exc_indep_all, window_all)

print(f"\n=== Traffic-Light Backtest (from {start_date.date()} to end, {window_all} days) ===")
print(f"Copula model:      exceptions = {n_exc_copula_all:3d}, TL = {TL_copula_all}")
print(f"Independent model: exceptions = {n_exc_indep_all:3d}, TL = {TL_indep_all}")

block_size = 250
start_date = pd.Timestamp("2020-01-08")
sub = var_df[var_df["VaR_date"] >= start_date].copy().reset_index(drop=True)

blocks = []
for start in range(0, len(sub), block_size):
    block = sub.iloc[start:start + block_size]
    if len(block) < block_size:
        break

    n_exc_cop = int(block["hit_copula"].sum())
    n_exc_ind = int(block["hit_indep"].sum())

    TL_cop = tl_color(n_exc_cop, block_size)
    TL_ind = tl_color(n_exc_ind, block_size)

    blocks.append({
        "block_id": len(blocks) + 1,
        "start_date": block["VaR_date"].iloc[0],
        "end_date": block["VaR_date"].iloc[-1],
        "n_days": len(block),
        "exc_copula": n_exc_cop,
        "TL_copula": TL_cop,
        "exc_indep": n_exc_ind,
        "TL_indep": TL_ind
    })

blocks_df = pd.DataFrame(blocks)
blocks_df
start_date = pd.Timestamp("2020-01-08")
sub = var_df[var_df["VaR_date"] >= start_date].copy()

print("=== Sample info ===")
print("Number of observations:", len(sub))
print("Start date:", sub["VaR_date"].min())
print("End date:", sub["VaR_date"].max())
print()

for col in ["VaR_copula", "VaR_indep"]:
    s = sub[col]
    print(f"=== {col} ===")
    print("Mean:", s.mean())
    print("Median:", s.median())
    print("5% quantile:", s.quantile(0.05))
    print("95% quantile:", s.quantile(0.95))
    print("Min:", s.min())
    print("Max:", s.max())
    print()

diff = sub["VaR_copula"] - sub["VaR_indep"]

print("=== Difference: VaR_copula - VaR_indep ===")
print("Mean diff:", diff.mean())
print("Median diff:", diff.median())
print("Min diff:", diff.min())
print("Max diff:", diff.max())
print("Share of days where copula VaR is more negative:",
      (sub["VaR_copula"] < sub["VaR_indep"]).mean())
print()

print("=== Exceptions summary (99% VaR) ===")
print("Total exceptions (copula):", int(sub["hit_copula"].sum()))
print("Total exceptions (indep):", int(sub["hit_indep"].sum()))
print("Exception rate copula:", sub["hit_copula"].mean())
print("Exception rate indep:", sub["hit_indep"].mean())

```