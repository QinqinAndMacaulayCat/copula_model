
# Proposal: Copula-Based Portfolio Risk Modeling Using S&P 500 Constituents

## 1. Introduction and Objective

Traditional Value-at-Risk (VaR) estimation methods often rely on simplifying assumptions such as normally distributed returns or independence among asset returns. However, in real financial markets, asset returns exhibit non-linear dependence and tail co-movements that such models fail to capture.

This project proposes to develop a portfolio VaR estimation framework based on **copula models** using daily return data of **S&P 500 constituent stocks**. The key objective is to model the joint dependence structure accurately—particularly in the tails—and to evaluate whether copula-based VaR improves risk estimation over the assumption of independence. The performance of the model will be assessed via backtesting on historical data.

## 2. Methodology

### 2.1 Data Preparation
- Obtain daily closing prices of S&P 500 constituents over a selected period.
- Calculate daily log returns and align the data.
- Define an equally weighted or pre-specified portfolio of these assets.

### 2.2 Marginal Distribution Modeling
- Fit appropriate univariate distributions to each stock’s return time series (e.g., empirical distribution, normal, or t-distribution).
- Transform each return series into uniform random variables using the **probability integral transform**.

### 2.3 Copula Construction
- Use transformed marginals to estimate a **multivariate copula model**, capturing the dependence structure across assets.
- Two primary candidates:
  - **Gaussian Copula** (captures linear dependence but lacks tail dependence).
  - **Student-t Copula** (captures both linear and symmetric tail dependence).
- Parameters will be estimated using **pseudo-maximum likelihood** or **Inference Functions for Margins (IFM)**.

### 2.4 VaR Estimation
- Simulate joint returns from the fitted copula and marginal distributions.
- Aggregate the simulated returns into portfolio-level returns.
- Compute empirical **VaR at multiple confidence levels** (e.g., 95%, 99%) from the simulated return distribution.

### 2.5 Benchmark and Backtesting
- Construct a baseline model assuming **independent asset returns** (i.e., product of marginals).
- Compare VaR estimates from both models.
- Perform **backtesting** using actual historical returns:
  - Count VaR violations (actual loss > predicted VaR).
  - Apply statistical tests (e.g., **Kupiec Test**) to evaluate predictive accuracy.

## 3. Implementation Plan

| Stage | Tasks | Tools |
|-------|-------|-------|
| Data Collection | Download prices, calculate returns | WRDS, Yahoo Finance, Python (pandas) |
| Marginal Modeling | Fit distributions, transform to uniforms | `scipy.stats`, `statsmodels` |
| Copula Estimation | Estimate Gaussian/t Copulas | `copulas`, `pyvinecopulib`, `numpy` |
| Simulation | Generate joint returns, estimate VaR | `numpy`, `matplotlib` |
| Backtesting | Violation counting, Kupiec test | Custom Python functions |

## 4. Expected Results

- **Copula-based VaR** is expected to be more conservative than the independence-based VaR, especially under market stress.
- **Student-t Copula** should outperform Gaussian Copula in capturing joint extreme losses.
- Violation ratios in backtesting are expected to be closer to the nominal levels in the copula models, validating the improved accuracy.
- The study will demonstrate the importance of modeling joint tail dependence in realistic portfolio risk management.

## 5. Significance

This project highlights the practical benefits of **copula modeling in portfolio risk estimation**. In contrast to classical models, copulas offer a flexible way to separate marginal behaviors from dependence structures, allowing for more realistic simulation of extreme market events. The framework can be extended to stress testing, portfolio optimization under risk constraints, or even to credit risk models such as CDOs.
