
# Proposal: Copula-Based Portfolio Risk Modeling Using S&P 500 Constituents

Team members: Yue Yang, Ziran Guo, Yuxi Geng, Qinqin Huang

## 1. Introduction and Objective

Traditional Value-at-Risk (VaR) estimation methods often rely on simplifying assumptions such as normally distributed returns or independence among asset returns. However, in real financial markets, asset returns exhibit non-linear dependence and tail co-movements that such models fail to capture.

This project proposes to develop a portfolio VaR estimation framework based on copula models using daily return data of S&P 500 constituent stocks. The key objective is to model the joint dependence structure accurately, particularly in the tails, and to evaluate whether copula-based VaR improves risk estimation over the assumption of independence. The performance of the model will be assessed via traffic light backtesting methods.

The specific steps include:

- Data collection and preprocessing of daily returns for S&P 500 stocks.
- Fitting marginal distributions to individual stock returns.
- Constructing copula models (Gaussian, Student-t or others based on fit) to capture dependencies.
- Simulating portfolio returns using the fitted copula and marginal distributions.
- Estimating VaR at different confidence levels from the simulated returns. 
- Backtesting how many times actual losses exceed predicted VaR and evaluating model performance by comparing with benchmarks assuming independence among assets.

## 2. Methodology

### 2.1 Data Preparation
- Obtain daily closing prices of S&P 500 constituents over a selected period. We simplify by using the current constituents rather than considering portfolio changes over time.
- Calculate daily log returns.

### 2.2 Marginal Distribution Modeling
- Fit appropriate univariate distributions to each stock’s return time series (e.g., normal, or t-distribution).
- Transform each return series into uniform random variables which are empirical marginals and will be used for copula fitting.

### 2.3 Copula Construction
- Use transformed marginals calibrated from the previous step to estimate a multivariate copula model, capturing the dependence structure across assets.
- Candidates:
  - Gaussian Copula (captures linear dependence but lacks tail dependence).
  - Student-t Copula (captures both linear and symmetric tail dependence).
  - Other Archimedean copulas (Clayton, Gumbel) if asymmetrical tail dependence is observed.

### 2.4 VaR Estimation
- Simulate joint marginal distributions using the fitted copula model and inverse back to original return space.
- Aggregate the simulated returns into portfolio-level returns.
- Compute VaR at multiple confidence levels (e.g., 95%, 99%) from the simulated return paths.

### 2.5 Benchmark and Backtesting
- Simulate return paths independently for each asset.
- Compute VaR under the independence assumption.
- Perform backtesting using actual historical returns: Count VaR violations (actual loss > predicted VaR) for both copula-based and independence-based VaR estimates. Check if copula-based VaR provides better coverage.


