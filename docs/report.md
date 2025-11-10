
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

We downloaded stock prices from yahoo finance and transformed to lognormal return rates.

### 2.2 Marginal Distribution Modeling

### 2.3 Copula Construction


### 2.4 Simulate Paths


### 2.5 VaR Estimation

### 2.6 Benchmark and Backtesting
