
# Copula-Based Portfolio Risk Modeling Using S&P 500 Constituents

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


#### 2.4.3 Archimedean Copulas (Clayton, Gumbel, Frank)

1. Generate a variate $V$ with distribution function $G$ such that  

   $$ 
   \hat{G}(t) = \mathcal{L}\{G\}(t) 
   $$  

   is the Laplace–Stieltjes transform of $G$, and is the inverse of the generator $\phi$ of the required copula.

2. Generate independent uniform variates  

   $$ 
   X_1, X_2, \dots, X_d \sim \mathcal{U}(0,1) 
   $$

3. Return  

   $$
   U = \left( \hat{G}^{-1}\left( -\ln(X_1)/V \right), \dots, \hat{G}^{-1}\left( -\ln(X_d)/V \right) \right)^T
   $$

a. Clayton Copula

- Generate a gamma variate  

  $$
  V \sim \text{Gamma}\left(\frac{1}{\theta}, 1\right), \quad \theta > 0 
  $$

- Laplace transform:  

  $$ 
  \hat{G}(t) = (1 + t)^{-1/\theta} 
  $$

- Inverse Laplace transform:  

  $$ 
  \hat{G}^{-1}(t) = t^{-\theta} - 1 
  $$


b. Gumbel Copula

- Generate a positive stable variate  

  $$ 
  V \sim \text{St}\left(\frac{1}{\theta}, 1, \gamma, 0\right), \quad \theta > 1 
  $$  

  $$ 
  \gamma = \left( \cos\left( \frac{\pi}{2\theta} \right) \right)^\theta 
  $$

- Laplace transform:  

  $$ 
  \hat{G}(t) = \exp\left( -t^{1/\theta} \right) 
  $$

c. Frank Copula

- Generate a discrete random variable $V$ with probability mass function:  

  $$
  p(k) = P(V = k) = \frac{(1 - e^{-\theta})^k}{k\theta}, \quad k = 1, 2, \dots, \quad \theta > 0
  $$

- Laplace transform:  

  $$ 
U_i = - \frac{1}{\theta} \log \left( 1 + \frac{(e^{-V \theta} - 1)(1 - e^{-\theta X_i})}{1 - e^{-\theta}} \right)
  $$


### 2.5 VaR Estimation

### 2.6 Benchmark and Backtesting

# 3. Results




# Note

1. We referrenced text book - "Quantitative Risk Management: Concepts, Techniques and Tools" by Alexander J. McNeil, Rüdiger Frey, and Paul Embrechts for the methodology of copula modeling and VaR estimation.
2. The code is in the zip file. We did not put code here because it is too long.
