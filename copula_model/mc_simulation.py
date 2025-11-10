
import numpy as np
from scipy.stats import norm, t, levy_stable


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
