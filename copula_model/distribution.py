
import numpy as np
from scipy.stats import norm, t

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