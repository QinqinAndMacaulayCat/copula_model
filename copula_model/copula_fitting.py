import numpy as np
from scipy.stats import kendalltau, t, multivariate_t, multivariate_normal, norm
from scipy.special import gammaln
from scipy.optimize import minimize
import time


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

def aic(log_likelihood: float,
        n_params: int
        ) -> float:
    """
    Computes the Akaike Information Criterion (AIC) for model selection.
    Parameters
    ----------
    copula_type : str
        Type of copula ('gaussian', 't')
    log_likelihood : float
        Log-likelihood of the fitted model.
    n_params : int
        Number of parameters in the model.
    Returns
    -------
    float
        AIC value.
    """
    return 2 * n_params - 2 * log_likelihood

def bic(log_likelihood: float,
        n_params: int) -> float:
    """
    Computes the Bayesian Information Criterion (BIC) for model selection.
    Parameters
    ----------
    log_likelihood : float
        Log-likelihood of the fitted model.
    n_params : int
        Number of parameters in the model.
    Returns
    -------
    float
        BIC value.
    """
    return n_params * np.log(log_likelihood) - 2 * log_likelihood

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


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    sample_data = np.random.rand(1000, 500)

    corr = correlation_matrix(sample_data)
    print("Correlation Matrix:\n", corr)
    ll_gaussian = log_likelihood_gaussian(corr, sample_data)
    print("Gaussian Log-Likelihood:", ll_gaussian)
    fitted_nu = fit_t_copula(sample_data, corr)
    print("Fitted nu for t-copula:", fitted_nu)
    ll_t = log_likelihood_t(fitted_nu, corr, sample_data)
    print("t Copula Log-Likelihood:", ll_t)



