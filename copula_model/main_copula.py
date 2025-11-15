import numpy as np
import pandas as pd
from .copula_fitting import model_selection
from .mc_simulation import simulate_gaussian_copula, simulate_t_copula, simulate_independent 
from .distribution_fitting import main_distr_fitting
from .distribution import inverse_gaussian, inverse_t, inverse_empirical
import time
import multiprocessing

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