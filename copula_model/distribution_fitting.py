"""
Distribution fitting and PIT utilities

This module fits marginal distributions (Normal / Student-t / Empirical) to
daily log-returns per ticker, selects the best marginal per ticker using
information criteria and goodness-of-fit tests, and computes Probability
Integral Transform (PIT) series (Uniform(0,1)) for downstream copula analysis.

Main methods

- run_fitting
    - Description: Fit candidate marginals for each column in a returns DataFrame,
        choose the best model per ticker, and return the PIT DataFrame and a
        summary table of selected models and parameters.
    - Parameters:
        - returns (pd.DataFrame): rows are dates, columns are tickers, values are
            daily log-returns.
    - Returns: (pit_df, best_table)
        - pit_df (pd.DataFrame): PIT values per ticker, index aligned to returns.
        - best_table (pd.DataFrame): per-ticker model choice and fitted parameters.

- get_pit_from_returns
    - Description: Convenience wrapper that runs fitting on an already-loaded
        returns DataFrame and returns only the PIT DataFrame (no file I/O).
    - Parameters:
        - returns (pd.DataFrame)
    - Returns: pit_df (pd.DataFrame)

- get_pit_from_path
    - Description: Load returns from a CSV file and return the PIT DataFrame
        without saving outputs to disk.
    - Parameters:
        - path (Path): path to the returns CSV file.
    - Returns: pit_df (pd.DataFrame)

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF
from pathlib import Path
from typing import Dict, Tuple
import argparse

data_path = Path('/Users/ziranguo/Desktop/MQF/2025 Fall/Risk Management/Project/sp500_log_returns.csv')

def load_returns_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Try to parse a date column if present
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    # Ensure numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    # Drop completely empty columns
    df = df.dropna(axis=1, how='all')
    return df

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


def main(path: Path = data_path, save: bool = True, out_dir: Path = None, use_parquet: bool = False, best_output: Path = None):
    """Load returns from path, run fitting, and optionally save/print PIT DataFrame and best_table.

    Args:
        path: Path to input returns CSV.
        save: whether to save outputs to disk.
        out_dir: directory to save outputs (defaults to input parent).
        use_parquet: if True try to save PIT as parquet (pyarrow required).
        best_output: explicit path for best_table CSV (overrides out_dir).

    Returns:
        pit_df, best_table
    """
    returns = load_returns_csv(path)
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
            print('Saved PIT to', pit_path)
        except Exception as e:
            # fallback to csv if parquet failed
            if use_parquet:
                pit_path = out_dir / 'pit_uniform.csv.gz'
                pit_df.to_csv(pit_path, compression='gzip')
                print('Parquet save failed, saved PIT to', pit_path)
            else:
                raise

        print('PIT shape:', pit_df.shape)
        print('PIT head:')
        print(pit_df.head())

        # save best_table
        if best_output is None:
            best_path = out_dir / 'best_marginals.csv'
        else:
            best_path = Path(best_output)
            best_path.parent.mkdir(parents=True, exist_ok=True)
        best_table.to_csv(best_path)
        print('Saved best_table to', best_path)

    return pit_df, best_table


def get_pit_from_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper: run fitting on an already-loaded returns DataFrame and return pit_df only."""
    pit_df, _ = run_fitting(returns)
    return pit_df


def get_pit_from_path(path: Path = data_path) -> pd.DataFrame:
    """Load returns from CSV path and return pit_df (no saving)."""
    returns = load_returns_csv(path)
    pit_df, _ = run_fitting(returns)
    return pit_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit marginals and compute PIT for returns CSV')
    parser.add_argument('--input', '-i', type=Path, default=data_path, help='Path to returns CSV')
    parser.add_argument('--out-dir', '-o', type=Path, default=None, help='Directory to save outputs (default: input parent)')
    parser.add_argument('--no-save', action='store_true', help="Don't save outputs to disk; just run and return")
    parser.add_argument('--parquet', action='store_true', help='Save PIT as parquet if possible')
    parser.add_argument('--best-output', type=Path, default=None, help='Path to save best_table CSV (overrides out-dir)')
    args = parser.parse_args()

    main(path=args.input, save=not args.no_save, out_dir=args.out_dir, use_parquet=args.parquet, best_output=args.best_output)
