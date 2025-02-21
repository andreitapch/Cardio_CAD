from scipy.stats import skew, kurtosis, ks_2samp
import numpy as np
import pandas as pd


def analyze_vendor_differences(speqle_traces_ge, speqle_traces_philips):
    # Ensure 'strain_trace' contains lists of floats
    speqle_traces_philips.tracesDF['strain_trace'] = speqle_traces_philips.tracesDF['strain_trace'].apply(
        lambda x: x if isinstance(x, list) else list(map(float, x.split(','))))
    speqle_traces_ge.tracesDF['strain_trace'] = speqle_traces_ge.tracesDF['strain_trace'].apply(
        lambda x: x if isinstance(x, list) else list(map(float, x.split(','))))

    # Calculate statistics for Philips
    philips_stats = speqle_traces_philips.tracesDF['strain_trace'].apply(lambda x: pd.Series({
        'min': np.min(x),
        'max': np.max(x),
        'magnitude': np.max(x) - np.min(x),
        'mean': np.mean(x),
        'std': np.std(x),
        'skew': skew(x),
        'kurtosis': kurtosis(x)
    }))

    # Calculate statistics for GE
    ge_stats = speqle_traces_ge.tracesDF['strain_trace'].apply(lambda x: pd.Series({
        'min': np.min(x),
        'max': np.max(x),
        'magnitude': np.max(x) - np.min(x),
        'mean': np.mean(x),
        'std': np.std(x),
        'skew': skew(x),
        'kurtosis': kurtosis(x)
    }))

    # Calculate differences
    differences = philips_stats.mean() - ge_stats.mean()

    # Perform Kolmogorov-Smirnov test for distribution differences
    ks_results = speqle_traces_philips.tracesDF['strain_trace'].apply(
        lambda x: ks_2samp(x, speqle_traces_ge.tracesDF['strain_trace'].iloc[0])
    )
    # Extract p-values from KS test results
    ks_pvalues = ks_results.map(lambda result: result.pvalue)

    # Print analysis
    print(f"Philips Statistics:\n{philips_stats.describe()}")
    print(f"GE Statistics:\n{ge_stats.describe()}")
    print(f"Mean Differences:\n{differences}")
    print(f"Kolmogorov-Smirnov Test p-values:\n{ks_pvalues}")

    # Recommendation based on differences and KS test
    if np.abs(differences['magnitude']) > 0.1 or ks_pvalues.mean() < 0.05:
        print("Recommendation: Perform imputation separately.")
    else:
        print("Recommendation: Strain traces are suitable for combined imputation.")
