import limix
import numpy as np
import pandas as pd
import re
from limix.varDecomp import VarianceDecomposition
import statsmodels.nonparametric.smoothers_lowess
import random
from generate_covariance import generate_covariance_dict


def run_variance_analysis(quant_df, metadata_df, transform_fcn=np.log2, fixed_effect_df=None):
    '''A function to perform variance decomposition, as well as computing overdispersion
    and mean abundance statistics.'''
    # Drop rows with any NA values in the metadata_df
    metadata_df = metadata_df.loc[quant_df.columns, :].dropna()
    random_effect_dict = generate_covariance_dict(metadata_df)

    selected_columns = random_effect_dict.keys()
    samples_w_metadata = list(set.intersection(
        *[set(x.index) for x in random_effect_dict.values()]))

    print 'Running variance decomposition for: {}'.format(selected_columns)
    if fixed_effect_df is not None:
        print 'Using fixed effects: {}'.format(list(fixed_effect_df.columns))

    # calculate variance components
    var_df = pd.DataFrame(index=quant_df.index,
                          columns=selected_columns + ['residual'])

    var_component_names = selected_columns + ['residual']

    for idx, feature_id in enumerate(quant_df.index):
        phenotype_ds = transform_fcn(quant_df.loc[feature_id, :])
        var_df.loc[feature_id, var_component_names] = variance_decomposition(
            phenotype_ds, random_effect_dict, fixed_effect_df)

    var_df['variance'] = quant_df.apply(lambda x: np.var(x.dropna()), axis=1)
    var_df['mean'] = quant_df.apply(lambda x: np.mean(x.dropna()), axis=1)
    var_df['cv'] = var_df['variance']**0.5/var_df['mean']
    var_df['overdispersion'] = calculate_empirical_overdispersion(
        var_df['mean'].values, var_df['variance'].values, transform_fcn)
    var_df['overdispersion_rank'] = rank_series_w_nans(
        var_df['overdispersion'])
    var_df['mean_rank'] = rank_series_w_nans(var_df['mean'])
    return var_df


def rank_series_w_nans(dataseries):
    '''Rank a dataseries from 0 to 1, keeping nans as nans.'''
    dataseries_dropped = dataseries.dropna()
    ranked_vals_dropped = dataseries_dropped.rank(
    ) / float(len(dataseries_dropped.index))
    ranked_vals = pd.Series(data=np.nan, index=dataseries.index)
    ranked_vals.loc[ranked_vals_dropped.index] = ranked_vals_dropped
    return ranked_vals


def calculate_empirical_overdispersion(mean, variance, transform_fcn):
    mean = transform_fcn(mean)
    variance = transform_fcn(variance)
    lowess = statsmodels.nonparametric.smoothers_lowess.lowess(
        variance, mean, frac=0.1, return_sorted=False)
    overdispersion = variance - lowess
    return overdispersion


def normalize_covariance_matrix(K):
    '''Gower normalization code borrowed from limix 2.0 '''
    K = np.asarray(K, float)
    c = (K.shape[0] - 1) / (K.trace() - K.mean(0).sum())
    return c * K

def run_variance_analysis_cross_validation(quant_df, metadata_df, transform_fcn=np.log2, cv_fraction=0.2):
    metadata_df.dropna(inplace=True)
    samples = list(set(metadata_df.index) & set(quant_df.columns))
    random.shuffle(samples)
    nS = len(samples)
    nLeftOut = int(cv_fraction * nS)
    nRuns = nS / nLeftOut
    print nS, nLeftOut, nRuns
    sample_subsets = [samples[:x * nLeftOut] +
                      samples[(x + 1) * nLeftOut:] for x in range(nRuns)]
    print len(sample_subsets)
    var_df_list = [run_variance_analysis(
        quant_df.loc[:, x], metadata_df.loc[x, :], transform_fcn=transform_fcn) for x in sample_subsets]
    return var_df_list


def variance_decomposition(phenotype_ds, random_effect_dict, fixed_effect_df=None):
    '''phenotype_ds is a pandas DataSeries, (samples, 1).
       random_effect_dict is a dictionary of pandas DataFrames, (samples, samples).
       fixed_effect_df is a pandas DataFrame, (samples, fixed effects).'''

    var_component_names = random_effect_dict.keys() + ['residual']

    samples_w_metadata = random_effect_dict.values()[0].index

    phenotype_ds = phenotype_ds.dropna()

    samples = list(set(phenotype_ds.index) & set(samples_w_metadata))
    if len(samples) == 0:
        var_ds = pd.Series(data=np.nan, index=var_component_names)
        return var_ds

    # variance component model
    vc = VarianceDecomposition(phenotype_ds.loc[samples].values)
    # add intercept
    vc.addFixedEffect()
    # add fixed effects
    if fixed_effect_df is not None:
        # subset fixed effects, and remove columns with variance == 0
        fixed_effect_df = fixed_effect_df.loc[samples, :]
        cols_to_keep = fixed_effect_df.apply(np.std) != 0.0
        fixed_effect_df = fixed_effect_df.loc[:, cols_to_keep]
        fixed_effect_matrix = fixed_effect_df.values
        vc.addFixedEffect(F=fixed_effect_matrix)
    # add random effects
    for key in random_effect_dict.keys():
        random_effect_matrix = random_effect_dict[key].loc[samples,
                                                           samples].values
        random_effect_matrix = normalize_covariance_matrix(random_effect_matrix)
        vc.addRandomEffect(K=random_effect_matrix)
    vc.addRandomEffect(is_noise=True)

    # fit the model
    try:
        vc.optimize()
        var_data = vc.getVarianceComps()[0]
        var_ds = pd.Series(data=var_data, index=var_component_names)
        var_ds = var_ds / var_ds.sum()
    except (np.linalg.linalg.LinAlgError, ValueError) as e:
        # This error is raised when the covariance of the phenotype is not positive definite
        # (e.g. if it is all zeros)
        var_ds = pd.Series(data=np.nan, index=var_component_names)

    return var_ds
