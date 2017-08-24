import limix
import numpy as np
import pandas as pd
import re
from limix.varDecomp import VarianceDecomposition
import statsmodels.nonparametric.smoothers_lowess
import random
from generate_covariance import generate_covariance_dict

def run_variance_analysis(quant_df,metadata_df,transform_fcn=np.log2):
    '''A function to perform variance decomposition, as well as computing overdispersion
    and mean abundance statistics.'''
    #Drop rows with any NA values in the metadata_df
    metadata_df.dropna(inplace=True)    
    random_effect_dict = generate_covariance_dict(metadata_df)
    var_df = variance_decomposition(transform_fcn(quant_df),random_effect_dict)
    var_df['variance'] = quant_df.apply(lambda x: np.var(x.dropna()),axis=1)
    var_df['mean'] = quant_df.apply(lambda x: np.mean(x.dropna()),axis=1)
    var_df['overdispersion'] = calculate_empirical_overdispersion(var_df['mean'].values,var_df['variance'].values,transform_fcn)
    var_df['overdispersion_rank'] = rank_series_w_nans(var_df['overdispersion'])
    var_df['mean_rank'] = rank_series_w_nans(var_df['mean'])
    return var_df

def rank_series_w_nans(dataseries):
    '''Rank a dataseries from 0 to 1, keeping nans as nans.'''
    dataseries_dropped = dataseries.dropna()
    ranked_vals_dropped = dataseries_dropped.rank()/float(len(dataseries_dropped.index))
    ranked_vals = pd.Series(data=np.nan,index=dataseries.index)
    ranked_vals.loc[ranked_vals_dropped.index] = ranked_vals_dropped
    return ranked_vals


def calculate_empirical_overdispersion(mean,variance,transform_fcn):
    mean = transform_fcn(mean)
    variance = transform_fcn(variance)
    lowess = statsmodels.nonparametric.smoothers_lowess.lowess(variance,mean,frac=0.1,return_sorted=False)
    overdispersion = variance-lowess
    return overdispersion

def run_variance_analysis_cross_validation(quant_df,metadata_df,transform_fcn=np.log2,cv_fraction=0.2):
    metadata_df.dropna(inplace=True)
    samples = list(set(metadata_df.index)&set(quant_df.columns))
    random.shuffle(samples)
    nS = len(samples)
    nLeftOut = int(cv_fraction*nS)
    nRuns = nS/nLeftOut
    print nS,nLeftOut,nRuns
    sample_subsets = [samples[:x*nLeftOut]+samples[(x+1)*nLeftOut:] for x in range(nRuns)]
    print len(sample_subsets)
    var_df_list = [run_variance_analysis(quant_df.loc[:,x],metadata_df.loc[x,:],transform_fcn=transform_fcn) for x in sample_subsets]
    return var_df_list

def variance_decomposition(quant_df,random_effect_dict): 

    selected_columns = random_effect_dict.keys()
    samples_w_metadata = random_effect_dict.values()[0].index
    print 'Running variance decomposition for: {}'.format(selected_columns)

    var_df = pd.DataFrame(index = quant_df.index,columns=selected_columns+['residual'])

    rel_var_columns = selected_columns+['residual']

    for idx,feature_id in enumerate(quant_df.index):

        phenotypes = quant_df.loc[feature_id,:].dropna()
        if len(phenotypes)==0:
            var_df.loc[feature_id,rel_var_columns] = np.nan
            continue

        samples = list(set(phenotypes.index)&set(samples_w_metadata))
        # variance component model
        vc = VarianceDecomposition(phenotypes.loc[samples].values)
        vc.addFixedEffect()
        for key in selected_columns:
            random_effect_matrix = random_effect_dict[key].loc[samples,samples].values
            vc.addRandomEffect(K=random_effect_matrix)
        vc.addRandomEffect(is_noise=True)
        try:
            vc.optimize()
            var_data = vc.getVarianceComps()[0]
            var_dataseries = pd.Series(data=var_data,index=rel_var_columns)
            var_dataseries = var_dataseries/var_dataseries.sum()
            var_df.loc[feature_id,rel_var_columns] = var_dataseries
        except np.linalg.linalg.LinAlgError:
            #This error is raised when the covariance of the phenotype is not positive definite
            # (e.g. if it is all zeros)
            var_df.loc[feature_id,rel_var_columns] = np.nan

        
    return var_df
