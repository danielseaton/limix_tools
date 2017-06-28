import limix
import numpy as np
import pandas as pd
import re
from limix.varDecomp import VarianceDecomposition
import statsmodels.nonparametric.smoothers_lowess


def run_variance_analysis(quant_df,metadata_df,var_stabilizing_transform=np.log2):
    '''A function to perform variance decomposition, as well as computing overdispersion
    and mean abundance statistics.'''
    var_df = variance_decomposition(var_stabilizing_transform(quant_df),metadata_df)
    var_df['variance'] = quant_df.apply(lambda x: np.var(x.dropna()),axis=1)
    var_df['mean'] = quant_df.apply(lambda x: np.mean(x.dropna()),axis=1)
    var_df['overdispersion'] = calculate_empirical_overdispersion(var_df['mean'].values,var_df['variance'].values)
    var_df['overdispersion_rank'] = var_df['overdispersion'].rank()/float(len(var_df.index))
    var_df['mean_rank'] = var_df['mean'].rank()/float(len(var_df.index))
    return var_df

def calculate_empirical_overdispersion(mean,variance):
    mean = np.log2(mean)
    variance = np.log2(variance)
    lowess = statsmodels.nonparametric.smoothers_lowess.lowess(variance,mean,frac=0.1,return_sorted=False)
    overdispersion = variance-lowess
    return overdispersion

def variance_decomposition(quant_df,metadata_df):

    #Drop rows with any NA values in the metadata_df
    metadata_df.dropna(inplace=True)
    
    selected_columns = list(metadata_df.columns)
    print 'Running variance decomposition for: {}'.format(selected_columns)
    random_effect_dict = dict()
    for column_name in selected_columns:
        random_effect_mat = []
        for categorical_value in metadata_df[column_name]:
            vector_of_matches = metadata_df[column_name].map(lambda x : int(x==categorical_value)).values
            if sum(vector_of_matches)==len(vector_of_matches):
                print 'All samples are identical'
            random_effect_mat.append(vector_of_matches)
        random_effect_mat = np.array(random_effect_mat)
        random_effect_df = pd.DataFrame(data=random_effect_mat,index=metadata_df.index,columns=metadata_df.index)
        random_effect_dict[column_name] = random_effect_df

    var_df = pd.DataFrame(index = quant_df.index,columns=selected_columns+['residual'])

    rel_var_columns = selected_columns+['residual']

    for idx,feature_id in enumerate(quant_df.index):
        phenotypes = quant_df.loc[feature_id,:].dropna()
        samples = list(set(phenotypes.index)&set(metadata_df.index))
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
