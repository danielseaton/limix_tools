import numpy as np
import pandas as pd
import random
import limix_tools

random.seed(0)
np.random.seed(0)

nS = 100
nF = 10

sample_list = ['sample'+str(x) for x in range(nS)]
feature_list = ['feature'+str(x) for x in range(nF)]

dummy_metadata = [random.choice(['yes','no']) for x in sample_list]
metadata_df = pd.DataFrame(data=dummy_metadata, index=sample_list, columns=['random0'])

dummy_covdata = [random.choice([0,1]) for x in sample_list]
fixed_effect_df = pd.DataFrame(data=dummy_covdata, index=sample_list, columns=['fixed0'])

quant_matrix = 2.5 * np.random.randn(nF, nS) + 3
quant_df = pd.DataFrame(data=quant_matrix, index=feature_list, columns=sample_list)

var_df_0 = limix_tools.run_variance_analysis(quant_df, metadata_df, transform_fcn=lambda x: x, fixed_effect_df=None)

quant_df.loc[:, metadata_df['random0']=='yes'] = quant_df.loc[:, metadata_df['random0']=='yes'] + 150

var_df_1 = limix_tools.run_variance_analysis(quant_df, metadata_df, transform_fcn=lambda x: x, fixed_effect_df=None)

var_df_2 = limix_tools.run_variance_analysis(quant_df, metadata_df, transform_fcn=lambda x: x, fixed_effect_df=fixed_effect_df)

assert(var_df_0['random0'].mean()<0.05)
assert(var_df_1['random0'].mean()>0.95)
assert(var_df_2['random0'].mean()>0.95)


