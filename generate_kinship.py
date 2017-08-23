import numpy as np
import pandas as pd
import scipy.stats
import re
import glob
import scipy
import limix
from sklearn.preprocessing import Imputer

def generate_kinship(snp_df):
    snp_df = snp_df.dropna(how='all',axis=1)
    snp_df = snp_df.dropna(how='all',axis=0)
    snp_df = pd.DataFrame(data=fill_NaN.fit_transform(snp_df),columns=snp_df.columns,index=snp_df.index)
    std = snp_df.apply(np.std,axis=0)
    #keep snps with some variation 
    snp_df = snp_df.loc[:,std>0]
    X = snp_df.values
    X -= X.mean(0)
    X /= X.std(0)
    X /= np.sqrt(X.shape[1])
    K = X.dot(X.T)
    index = snp_df.index
    kinship_df = pd.DataFrame(data=K,index=index,columns=index)
    return kinship_df
