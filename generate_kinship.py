import numpy as np
import pandas as pd
import scipy.stats
import re
import scipy
import limix
from sklearn.preprocessing import Imputer

def generate_kinship(snp_df):
    fill_NaN = Imputer(missing_values=np.nan, strategy='mean', axis=0)

    snp_df = snp_df.dropna(how='all',axis=1)
    snp_df = snp_df.dropna(how='all',axis=0)
    snp_df = pd.DataFrame(data=fill_NaN.fit_transform(snp_df),columns=snp_df.columns,index=snp_df.index)
    std = snp_df.apply(np.std,axis=0)
    #keep snps with some variation 
    snp_df = snp_df.loc[:,std>0]
    nI = len(snp_df.index)
    K = np.zeros((nI,nI))
    
    chunk_size = 100000
    nChunks = len(snp_df.columns)/chunk_size + 1
    for idx in range(nChunks):
        start = idx*chunk_size
        stop = (idx+1)*chunk_size
        X = snp_df.values[:,start:stop]
        X -= X.mean(0)
        X /= X.std(0)
        Ktemp = X.dot(X.T)
        K += Ktemp
    K /= snp_df.shape[1]
    index = snp_df.index
    kinship_df = pd.DataFrame(data=K,index=index,columns=index)
    return kinship_df
