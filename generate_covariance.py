import numpy as np
import pandas as pd
import scipy.stats
import re
import scipy
import limix
from sklearn.preprocessing import Imputer


def generate_cis_kinship(bed, fam, bim, chrom, start, end, window_size):
    lowest = min([start, end])
    highest = max([start, end])
    cis_snp_query = bim.query("chrom == '%s' & pos > %d & pos < %d" % (
        chrom, lowest - window_size, highest + window_size))
    iid_idxs = fam.index
    snp_idxs = cis_snp_query['i'].values
    if len(snp_idxs) == 0:
        return None
    kinship_cis_df = generate_kinship(bed, fam, snp_idxs, iid_idxs)
    index = pd.Index(fam.loc[kinship_cis_df.index, 'iid'])
    kinship_cis_df.index = index
    kinship_cis_df.columns = index
    return kinship_cis_df


def generate_kinship(bed, fam, snp_idxs, iid_idxs, chunk_size=100000):
    assert(all([x in fam.index for x in iid_idxs]))

    fill_NaN = Imputer(missing_values=np.nan, strategy='mean', axis=0)

    nI = len(iid_idxs)
    K = np.zeros((nI, nI))

    nChunks = nI / chunk_size + 1
    snp_count = 0
    for idx in range(nChunks):
        start = idx * chunk_size
        stop = (idx + 1) * chunk_size
        snp_idxs_subset = snp_idxs[start:stop]

        # limit snp_df to snp subset, and individual subset
        snp_df = pd.DataFrame(data=bed[snp_idxs_subset, :].compute(
        ).transpose(), index=fam.index, columns=snp_idxs_subset)
        snp_df = snp_df.loc[iid_idxs, :]

        # drop snps that are all na
        snp_df = snp_df.dropna(how='all', axis=0)
        # impute missing snps
        snp_df = pd.DataFrame(data=fill_NaN.fit_transform(
            snp_df), columns=snp_df.columns, index=snp_df.index)
        # remove snps that don't vary
        std = snp_df.apply(np.std, axis=0)
        snp_df = snp_df.loc[:, std > 0]
        # number of snps we're using after dropping some (should be close to chunk_size)
        snp_count += len(snp_df.columns)

        X = snp_df.values
        X -= X.mean(0)
        X /= X.std(0)
        Ktemp = X.dot(X.T)
        K += Ktemp

    K /= snp_count
    kinship_df = pd.DataFrame(data=K, index=iid_idxs, columns=iid_idxs)
    return kinship_df


def generate_covariance_dict(metadata_df):
    '''Return a dictionary of random effect covariance matrices based on a metadata dataframe.'''

    # Drop rows with any NA values in the metadata_df
    metadata_df.dropna(inplace=True)

    selected_columns = list(metadata_df.columns)

    random_effect_dict = dict()
    for column_name in selected_columns:
        random_effect_mat = []
        for categorical_value in metadata_df[column_name]:
            vector_of_matches = metadata_df[column_name].map(
                lambda x: int(x == categorical_value)).values
            if sum(vector_of_matches) == len(vector_of_matches):
                print 'All samples are identical in {}'.format(column_name)
            random_effect_mat.append(vector_of_matches)
        random_effect_mat = np.array(random_effect_mat)
        random_effect_df = pd.DataFrame(
            data=random_effect_mat, index=metadata_df.index, columns=metadata_df.index)
        random_effect_dict[column_name] = random_effect_df

    return random_effect_dict
