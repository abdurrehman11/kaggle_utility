# Author: Abdur Rehman

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

class CategoricalEncoder():
    """Transform categorical features into a given encoding scheme."""

    def __init__(self, encoding_type='mean', n_folds=5):
        self.encoding_type = encoding_type
        self.n_folds = n_folds

    def likelihood_encoding(self, data):
        encoding = data.mean()
        return encoding

    def weight_of_evidence_encoding(self, data):
        ones_count = data.sum()
        zeros_count = data.size() - ones_count
        
        # make demoninator valid
        if zeros_count == 0:
            zeros_count = 1
        
        encoding = np.log((ones_count / zeros_count) + 1) * 100
        return encoding

    def count_encoding(self, data):
        encoding = data.sum()
        return encoding

    def diff_encoding(self, data):
        ones_count = data.sum()
        zeros_count = data.size() - ones_count
        encoding = (ones_count - zeros_count).abs()
        return encoding

    def KFold_target_encoding(
        self, 
        train_df, 
        test_df, 
        target, 
        features,
        n_folds=None, 
        encoding_type=None,
    ):
        """ 
        Transform categorical features of train and test dataframe 
        into given target encoding using KFold cross-validation.

        Parameters
        ----------
        train_df : dataframe-like, shape (n_samples, n_features)
            Train Dataframe of categorical features to be encoded.

        test_df : dataframe-like, shape (n_samples, n_features)
            Test Dataframe of categorical features to be encoded.

        features : list-like
            Names of categorical columns to be encoded.

        n_folds: int
            No. of folds used for finding encoding of categorical features.

        encoding_type: str, {'mean', 'evidence', 'count', 'diff'}
            Encoding type to encode categorical features.

        Returns
        -------
        DataFrame
            Encoded categorical features of train and test set.
        """

        if n_folds is None:
            n_folds = self.n_folds

        if encoding_type is None:
            encoding_type = self.encoding_type

        # select the appropriate encoding function
        agg_func = None
        if encoding_type == 'mean':
            agg_func = self.likelihood_encoding
        elif encoding_type == 'evidence':
            agg_func = self.weight_of_evidence_encoding
        elif encoding_type == 'count':
            agg_func = self.count_encoding
        elif encoding_type == 'diff':
            agg_func = self.diff_encoding

        kf = KFold(n_splits=n_folds)
    
        # copy features for mean encoding
        for feature in categorical_features:
            train_df[feature + '_' + encoding_type + '_target'] = train_df[feature]
            test_df[feature + '_' + encoding_type + '_target'] = test_df[feature]
    
        # populate features of train set with mean encoding
        for train_index, valid_index in kf.split(train_df):
            train_data = train_df.iloc[train_index]
        
            for feature in features:
                feature_encoding = train_data.groupby([feature])[target].apply(agg_func)
                train_df[feature + '_' + encoding_type + '_target'].iloc[valid_index] = \
                train_df[feature + '_' + encoding_type + '_target'].iloc[valid_index].map(feature_encoding)
    
        # populate features of test set with mean encoding
        for feature in features:
            train_df[feature + '_' + encoding_type + '_target'] = \
            pd.to_numeric(train_df[feature + '_' + encoding_type + '_target'])

            feature_encoding = train_df.groupby([feature])[feature + '_' + encoding_type + '_target'] \
                                .apply(agg_func)
            test_df[feature + '_' + encoding_type + '_target'] = \
            test_df[feature + '_' + encoding_type + '_target'].map(feature_encoding)

        train_df = train_df.drop(features, axis=1)
        test_df = test_df.drop(features, axis=1)
    
        return train_df, test_df