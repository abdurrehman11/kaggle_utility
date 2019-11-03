import gc
from time import time
import datetime
import pandas as pd

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import roc_auc_score

class TrainXGBoost:
    def __init__(
        self, 
        params,
        X_train,
        y_train,
        X_test,
        submission,
        target,
        metric,
        n_folds=5,
        verbose=True
    ):
        self.params = params
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.submission = submission
        self.target = target
        self.verbose = verbose
        self.n_folds = n_folds
        self.metric = metric
        self.feature_importances = pd.DataFrame()
        self.train_aucs = []
        self.valid_aucs = []
        
    def train_model(self):
        # KFold for cross-validation
        folds = KFold(n_splits=self.n_folds)
        
        self.submission[target] = 0
            
        training_start_time = time()
        for fold, (train_index, valid_index) in enumerate(folds.split(self.X_train)):
            start_time = time()
            print('Training on Fold {}'.format(fold + 1))

            model = XGBClassifier(**self.params)
            
            # make train and valid set
            X_train, X_valid = self.X_train.iloc[train_index], self.X_train.iloc[valid_index]
            y_train, y_valid = self.y_train.iloc[train_index], self.y_train.iloc[valid_index]

            # train the model
            model.fit(
                X_train, 
                y_train,
                eval_set=[(X_train, y_train), (X_valid, y_valid)],
                eval_metric=self.metric, 
                verbose=self.verbose
            )
            
            train_pred = model.predict_proba(X_train)[:, 1]
            del X_train
    
            valid_pred = model.predict_proba(X_valid)[:, 1]
            del X_valid

            # train and valid roc_auc
            self.train_aucs.append(roc_auc_score(y_train, train_pred))
            self.valid_aucs.append(roc_auc_score(y_valid, valid_pred))
            
            del y_train, train_pred
            del y_valid, valid_pred
            
            print('ROC AUC on Train: {}'.format(self.train_aucs[fold]))
            print('ROC AUC on Validation: {}'.format(self.valid_aucs[fold]))
            
            # test set predictions for KFold
            test_pred = model.predict_proba(self.X_test)[:, 1]
            self.submission[self.target] = self.submission[self.target] + test_pred / self.n_folds

            gc.collect()
            
            print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))))
            print("=" * 30)
            print()

            self.feature_importances['fold_{}'.format(fold + 1)] = pd.Series(model.get_booster().get_fscore())
            
        print('-' * 30)
        print('Training has finished!')
        print('Total training time is {}'.format(str(datetime.timedelta(seconds=time() - training_start_time))))
        print('Mean AUC on Train: ', np.mean(self.train_aucs))
        print('Mean AUC on Validation: ', np.mean(self.valid_aucs))
        print('-' * 30)
            
        return model
    
    def get_feature_importances(self):
        self.feature_importances['average'] = self.feature_importances.mean(axis=1)
        return self.feature_importances
    
    def auc_score(self):
        folds = ['fold_{}'.format(fold+1) for fold in range(self.n_folds)]
        auc_df = pd.DataFrame({
            'Fold_no': folds,
            'Train AUC': self.train_aucs,
            'Valid AUC': self.valid_aucs,
        })
        
        auc_df.set_index('Fold_no', inplace=True)
        
        return auc_df
        