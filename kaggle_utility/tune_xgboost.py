
class TuneXGBoost:
    def __init__():
        self.params = {
            silent: False,
            sacle_pos_weight: 1,
            learning_rate: 0.01,
            colsample_bytree: 0.4,
            subsample: 0.8,
            n_estimators: 1000,
            reg_alpha: 0.3,
            max_depth: 3,
            gamma: 10,
            nthread: 4,
            seed: 27,
            objective: 'binary:logistic',
        }