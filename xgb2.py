import xgboost as xgb

class EarlyStoppingDelta(xgb.callback.TrainingCallback):
    def __init__(self, rounds, min_delta, metric="auc", maximize=True):
        self.rounds    = rounds
        self.min_delta = min_delta
        self.metric    = metric
        self.maximize  = maximize
        self.best      = None
        self.best_iter = 0

    def after_iteration(self, model, epoch, evals_log):
        score = evals_log["val"][self.metric][-1]

        if self.best is None:
            self.best, self.best_iter = score, epoch
        else:
            gain = score - self.best if self.maximize else self.best - score
            if gain >= self.min_delta:
                self.best, self.best_iter = score, epoch

        if epoch - self.best_iter >= self.rounds:
            return True   # stoppe l'entraînement
        return False

# Usage
dtrain = xgb.DMatrix(X_train, label=y_train)
dval   = xgb.DMatrix(X_val,   label=y_val)

params = {"objective": "binary:logistic", "eval_metric": "auc", "max_depth": 3}

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=[(dval, "val")],
    callbacks=[EarlyStoppingDelta(rounds=10, min_delta=0.05)],
    verbose_eval=False,
)
