from challenger_benchmark.tuning import tune, subsample_for_tuning

X_tune, y_tune = subsample_for_tuning(X_train, y_train, cfg.tuning.sample_frac, cfg.tuning.seed)
best_params, study = tune(model, X_tune, y_tune, cfg.tuning, callbacks=[trace])
del X_tune, y_tune

X_tune, y_tune = subsample_for_tuning(X_train, y_train, cfg.tuning.sample_frac, cfg.tuning.seed)
bp, st = tune(m, X_tune, y_tune, cfg.tuning)
del X_tune, y_tune
