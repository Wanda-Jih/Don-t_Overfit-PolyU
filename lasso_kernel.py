import gc
gc.enable()
gc.collect()

import sys, os
import pandas as pd
import numpy as np

from sklearn.linear_model import Lasso, LassoCV
from sklearn.feature_selection import RFECV
from sklearn import preprocessing as pp
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, ShuffleSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, r2_score, make_scorer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# some heuristic settings
rfe_min_features = 11
rfe_step = 15
rfe_cv = 20
sss_n_splits = 20
sss_test_size = 0.35
noise_std = 0.01
r2_threshold = 0.185
random_seed = 213
gridsearchCV = 20

np.random.seed(random_seed)

# import data
train = pd.read_csv("/Users/JoonH/dont-overfit-ii/train.csv")
train_y = train['target']
train_X = train.drop(['id','target'], axis=1).values

test = pd.read_csv("/Users/JoonH/dont-overfit-ii/test.csv")
test = test.drop(['id'], axis=1).values

# scale using RobustScaler
# fitting scaler on full data outperforms fitting on test_X only (+0.006 kaggle score)
data = pp.RobustScaler().fit_transform(np.concatenate((train_X, test), axis=0))
train_X = data[:250]
test = data[250:]

# add a bit of noise to train_X to reduce overfitting
train_X += np.random.normal(0, noise_std, train_X.shape)


# define roc_auc_metric robust to only one class in y_pred
def scoring_roc_auc(y, y_pred):
    try:
        return roc_auc_score(y, y_pred)
    except:
        return 0.5


robust_roc_auc = make_scorer(scoring_roc_auc)


def build_and_train_model(mtype, train_X, train_y):
    if mtype == 'lasso':
        model = Lasso(alpha=0.031, tol=0.01, random_state=random_seed, selection='random')
        param_grid = {
            'alpha': [0.019, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.029, 0.031],
            'tol': [0.001, 0.0011, 0.0012, 0.0013, 0.0014, 0.0015, 0.0016, 0.0017],
            'max_iter': [250, 500, 750, 1000, 1250, 1500, 1750, 2000]
        }

        # define recursive elimination feature selector
        feature_selector = RFECV(model, min_features_to_select=rfe_min_features, scoring=robust_roc_auc, step=rfe_step,
                                 verbose=0, cv=rfe_cv, n_jobs=-1)
    print("counter | val_mse  |  val_mae  |  val_roc  |  val_cos  |  val_dist  |  val_r2    | feature_count ")
    print("-------------------------------------------------------------------------------------------------")

    predictions = pd.DataFrame()
    counter = 0
    # split training data to build one model on each traing-data-subset

    for train_index, val_index in StratifiedShuffleSplit(n_splits=sss_n_splits, test_size=sss_test_size,
                                                         random_state=random_seed).split(train_X, train_y):

        X, val_X = train_X[train_index], train_X[val_index]
        y, val_y = train_y[train_index], train_y[val_index]

        # get the best features for this data set
        feature_selector.fit(X, y)
        # remove irrelevant features from X, val_X and test
        X_important_features = feature_selector.transform(X)
        val_X_important_features = feature_selector.transform(val_X)
        test_important_features = feature_selector.transform(test)

        # run grid search to find the best Lasso parameters for this subset of training data and subset of features
        # grid_search = GridSearchCV(feature_selector.estimator_, param_grid=param_grid, verbose=0, n_jobs=-1, scoring=robust_roc_auc, cv=gridsearchCV)
        grid_search = RandomizedSearchCV(feature_selector.estimator_, param_distributions=param_grid, verbose=0,
                                         n_jobs=-1, scoring=robust_roc_auc, cv=gridsearchCV)  # , n_iter = 100)

        grid_search.fit(X_important_features, y)
        print(grid_search.best_params_)
        # score our fitted model on validation data
        val_y_pred = grid_search.best_estimator_.predict(val_X_important_features)
        val_mse = mean_squared_error(val_y, val_y_pred)
        val_mae = mean_absolute_error(val_y, val_y_pred)
        val_roc = roc_auc_score(val_y, val_y_pred)
        val_cos = cosine_similarity(val_y.values.reshape(1, -1), val_y_pred.reshape(1, -1))[0][0]
        val_dst = euclidean_distances(val_y.values.reshape(1, -1), val_y_pred.reshape(1, -1))[0][0]
        val_r2 = r2_score(val_y, val_y_pred)

        # if model did well on validation, save its prediction on test data, using only important features
        # r2_threshold (0.185) is a heuristic threshold for r2 error
        # you can use any other metric/metric combination that works for you
        if val_r2 > r2_threshold:
            message = '<-- OK'
            prediction = grid_search.best_estimator_.predict(test_important_features)
            predictions = pd.concat([predictions, pd.DataFrame(prediction)], axis=1)
        else:
            message = '<-- skipping'

        print(
            "{0:2}      | {1:.4f}   |  {2:.4f}   |  {3:.4f}   |  {4:.4f}   |  {5:.4f}    |  {6:.4f}    |  {7:3}         {8}  ".format(
                counter, val_mse, val_mae, val_roc, val_cos, val_dst, val_r2, feature_selector.n_features_, message))

        counter += 1

    print("-------------------------------------------------------------------------------------------------")
    print("{}/{} models passed validation threshold and will be ensembled.".format(len(predictions.columns),
                                                                                   sss_n_splits))

    return predictions


predictions = build_and_train_model('lasso', train_X, train_y)
mean_pred = pd.DataFrame(predictions.mean(axis=1))
mean_pred.index += 250
mean_pred.columns = ['target']
mean_pred.to_csv('/Users/JoonH/DO2_lasso_kernel_submission1.csv', index_label='id', index=True)
# original approach saved without 1 suffix, 0.870 LB score
# mean_pred.to_csv('/Users/JoonH/DO2_lasso_kernel_submission.csv', index_label='id', index=True)


# define roc_auc_metric robust to only one class in y_pred
def scoring_roc_auc(y, y_pred):
    try:
        return roc_auc_score(y, y_pred)
    except:
        return 0.5


robust_roc_auc = make_scorer(scoring_roc_auc)


def build_and_train_model(mtype, train_X, train_y):
    if mtype == 'lasso':
        model = Lasso(alpha=0.031, tol=0.01, random_state=random_seed, selection='random')
        param_grid = {
            'alpha': [0.019, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.029, 0.031],
            'tol': [0.001, 0.0011, 0.0012, 0.0013, 0.0014, 0.0015, 0.0016, 0.0017],
            'max_iter': [250, 500, 750, 1000, 1250, 1500, 1750, 2000]
        }

        print("counter | val_mse  |  val_mae  |  val_roc  |  val_cos  |  val_dist  |  val_r2    | feature_count ")
    print("-------------------------------------------------------------------------------------------------")

    predictions = pd.DataFrame()
    counter = 0
    # split training data to build one model on each traing-data-subset

    for train_index, val_index in StratifiedShuffleSplit(n_splits=sss_n_splits, test_size=sss_test_size,
                                                         random_state=random_seed).split(train_X, train_y):

        X, val_X = train_X[train_index], train_X[val_index]
        y, val_y = train_y[train_index], train_y[val_index]
        # define recursive elimination feature selector
        grid_search = GridSearchCV(model, param_grid=param_grid, verbose=0, n_jobs=-1, scoring=robust_roc_auc,
                                   cv=gridsearchCV)
        grid_search.fit(X, y)
        feature_selector = RFECV(grid_search.best_estimator_, min_features_to_select=rfe_min_features,
                                 scoring=robust_roc_auc, step=rfe_step, verbose=0, cv=rfe_cv, n_jobs=-1)

        # get the best features for this data set
        feature_selector.fit(X, y)
        # remove irrelevant features from X, val_X and test
        X_important_features = feature_selector.transform(X)
        val_X_important_features = feature_selector.transform(val_X)
        test_important_features = feature_selector.transform(test)

        # run grid search to find the best Lasso parameters for this subset of training data and subset of features
        # grid_search = GridSearchCV(feature_selector.estimator_, param_grid=param_grid, verbose=0, n_jobs=-1, scoring=robust_roc_auc, cv=gridsearchCV)
        grid_search = RandomizedSearchCV(feature_selector.estimator_, param_distributions=param_grid, verbose=0,
                                         n_jobs=-1, scoring=robust_roc_auc, cv=gridsearchCV)

        grid_search.fit(X_important_features, y)
        print(grid_search.best_params_)
        # score our fitted model on validation data
        val_y_pred = grid_search.best_estimator_.predict(val_X_important_features)
        val_mse = mean_squared_error(val_y, val_y_pred)
        val_mae = mean_absolute_error(val_y, val_y_pred)
        val_roc = roc_auc_score(val_y, val_y_pred)
        val_cos = cosine_similarity(val_y.values.reshape(1, -1), val_y_pred.reshape(1, -1))[0][0]
        val_dst = euclidean_distances(val_y.values.reshape(1, -1), val_y_pred.reshape(1, -1))[0][0]
        val_r2 = r2_score(val_y, val_y_pred)

        # if model did well on validation, save its prediction on test data, using only important features
        # r2_threshold (0.185) is a heuristic threshold for r2 error
        # you can use any other metric/metric combination that works for you
        if val_r2 > r2_threshold:
            message = '<-- OK'
            prediction = grid_search.best_estimator_.predict(test_important_features)
            predictions = pd.concat([predictions, pd.DataFrame(prediction)], axis=1)
        else:
            message = '<-- skipping'

        print(
            "{0:2}      | {1:.4f}   |  {2:.4f}   |  {3:.4f}   |  {4:.4f}   |  {5:.4f}    |  {6:.4f}    |  {7:3}         {8}  ".format(
                counter, val_mse, val_mae, val_roc, val_cos, val_dst, val_r2, feature_selector.n_features_, message))

        counter += 1

    print("-------------------------------------------------------------------------------------------------")
    print("{}/{} models passed validation threshold and will be ensembled.".format(len(predictions.columns),
                                                                                   sss_n_splits))

    return predictions


predictions = build_and_train_model('lasso', train_X, train_y)
mean_pred = pd.DataFrame(predictions.mean(axis=1))
mean_pred.index += 250
mean_pred.columns = ['target']
mean_pred.to_csv('/Users/JoonH/DO2_lasso_kernel_submission1.csv', index_label='id', index=True)


#bootstrapping?
mean_pred.head()
#Bootstrapping
predictions = mean_pred.sort_values(['target'], ascending = False)
top = predictions[:5]
predictions = mean_pred.sort_values(['target'], ascending = True)
btm = predictions[:5]
bootstrap = pd.concat([top, btm],axis = 0)
test_df = pd.read_csv("/Users/JoonH/dont-overfit-ii/test.csv", index_col = 'id')
test_df.head()

bootstrap_features = test_df.iloc[list(bootstrap.index - 250)]
bootstrap_features.head()
bootstrap['id'] = bootstrap.index
bootstrap = bootstrap.set_index(['id'])
bootstrap_data = pd.concat([bootstrap,bootstrap_features],axis=1)


def special_round(n):
    if (n > 0.5):
        return 1
    else:
        return 0


bootstrap_data['target'] = [special_round(x) for x in bootstrap['target']]
bootstrap_data.head()

bootstrap_data = bootstrap_data.reset_index(drop = True)
bootstrap_data.head()

comb_df = pd.concat([bootstrap_data,train.drop(['id'],axis=1)],axis=0).reset_index(drop = True)
comb_df.head()

x_train = comb_df.drop(['target'],axis=1)
y_train = comb_df['target']
x_test = test_df.reset_index(drop = True)

x_train = np.array(x_train)
x_test = np.array(x_test)


data_boot = pp.RobustScaler().fit_transform(np.concatenate((x_train, x_test), axis=0))
x_train = data_boot[:260]
x_test = data_boot[260:]

# add a bit of noise to train_X to reduce overfitting
x_train += np.random.normal(0, noise_std, x_train.shape)

r2_threshold = 0.185
predictions = build_and_train_model('lasso',x_train, y_train)
mean_pred = pd.DataFrame(predictions.mean(axis=1))
mean_pred.index += 250
mean_pred.columns = ['target']
mean_pred.to_csv('/Users/JoonH/DO2_lasso_kernel_submission1_bootstrap.csv', index_label='id', index=True)