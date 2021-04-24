import os, glob, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import sem
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix, classification_report, plot_confusion_matrix
from sklearn.metrics import f1_score, make_scorer, adjusted_rand_score,  silhouette_score

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_validate, cross_val_predict, cross_val_score, RepeatedStratifiedKFold
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, LabelEncoder, OrdinalEncoder

from kneed import KneeLocator

def evaluate_model(clf, X_train, y_train, repeats = 5):
    ''' Evaluate model with Cross validation

    Args:
        X_train (pd.DataFrame, np.array): Features of the training set
        y_train (pd.Series, np.array): Target of the training set
        clf (sklearn.base.BaseEstimator): Estimator to train and use

    Returns:
        results_train (list, None): List of alculated scores based on cross validation
    '''
	# prepare the cross-validation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=repeats, random_state=42)
	
    # evaluate model
    scores = cross_val_score(clf, X_train, y_train, scoring='precision', cv=cv, n_jobs=-1)

    return scores


def rs_model(clf, X_train, y_train, params_grid, repeats = 3, n_splits = 10):
    ''' Randomized Grid Search

    Args:
        X_train (pd.DataFrame, np.array): Features of the training set
        X_test (pd.DataFrame, np.array): Features of thee test set
        y_train (pd.Series, np.array): Target of the training set
        y_teset (pd.Seeries, np.array): Target of the test set
        clf (sklearn.base.BaseEstimator): Estimator to train and use

    Returns:
        rs_model.best_score_: Score of best model
        rs_model.best_params_: Parameters of best model
    '''

    cv = RepeatedStratifiedKFold(n_splits= n_splits, n_repeats= repeats, random_state=42)
    # define search
    rs = RandomizedSearchCV(clf, params_grid, n_iter=500, scoring='precision', n_jobs=-1, cv=cv, random_state=42)
    # execute search
    rs_model = rs.fit(X_train, y_train)
    # summarize result
    print('Best Score: %s' % rs_model.best_score_)
    print('Best Hyperparameters: %s' % rs_model.best_params_)

    return rs_model.best_score_, rs_model.best_params_