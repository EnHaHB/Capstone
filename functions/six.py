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
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=42)
	
    # evaluate model
    scores = cross_val_score(clf, X_train, y_train, scoring='recall', cv=cv, n_jobs=-1)
    print(f'Average Score: {round(np.mean(scores),3)} with an Standard Error of {round(sem(scores), 4)}')

    return scores

    
def run_rand_grid_search(clf, X_train, y_train, params_grid, n_iter=10):
    """Perform a randomized grid search and calculate performance metrics.
    
    Args:
        X_train (pd.DataFrame, np.array): Features of the training set
        X_test (pd.DataFrame, np.array): Features of thee test set
        y_train (pd.Series, np.array): Target of the training set
        y_teset (pd.Seeries, np.array): Target of the test set
        clf (sklearn.base.BaseEstimator): Estimator to train and use
        params_grid (dict): Dictionary defining the parameters for the grid search
        n_iter (int): Number of grid search combinations to run
        cv (int, None): Number of cross-validations, default=None
        
    Returns:
        model (BaseSearchCV): The trained grid search
    """
    # cross validation procedure
    cv = RepeatedStratifiedKFold(n_splits= 5, n_repeats= 3, random_state=42)

    #randomized search
    gs = RandomizedSearchCV(clf, params_grid, n_iter=n_iter, cv=cv, random_state=42, verbose=5, scoring = 'recall')

    #model fitting
    gs_model = gs.fit(X_train, y_train)

    print(f"--- GRID SEARCH RESULTS {'-'*10}")
    print(f"Best model: {gs.best_params_}")
    print(f"Best cross-validated Score: {gs.best_score_}")

    return gs_model
