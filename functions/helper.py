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


def evaluate_model(clf, X_train, y_train, scoring = 'recall', repeats = 5):
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
    scores = cross_val_score(clf, X_train, y_train, scoring=scoring, cv=cv, n_jobs=-1)
    print(f'Average Score: {round(np.mean(scores),3)} with an Standard Error of {round(sem(scores), 4)}')

    return scores

def pred_model(clf, X_train, X_test, y_train, y_test):
    ''' Predict model

    Args:
        X_train (pd.DataFrame, np.array): Features of the training set
        y_train (pd.Series, np.array): Target of the training set
        X_test (pd.DataFrame, np.array): Features of the test set
        y_test (pd.Series, np.array): Target of the test set
        clf (sklearn.base.BaseEstimator): Estimator to train and use

    Returns:
        model (list, None): The predicted model
        y_pred (pd.DataFrame, np.array): Predicted values of the test set
    '''
    model = clf.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(classification_report(y_test,y_pred))
    print(confusion_matrix(y_test,y_pred))
    plot_confusion_matrix(model, X_test, y_test)
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    print (f'F1-Score : {round(f1_score(y_test, y_pred), 4)}')
    return model, y_pred


def rs_model(clf, X_train, y_train, params_grid, scoring = 'recall', repeats = 3, n_splits = 10):
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
    rs = RandomizedSearchCV(clf, params_grid, n_iter=500, scoring=scoring, n_jobs=-1, cv=cv, random_state=42, verbose = 5)
    # execute search
    rs_model = rs.fit(X_train, y_train)
    # summarize result
    print('Best Score: %s' % rs_model.best_score_)
    print('Best Hyperparameters: %s' % rs_model.best_params_)

    return rs_model.best_score_, rs_model.best_params_


def ari_scores(pipe, pipe_preprocessor, pipe_clusterer, X_train_trans, y_train):

    """Calculate Adjusted Rand Score, plot ARI for a range(2 to 11) of PC components
    
    Args:
        pipe: Pipe for entire model
        pipe_preprocessor: Pipe for PCA
        pipe_clusterer: Pipe for clusterer
        y_train (pd.Series, np.array): Target of the training set
        X_train_trans (): pre-transformed X_train (scaled and encoded)
    
    Returns:
        Plot of ARI
    """

    ari_scores = []

    for n in range(2, 11):
        # This set the number of components for pca,
        # but leaves other steps unchanged
        pipe_preprocessor.n_components = n
        pipe.fit(X_train_trans)

        ari = adjusted_rand_score(y_train, pipe_clusterer.labels_,)
        # Add ari to the lists
        ari_scores.append(ari)

    plt.plot(range(2, 11), ari_scores, c="#fc4f30")
    plt.xlabel("Principal Components")
    plt.tight_layout()
    plt.show()

def knee_loc(pipe, pipe_clusterer, X_train_trans):

    """Locate knee for clusterer and plot WCSS
    
    Args:
        pipe: Pipe for the entire model
        pipe_clusterer: Pipe for clusterer
        X_train_trans (): pre-transformed X_train (scaled and encoded)
    
    Returns:
        knee/elbow
        Plot of WCSS
    """

    wcss = []
    for i in range(1, 11):
        pipe_clusterer.n_clusters = i
        pipe.fit(X_train_trans)
        wcss.append(pipe_clusterer.inertia_)

    kl = KneeLocator(range(1, 11), wcss, curve="convex", direction="decreasing")
    print(kl.elbow)

    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Within cluster sum of squares (WCSS)')
    plt.show()