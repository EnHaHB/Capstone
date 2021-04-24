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


def pred_model(clf, X_train, X_test, y_train, y_test):
    model = clf.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(classification_report(y_test,y_pred))
    print(confusion_matrix(y_test,y_pred))
    plot_confusion_matrix(model, X_test, y_test)
    return model

def pred_eval_plot_model(X_train, X_test, y_train, y_test, clf, cv=None):
    """Train a single model and print evaluation metrics.

    Args:
        X_train (pd.DataFrame, np.array): Features of the training set
        X_test (pd.DataFrame, np.array): Features of thee test set
        y_train (pd.Series, np.array): Target of the training set
        y_teset (pd.Seeries, np.array): Target of the test set
        clf (sklearn.base.BaseEstimator): Estimator to train and use
        cv (int, None): Number of cross-validations, default=None
    
    Returns:
        model (sklearn.base.BaseEstimator): The trained model
    """


    if cv:
        #model = cross_validate(clf, X_train, y_train, cv=5, verbose=5)
        #print(f"Best cross-validated score: {model['test_score'].mean()}")
        y_train_pred = cross_val_predict(clf, X_train, y_train, cv=5)
        y_pred = cross_val_predict(clf, X_test, y_test, cv=5)
    else: 
        model = clf.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_pred = model.predict(X_test)
        print(f"--- MODEL PARAMETERS {'-'*10}")
        #print(json.dumps(model.get_params(), indent=4))
        print(model.get_params())
        plot_confusion_matrix(model, X_test, y_test)
    
    print(f"--- CLASSIFICATION REPORT {'-'*10}")
    print(classification_report(y_test,y_pred))
    print(f"--- CONFUSION MATRIX {'-'*10}")
    print(confusion_matrix(y_test,y_pred))
    return model

def _pred_eval_plot_grid(X_train, X_test, y_train, y_test, gs):
    """Helper function to perform a grid search and calculate performance metrics.
    
    Args:
        X_train (pd.DataFrame, np.array): Features of the training set
        X_test (pd.DataFrame, np.array): Features of thee test set
        y_train (pd.Series, np.array): Target of the training set
        y_test (pd.Seeries, np.array): Target of the test set
        gs (BaseSearchCV): SearchCV to train and use
    
    Returns:
        model (BaseSearchCV): The trained grid search
    """
    gs = gs.fit(X_train, y_train)
    
    # Testing predictions (to determine performance)
    y_pred = gs.best_estimator_.predict(X_test)

    print(f"--- GRID SEARCH RESULTS {'-'*10}")
    print(f"Best model: {gs.best_params_}")
    print(f"Best cross-validated Score: {gs.best_score_}")
    print(f"--- CLASSIFICATION REPORT {'-'*10}")
    print(classification_report(y_test,y_pred))
    print(f"--- CONFUSION MATRIX {'-'*10}")
    print(confusion_matrix(y_test,y_pred))
    plot_confusion_matrix(gs.best_estimator_, X_test, y_test)
    return gs
    

def run_rand_grid_search(X_train, X_test, y_train, y_test, clf, params_grid, n_iter=10, cv=5):
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
    gs = RandomizedSearchCV(clf, params_grid, n_iter=n_iter, cv=cv, random_state=42, verbose=5, scoring = 'precision')
    return _pred_eval_plot_grid(X_train, X_test, y_train, y_test, gs)
    
def run_grid_search(X_train, X_test, y_train, y_test, clf, params_grid, cv=5):
    """Perform a grid search and calculate performance metrics.
    
    Args:
        X_train (pd.DataFrame, np.array): Features of the training set
        X_test (pd.DataFrame, np.array): Features of thee test set
        y_train (pd.Series, np.array): Target of the training set
        y_teset (pd.Seeries, np.array): Target of the test set
        clf (sklearn.base.BaseEstimator): Estimator to train and use
        params_grid (dict): Dictionary defining the parameters for the grid search
        cv (int, None): Number of cross-validations, default=None
        
    Returns:
        model (BaseSearchCV): The trained grid search
    """
    gs = GridSearchCV(clf, params_grid, cv=cv, verbose=5)
    return _pred_eval_plot_grid(X_train, X_test, y_train, y_test, gs)


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

        ari = adjusted_rand_score(
        y_train,
        pipe_clusterer.labels_,
        )
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

def pred_cluster(pipe, pipe_preprocessor, pipe_clusterer, X_train_trans, X_test_trans, y_test):
    """ Predict clusters of X_train and generate plot of outcome (based on PCA)
    
    Args:
        pipe: Pipe for the entire model
        pipe_preprocessor: Pipe for PCA
        pipe_clusterer: Pipe for clusterer
        y_test: true labels
        X_train_test (): pre-transformed X_train (scaled and encoded)
    
    Returns:
        Plot of first 2 PCs
    """
    pipe_preprocessor.n_components = 2
    pipe.fit(X_train_trans)
    y_pred = pipe.predict(X_test_trans)
    y_test = y_test.to_numpy()

    pcadf = pd.DataFrame(pipe_preprocessor.transform(X_test_trans),columns=["PC 1", "PC 2"])
    pcadf["predicted_cluster"] = y_pred
    pcadf["true_label"] = y_test

    scat = sns.scatterplot("PC 1", "PC 2", s=50, data=pcadf, hue="predicted_cluster", style="true_label")
    scat.set_title("Clustering results of test data")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.show()

def _f_score_i(cl_real_i, cl_pred_i):
    ''' Calculate f-score for a single posting_id
        f1-score is the mean of all f-scores
    Args:
        cl_real_i (list): list of IDs belonging to the real cluster
        cl_pred_i (list): list of IDs belonging to the predicted cluster

    Returns:
    float value of f-score
    '''
    s_pred = set(cl_pred_i)
    s_real = set(cl_real_i)
    s_intsec = s_pred.intersection(s_real)
    return 2*len(s_intsec) / (len(s_pred)+len(s_real))

def _recall_i(cl_real_i, cl_pred_i):
    ''' Calculate recall for a single posting_id
    Args:
        cl_real_i (list): list of IDs belonging to the real cluster
        cl_pred_i (list): list of IDs belonging to the predicted cluster
    
    Returns:
    float value of recall
    '''

    s_pred = set(cl_pred_i)
    s_real = set(cl_real_i)
    s_diff_r_p = s_real.difference(s_pred)
    return (len(s_real) - len(s_diff_r_p)) / len(s_real)

def _precision_i(cl_real_i, cl_pred_i):
    ''' Calculate precision for a single posting_id
    Args:
        cl_real_i (list): list of IDs belonging to the real cluster
        cl_pred_i (list): list of IDs belonging to the predicted cluster
    
    Returns:
    float value of precision
    '''
    
    s_pred = set(cl_pred_i)
    s_real = set(cl_real_i)
    s_diff_p_r = s_pred.difference(s_real)
    return (len(s_pred) - len(s_diff_p_r)) / len(s_pred)


def get_cluster_metrics(cl_real_i, cl_pred_i):
    ''' Calculate cluster metrics and return average of each
    Args:
        cl_real_i (list): list of IDs belonging to the real cluster
        cl_pred_i (list): list of IDs belonging to the predicted cluster
    
    Returns:
    Average value of F-Score, Recall, Precision and Adj. Rand Index
    '''


    f_scores = []
    for i in range(len(cl_real_i)):
        f_score = _f_score_i(cl_real_i[i], cl_pred_i[i])
        f_scores.append(f_score)

    recalls = []
    for i in range(len(cl_real_i)):
        recall = _recall_i(cl_real_i[i], cl_pred_i[i])
        recalls.append(recall)

    precisions = []
    for i in range(len(cl_real_i)):
        precision = _precision_i(cl_real_i[i], cl_pred_i[i])
        precisions.append(precision)
    

    print(f"           Average F_score: {round(np.mean(f_scores), 3)}")
    print(f"            Average Recall: {round(np.mean(recalls), 3)}")
    print(f"         Average Precision: {round(np.mean(precisions), 3)}")
    return np.mean(f_scores), np.mean(recalls), np.mean(precisions)
    
