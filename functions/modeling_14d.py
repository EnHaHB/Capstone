import os, glob
import json

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
    model = clf.fit(X_train, y_train)

    if cv:
        cv = cross_validate(m_rf, X_train_trans, y_train, cv=5, verbose=5)
        print(f"Best cross-validated score: {cv['test_score'].mean()}")
    
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)
    
    print(f"--- MODEL PARAMETERS {'-'*10}")
    print(json.dump(model.get_params(), indent=4))
    print(f"--- F1-Score {'-'*10}")
    print(f1_score(y_test, y_pred, labels=np.unique(y_pred), pos_label= 'Y'))
    print(f"--- CLASSIFICATION REPORT {'-'*10}")
    print(classification_report(y_test,y_pred))
    print(f"--- CONFUSION MATRIX {'-'*10}")
    print(confusion_matrix(y_test,y_pred))
    plot_confusion_matrix(model, X_test, y_test, display_labels=["deceased", "alive"])
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
    print(f"Best cross-validated F1-Score: {gs.f1}")
    print(f"--- CLASSIFICATION REPORT {'-'*10}")
    print(classification_report(y_test,y_pred))
    print(f"--- CONFUSION MATRIX {'-'*10}")
    print(confusion_matrix(y_test,y_pred))
    plot_confusion_matrix(gs.best_estimator_, X_test, y_test, y_test, display_labels=["deceased", "alive"])
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
    gs = RandomizedSearchCV(clf, params_grid, n_iter=n_iter, cv=cv, random_state=42, verbose=5, scoring = f1)
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
    gs = GridSearchCV(clf, params_grid, cv=cv, verbose=5, scoring = f1)
    return _pred_eval_plot_grid(X_train, X_test, y_train, y_test, gs)
    