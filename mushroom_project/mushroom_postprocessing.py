import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, fbeta_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def kfold_predict_proba(X, y, model):
    """Trains a model using KFold cross-validation on X and y and returns a 
    dataframe with the labels and concatinated predict_probas for each split. 
    The idea is to use it to find an optimal threshold without an overfitting risk on the test set"""
    
    kf = KFold(n_splits=5, shuffle=True, random_state=10)
    clf = model
    pp_df = pd.DataFrame(columns=["Label", "Predict_proba"])
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        clf.fit(X_train, y_train)
        pp_df = pd.concat([pp_df, pd.DataFrame({"Label": y_test, "Predict_proba": clf.predict_proba(X_test)[:,1]})], axis=0)
    return pp_df

def roc_curves(X, y, estimators):
    """Plots ROC curves of estimators using K-fold predcit_proba"""
    
    estimators2 = dict(estimators)
    estimators2.pop("SVM")
    fig, axs = plt.subplots(2, 3, figsize=(15,9), constrained_layout=True)
    for i, clf in enumerate(estimators2.values()):
        clf_pp = kfold_predict_proba(X, y, clf)
        label = clf_pp["Label"].tolist()
        pp = clf_pp["Predict_proba"].tolist()
        fpr, tpr, thresholds = roc_curve(label, pp)
        area = auc(fpr, tpr)
        axs[i//3][i%3].plot(fpr, tpr, linestyle="-", label='ROC curve (area = %0.2f)' % area)
        axs[i//3][i%3].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title=list(estimators2.keys())[i])
        axs[i//3][i%3].legend()
    fig.suptitle("ROC Curves", fontsize=20)
    plt.show()

def untrain_models(estimators):
    """Gets a dictionary of trained estimators and returns a dictionary of the 
    same estimators but untrained and with the same parameters"""

    new_clfs = []
    for clf in estimators.values():
        new_clfs.append(eval(type(clf).__name__)().set_params(**clf.get_params()))
    return dict(zip(estimators.keys(), new_clfs))

def best_threshold(pred_df):
    """Given lists of predict_proba and labels, returns the best threshold under f_0.5 scoring"""
    
    label = pred_df["Label"].tolist()
    pred_proba = pred_df["Predict_proba"].tolist()
    fpr, tpr, thresholds = roc_curve(label, pred_proba)
    scores = np.array([fbeta_score(label, pred_proba >= thr, beta=.5) for thr in thresholds])
    return thresholds[scores.argmax()]

def best_thresholds_dict(X, y, estimators):
    """Gets a dictionary of optimal thresholds for given estimators and X, y train sets"""
    
    thresholds = []
    for clf in untrain_models(estimators).values():
        if type(clf).__name__ == "SVC":
            thresholds.append(0.5)
        else:
            thresholds.append(best_threshold(kfold_predict_proba(X, y, clf)))
    return dict(zip(estimators.keys(), thresholds))

def predict_by_threshold(model, threshold, test):
    """Model prediction by a given decision threshold"""
    
    if type(model).__name__ == "SVC":
        return model.predict(test)
    return (model.predict_proba(test)[:,1] >= threshold).astype("int")

def pred_by_threshold_dict(estimators, thresholds, test):
    """Returns a dictionary of models predictions by suitable decision thresholds"""
    
    preds = []
    for i, clf in enumerate(estimators.values()):
        preds.append(predict_by_threshold(clf, thresholds[list(thresholds.keys())[i]], test))
    return dict(zip(estimators.keys(), preds))
