from mushroom_preprocessing import preprocessing, removed_indices
from mushroom_postprocessing import best_thresholds_dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import make_scorer, fbeta_score, confusion_matrix


def get_model(original_df, n):
    """Gets a dataframe and a serial and returns the corresponding model"""
    
    df_encoder, df = preprocessing(original_df)
    X_train, X_test, y_train, y_test = data_split(df)
    estimators = run_models(X_train, y_train, GridSearchCV)
    thresholds = best_thresholds_dict(X_train, y_train, estimators)
    if n < 5:
        return estimators
    elif n == 5:
        clf = "Random Forest"
    elif n == 6:
        clf = "KNN"
    return MushroomModel(estimators[clf], thresholds[clf], original_df, 
                         df_encoder, removed_indices(original_df, df))
        
def data_split(df):
    """Train-Test splits the data"""
    
    X = df.drop("class", axis=1)
    y = df["class"]
    return train_test_split(X, y, test_size=0.3, random_state=10)

def run_models(X, y, searchCV):
    """Runs classification models using GridSearchCV or RandomizedSearhCV with
    f_0.5 scoring function and outputs their best estimators and a table with 
    their best scores and parameters"""
    
    grid_params = {"Logistic Regression": {"model": LogisticRegression(), "params": {"C": [.1, 1, 8, 9, 10], "max_iter": [300], "solver": ["lbfgs", "liblinear"]}},
                     "SGD": {"model": SGDClassifier(), "params": {"loss": ["modified_huber"]}},
                     "KNN": {"model": KNeighborsClassifier(), "params": {"n_neighbors": range(1,10), "weights": ["uniform", "distance"]}},
                     "SVM": {"model": SVC(), "params": {"C": [.01, .1, .5, 1, 5, 10], "kernel": ["linear", "poly", "rbf", "sigmoid"]}},
                     "Naive Bayes": {"model": GaussianNB(), "params": {}},
                     "Decision Tree": {"model": DecisionTreeClassifier(), "params": {"criterion": ["gini", "entropy"]}},
                     "Random Forest": {"model": RandomForestClassifier(), "params": {"n_estimators": [50, 100, 200], "criterion": ["gini", "entropy"]}}}
    scores = []
    estimators = {}
    for model, mp in grid_params.items():
        clf = searchCV(mp["model"], mp["params"], scoring=make_scorer(fbeta_score, beta=.5), n_jobs=-1, cv=5)
        clf.fit(X, y)
        scores.append({"Model": model, "Best Score": clf.best_score_, "Best Parameters": clf.best_params_})
        estimators[model] = clf.best_estimator_
    print(pd.DataFrame(scores, columns=["Model", "Best Score", "Best Parameters"]).to_string())
    return estimators
    
def best_knn(X, y, n):
    """Runs the KNN model and plots a chart of K values from 1 to n with their scores"""
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
    scoreList = []
    for i in range(1,n + 1):
        knn = GridSearchCV(KNeighborsClassifier(n_neighbors = i, n_jobs=-1), {"weights": ["uniform", "distance"]}, scoring=make_scorer(fbeta_score, beta=.5), cv=5)
        knn.fit(X_train, y_train)
        scoreList.append(knn.best_score_)
    plt.plot(range(1,n + 1), scoreList)
    plt.xticks(np.arange(1,n + 1,1))
    plt.xlabel("K value")
    plt.ylabel("Score")
    plt.show()
    print("Maximum KNN Score:", max(scoreList), "for K =", scoreList.index(max(scoreList)) + 1)
    
def pred_dict(estimators, X_test):
    """Gets a dictionary of estimators and returns a dictionary of their predictions over a test set"""
    
    return dict(zip(estimators.keys(), [x.predict(X_test) for x in estimators.values()]))

def xgboost_model(X, y, searchCV):
    """Runs an XGBoost classifier on X and y using GridSearchCV or RandomizedSearhCV 
    with f_0.5 scoring function and returns the best score with the best estimator"""
    
    params = {
        'min_child_weight': [0.1, 0.5, 1, 5],
        'gamma': [0.01, 0.1, 0.5, 1, 1.5, 2],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 2],
        'max_depth': [2, 3, 4, 10]
        }
    xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic', silent=True, nthread=1)
    clf = searchCV(xgb, params, scoring=make_scorer(fbeta_score, beta=.5), n_jobs=-1, cv=5)
    clf.fit(X, y)
    print("XGBoost best score:", clf.best_score_)
    print("XGBoost best parameters:", clf.best_params_)
    return clf.best_estimator_

def clustering_score(labels, cl_labels):
    """Calculates an f_0.5 score between the original labels feature and a given clustering labels vector"""
    
    cm = confusion_matrix(labels, cl_labels)
    tp = cm[1][1]
    fp = cm[0][1]
    fn = cm[1][0]
    return (1.25 * tp) / (1.25 * tp + 0.25 * fn + fp)

def clustering_model(original_df):
    """Creates clustering models for the dataframe by using 2 clusters that 
    would substitute the original binary target, and outputs the clustering 
    labels vectors and a table with the models scores"""
    
    df_encoder, df = preprocessing(original_df)
    kmeans = KMeans(n_clusters=2, n_jobs=-1, random_state=0).fit(df.drop("class", axis=1))
    gmm = GaussianMixture(n_components=2, random_state=0).fit(df.drop("class", axis=1))
    kmeans_pred = kmeans.predict(df.drop("class", axis=1))
    gmm_pred = gmm.predict(df.drop("class", axis=1))
    kmeans_score = clustering_score(df["class"], kmeans_pred)
    gmm_score = clustering_score(df["class"], gmm_pred)
    estimators = {"KMeans": kmeans, "Gaussian Mixture": gmm}
    print(pd.DataFrame([["KMeans", kmeans_score],["Gaussian Mixture", gmm_score]], columns=["Model", "Score"]))
    return MushroomModel(estimators["KMeans"], 0.5, original_df, df_encoder, 
                         removed_indices(original_df, df))

def unique_values(df, del_idx):
    """Gets a dataframe and its deleted indices during the preprocessing and 
    returns the updated unique values as a matrix"""
    
    unique_mat = []
    del_col = pd.get_dummies(df.drop("class", axis=1), drop_first=True).columns[del_idx]
    for f in df.drop("class", axis=1).columns:
        values = df[f].unique().tolist()
        for c in del_col:
            if f == c[:-2]:
                values.remove(c[-1])
        unique_mat.append(values)
    return unique_mat


class MushroomModel:
    """Class for the final model we achieved after all the analysis. 
    Gets the estimator, its best decision threshold, the original dataframe, 
    the OHE fitted object of the original dataframe and the indices of the 
    removed columns during the preprocessing"""

    
    def __init__(self, estimator, threshold, df, encoder, del_idx):
        self.estimator = estimator
        self.threshold = threshold
        self.df = df
        self.encoder = encoder
        self.del_idx = del_idx
        self.uniques = unique_values(df, del_idx)
        
    def random_predict(self):
        sample = []
        for i, col in enumerate(self.df.drop("class", axis=1).columns):
            sample.append(random.choice(self.uniques[i]))
            print(col + ":", sample[-1])
        return self.predict(sample)
        
    def sample_fit(self, sample):
        sample = self.encoder.transform(np.array(sample).reshape(1,-1))[0]
        sum_check = sample.sum()
        sample = np.array([x for i, x in enumerate(sample) if i not in self.del_idx]).reshape(1,-1)
        if sum_check != sample.sum():
            raise ValueError("Invalid input")
        return sample
    
    def predict(self, sample):
        sample = self.sample_fit(sample)
        est_name = type(self.estimator).__name__
        if  est_name == "SVC" or est_name == "KMeans":
            return self.estimator.predict(sample)
        return (self.estimator.predict_proba(sample)[:,1] >= self.threshold).astype("int")
