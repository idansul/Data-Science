import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import VarianceThreshold, f_classif, chi2, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.cluster import KMeans


def unique_values(df):
    """Gets unuique values in the dataframe"""
    
    print("Unique values:")
    for col in df.columns:
        print(col + ":", df[col].unique())

def features_countplot(df):
    """Count-plots all of the features in the dataframe"""

    fig, axs = plt.subplots(8, 3, figsize=(14,30), constrained_layout=True)
    for i, f in enumerate(df.columns):
        sb.countplot(x=f, data=df, ax=axs[i//3][i%3])
    plt.show()

def values_freq(df, n):
    """Prints a frequency of values under a threshold"""
    
    print("Number of values with frequency of less than", str(n) + ":")
    for col in df.columns:
        for val in list(df[col].unique()):
            if df[col][df[col] == val].count() < n:
                print(col, "of type", val + ":", df[col][df[col] == val].count())
    
def heatmap(df):
    """Plots a heatmap of a dataframe"""
    
    plt.figure(figsize=(22,22))
    sb.heatmap(df.corr(), cmap='coolwarm')
    
def confusion_mat(y_test, predictions):
    """Plots confusion matrices for a dictionary of estimators"""
    
    plt.figure(figsize=(18,12))
    plt.suptitle("Confusion Matrices",fontsize=30)
    plt.subplots_adjust(wspace = 0.4, hspace= 0.4)
    
    for i, model in enumerate(predictions.keys()):
        plt.subplot(3,3,i + 1)
        plt.title(model, fontsize=15, pad=10)
        sb.heatmap(confusion_matrix(y_test, predictions[model]), annot=True, cmap="Blues", fmt="d",cbar=False, annot_kws={"size": 25})
    plt.show()
    
def classification_rep(y_test, predictions):
    """Prints classification reports for a dictionary of estimators"""
    
    print("Classification Reports".center(125, "="))
    for i, model in enumerate(predictions.keys()):
        print(model.center(55, " "))
        print(classification_report(y_test, predictions[model]))
        print("*" * 54)
        
def feature_target_dist(df):
    """Plots the distribution of the features in the target"""
    
    fig, axs = plt.subplots(7, 3, figsize=(14,30), constrained_layout=True)
    for i, f in enumerate(df.drop("class", axis=1).columns):
        sb.countplot(x=f, hue="class", data=df, ax=axs[i//3][i%3])
    plt.show()
    
def feature_freq(df, feature):
    """Prints the frequency of a given feature of a dataframe when it's grouped by the target"""
    
    df = pd.get_dummies(df[["class", feature]], columns=[feature])
    for f in df.drop("class", axis=1).columns:
        fd = pd.DataFrame(df[["class", f]][df[f] == 1].groupby("class").count())
        fd["percent"] = fd[f] / fd[f].sum() * 100
        print(fd)
        print("*************************")
        
def var_threshold(X, threshold):
    """Runs a variance threshold algorithm and returns the features with a 
    variance that is lower than the threshold"""
    
    vt = VarianceThreshold(threshold)
    vt.fit(X)
    low_var_features = [x for x in X.columns if x not in X.columns[vt.get_support()]]
    print("Feaures with variance of less than " + str(threshold) + ":", ", ".join(low_var_features))
    print("Number of remaining features:", sum(vt.get_support()))
    return low_var_features

def stat_scores(X, y, n):
    """Plots statistical tests between the feature matrix X and the target 
    vector y for n features with the highest scores"""
    
    fc = f_classif(X, y)
    chi = chi2(X, y)
    mi = mutual_info_classif(X, y, random_state=10)
    rfi = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=10).fit(X, y).feature_importances_
    eti = ExtraTreesClassifier(n_estimators=100, random_state=10).fit(X, y).feature_importances_
    plt.title("ANOVA F-value stats")
    plot_feature_rank(X, fc[0], n)
    plt.title("Chi-squared stats")
    plot_feature_rank(X, chi[0], n)
    plt.title("Mutual Information stats")
    plot_feature_rank(X, mi, n)
    plt.title("Random Forest Importance")
    plot_feature_rank(X, rfi, n)
    plt.title("Extra Trees Importance")
    plot_feature_rank(X, eti, n)
    
def plot_feature_rank(X, stat, n):
    """Plots a chart of n important features from X using stat test"""
        
    pd.Series(stat).sort_values(ascending=False).nlargest(n).plot.bar(figsize = (16,4))
    plt.xticks(range(0,n), X.columns[pd.Series(stat).sort_values(ascending=False).nlargest(n).index], rotation=20)
    plt.show()
    
def kmeans_elbow(df, n):
    """Plots the clustering Elbow method for a dataframe in the range of 1 to n"""
    
    norm = []
    for k in range(1, n + 1):
        km = KMeans(n_clusters=k, n_jobs=-1)
        km.fit(df.drop("class", axis=1))
        norm.append(km.inertia_)
    plt.plot(range(1, n + 1), norm, 'bx-')
    plt.xlabel("Number of clusters")
    plt.ylabel("Norm (Sum of squared distances)")
    plt.title("Elbow method for optimal k")
