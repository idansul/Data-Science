import numpy as np
import pandas as pd
from urllib import request
from sklearn.preprocessing import OneHotEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor


def preprocessing(df=None):
    """Returns either the original dataframe after cleaning or a preprocessed 
    dataframe after encoding and deleting its features with perfect correlation"""
    
    if df is None:
        df = get_dataset()
        data_cleaning(df)
        return df
    else:
        df_encoder, df = one_hot_encoding(df)
        df = del_perfect_corr(df)
        return df_encoder, df
                
def get_dataset():
    """Returns the original Mushroom dataset from the UCI repository website"""
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
    data = request.urlopen(url)
    dfcolumns = ["class", "cap-shape", "cap-surface", "cap-color", "bruises", 
                 "odor", "gill-attachment", "gill-spacing", "gill-size", 
                 "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring",
                 "stalk-surface-below-ring", "stalk-color-above-ring",
                 "stalk-color-below-ring", "veil-type", "veil-color", "ring-number",
                 "ring-type", "spore-print-color", "population", "habitat"]
    df = pd.read_csv(data, sep=",", skiprows=0, names=dfcolumns)
    # df.to_csv(os.getcwd() + "\\Mushrooms.csv", index=False)
    return df
    
def data_cleaning(df):
    """Cleans the data as part of the preprocessing stage"""  
    
    df.drop("veil-type", axis=1, inplace=True)
    
def ohe_to_df(ohe, df):
    """Gets a OneHotEncoder matrix and returns it as a dataframe with the dummy columns of df"""
    
    columns = pd.get_dummies(df, drop_first=True).columns
    ohe_df = pd.DataFrame(ohe, columns=columns[1:]).astype("int")
    ohe_df.insert(0, "class", df["class"])
    ohe_df["class"].replace({"e":1, "p":0}, inplace=True)
    return ohe_df

def one_hot_encoding(df):
    """Gets a dataframe, encodes it with OneHotEncoder and returns the fitted
    object with a dataframe version of it"""
    
    df_encoder = OneHotEncoder(sparse=False, drop="first").fit(df.drop("class", axis=1))
    df = ohe_to_df(df_encoder.transform(df.drop("class", axis=1)), df)
    return df_encoder, df

def removed_indices(original_df, df):
    """Returns the indices of the removed columns during the preprocessing stage"""
    
    cols = pd.get_dummies(original_df.drop("class", axis=1), drop_first=True).columns
    return [cols.tolist().index(x) for x in cols if x not in df.columns]
    
def df_corr_coeff(df, coeff):
    """Prints pairs of features with a correlation that is greater than or
    equal to the given coefficient (in absolute value)"""
    
    table = []
    upper = df.corr().where(np.triu(np.ones(df.corr().shape), k=1).astype(np.bool))
    for col in df.corr().columns:
        for i, val in enumerate(list(upper[col].dropna().values)):
            if abs(val) >= coeff:
                table.append((upper[col].dropna().name, df.corr().columns[i], val))
    table = pd.DataFrame(table, columns=["Feature 1", "Feature 2", "Correlation"])
    return table

def del_perfect_corr(df):
    """Deletes a feature for every pair of perfectly correlated features 
    (disregarding the target feature)"""
    
    features = df_corr_coeff(df, 1)
    if features.shape[0] == 0:
        return df
    features_groups = [{features["Feature 1"][0], features["Feature 2"][0]}]
    for i in range(1, features.shape[0]):
        if features["Feature 1"][i] not in set.union(*features_groups) and features["Feature 2"][i] not in set.union(*features_groups):
            features_groups.append({features["Feature 1"][i], features["Feature 2"][i]})
        elif features["Feature 1"][i] in set.union(*features_groups) and features["Feature 2"][i] in set.union(*features_groups):
            continue
        else:
            for group in features_groups:
                if features["Feature 1"][i] in group or features["Feature 2"][i] in group:
                    group.add(features["Feature 1"][i])
                    group.add(features["Feature 2"][i])
                    break
    for group in features_groups:
        group.pop()
        df = df.drop(group, axis=1)        
    return df

def fix_multicolinearity(df, n):
    """Fixes the multicolinearity of a dataframe by dropping features with 
    high Variance Inflation Factor (VIF) until all features have a VIF of less 
    than n. Returns the updated dataframe and the VIF dataframe with VIF values
    of the remaining features"""
    
    vif = pd.DataFrame()
    vif["features"] = df.columns
    vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    while any(i > n for i in vif["VIF"].values):
        for f in range(1, df.shape[1]):
            if vif.iloc[f][1] > n:
                df = df.drop(vif.iloc[f][0], axis=1)
                break
        vif = pd.DataFrame()
        vif["features"] = df.columns
        vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return df, vif