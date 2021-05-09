import streamlit as st
import pandas as pd
import timeit
import pickle
import random

start = timeit.default_timer()

# Setting configuration and background
st.set_page_config(page_title="Mushroom Predictions", page_icon=":mushroom:")

page_bg_img = '''
<style>
body {
background-image: url("https://www.linkpicture.com/q/test2_3.jpg");
background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Setting title and description
st.title("Mushroom Prediction App")
st.write("This app predicts the edibility of a mushroom.")
st.write("""The data set includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota families.
         The app is based on 3 ML models that I built and that can be found in my [Github](https://github.com/idansul/data_science/tree/main/mushroom_project).""")

# Loading the dataframe
@st.cache
def get_data():
    return pd.read_csv("mushroom_project/Mushrooms.csv")

data_exp = st.beta_expander("Mushroom Data")
df = get_data()
data_exp.dataframe(df)
data_exp.write("To see the full attribute information [click here](https://archive.ics.uci.edu/ml/datasets/mushroom).")

# Loading the models
@st.cache(allow_output_mutation=True)
def get_models():
    models = []
    with open("mushroom_project/colors_model", "rb") as pkl:
        models.append(pickle.load(pkl))
    with open("mushroom_project/area_model", "rb") as pkl:
        models.append(pickle.load(pkl))
    with open("mushroom_project/clustering_model", "rb") as pkl:
        models.append(pickle.load(pkl))
    return models

models = get_models()
models = dict(zip(["Colors", "Area", "Clustering"], models))

# Selecting a model
model = models[st.selectbox("Select Model:", list(models.keys()))]

# Making customized and random predictions
st.subheader("Make predictions:")
features = {"cap-shape": {"b": "bell", "c": "conical", "x": "convex", "f": "flat", "k": "knobbed", "s": "sunken"},
            "cap-surface": {"f": "fibrous", "g": "grooves", "y": "scaly", "s": "smooth"},
            "cap-color": {"n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "r": "green", "p": "pink", "u": "purple", "e": "red", "w": "white", "y": "yellow"},
            "bruises": {"t": "bruises", "f": "no"},
            "odor": {"a": "almond", "l": "anise", "c": "creosote", "y": "fishy", "f": "foul", "m": "musty", "n": "none", "p": "pungent", "s": "spicy"},
            "gill-attachment": {"a": "attached", "d": "descending", "f": "free", "n": "notched"},
            "gill-spacing": {"c": "close", "w": "crowded", "d": "distant"},
            "gill-size": {"b": "broad", "n": "narrow"},
            "gill-color": {"k": "black", "n": "brown", "b": "buff", "h": "chocolate", "g": "gray", "r": "green", "o": "orange", "p": "pink", "u": "purple", "e": "red", "w": "white", "y": "yellow"},
            "stalk-shape": {"e": "enlarging", "t": "tapering"},
            "stalk-root": {"b": "bulbous", "c": "club", "u": "cup", "e": "equal", "z": "rhizomorphs", "r": "rooted", "?": "missing"},
            "stalk-surface-above-ring": {"f": "fibrous", "y": "scaly", "k": "silky", "s": "smooth"},
            "stalk-surface-below-ring": {"f": "fibrous", "y": "scaly", "k": "silky", "s": "smooth"},
            "stalk-color-above-ring": {"n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "o": "orange", "p": "pink", "e": "red", "w": "white", "y": "yellow"},
            "stalk-color-below-ring": {"n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "o": "orange", "p": "pink", "e": "red", "w": "white", "y": "yellow"},
            "veil-color": {"n": "brown", "o": "orange", "w": "white", "y": "yellow"},
            "ring-number": {"n": "none", "o": "one", "t": "two"},
            "ring-type": {"e": "evanescent", "f": "flaring", "l": "large", "n": "none", "p": "pendant", "s": "sheathing", "z": "zone"},
            "spore-print-color": {"k": "black", "n": "brown", "b": "buff", "h": "chocolate", "r": "green", "o": "orange", "u": "purple", "w": "white", "y": "yellow"},
            "population": {"a": "abundant", "c": "clustered", "n": "numerous", "s": "scattered", "v": "several", "y": "solitary"},
            "habitat": {"g": "grasses", "l": "leaves", "m": "meadows", "p": "paths", "u": "urban", "w": "waste", "d": "woods"}}

def predict(p):
    if p:
        st.success("Edible")
    else:
        st.error("Poisonous")

sample = {}
original_sample = {}
for i, col in enumerate(model.df.drop("class", axis=1).columns):
    sample[col] = st.selectbox(col, list(features[col].values()))
    original_sample[col] = list(features[col].keys())[list(features[col].values()).index(sample[col])]
predict(model.predict(list(original_sample.values())))

if st.button("Random Prediction"):
    sample = []
    for i, col in enumerate(model.df.drop("class", axis=1).columns):
        sample.append(random.choice(model.uniques[i]))
        st.write(col + ":", features[col][sample[-1]])
    predict(model.predict(sample)[0])

stop = timeit.default_timer()
st.text(f"Running time: {round(stop - start, 4)} sec")

# Credit for the drawings
credit = st.beta_columns([2.2, 0.7])
credit[1].markdown("###### Drawings by [Martha Iserman](https://www.bigredsharks.com/)")
