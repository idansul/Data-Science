This data science project is about a classification of mushrooms into edible and poisonous categories. The dataset is taken from the UCI repository website: https://archive.ics.uci.edu/ml/datasets/Mushroom.

The python libraries that should be installed for this project are: numpy, pandas, matplotlib, seaborn, sklearn, xgboost. The notebook also makes use of the library lime.

The notebook describes with comments a step-by-step process of selecting 3 variations of models. The variations are: Colors - based only on features that involve colors; Area - based on population and habitat; Clustering - based on all of the features, and uses a clustering method for the model. All of these variations are accesible to use through the main script "mushrooms_main.py". The main script allows to either reconstruct all of the models from the beginning (takes about 2-3 minutes) or to load them directly from their pickles for a quick use. When choosing a model to use it's possible to either test it on random samples automatically or to enter samples manually. Entering samples manually should be done by choosing one of the possible values that are present in parentheses next to each feature. The output shows the class of the mushroom according to the model prediction; 1 is edible, 0 is poisonous.

To activate the code in the command line run the command "python mushroom_main.py".

### Web application
I created a Streamlit web application for getting predictions from the models: https://share.streamlit.io/idansul/data_science/main/mushroom_project/web_app.py.
