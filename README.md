# matminer-webapp

A streamlit (Python) dashboard that takes a dataset from the open-source Materials Project and uses a trained ML model to predict a material (e.g. band gap) property from material composition provided by the user. The ML model is a random forest regressor. The dataset is obtained and featurized using  the matminer package. Pandas dataframes and SQL tables are used to handle the data.
