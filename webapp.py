import streamlit as st
import mysql.connector
import pandas as pd
import numpy as np
from app_utils import *
from joblib import dump, load

# streamlit
print("Running Streamlit dashboard...")
st.title('Matbench Expt Gap Regression')
option = st.sidebar.selectbox('Options:', ['View data', 'View ML model', 'Input user-specified formula'])
st.header(option)

if option == "View data":
    st.write("The data used in this app is the Matbench Expt Gap dataset.")
    matbench_df = load_matbench_data("matbench_expt_gap")
    feature_data = pd.read_csv("feature_data_sql.csv")
    st.dataframe(matbench_df)
    st.write("Multiple features of the composition were introduced by using the matminer package.")
    st.dataframe(feature_data)

if option == "View ML model":
    st.write("The ML model used is a Random Forest Regressor based on composition features.")
    # load trained random forest model
    rf = load('rf.joblib')
    st.write("Plots and metrics showing the performance of the model are now displayed.")
    st.image("plots/randomsearchcv_train_prediction.png", caption="Train data prediction accuracy of RF model via hyperparameter tuning with RandomSearchCV.")
    st.image("plots/gridsearchcv_train_prediction.png", caption="Train data prediction accuracy of RF model via further hyperparameter tuning with GridSearchCV.")
    st.image("plots/gridsearchcv_test_prediction.png", caption="Test data prediction accuracy of RF model.")
    st.write("Parameters of the RF model:")
    st.write(rf.get_params())

if option == "Input user-specified formula":
    st.write("Please input the formula of the material you want to predict the experimental band gap for. Some formatting examples: Ag(AuS)2, Ag0.5Ge1Pb1.75S4, Ag2BBr, ZrSbRu")
    try:
        formula = st.text_input("Formula:")
        user_data = featurize_formula(formula)
        st.write(f"Featurized formula.")
        # load trained random forest model
        rf = load('rf.joblib')
        st.write("Predicting band gap using trained ML model...")
        pred_gap = float(rf.predict(user_data)[0])
        # pred_gap = 0
        st.write(f"The predicted band gap for the material with formula {formula} is: {pred_gap:.6f} eV.")
    except:
        st.write("INVALID formula. Please try again.")
