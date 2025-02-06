import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
le=LabelEncoder()
filename ="QuadraticDiscriminantAnalysis_model_strock.pkl"
with open(filename,"rb")as file:
    loaded_model=pickle.load(file)


st.title('Strock Prediction App')
st.subheader('Please enter your data:')


df = pd.read_csv('healthcare-dataset-stroke-data.csv')
columns_list = df.columns.to_list()


uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])



if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    df["bmi"].fillna(df["bmi"].median(),inplace=True)
    

    work_type_mapping = {
    'Private': 0,
    'Self-employed': 1,
    'Govt_job': 2,
    'children': 3,
    'Never_worked': 4}

    df["gender"]=le.fit_transform(df["gender"])
    df["ever_married"]=le.fit_transform(df['ever_married'])
    df["smoking_status"]=le.fit_transform(df["smoking_status"])

    df['work_type'] = df['work_type'].map(work_type_mapping)    
    


    df["Residence_type"]=le.fit_transform(df["Residence_type"])

    
    
    prediction = loaded_model.predict(df)
    prediction_text = np.where(prediction == 1, 'Yes', 'No')
    st.subheader('stroke prection')
    st.write(prediction_text)