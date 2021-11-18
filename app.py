import streamlit as st
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(layout="wide")
df = pd.read_csv('winequality-red.csv')

column_transformer = pickle.load(open('column_transformer.pkl','rb'))
pipeline = pickle.load(open('pipeline.pkl','rb'))
selected_features = pickle.load(open('selected_features','rb'))

'''
Created on friday Nov 18, 2021 by Sujan Neupane
'''

st.title("Wine Quality Prediction")

fixed_acidity = st.number_input("Enter the fixed acitity of the wine: ")
volatile_acidity = st.number_input("Enter the volatile acidity of the wine: ")
citric_acid = st.number_input("Enter the citric acid present in the wine: ")
chlorides = st.number_input("Enter the amount of chlorides present in the wine: ")
total_sulphur_dioxde = st.number_input("Enter the amount of total sulphur dioxide present in the wine: ")
density = st.number_input("Enter the density of the wine: ")
sulphates = st.number_input("Enter the amount of sulphates present in the wine: ")
alcohol = st.number_input("Enter the amount of alcohol present in the wine: ")


X = pd.DataFrame({
    selected_features[0]: [fixed_acidity],
    selected_features[1]: [volatile_acidity],
    selected_features[2]: [citric_acid],
    selected_features[3]: [chlorides],
    selected_features[4]: [total_sulphur_dioxde],
    selected_features[5]: [density],
    selected_features[6]: [sulphates],
    selected_features[7]: [alcohol]
})
if st.button("Predict the quality"):
    if (pipeline.predict(X) == 1):
        st.write("Good Quality Wine")
    else:
        st.write("Bad Quality Wine")

    
