# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 12:20:07 2020

@author: Rajan
"""


import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import random
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize 
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans
import plotly.express as px
import streamlit as st
import pydeck as pdk 
import altair as alt 
from datetime import datetime

# disable warnings
st.set_option('deprecation.showfileUploaderEncoding', False)

st.write("""
# Predictive modelling application
### This application is used to build any machine learning model and generate predictions.
""")

st.sidebar.header('User Interaction Pane')

# taking input
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file - train file", type=["csv"])


if uploaded_file is None:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    
else:
    input_df = pd.read_csv(uploaded_file)
    st.write(input_df)
    
# ml problem selector
ml_prob = st.sidebar.selectbox('Choose the ML category',('Select','Regression','Classification','Clustering'))

if ml_prob in ['Regression','Classification']:
    dep_var = st.sidebar.selectbox('Choose the dependent variable',(list(input_df.columns)))
elif ml_prob in ['Clustering']:
    algo = st.sidebar.selectbox('Choose the required algorithm',['Kmeans','Dbscan'])
    if algo=='Kmeans':
        numclust = st.sidebar.selectbox('Choose the required num of clust',[2,3,4,5,6,7,8])

# columns removal/retention
yesno = st.sidebar.selectbox('Do you want to add/remove columns?',('Select','No','Yes'))

if yesno == 'Yes':
    cols_selected = st.sidebar.multiselect('Choose the columns',(list(input_df.columns)))
    action = st.sidebar.selectbox('Choose the required action',('Remove','Retain'))
    if action == 'Remove':
        input_df2 = input_df.drop(cols_selected, axis = 1)
        st.write('Transformed dataframe')
        st.write(input_df2)
    elif action == 'Retain':
        input_df2 = input_df[cols_selected]
        st.write('Transformed dataframe')
        st.write(input_df2)

# other actions
if yesno == 'Yes':
    if st.sidebar.button('Export'):
        input_df2.to_csv('transformed_data.csv')

# EDA
chart_type = st.sidebar.selectbox('Which plot do you need?',['Select','Bar','Scatter'])

if chart_type == 'Bar':
    x = st.sidebar.selectbox('select a categorical variable',(list(input_df.columns)))
    st.write('Exploratory Analysis')
    st.bar_chart(input_df[[x]])
elif chart_type == 'Scatter':
    x = st.sidebar.selectbox('select a continuous variable - x',(list(input_df.columns)),key="x")
    y = st.sidebar.selectbox('select a continuous variable - y',(list(input_df.columns)),key="y")
    c = alt.Chart(input_df).mark_circle().encode(x=x, y=y)
    st.write('Exploratory Analysis')
    st.altair_chart(c, use_container_width=True)
    
    
# model building phase
if st.sidebar.button('Build model'):
    if ml_prob=='Regression':
        if yesno == 'Yes':
            input_df2 = input_df2.copy()
        elif yesno == 'No':
            input_df2 = input_df.copy()
        indep_df =  input_df2.drop(dep_var, axis = 1)
        indep_var = list(indep_df.columns)
    
    
        st.subheader('dep_var')
        st.write(dep_var)
        st.subheader('indep_vars')
        st.write(indep_var)
        # onehot encoding - using getdummies
        indep = input_df2.drop(dep_var, axis = 1)
        dep = input_df2[[dep_var]]
        interim = pd.get_dummies(indep)
        input_df3 = pd.concat([interim,dep],axis=1)

        # splitting into test and train
        X = input_df3.drop(dep_var, axis = 1)
        Y = input_df3[[dep_var]]
        random.seed(101)
        X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.10, random_state=1)
        
        # Randomforest model building
        rf = RandomForestRegressor()
        rf.fit(X_train, Y_train)
        pred1 = rf.predict(X_validation)
        fdf = Y_validation.copy()
        fdf['Predicted'] = pred1
        fdf['error2'] = abs(fdf['Predicted'] - fdf[dep_var])
        fdf['mape2'] = 100 * (fdf['error2']/fdf[dep_var])
        acc = round((100 - np.mean(fdf['mape2'])), 2)
        st.write('Prediction Accuracy')
        st.write(acc)
        st.write('Mean Absolute Error:', metrics.mean_absolute_error(Y_validation, pred1))
        st.write('Mean Squared Error:', metrics.mean_squared_error(Y_validation, pred1))
        st.write('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_validation, pred1)))
        fdf = Y_validation.copy()
        fdf['Predicted'] = pred1
        st.write('Predictions')
        st.write(fdf)
        st.write('Feature Importance')
        overall_imp = pd.DataFrame()
        overall_imp['variable'] = X.columns
        overall_imp['importance'] = rf.feature_importances_
        st.write(overall_imp.sort_values('importance',ascending=False))
        
    elif ml_prob == 'Classification':
        if yesno == 'Yes':
            input_df2 = input_df2.copy()
        elif yesno == 'No':
            input_df2 = input_df.copy()
        indep_df =  input_df2.drop(dep_var, axis = 1)
        indep_var = list(indep_df.columns)
    
    
        st.subheader('dep_var')
        st.write(dep_var)
        st.subheader('indep_vars')
        st.write(indep_var)
        # onehot encoding - using getdummies
        indep = input_df2.drop(dep_var, axis = 1)
        dep = input_df2[[dep_var]]
        interim = pd.get_dummies(indep)
        input_df3 = pd.concat([interim,dep],axis=1)

        # splitting into test and train
        X = input_df3.drop(dep_var, axis = 1)
        Y = input_df3[[dep_var]]
        random.seed(101)
        X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.10, random_state=1)
        
        clf = RandomForestClassifier()
        clf.fit(X_train, Y_train)
        prediction = clf.predict(X_validation)
        prediction_proba = clf.predict_proba(X_validation)
        confmat_labels = input_df[dep_var].unique().tolist()
        cm = confusion_matrix(Y_validation, prediction,labels=confmat_labels)
        
        st.subheader('Prediction Accuracy')
        st.write(metrics.accuracy_score(Y_validation, prediction))
        st.subheader('Prediction Probability')
        st.write(prediction_proba)
        st.subheader('Confusion Matrix')
        st.write(cm)
        st.subheader('Classification Report')
        st.write(metrics.classification_report(Y_validation, prediction))
        #st.bar_chart(input_df2[[dep_var]])
        
    elif ml_prob == 'Clustering':
        if yesno == 'Yes':
            input_df2 = input_df2.copy()
        elif yesno == 'No':
            input_df2 = input_df.copy()
        if algo=='Dbscan':
            # handle nan, inf, -inf
            input_df2 = input_df2.replace([np.inf, -np.inf], np.nan)
            # handle missing
            input_df2.fillna(method ='ffill', inplace = True)
            # onehot encoding - using getdummies
            input_df3 = pd.get_dummies(input_df2)
            # standardization
            scaler = StandardScaler()
            input_df3_scaled = scaler.fit_transform(input_df3)
            input_df3_norm = normalize(input_df3_scaled) 
            input_df3_norm = pd.DataFrame(input_df3_norm) 
            # dim reduction
            pca = PCA(n_components = 2) 
            X_principal = pca.fit_transform(input_df3_norm) 
            X_principal = pd.DataFrame(X_principal) 
            X_principal.columns = ['P1', 'P2']
            # clustering
            dbscan = DBSCAN(eps=0.0375, min_samples = 3).fit(X_principal)
            labels = dbscan.labels_
            y_pred = dbscan.fit_predict(X_principal)
            input_df3_norm['clusters'] = pd.Series(y_pred, index=input_df3_norm.index)
            st.write(input_df3_norm['clusters'].value_counts())
            
        elif algo=='Kmeans':
            # handle nan, inf, -inf
            input_df2 = input_df2.replace([np.inf, -np.inf], np.nan)
            # handle missing
            input_df2.fillna(method ='ffill', inplace = True)
            # onehot encoding - using getdummies
            input_df3 = pd.get_dummies(input_df2)
            # standardization
            scaler = StandardScaler()
            input_df3_scaled = scaler.fit_transform(input_df3)
            input_df3_norm = normalize(input_df3_scaled) 
            input_df3_norm = pd.DataFrame(input_df3_norm)
            kmeans = KMeans(n_clusters=numclust)
            kmeans.fit(input_df3_norm)
            y_kmeans = kmeans.predict(input_df3_norm)
            centers = kmeans.cluster_centers_
            input_df3_norm['clusters'] = pd.Series(y_kmeans, index=input_df3_norm.index)
            st.write(input_df3_norm['clusters'].value_counts())