# -*- coding: utf-8 -*-
import os
import time
import random
import chardet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
import streamlit as st
from modules import DataLoader, EDA, CommentBox, TrainingLogger, NeuralNetwork, preprocess_data, ModelTraining, plot_training_history

st.set_option('deprecation.showPyplotGlobalUse', False)

# MongoDB connection details
mongo_id = 'mongodb+srv://ndhu:ndhu@cluster0.vnqxzcd.mongodb.net/?retryWrites=true&w=majority'
mongo_client = MongoClient(mongo_id)
mongo_db = mongo_client["ndhu"]
mongo_collection = mongo_db['comment']  # mini database
 


# this is the main function
def implied_volatility_predictor():  
 
 
    # Load the custom CSS file
    custom_css = open("styles.css").read()
    st.markdown(f'<style>{custom_css}</style>', unsafe_allow_html=True)

    
    # App's title and description
    st.title('Implied Volatility Predictor')
    #st.text('By Kuanlin Lai')
    st.markdown("##### You can customize and train a neural network for implied volatility prediction using this web app!")
    
    # comment_box  
    comment_box = CommentBox(mongo_collection)
    comment_box.display()
    
    # In the sidebar, allow users to enable or disable random seed initialization
    allow_randomness = st.sidebar.checkbox("Check this to Allow Randomization")
    model_training_times = 1 #default value
    
    if allow_randomness:  #Allow Randomness and input field for Model Training Times
        model_training_times = st.sidebar.number_input("Model Training Times", min_value=1, value=1)
        
    else: # Set the random seeds for reproducibility
        random.seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)
    
    # checkbox in the sidebar for 'See Training Details'
    see_training_details = st.sidebar.checkbox("See Training Details")
    
    # Data reading part: Allow user to choose the source of the input file
    bottom1 = "Use Default Data File"
    bottom2 = "Upload Data File"
    csv_choice = st.sidebar.radio("Select Input", (bottom1, bottom2))
    data_loader = DataLoader(csv_choice)
    D = data_loader.load_data()
    
 
    # Proceed if any CSV or excel file is selected (uploaded or default)
    if D is None:
        # No file selected, prompt user to upload a file
        st.warning("Please upload a CSV or Excel file.") #檢查是否有文件被選中
    else:      
        D, columns = preprocess_data(D)
        
        # Initialize session state for button
        if "show_data" not in st.session_state:
            st.session_state.show_data = False
    
        # Show basic info and EDA of the data on button click
        if st.button("See Basic Info and EDA of the Data"):
            st.session_state.show_data = not st.session_state.show_data
            
        # Show EDA if button is clicked
        if st.session_state.show_data:
            eda = EDA(D, columns)
            eda.basic_info()
            eda.plot()
             
        # User-defined test size
        test_size = st.slider('Test size (percentage)', min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    
        # Train/test split
        x_train, x_test, y_train, y_test = train_test_split(
            D.drop('Target', axis=1),
            D['Target'],
            test_size=test_size,
            random_state=None if allow_randomness else 0,
        )    
        
        
        category = max(y_train.nunique(), y_test.nunique())
        dim = len(columns) - 1
        
        # Convert target to one-hot encoding
        y_train2 = tf.keras.utils.to_categorical(y_train, num_classes=(category))
        y_test2 = tf.keras.utils.to_categorical(y_test, num_classes=(category))
        
        # Get user-defined parameters
        num_layers = st.number_input('Number of hidden layers', min_value=1, max_value=10, value=1)
        neurons = [st.number_input(f'Number of neurons for hidden layer {i + 1}', min_value=1, max_value=100, value=40) for i in range(num_layers)]
        num_epochs = st.number_input('Number of epochs', min_value=1, max_value=100000, value=2500)
        batch_size = st.number_input("Batch Size", min_value=1, max_value=10000, value=100)
    
    try: 
        if st.button('Train'): #if click the training bottom
            if D is None:
                st.write('請先上傳你的資料集') 
            else:
                neural_network = NeuralNetwork(dim, neurons, category)
                model_training = ModelTraining(neural_network, model_training_times, num_epochs, batch_size, see_training_details)
                accuracies, history, total_time, score = model_training.train_and_evaluate(x_train, y_train2, x_test, y_test2)
                model_training.display_results(accuracies, history, total_time, score)
    except: 
        st.write('發生錯誤，請再重新按一次Train按鈕，請勿在訓練過程中變更參數或其他設定') 
        st.write('若要在訓練過程中變更參數或其他設定，請先按右上角Stop按鈕停止訓練，再進行變更')
        
if __name__ == "__main__":
    implied_volatility_predictor()
