# -*- coding: utf-8 -*-
#add chatbox
import streamlit as st
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import chardet
import os
import time
from pymongo import MongoClient

# MongoDB connection details
mongo_id = 'mongodb+srv://ndhu:ndhu@cluster0.vnqxzcd.mongodb.net/?retryWrites=true&w=majority'
mongo_client = MongoClient(mongo_id)
mongo_db = mongo_client["ndhu"]
mongo_collection = mongo_db['comment']  # mini database


def implied_volatility_predictor():  

    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    # Add title and description
    st.title('Implied Volatility Predictor')
    st.text('By Kuanlin Lai')
    st.markdown("##### You can customize and train a neural network for implied volatility prediction using this web app!")
    
    
    # In the sidebar, allow users to enable or disable random seed initialization
    allow_randomness = st.sidebar.checkbox("Check this to Allow Randomization")
    
    # Add new feature: Allow Randomness and input field for Model Training Times
    model_training_times = 1
    if allow_randomness:
        model_training_times = st.sidebar.number_input("Model Training Times", min_value=1, value=1)
    else:
        # Set the random seeds for reproducibility
        random.seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)
        
 
    # Allow user to choose input file
    bottom1 = "Use Default Data File"
    bottom2 = "Upload Data File"
    csv_choice = st.sidebar.radio("Select Input", (bottom1, bottom2))
    
    # If user uploads a file, read it in and process it
    if csv_choice == bottom2:
        uploaded_file = st.sidebar.file_uploader("Choose a CSV/Excel file")
    
        if uploaded_file is not None:
            # Determine the file format based on the file extension
            file_extension = uploaded_file.name.split(".")[-1].lower()
        
            if file_extension == "csv":
                D = pd.read_csv(uploaded_file, low_memory=False)
            elif file_extension == "xlsx":
                D = pd.read_excel(uploaded_file, engine="openpyxl")
            else:
                st.sidebar.error("Unsupported file format. Please upload a CSV or Excel file.")
                
            # Rename the last column to 'Target'
            D.rename(columns={D.columns[-1]: 'Target'}, inplace=True)
    
    # If user chooses to use the default file, read it in and process it
    elif csv_choice == bottom1:
        # Read the default CSV file
        with open('ImplyVolatility_漲跌0.015_.csv', 'rb') as f:
            result = chardet.detect(f.read())
        D = pd.read_csv('ImplyVolatility_漲跌0.015_.csv', encoding=result['encoding'], low_memory=False)
    
    
    # Proceed if any CSV file is selected (uploaded or default)
    if 'D' in locals():
        # Convert the date column to datetime format
        D['date'] = pd.to_datetime(D['date'])
        # Set the date column as the index
        D = D.set_index('date')
    
        # Convert the input data to numerical values
        for column in D.columns:
            if column != 'date':
                D[column] = pd.to_numeric(D[column], errors='coerce').fillna(0).astype(float)
    
        feature_columns = D.columns[:-1].tolist()
        selected_features = st.sidebar.multiselect("Select features in dataset for model training", feature_columns, default=feature_columns)
    
        # Update the dataset with the selected features
        D = D.filter(selected_features + ['Target'])
    
    
        columns = list(D.columns)
        
        
        # Initialize session state for button
        if "show_data" not in st.session_state:
            st.session_state.show_data = False
    
        # Toggle basic info of the data
        if st.button("See Basic Info of the Data"):
            st.session_state.show_data = not st.session_state.show_data
    
        if st.session_state.show_data:
            
            st.write(f"Number of rows: {len(D)}")
            st.write(f"Number of columns: {len(columns)}")
            st.write(f"Target variable classes: {D.Target.unique()}")
            
            st.subheader("前五行")
            st.write(D.head())
    
            st.subheader("Correlation")
            st.write(D.corr())
    
            st.subheader("Descriptive Statistics")
            st.write(D.describe())
    
            st.markdown("Correlation Heatmap")
            sns.set(style="white")
            corr = D.corr()
            mask = np.zeros_like(corr, dtype=np.bool)
            mask[np.triu_indices_from(mask)] = True
            f, ax = plt.subplots(figsize=(11, 9))
            cmap = sns.diverging_palette(220, 10, as_cmap=True)
            sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
            st.pyplot(f)
            
            st.markdown("Pairplot")
            sns.pairplot(D, hue="Target", corner=True)
            st.pyplot()
            
        # Initialize session state for comment box
        if "show_comment_box" not in st.session_state:
            st.session_state.show_comment_box = False
        
        # Show or hide comment box with a button
        if st.sidebar.button("點這裡開啟留言區"):
            st.session_state.show_comment_box = not st.session_state.show_comment_box
        
            # Comment section in the right sidebar
        if st.session_state.show_comment_box:
            st.sidebar.subheader("歡迎給我們一些建議！")
        
            # Input new comment
            new_comment = st.sidebar.text_area("Leave a comment here", value="", height=100)
        
            # Save and display new comment
            if st.sidebar.button("Submit comment"):
                # Fetch comments from MongoDB collection
                comments_count = mongo_collection.count_documents({})
        
                # Create a comment document
                comment_doc = {
                    "_id": comments_count,
                    "name": "Anonymous",
                    "comment": new_comment,
                }
        
                # Add this line to remove the '_id' field from the comment_doc
                comment_doc.pop('_id', None)
                # Insert the comment into the MongoDB collection
                mongo_collection.insert_one(comment_doc)
                st.sidebar.write("留言已送出，感謝您的寶貴建議")
                # Clear the comment box after submission
                st.session_state.new_comment = ''


            
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
        neurons = [
            st.number_input(f'Number of neurons for hidden layer {i + 1}', min_value=1, max_value=100, value=40)
            for i in range(num_layers)
        ]
        num_epochs = st.number_input('Number of epochs', min_value=1, max_value=10000, value=2500)
        # Allow users to set the batch size
        batch_size = st.number_input("Batch Size", min_value=1, max_value=10000, value=100)
        
    # Add a new checkbox in the sidebar for 'See Training Details'
    see_training_details = st.sidebar.checkbox("See Training Details")
    
    # Create the TrainingLogger class
    class TrainingLogger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            if see_training_details:
                st.sidebar.write(f"Epoch {epoch+1}/{num_epochs}")
                for key, value in logs.items():
                    st.sidebar.write(f"{key}: {value:.4f}")
    
    # Initialize the logger
    logger = TrainingLogger()
        
        #if click the training bottom
 
    if st.button('Train'):
        start_time = time.time()
        accuracies = []
    
        for training_iteration in range(model_training_times):
            # Create the model
            model = tf.keras.models.Sequential()
    
            # Input layer
            model.add(
                tf.keras.layers.Dense(units=40, activation=tf.nn.relu, input_dim=dim)
            )
    
            # Hidden layers
            for num_neurons in neurons:
                model.add(
                    tf.keras.layers.Dense(units=num_neurons, activation=tf.nn.relu)
                )
    
            # Output layer
            model.add(
                tf.keras.layers.Dense(units=category, activation=tf.nn.softmax)
            )
    
            # Compile the model
            model.compile(
                optimizer='adam',
                loss=tf.keras.losses.categorical_crossentropy,
                metrics=['accuracy'],
            )
    
            # Train the model
            if see_training_details:
                history = model.fit(
                    x_train,
                    y_train2,
                    epochs=num_epochs,
                    batch_size=100,
                    callbacks=[logger],
                    verbose=0,
                )
            else:
                history = model.fit(
                    x_train,
                    y_train2,
                    epochs=num_epochs,
                    batch_size=batch_size,
                    verbose=0,
                )
    
            # Evaluate the model
            score = model.evaluate(x_test, y_test2, batch_size=len(y_test))
            #st.write(f"Score: {score}")
    
            # Make predictions
            predict = model.predict(x_test)
            predict2 = np.argmax(predict, axis=1)
    
            # Calculate accuracy
            correct_predictions = np.sum(predict2 == np.argmax(y_test2, axis=1))
            accuracy = correct_predictions / len(y_test)
            accuracies.append(accuracy)
            
            if model_training_times > 1:
                st.write(f"完成第{training_iteration + 1}次訓練，模型準確率為： {accuracy:.6f}")

        end_time = time.time()
        total_time = end_time - start_time
        
        st.write(f'運行程式共花費了: {total_time:.2f} 秒')

        # Display the accuracies dataframe and the average accuracy at the bottom of the webpage
        if model_training_times > 1:
 
            st.subheader("All Accuracies")
            accuracies_df = pd.DataFrame({"Training Iteration": range(1, model_training_times + 1), "Accuracy": accuracies})
            st.write(accuracies_df)
    
            # Calculate and display the average accuracy
            average_accuracy = np.mean(accuracies)
            st.subheader("Average Accuracy")
            st.write(f"Average Accuracy:　{average_accuracy:.6f}")
    
        # Show individual score and accuracy if model_training_times is 1
        else:
            st.write(f"Score: {score}")
            st.write(f"Accuracy = {accuracies[0]:.6f}")

        
        # Plot accuracy and loss
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
        ax1.plot(history.history['accuracy'])
        ax1.set_title('Implied Volatility classification')
        ax1.set_ylabel('accuracy')
        ax1.set_xlabel('epoch')
        ax1.legend(['train accuracy'], loc='lower left')
    
        ax2.plot(history.history['loss'])
        ax2.set_title('Implied Volatility classification')
        ax2.set_ylabel('loss')
        ax2.set_xlabel('epoch')
        ax2.legend(['train loss'], loc='upper left')
        st.pyplot(fig)

if __name__ == "__main__":
    implied_volatility_predictor()
