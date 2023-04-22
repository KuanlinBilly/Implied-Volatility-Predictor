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
st.set_option('deprecation.showPyplotGlobalUse', False)

# MongoDB connection details
mongo_id = 'mongodb+srv://ndhu:ndhu@cluster0.vnqxzcd.mongodb.net/?retryWrites=true&w=majority'
mongo_client = MongoClient(mongo_id)
mongo_db = mongo_client["ndhu"]
mongo_collection = mongo_db['comment']  # mini database

# DataLoader: Responsible for loading the data from a CSV file or user upload
# DataLoader: 負責從CSV文件或用戶上傳中加載數據
class DataLoader:
    def __init__(self, csv_choice):
        self.csv_choice = csv_choice
    
    def read_csv_file(self, file_path):
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        return pd.read_csv(file_path, encoding=result['encoding'], low_memory=False)

    def load_data(self):
        # Allow user to choose input file
        bottom1 = "Use Default Data File"
        bottom2 = "Upload Data File"
        
        if self.csv_choice == bottom2:
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
                    return None
                    
                # Rename the last column to 'Target'
                D.rename(columns={D.columns[-1]: 'Target'}, inplace=True)
                return D
        
        # If user chooses to use the default file, read it in and process it
        elif self.csv_choice == bottom1:
            # Read the default CSV file
            D = self.read_csv_file('ImplyVolatility_漲跌0.015_.csv')
            D.rename(columns={D.columns[-1]: 'Target'}, inplace=True)
            return D
        
        return None

#EDA: Responsible for displaying exploratory data analysis results.
class EDA:
    def __init__(self, D, columns):
        self.data = D
        self.columns = columns
    
    def basic_info(self): #basic analysis
        st.write(f"Number of rows: {len(self.data)}")
        st.write(f"Number of columns: {len(self.columns)}")
        st.write(f"Target variable classes: {self.data.Target.unique()}")
        
        st.subheader("前五行")
        st.write(self.data.head())

        st.subheader("Correlation")
        st.write(self.data.corr())

        st.subheader("Descriptive Statistics")
        st.write(self.data.describe())
    
    def plot(self):
        st.markdown("Correlation Heatmap")
        sns.set(style="white")
        corr = self.data.corr()
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        f, ax = plt.subplots(figsize=(11, 9))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
        st.pyplot(f)

        st.markdown("Pairplot")
        sns.pairplot(self.data, hue="Target", corner=True)
        st.pyplot()

#CommentBox: Handles the comment box for users to leave comments.
class CommentBox:
    def __init__(self, mongo_collection):
        self.mongo_collection = mongo_collection

    def display(self):
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
                comments_count = self.mongo_collection.count_documents({})

                # Create a comment document
                comment_doc = {
                    "_id": comments_count,
                    "name": "Anonymous",
                    "comment": new_comment,
                }

                # Add this line to remove the '_id' field from the comment_doc
                comment_doc.pop('_id', None)
                # Insert the comment into the MongoDB collection
                self.mongo_collection.insert_one(comment_doc)
                st.sidebar.write("留言已送出，感謝您的寶貴建議")
                # Clear the comment box after submission
                st.session_state.new_comment = ''



def preprocess_data(D):
    D['date'] = pd.to_datetime(D['date'])
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
    
    return D, columns

# 定義 TrainingLogger 類別，用於在訓練過程中顯示訓練細節
# TrainingLogger: Logs training progress during the training of the neural network.
class TrainingLogger(tf.keras.callbacks.Callback):
    def __init__(self, progress_bar, see_training_details, num_epochs):
        super(TrainingLogger, self).__init__()
        self.progress_bar = progress_bar
        self.see_training_details = see_training_details
        self.num_epochs = num_epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.progress_bar.progress((epoch + 1) / self.num_epochs)
        if self.see_training_details:
            st.sidebar.write(f"Epoch {epoch+1}/{self.num_epochs}")
            for key, value in logs.items():
                st.sidebar.write(f"{key}: {value:.4f}")

# NeuralNetwork: Represents the neural network model and its associated methods.
class NeuralNetwork:
    def __init__(self, dim, neurons, category):
        self.model = tf.keras.models.Sequential()
        self.build_model(dim, neurons, category)
    
    def build_model(self, dim, neurons, category):
        self.model.add(
            tf.keras.layers.Dense(units=40, activation=tf.nn.relu, input_dim=dim)
        )
        for num_neurons in neurons:
            self.model.add(
                tf.keras.layers.Dense(units=num_neurons, activation=tf.nn.relu)
            )
        self.model.add(
            tf.keras.layers.Dense(units=category, activation=tf.nn.softmax)
        )
        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.categorical_crossentropy,
            metrics=['accuracy'],
        )

    def train(self, x_train, y_train2, num_epochs, batch_size, see_training_details, logger=None):
        if logger is not None:
            return self.model.fit(
                x_train,
                y_train2,
                epochs=num_epochs,
                batch_size=batch_size,
                callbacks=[logger],
                verbose=0,
            )

    def evaluate(self, x_test, y_test2):
        return self.model.evaluate(x_test, y_test2, batch_size=len(y_test2))

    def predict(self, x_test):
        return self.model.predict(x_test)


class ModelTraining:
    def __init__(self, neural_network, model_training_times, num_epochs, batch_size, see_training_details):
        self.neural_network = neural_network
        self.model_training_times = model_training_times
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.see_training_details = see_training_details

    def train_and_evaluate(self, x_train, y_train, x_test, y_test):
        accuracies = []

        for training_iteration in range(self.model_training_times):
            # Add progress bar 進度條
            progress_bar = st.progress(0)
            logger = TrainingLogger(progress_bar, self.see_training_details, self.num_epochs)
            start_time = time.time()

            # model training
            history = self.neural_network.train(x_train, y_train, self.num_epochs, self.batch_size, self.see_training_details, logger)
            score = self.neural_network.evaluate(x_test, y_test)
            #predict = neural_network.predict(x_test) # we don't need to show this in app for now

            # Store each time's accuracy when we run the model into an accuracy list
            _, accuracy = score[0], score[1]
            accuracies.append(accuracy)

            if self.model_training_times > 1:
                st.write(f"完成第{training_iteration + 1}次訓練，模型準確率為： {accuracy:.6f}")

            # Clear progress bar when training is finished
            progress_bar.empty()

        end_time = time.time()
        total_time = end_time - start_time

        return accuracies, history, total_time, score

    def display_results(self, accuracies, history, total_time, score):
        st.write(f'運行程式共花費了: {total_time:.2f} 秒')

        # Display the accuracies dataframe and the average accuracy at the bottom of the webpage
        if self.model_training_times > 1:
            st.subheader("All Accuracies")
            accuracies_df = pd.DataFrame({"Training Iteration": range(1, self.model_training_times + 1), "Accuracy": accuracies})
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
        plot_training_history(history)
        
# 绘制模型的训练历史，以帮助用户了解模型的训练过程和性能表现
def plot_training_history(history):
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


# this is the main function
def implied_volatility_predictor():  
 
 
    # Load the custom CSS file
    custom_css = open("styles.css").read()
    st.markdown(f'<style>{custom_css}</style>', unsafe_allow_html=True)

    
    # Add App's title and description
    st.title('Implied Volatility Predictor')
    #st.text('By Kuanlin Lai')
    st.markdown("##### You can customize and train a neural network for implied volatility prediction using this web app!")
    
    
    # In the sidebar, allow users to enable or disable random seed initialization
    allow_randomness = st.sidebar.checkbox("Check this to Allow Randomization")
    model_training_times = 1
    if allow_randomness:
        #Allow Randomness and input field for Model Training Times
        model_training_times = st.sidebar.number_input("Model Training Times", min_value=1, value=1)
    else:
        # Set the random seeds for reproducibility
        random.seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)
    
        
    # Data reading part: Allow user to choose the source of the input file
    bottom1 = "Use Default Data File"
    bottom2 = "Upload Data File"
    csv_choice = st.sidebar.radio("Select Input", (bottom1, bottom2))
    
    data_loader = DataLoader(csv_choice)
    D = data_loader.load_data()
    
    
    # Proceed if any CSV or excel file is selected (uploaded or default)
    if D is not None:      
        D, columns = preprocess_data(D)
        
        # Initialize session state for button
        if "show_data" not in st.session_state:
            st.session_state.show_data = False
    
        # Toggle basic info of the data
        if st.button("See Basic Info and EDA of the Data"):
            st.session_state.show_data = not st.session_state.show_data
            
        #exploratory data analysis
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
        neurons = [
            st.number_input(f'Number of neurons for hidden layer {i + 1}', min_value=1, max_value=100, value=40)
            for i in range(num_layers)
        ]
        num_epochs = st.number_input('Number of epochs', min_value=1, max_value=100000, value=2500)
        # Allow users to set the batch size
        batch_size = st.number_input("Batch Size", min_value=1, max_value=10000, value=100)
        
    # Add a new checkbox in the sidebar for 'See Training Details'
    see_training_details = st.sidebar.checkbox("See Training Details")
    
    # comment_box  
    comment_box = CommentBox(mongo_collection)
    comment_box.display()
    
    
    try: 
        if st.button('Train'): #if click the training bottom
            neural_network = NeuralNetwork(dim, neurons, category)
            model_training = ModelTraining(neural_network, model_training_times, num_epochs, batch_size, see_training_details)
            accuracies, history, total_time, score = model_training.train_and_evaluate(x_train, y_train2, x_test, y_test2)
            model_training.display_results(accuracies, history, total_time, score)
    except: 
        st.write('發生錯誤，請再重新按一次Train按鈕，請勿在訓練過程中變更參數或其他設定') 
        st.write('若要在訓練過程中變更參數或其他設定，請先按右上角Stop按鈕停止訓練，再進行變更')
        
if __name__ == "__main__":
    implied_volatility_predictor()
