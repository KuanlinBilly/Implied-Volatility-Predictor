# -*- coding: utf-8 -*-
import streamlit as st
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#using the chardet library to detect the encoding of the file automatically.
import chardet
with open('ImplyVolatility_漲跌0.015_.csv', 'rb') as f:
    result = chardet.detect(f.read()) 

st.title('Implied Volatility Classification')

# Set the random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Read CSV
D = pd.read_csv('ImplyVolatility_漲跌0.015_.csv', encoding=result['encoding'], low_memory=False)

# Convert the date column to datetime format
D['date'] = pd.to_datetime(D['date'])

# Set the date column as the index
D = D.set_index('date')

# Train/test split
columns = list(D.columns)
x_train, x_test, y_train, y_test = train_test_split(
    D.drop('Target', axis=1),
    D['Target'],
    test_size=0.2,
    random_state=0,
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

if st.button('Train'):
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
    history = model.fit(
        x_train,
        y_train2,
        epochs=num_epochs,
        batch_size=100,
    )

    # Evaluate the model
    score = model.evaluate(x_test, y_test2, batch_size=len(y_test))
    st.write(f"Score: {score}")

    # Make predictions
    predict = model.predict(x_test)
    predict2 = np.argmax(predict, axis=1)

    # Calculate accuracy
    correct_predictions = np.sum(predict2 == np.argmax(y_test2, axis=1))
    accuracy = correct_predictions / len(y_test)
    st.write(f"Accuracy = {accuracy:.6f}")

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

