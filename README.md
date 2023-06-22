# Implied-Volatility-Predictor
[Web App Link](https://kuanlinbilly-implied-volatility-predictor-login-app-1f7re4.streamlit.app/)
 
<div align=center>
<img src="https://github.com/KuanlinBilly/Implied-Volatility-Predictor/blob/main/img-folder/webapp.jpg">
</div>

## Overview
* This Implied Volatility Predictor Web App is designed to help users predict implied volatility, which is a crucial element in options pricing and quant trading strategies.   
* This app provides a user-friendly interface to train and customize a neural network for implied volatility prediction. 
* The app employs MongoDB to store user comments, enhancing the project's overall functionality and allowing for seamless collaboration and feedback.

## Features
* Login page for secure access
* Implied volatility prediction using a customizable neural network model
* Support for both default and user-uploaded CSV or Excel data files
* User-selectable input features for model training
* Adjustable test size, hidden layers, neurons, and epochs for model training
* Option to enable or disable random seed initialization
* Model training details toggle
* Polished data visualization for Exploratory data analysis (EDA), such as correlation heatmap and pairplot
* Comment box for users to provide feedback and suggestions

## How this App can help you?
### Quant Trading
The Implied Volatility Predictor Web App is a valuable tool for traders looking to better understand the potential movement of options prices. Accurate predictions of implied volatility help users make informed decisions, leading to improved risk management and potentially more profitable trades.

### Customizable Model
The app's neural network model can be tailored to user requirements by selecting and adjusting various hyperparameters, optimizing model performance and giving users more control over the prediction process.

### Secure Access
The login page ensures only authorized users can access the app, protecting user data and preventing unauthorized access to the model and its predictions.

### Collaboration and Feedback
The integration of MongoDB for storing user comments enables seamless collaboration and feedback among users, fostering a sense of community and encouraging continuous improvement.

## How to Use the App
<div align=center>
<img src="https://github.com/KuanlinBilly/Implied-Volatility-Predictor/blob/main/img-folder/login.jpg">
</div>

1. Sign in to the app using your credentials. You can enter the app with Username: ndhu, passwords: 1234.   
2. Upload your data file (in CSV or Excel format) or use the default data file provided.   
3. Select the input features to be used for model training.    
4. Adjust the model hyperparameters as desired (test size, hidden layers, neurons, and epochs).   
5. Toggle the "Model Training Details" switch to view the model's performance metrics.    
6. Click the "Predict" button to generate implied volatility predictions.    
7. View the predicted results on the app's output page.    

## Data Used for Training the Model
The app's default dataset contains options prices and corresponding implied volatility values for the TSEC weighted index (^TWII).

## Conclusion
The Implied Volatility Predictor Web App demonstrates the use of Streamlit for web app development, neural networks for predictive modeling, and MongoDB for data storage and retrieval. It offers a practical solution for predicting implied volatility in options trading and serves as a customizable and collaborative platform for users.

## Limitations and Future Improvements
The app's accuracy depends on the quality and relevance of the training data, and it currently only supports CSV and Excel data files. Possible future improvements include additional data file format support, real-time data feeds, and implementing more machine learning algorithms for predicting implied volatility.
