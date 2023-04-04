# Implied-Volatility-Predictor
[Web App Link](https://kuanlinbilly-implied-volatility-predictor-login-app-1f7re4.streamlit.app/)
<div align=center>
<img src="https://github.com/KuanlinBilly/Implied-Volatility-Predictor/blob/main/img-folder/webapp.jpg">
</div>

## Overview
This Implied Volatility Predictor Web App is designed to help users predict implied volatility, which is a crucial element in options pricing and quant trading strategies.   
This app provides a user-friendly interface to train and customize a neural network for implied volatility prediction. The app employs MongoDB to store user comments, enhancing the project's overall functionality and allowing for seamless collaboration and feedback.

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
The Implied Volatility Predictor Web App can be a valuable tool for traders who want to better understand the potential movement of options prices. By providing accurate predictions of implied volatility, the app can help users make more informed decisions when trading options, leading to better risk management and potentially more profitable trades.

### Customizable Model
The app's customizable neural network model is a key feature that allows users to tailor the model to their specific requirements. Users can select and adjust various hyperparameters to optimize the model's performance, giving them more control over the prediction process.

### Secure Access
The app includes a login page to ensure only authorized users have access. This feature protects the user's data and prevents unauthorized access to the model and its predictions.

### Collaboration and Feedback
The integration of MongoDB for storing user comments enables seamless collaboration and feedback among users. This feature allows users to share insights, observations, and suggestions related to the app and the model's performance, fostering a sense of community and encouraging continuous improvement.

## How to Use the App
1. Sign in to the app using your credentials. You can enter the app with Username: ndhu, passwords: 1234.
<div align=center>
<img src="https://github.com/KuanlinBilly/Implied-Volatility-Predictor/blob/main/img-folder/login.jpg">
</div>
3. Upload your data file (in CSV or Excel format) or use the default data file provided.
4. Select the input features to be used for model training.
5. Adjust the model hyperparameters as desired (test size, hidden layers, neurons, and epochs).
6. Toggle the "Model Training Details" switch to view the model's performance metrics.
7. Click the "Predict" button to generate implied volatility predictions.
8. View the predicted results on the app's output page.

## Data Used for Training the Model
The default dataset used in the app is the options prices and corresponding implied volatility values of **TSEC weighted index (^TWII)**.

## Conclusion
The project showcases how to use Streamlit for web app development, neural networks for predictive modeling, and MongoDB for data storage and retrieval. The app provides a practical solution for predicting implied volatility in options trading, while also offering a customizable model and a collaborative platform for users.

## Limitations and Future Improvements
While the Implied Volatility Predictor Web App provides a practical application for predicting implied volatility in options trading, it has some limitations. For instance, the accuracy of the model is dependent on the quality and relevance of the data used for training. In addition, the app currently only supports CSV and Excel data files, limiting the types of data that can be used for training.

Some possible future improvements for the app include adding support for additional data file formats, incorporating real-time data feeds, and implementing additional machine learning algorithms for predicting implied volatility.
