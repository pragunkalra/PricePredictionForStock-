STOCK PRICE PREDICTION
======================
This code performs several tasks related to data analysis and modeling of stock prices. The code assumes that a dataset with stock prices is stored in a pandas DataFrame named dataset.
For the purpose of the demo I have stored a csv file containing historic prices of cocacola stock

Checking and Handling Missing Values
====================================
The first task is to check for any missing values in the dataset using the isna().sum().sum() method. This helps to ensure that the dataset is complete and ready for analysis.

Plotting/Visualization
======================

The code then creates several plots to visualize the data.

A box plot is created using the boxplot() method from matplotlib. The plot shows the frequency distribution of the data for the columns 'Open', 'High', 'Low', 'Close', and 'Adj Close'.
A line plot is created using the plot() method from matplotlib to show the closing and opening prices over time.
The 100-day and 200-day moving averages of the closing price are calculated using the rolling() method and plotted on the same graph.
Importing Libraries for Building Models

The code then imports several libraries that are commonly used for building and evaluating machine learning models, including train_test_split for splitting the dataset into training and testing sets, LinearRegression for building a linear regression model, MinMaxScaler and StandardScaler for scaling the data, and mean_squared_error and r2_score for evaluating the model.

Data Split into Training and Testing
======================
The final task is to split the dataset into training and testing sets. The code uses the closing price column to make predictions and creates two new DataFrames, train_data and test_data, that contain 70% and 30% of the original data, respectively. The shapes of the new DataFrames are printed to confirm that the data was split correctly.

Requirements
================
Python 3.x
Pandas
Numpy
Scikit-learn
Keras
Tensorflow
Matplotlib

Usage
==================
1) Clone or download the repository.
2) Open the predicting_stock_prices.ipynb file in Jupyter Notebook or Google Colab.
3) Follow the instructions provided in the notebook to load the dataset and preprocess it.
4) Train and test the machine learning models to predict the stock prices.
5) Visualize the results using Matplotlib.



Machine Learning Models
=======================
**LSTM**
The LSTM model is trained to predict the closing stock price based on the previous 100 days' stock prices. The model has four LSTM layers with 50, 60, 80 and 120 neurons respectively. Each LSTM layer has a dropout layer with a rate of 0.2, 0.3, 0.4 and 0.4 respectively. The model has a Dense layer with 1 neuron as output. The model is trained for 50 epochs.

**Linear Regression**
The Linear Regression model is trained using the Open, High, Low, and Volume data of the stock. The model is trained on 60% of the data and tested on 40% of the data. The accuracy score of the model is calculated using the score method.

**Logistic Regression**
The Logistic Regression model is trained using the Open, High, Low, and Volume data of the stock. As the target variable is continuous, it is converted into a categorical variable by rounding it off to the nearest integer. The model is trained on 60% of the data and tested on 40% of the data. The accuracy score of the model is calculated using the accuracy_score method from scikit-learn.

Conclusion
===========
This project demonstrates how to preprocess and train machine learning models to predict stock prices. The LSTM model gives the best accuracy compared to Linear Regression and Logistic Regression.
