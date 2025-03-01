import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Sigmoid activation function
def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))


# Loss function: Binary Cross-Entropy
def calculate_loss(y_true, y_predicted):
    epsilon = 1e-5  # To prevent log(0) errors
    return -np.mean(
        y_true * np.log(y_predicted + epsilon) + (1 - y_true) * np.log(1 - y_predicted + epsilon)
    )


# Batch Gradient Descent for Logistic Regression
def batch_gd(X_train, y_train, lr=0.0001, iterations=5000):
    samples, features = X_train.shape
    weights = np.zeros(features)
    intercept = 0
    loss_history = []

    for _ in range(iterations):
        linear_output = np.dot(X_train, weights) + intercept
        predictions = sigmoid_function(linear_output)

        weight_gradient = (1 / samples) * np.dot(X_train.T, (predictions - y_train))
        bias_gradient = (1 / samples) * np.sum(predictions - y_train)

        weights -= lr * weight_gradient
        intercept -= lr * bias_gradient

        loss = calculate_loss(y_train, predictions)
        loss_history.append(loss)

    return weights, intercept, loss_history


# Stochastic Gradient Descent for Logistic Regression
def stochastic_gd(X_train, y_train, lr=0.0001, iterations=5000):
    samples, features = X_train.shape
    weights = np.zeros(features)
    intercept = 0
    loss_history = []

    for _ in range(iterations):
        for idx in range(samples):
            x_sample = X_train[idx].reshape(1, -1)
            y_actual = y_train.iloc[idx]

            linear_output = np.dot(x_sample, weights) + intercept
            prediction = sigmoid_function(linear_output)

            weight_update = x_sample.T * (prediction - y_actual)
            bias_update = prediction - y_actual

            weights -= lr * weight_update.flatten()
            intercept -= lr * bias_update

        epoch_predictions = sigmoid_function(np.dot(X_train, weights) + intercept)
        epoch_loss = calculate_loss(y_train, epoch_predictions)
        loss_history.append(epoch_loss)

    return weights, intercept, loss_history


# Load and preprocess dataset
df = pd.read_csv("Placement.csv")
df.drop(columns=["Student_ID"], inplace=True)

X_data = df.drop(columns=["Placement"])
y_data = df["Placement"]

scaler = StandardScaler()
X_scaled_data = scaler.fit_transform(X_data)

X_train, X_test, y_train, y_test = train_test_split(X_scaled_data, y_data, test_size=0.2, random_state=42)

# Train models using both gradient descent methods
weights_batch, bias_batch, loss_batch = batch_gd(X_train, y_train)
weights_stochastic, bias_stochastic, loss_stochastic = stochastic_gd(X_train, y_train)

# Print model parameters
print("Batch Gradient Descent Weights:", weights_batch)
print("Batch Gradient Descent Bias:", bias_batch)
print("Stochastic Gradient Descent Weights:", weights_stochastic)
print("Stochastic Gradient Descent Bias:", bias_stochastic)
