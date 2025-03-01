from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score

# Diabetes Dataset - Linear Regression
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Train Linear Regression Model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Evaluate Performance
train_r2 = lin_reg.score(X_train, y_train)
test_r2 = lin_reg.score(X_test, y_test)
train_mse = mean_squared_error(y_train, lin_reg.predict(X_train))
test_mse = mean_squared_error(y_test, lin_reg.predict(X_test))

print("\nLinear Regression on Diabetes Dataset")
print(f"Train R² Score: {train_r2:.4f}")
print(f"Test R² Score: {test_r2:.4f}")
print(f"Train Mean Squared Error: {train_mse:.4f}")
print(f"Test Mean Squared Error: {test_mse:.4f}")

# Breast Cancer Dataset - Logistic Regression
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Train Logistic Regression Model
log_reg = LogisticRegression(max_iter=7000)
log_reg.fit(X_train, y_train)

# Evaluate Performance
train_accuracy = log_reg.score(X_train, y_train)
test_accuracy = log_reg.score(X_test, y_test)

y_pred_train = log_reg.predict(X_train)
y_pred_test = log_reg.predict(X_test)

print("\nLogistic Regression on Breast Cancer Dataset")
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")