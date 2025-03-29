import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Split into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- LOGISTIC REGRESSION ---
solvers = ['liblinear', 'lbfgs', 'saga', 'newton-cg']
regularization_strengths = [0.01, 0.1, 1, 10]

logistic_results = []

for solver in solvers:
    for C in regularization_strengths:
        log_reg = LogisticRegression(solver=solver, C=C, max_iter=200)
        log_reg.fit(X_train_scaled, y_train)
        y_pred = log_reg.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        logistic_results.append([solver, C, accuracy])

# Convert results to a DataFrame
logistic_df = pd.DataFrame(logistic_results, columns=["Solver", "Regularization Strength (C)", "Accuracy"])
print("\n=== Logistic Regression Results ===")
print(logistic_df)

# --- MLP in scikit-learn ---
mlp_sklearn = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam',
                            max_iter=1000, learning_rate_init=0.001, random_state=42)
mlp_sklearn.fit(X_train_scaled, y_train)
y_pred_mlp_sklearn = mlp_sklearn.predict(X_test_scaled)
mlp_sklearn_accuracy = accuracy_score(y_test, y_pred_mlp_sklearn)
print(f"\nMLP (scikit-learn) Accuracy: {mlp_sklearn_accuracy:.4f}")

# --- MLP in Keras ---
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# One-hot encode the targets
y_train_cat = to_categorical(y_train, num_classes=3)
y_test_cat = to_categorical(y_test, num_classes=3)

# Build the Keras model
keras_model = Sequential([
    Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),  # Regularization
    Dense(3, activation='softmax')
])

keras_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train with validation data
history = keras_model.fit(X_train_scaled, y_train_cat, validation_data=(X_test_scaled, y_test_cat),
                          epochs=100, batch_size=5, verbose=0)

# Evaluate the Keras model
loss, acc = keras_model.evaluate(X_test_scaled, y_test_cat, verbose=0)
print(f"MLP (Keras) Accuracy: {acc:.4f}")

# --- Plot Training vs Validation Accuracy ---
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy (Keras MLP)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
