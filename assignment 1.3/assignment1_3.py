#Student name: Sahil Sajan
#Student ID: 202405476

import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor

X, y = make_regression(n_samples=200, n_features=1, noise=10, random_state=37)
X = StandardScaler().fit_transform(X)

y = y.ravel()

plt.figure(figsize=(10, 6))

for learning_rate in [0.001, 0.021, 0.2, 0.35]:
    model = SGDRegressor(learning_rate='constant', eta0=learning_rate, max_iter=1000, tol=1e-3)
    learning_rate_errors = []
    for _ in range(0, 2000, 55):
        model.partial_fit(X, y)
        error = ((model.predict(X) - y) ** 2).mean()
        learning_rate_errors.append(error)
    plt.plot(range(0, 2000, 55), learning_rate_errors, label=f'learning rate = {learning_rate}')

plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Sochastic Gradient Descent Convergence')
plt.legend()
plt.grid(True)
plt.show()
