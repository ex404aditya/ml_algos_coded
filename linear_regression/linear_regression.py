import numpy as np


def r2_score(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0, 1]
    return corr**2

class LinearRegressor:
    def __init__(self, LR_rate=0.001, n_iters=1000):  
        self.lr = LR_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.cost_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        #init params
        self.weights = np.zeros(n_features)
        self.bias = 0

        #gradient descent
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias

            #compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            #update params
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            #calculate cost
            cost = (1 / (2 * n_samples)) * np.sum((y_predicted - y)**2)
            self.cost_history.append(cost)

    def predict(self, X):
        y_approximated = np.dot(X, self.weights) + self.bias
        return y_approximated

    def get_cost_history(self):
        return self.cost_history

