import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.01, num_iters=1000):
        self.lr = lr
        self.num_iters = num_iters
        self.weights = None
        self.bias = None
        self.cost_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        #init params
        self.weights = np.zeros(n_features)
        self.bias = 0

        #gradient descent
        for _ in range(self.num_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            
            #sigmoid
            y_predicted = self._sigmoid(linear_model)

            #compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            #update params
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            #calculate cost 
            cost = self._binary_cross_entropy(y, y_predicted)
            self.cost_history.append(cost)

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _binary_cross_entropy(self, y_true, y_pred):
        n_samples = len(y_true)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        cost = -1 / n_samples * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return cost

    def get_cost_history(self):
        return self.cost_history
