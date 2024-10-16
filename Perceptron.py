import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, learning_rate=0.1, maximum_iterations=1000):
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        self.max_iterations = maximum_iterations

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.max_iterations):
            for idx, xi in enumerate(X):
                linear_output = np.dot(xi, self.weights) + self.bias
                y_pred = np.sign(linear_output)
                
                if y[idx] * y_pred <= 0:
                    self.weights += self.learning_rate * y[idx] * xi
                    self.bias += self.learning_rate * y[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.sign(linear_output)

if __name__ == "__main__":
    X = np.array([[2, 3], [1, 1], [2, 1], [-1, -2], [-2, -3], [-3, -2]])
    y = np.array([1, 1, 1, -1, -1, -1])

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, marker='o')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Scatter plot of X with labels y')
    plt.show()
    
    perceptron = Perceptron(learning_rate=0.1, maximum_iterations=1000)
    perceptron.fit(X, y)
    print(f"Weights :{perceptron.weights}")
    print(f"Bias :{perceptron.bias}")
    predictions = perceptron.predict(X)
    print(predictions)
