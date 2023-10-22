import numpy as np
import matplotlib.pyplot as plt
import pickle

np.random.seed(76)


class NeuralNetwork:

    def __init__(self, layers, learning_rate):
        self.layers = layers
        self.learning_rate = learning_rate
        self.num_layers = len(layers)
        self.weights = [
            np.random.randn(self.layers[i], self.layers[i + 1])
            for i in range(self.num_layers - 1)
        ]
        self.biases = [np.random.randn(1, size) for size in self.layers[1:]]

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.square(np.tanh(x))

    def forward(self, X):
        self.activations = [X]
        for i in range(self.num_layers - 1):
            z = np.dot(self.activations[i], self.weights[i]) + self.biases[i]
            a = self.tanh(z)
            self.activations.append(a)
        return self.activations[-1]

    def backward(self, Y):
        self.delta = (self.activations[-1] - Y)
        for i in reversed(range(self.num_layers - 1)):
            dz = self.delta * self.tanh_derivative(self.activations[i + 1])
            dw = np.dot(self.activations[i].T, dz)
            db = dz
            self.delta = np.dot(dz, self.weights[i].T)
            self.weights[i] -= self.learning_rate * dw
            self.biases[i] -= self.learning_rate * db

    def train(self, X_train, Y_train, X_test, Y_test, epochs):
        for epoch in range(epochs):
            for x, y in zip(X_train, Y_train):
                self.forward(x)
                self.backward(y)
            if epoch % 50 == 0:
                err = self.predict(X_test, Y_test)
                print(f'epoch:{epoch}, err:{err}')

    def predict(self, X_test, Y_test):
        pred = np.array([model.forward(x)[0][0] for x in X_test])
        return np.mean(abs(pred - Y_test))

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump((self.weights, self.biases), file)

    def load(self, filename):
        with open(filename, 'rb') as file:
            self.weights, self.biases = pickle.load(file)


if __name__ == "__main__":
    # 自定义数据
    hidden_layers = [50, 50]
    learning_rate = 0.01
    epochs = 5000

    # 生成数据
    X_train = np.linspace(-np.pi, np.pi, 1000)
    X_test = np.random.uniform(-np.pi, np.pi, 1000)
    Y_train = np.sin(X_train)
    Y_test = np.sin(X_test)

    # 训练
    layers = [1] + hidden_layers + [1]
    model = NeuralNetwork(layers, learning_rate=learning_rate)
    # model.train(X_train, Y_train, X_test, Y_test, epochs)
    # model.save("./task1.pkl")
    model.load("./task1.pkl")

    # 测试并绘图
    pred = np.array([model.forward(x)[0][0] for x in X_test])
    print(f'\naverage error:{np.mean(abs(pred-Y_test))}')

    plt.scatter(X_test, Y_test, label='sin(x)', s=5)
    plt.scatter(X_test, pred, label='trained', s=5)
    plt.legend()
    plt.show()
