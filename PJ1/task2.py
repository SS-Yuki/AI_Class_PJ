import os
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

np.random.seed(76)


class NeuralNetwork:

    def __init__(self, layers, learning_rate):
        self.layers = layers
        self.learning_rate = learning_rate
        self.num_layers = len(layers)
        # [] layers[i]行，layers[i+1]列
        # self.weights = [np.random.randn(self.layers[i], self.layers[i + 1]) for i in range(self.num_layers - 1)]
        self.weights = [
            np.random.uniform(-0.1,
                              0.1,
                              size=(self.layers[i], self.layers[i + 1]))
            for i in range(self.num_layers - 1)
        ]
        # [] 1行，layers[i]列
        # self.biases = [np.random.randn(1, size) for size in self.layers[1:]]
        self.biases = [np.full((1, size), -0.1) for size in self.layers[1:]]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.activations = [X]
        for i in range(self.num_layers - 1):
            # [n, 784]x[784, *]+(n)[1, *]   [n, *]x[*, 12]+(n)[1, 12]
            z = np.dot(self.activations[i], self.weights[i]) + self.biases[i]
            a = self.sigmoid(z)
            self.activations.append(a)
        # print(np.shape(self.activations[-1]))
        # print(self.activations[-1])
        return self.activations[-1]

    def backward(self, Y):
        # [n, 12]
        self.delta = (self.activations[-1] - Y) / Y.shape[0]
        for i in reversed(range(self.num_layers - 1)):
            # [n, 12] [n, 12] 直接乘
            dz = self.delta * self.sigmoid_derivative(self.activations[i + 1])
            # [*, n]x[n, 12]
            dw = np.dot(self.activations[i].T, dz)
            db = np.sum(dz, axis=0, keepdims=True)
            # [n, 12]x[12, *]
            self.delta = np.dot(dz, self.weights[i].T)
            self.weights[i] -= self.learning_rate * dw
            self.biases[i] -= self.learning_rate * db

    def train(self, X_train, Y_train, X_test, Y_test, epochs, batch):
        E = []
        A = []
        max = 0
        for epoch in range(epochs):
            # print(epoch)
            X_, Y_ = shuffle(X_train, Y_train, random_state=1)
            for i in range(0, len(X_), batch):
                x = X_[i:i + batch]
                y = Y_[i:i + batch]
                model.forward(x)
                model.backward(y)
            if (epoch % 10 == 0):
                acc = self.predict(X_test, Y_test)
                print(f'epoch:{epoch}, acc:{acc}')
                # print(self.weights)
                if acc > max:
                    self.save("./task2.pkl")
                    max = acc
                E.append(epoch)
                A.append(acc)
        print(f'max acc:{max}')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.plot(E, A, label='acc')
        plt.legend()
        plt.show()

    def predict(self, X, Y):
        prediction = self.forward(X)
        num = 0
        for i in range(X.shape[0]):
            if (np.argmax(prediction[i]) == np.argmax(Y[i])):
                num += 1
        return num / len(X)

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump((self.weights, self.biases), file)

    def load(self, filename):
        with open(filename, 'rb') as file:
            self.weights, self.biases = pickle.load(file)


def load_dataset(path):
    inputs = []
    labels = []
    for i in range(1, 13):
        subfolder = path + f"{i}/"
        for filename in os.listdir(subfolder):
            image = cv2.imread(subfolder + filename, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            image = np.array(image).flatten() / 255.0
            inputs.append(image)
            label = np.zeros(12)
            label[i - 1] = 1
            labels.append(label)
    inputs = np.array(inputs)
    labels = np.array(labels)
    # print(np.shape(inputs), np.shape(labels))
    return inputs, labels


if __name__ == "__main__":
    # 自定义数据
    hidden_layers = [128, 64]
    learning_rate = 0.05
    epochs = 5000
    batch = 100

    # 加载数据
    X, Y = load_dataset("/Users/yuki/Desktop/ai/Project1/test_data/")
    X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                        Y,
                                                        test_size=0.1,
                                                        random_state=76)

    # 训练
    layers = [28 * 28] + hidden_layers + [12]
    model = NeuralNetwork(layers, learning_rate=learning_rate)
    # model.train(X_train, Y_train, X_test, Y_test, epochs, batch)
    model.load('./task2.pkl')
    print(f'acc:{model.predict(X_test, Y_test)}')

    # X_, Y_ = load_dataset("/Users/yuki/Desktop/ai/Project1/test_data/")
    # print(f'acc:{model.predict(X_, Y_)}')