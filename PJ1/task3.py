import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import os
import cv2
import matplotlib.pyplot as plt


class My_CNN(nn.Module):

    def __init__(self) -> None:
        super(My_CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.out = nn.Linear(32 * 7 * 7, 12)

    def forward(self, x):
        x = x.float()
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


def train(model, train_dl, test_dl, epochs):
    E = []
    A = []
    max = 0
    for epoch in range(epochs):
        for i, data in enumerate(train_dl):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(outputs.size())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print(f'epoch:{epoch}-{i}, loss: {loss}')

        correct = 0
        total = 0
        for data in test_dl:
            inputs_t, labels_t = data
            outputs_t = model(inputs_t)
            pred = torch.max(outputs_t.data, 1)[1]
            labels_t = torch.max(labels_t.data, 1)[1]
            correct += (pred == labels_t).sum().item()
            total += labels_t.size()[0]

        acc = correct / total
        print(f'epoch:{epoch}, acc:{acc}')
        if acc > max:
            torch.save(model.state_dict(), './task3.pth')
            max = acc
        E.append(epoch)
        A.append(acc)
    print(f'max acc:{max}')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.plot(E, A, label='acc')
    plt.legend()
    plt.show()


def predict(model, test_dl):
    correct = 0
    total = 0
    for data in test_dl:
        inputs_t, labels_t = data
        outputs_t = model(inputs_t)
        pred = torch.max(outputs_t.data, 1)[1]
        labels_t = torch.max(labels_t.data, 1)[1]
        correct += (pred == labels_t).sum().item()
        total += labels_t.size()[0]
    acc = correct / total
    print(f'acc:{acc}')


def load_dataset(path):
    data = []
    labels = []
    for i in range(1, 13):
        subfolder = path + f"{i}/"
        for filename in os.listdir(subfolder):
            image = cv2.imread(subfolder + filename, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            image = (np.array(image) / 255.0).reshape(1, 28, 28)
            data.append(image)
            label = np.zeros(12)
            label[i - 1] = 1
            labels.append(label)
    data = np.array(data)
    labels = np.array(labels)
    # print(np.shape(data), np.shape(labels))
    return data, labels


if __name__ == "__main__":
    # 自定义数据
    epochs = 1000
    batch = 100

    X, Y = load_dataset("/Users/yuki/Desktop/ai/Project1/train/")
    X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                        Y,
                                                        test_size=0.1,
                                                        random_state=76)
    train_ds = TensorDataset(torch.from_numpy(X_train),
                             torch.from_numpy(Y_train))
    train_dl = DataLoader(train_ds, batch, shuffle=True)
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test))
    test_dl = DataLoader(test_ds, batch * 2)

    model = My_CNN()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    # train(model, train_dl, test_dl, epochs)

    model.load_state_dict(torch.load('./task3.pth'))
    predict(model, test_dl)

    # X_, Y_ = load_dataset("/Users/yuki/Desktop/ai/20302010040-于康/test_data/")
    # test_ds_ = TensorDataset(torch.from_numpy(X_), torch.from_numpy(Y_))
    # test_dl_ = DataLoader(test_ds_, batch)
    # predict(model, test_dl_)
