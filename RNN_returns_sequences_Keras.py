import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
import random


def get_X_y(n):
    X = np.zeros((n, 4))
    z = np.array([random.randint(0, 2) for i in range(n)])
    X[z == 0, :] = [0, 0, 1, 1]
    X[z == 1, :] = [0, 1, 0, 1]
    X[z == 2, :] = [0, 1, 1, 0]
    y = np.zeros((n, 4))
    y[:, :3] = X[:, 1:]
    print(X[:5, :])
    print(y[:5, :])
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    y = np.reshape(y, (y.shape[0], y.shape[1], 1))
    return X, y


model = Sequential()
model.add(LSTM(units=20, return_sequences=True))
model.add(Dense(units=1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

X_train, y_train = get_X_y(1000)
print(X_train.shape)
print(y_train.shape)
model.fit(X_train, y_train, epochs = 100, batch_size = 32)

X_test = np.zeros((3, 4))
X_test[0, :] = [0, 0, 1, 1]
X_test[1, :] = [0, 1, 0, 1]
X_test[2, :] = [0, 1, 1, 0]
X_test = np.reshape(X_test, (3, 4, 1))

y_predicted = model.predict(X_test)
print(y_predicted)
