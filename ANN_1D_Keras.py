import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense


def get_X_y(n):
    X = np.random.uniform(0, 1, n)
    y = np.zeros(n)

    for i in range(n):
        if X[i] < 0.6:
            y[i] = np.random.binomial(size=1, n=1, p=0.25)[0]
        else:
            y[i] = np.random.binomial(size=1, n=1, p=0.7)[0]

    return X, y


n_inputs = 1
n_hidden1 = 100
n_hidden2 = 40
n_outputs = 1

n = 1000

X, y = get_X_y(n)
print("X shape:", X.shape)

model = Sequential()
model.add(Dense(n_hidden1, input_dim=1, activation='relu'))
model.add(Dense(n_hidden2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=200, batch_size=100)

X_test = np.linspace(start=0, stop=1, num=100)
print("X test shape:", X_test.shape)
y_test = model.predict(X_test)

font = {'weight': 'bold',
        'size': 25}

matplotlib.rc('font', **font)
axes = plt.gca()
axes.set_ylim(0, 1)
plt.plot(X_test, y_test, c='green', marker='o', markersize=5)
plt.title("Function computed by neural network")
plt.yticks(np.arange(0, 1, 0.1))
plt.grid()
plt.show()