import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense


def get_X_y(n):
    X = np.random.uniform(0, np.pi, n)
    y = np.sin(X)
    return X, y


n = 40
X, y = get_X_y(n)
print("X shape:", X.shape)

model = Sequential()
model.add(Dense(6, input_dim=1, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

model.fit(X, y, epochs=1000, batch_size=4)

X_test = np.linspace(start=0, stop=np.pi, num=500)
print("X test shape:", X_test.shape)
y_test = model.predict(X_test)

font = {'weight': 'bold',
        'size': 25}

matplotlib.rc('font', **font)
axes = plt.gca()
axes.set_ylim(0, 1)
plt.plot(X_test, y_test, c='green', marker='o', markersize=5)
plt.title("Sinus approximated by neural network")
plt.yticks(np.arange(0, 1, 0.1))
plt.grid()
plt.show()