import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense


def get_X_y(m, n):
    X = np.random.binomial(size=(m,n), n=1, p=0.5)
    y0 = np.ones(m)
    y1 = np.random.binomial(size=m, n=1, p=0.5)
    y = np.where(X[:, 1]==0, y0, y1)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y


model = Sequential()
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

X_train, y_train = get_X_y(1000, 10)
model.fit(X_train, y_train, epochs = 100, batch_size = 32)

m_test = 12
n_test = 10
X_test, y_test = get_X_y(m_test, n_test)
y_predicted = model.predict(X_test)

for i in range(m_test):
    print("i=", i, "x_last=",  X_test[i, 1, 0], "y_predicted=", y_predicted[i, 0])

