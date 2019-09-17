import tensorflow as tf
import numpy as np


def get_new_X_1(X, y, n_steps):
    X[0][0][0] = np.random.binomial(size=1, n=1, p=0.5)[0]
    for i in range(1, n_steps):
        if X[0][i - 1][0] == 0:
            X[0][i][0] = 1
        else:
            X[0][i][0] = np.random.binomial(size=1, n=1, p=0.5)[0]

    for i in range(n_steps - 1):
        y[0][i][0] = X[0][i + 1][0]

    if y[0][n_steps - 2][0] == 0:
        y[0][n_steps - 1][0] = 1
    else:
        y[0][n_steps - 1][0] = np.random.binomial(size=1, n=1, p=0.5)[0]
    return X, y


def get_new_X_2(X, y, n_steps):
    X[0][0][0] = np.random.uniform(-1, 1, 1)[0]
    X[0][1][0] = np.random.uniform(-1, 1, 1)[0]
    for i in range(2, n_steps):
        X[0][i][0] = X[0][i - 1][0] + X[0][i - 2][0]

    for i in range(n_steps - 1):
        y[0][i][0] = X[0][i + 1][0]

    y[0][n_steps - 1][0] = X[0][n_steps - 1][0] + X[0][n_steps - 2][0]

    return X, y


def get_new_X_3(X, y, n_steps):
    X[0][0][0] = np.random.binomial(size=1, n=1, p=0.75)[0]
    for i in range(1, n_steps):
        if X[0][i - 1][0] == 0:
            X[0][i][0] = np.random.binomial(size=1, n=1, p=0.667)[0]
        else:
            X[0][i][0] = np.random.binomial(size=1, n=1, p=0.333)[0]

    for i in range(n_steps - 1):
        y[0][i][0] = X[0][i + 1][0]

    if X[0][n_steps - 1][0] == 0:
        y[0][n_steps - 1][0] = np.random.binomial(size=1, n=1, p=0.667)[0]
    else:
        y[0][n_steps - 1][0] = np.random.binomial(size=1, n=1, p=0.333)[0]

    return X, y


def get_new_X_4(X, y, n_steps):
    X[0][0][0] = np.random.binomial(size=1, n=1, p=0.5)[0]
    X[0][1][0] = 1 - X[0][0][0]
    X[0][2][0] = np.random.binomial(size=1, n=1, p=0.5)[0]

    for i in range(2):
        y[0][i][0] = X[0][i + 1][0]

    y[0][2][0] = (X[0][1][0] + X[0][2][0]) % 2

    return X, y


n_steps = 3
n_inputs = 1
n_neurons = 100
n_outputs = 1
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
cell = tf.contrib.rnn.OutputProjectionWrapper(
tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu), output_size=n_outputs)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

learning_rate = 0.001
loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
init = tf.global_variables_initializer()

X1 = np.zeros((1, n_steps, 1))
X2 = np.zeros((1, n_steps, 1))

n_iterations = 10001
# batch_size = 50
with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        X_batch, y_batch = get_new_X_4(X1, X2, n_steps)
        # X_batch, y_batch = [...] # fetch the next training batch
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        if iteration % 10000 == 0:
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iteration, "\tMSE:", mse)
            print("X:")
            print(X_batch)
            print("y:")
            print(y_batch)
            out = outputs.eval(feed_dict={X: X_batch, y: y_batch})
            print("Outputs:")
            print(out)


