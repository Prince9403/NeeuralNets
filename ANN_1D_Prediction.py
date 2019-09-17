import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np
import matplotlib.pyplot as plt


def get_new_X_1(X, y):
    X = np.random.uniform(0, 1, 1)[0]

    if X < 0.6:
        y = np.random.binomial(size=1, n=1, p=0.25)[0]
    else:
        y = np.random.binomial(size=1, n=1, p=0.7)[0]

    return np.array([[X]]), y


n_inputs = 1
n_hidden1 = 100
n_hidden2 = 40
n_outputs = 1

X = tf.compat.v1.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.compat.v1.placeholder(tf.float32, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = fully_connected(X, n_hidden1, scope="hidden1", activation_fn=tf.nn.relu)
    hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2", activation_fn=tf.nn.relu)
    logits = fully_connected(hidden2, n_outputs, scope="outputs", activation_fn=tf.nn.relu)

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.square(logits - y), name="loss")

learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
    init = tf.compat.v1.global_variables_initializer()

n_iterations = 100001
# batch_size = 50

X1 = 0.0
X2 = 0

z = np.zeros(101)

with tf.compat.v1.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        X_batch, y_batch = get_new_X_1(X1, X2)
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        if iteration % 10000 == 0:
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iteration, "\tMSE:", mse)
            out = logits.eval(feed_dict={X: X_batch})
            print("X:", X_batch[0][0], "y:", y_batch, "Outputs:", out)

    for i in range(101):
        x_next = np.array([[i * 0.01]])
        y_next = 0.2
        out = logits.eval(feed_dict={X: x_next})
        z[i] = out

plt.plot(np.arange(0, 1.001, 0.01), z, c='green', marker='o', markersize=5)
plt.grid()
plt.show()