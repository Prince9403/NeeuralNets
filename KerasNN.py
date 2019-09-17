import tensorflow as tf
from tensorflow.keras import layers

print(tf.VERSION)
print(tf.keras.__version__)

model = tf.keras.Sequential()
model.add(layers.Dense(64, activation='relu'))