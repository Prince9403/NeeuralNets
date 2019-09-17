import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense
import numpy as np


df = pd.read_csv("iris.csv")

# print(df.head(30))

X = df.drop(['variety'],axis=1)
y = df['variety'].map({'Setosa': 0, 'Versicolor': 1, 'Virginica': 2})

print(X.shape)
print(y.shape)

encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)

dummy_y = np_utils.to_categorical(encoded_y)

# print(dummy_y)

X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size=0.33, random_state=42)

model = Sequential()
model.add(Dense(7, input_dim=4, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size=10)

print("\n\nEvaluation:")
print(model.metrics_names)
print(model.evaluate(X_test, y_test))
print("Prediction:")
predictions = model.predict(X_test)
# print(predictions)
# print(np.argmax(predictions, axis=1))

intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[0].output)
intermediate_output = intermediate_layer_model.predict(X_test)
# print(intermediate_output)

#m = intermediate_output.shape[0]
#for i in range(m):
    # print(np.argmax(predictions, axis=1)[i], intermediate_output[i][0])