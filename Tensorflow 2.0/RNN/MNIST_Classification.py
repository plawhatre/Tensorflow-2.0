import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
#load data
data = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = data.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
K = np.size(np.unique(y_train))

#build model
i = tf.keras.layers.Input(shape=x_train[0].shape)
x = tf.keras.layers.LSTM(10, activation='relu')(i)
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.Dense(K, activation='softmax')(x)
model = tf.keras.Model(i, x)

#compile model
model.compile(optimizer='adam',
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy'])

#fit model
r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100)

#predict
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)

#performance
print(classification_report(y_test, y_pred))
plt.figure(figsize=(8,8))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
plt.show()

#plot
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show() 