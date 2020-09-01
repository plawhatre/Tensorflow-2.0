import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# looad data
data = load_breast_cancer()

#print the keys of the dataset
print(data.keys())

#train and test data
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33)

N, D = x_train.shape

#scale the data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

#build the model
model = tf.keras.models.Sequential([
	tf.keras.layers.Input(shape=(D,)),
	tf.keras.layers.Dense(1, activation='sigmoid')
	])

#compile the model
model.compile(optimizer='adam',
	loss='binary_crossentropy',
	metrics=['accuracy']
	)

#train the model
r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100)

#print
print('train score:', model.evaluate(x_train, y_train))
print('test score:', model.evaluate(x_test, y_test))

#plotting loss
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

#plotting accuracy
plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()

#prediction
y_pred = model.predict(x_test)
y_pred = (y_pred>0.5)
print(classification_report(y_test, y_pred))

#save and load the model
model.save('classification.h5')
model = tf.keras.models.load_model('classification.h5')