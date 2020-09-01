import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

#Load data
data = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = data.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

'''
tensorflow expects input of dimension: 
n_samples*heigth*width*channels
'''
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# number of classes
K = len(set(y_train))

#build model(FUNCTIONAL API)
i = tf.keras.layers.Input(shape=x_train[0].shape)
x = tf.keras.layers.Conv2D(32, (3, 3), strides=2, activation='relu')(i)
x = tf.keras.layers.Conv2D(64, (3, 3), strides=2, activation='relu')(x)
x = tf.keras.layers.Conv2D(128, (3, 3), strides=2, activation='relu')(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(K, activation='softmax')(x)

model = tf.keras.Model(i, x)

# compile model
model.compile(optimizer='adam',
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy'])

#fit the model
r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=15)

#plot
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='accurcy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()

#prediction
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)

#print
print(classification_report(y_test, y_pred))
plt.figure(figsize=(10,10))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
plt.show()
