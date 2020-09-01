import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255, x_test / 255

x_train = x_train.reshape(-1,28**2)
x_test = x_test.reshape(-1,28**2)

#build model
h_dim = 30
i = tf.keras.layers.Input(shape=x_train[0].shape)
h1 = tf.keras.layers.Dense(h_dim, activation='relu')(i)
h = tf.keras.layers.Dense(h_dim, activation='relu')(h1)
h2 = tf.keras.layers.Dense(h_dim, activation='relu')(h)
z = tf.keras.layers.Dense(28**2, activation='sigmoid')(h2)

model = tf.keras.Model(i, z)
#compile model
model.compile(optimizer='adam',
  loss='binary_crossentropy')

#train
r = model.fit(x_train, x_train, validation_data=(x_test, x_test), epochs=5)

x_hat = model.predict(x_test)

i = np.random.randint(x_test.shape[0]) 
plt.figure()
plt.subplot(131)
plt.imshow(x_test[i].reshape(28,28), cmap='gray')
plt.title('Original')
plt.subplot(132)
plt.imshow(x_hat[i].reshape(28,28), cmap='gray')
plt.title('Reconstructed')
plt.subplot(133)
plt.imshow((x_test[i]-x_hat[i]).reshape(28,28), cmap='gray')
plt.title('DIfference')
plt.show()
