import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255, x_test / 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

#build model
i = tf.keras.layers.Input(shape=x_train[0].shape)
#Encoder
x = tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same')(i)
x = tf.keras.layers.MaxPooling2D((2,2), padding='same')(x)
x = tf.keras.layers.Conv2D(8, (3,3), activation='relu', padding='same')(x)
h = tf.keras.layers.MaxPooling2D((2,2), padding='same')(x)
#Decoder
x = tf.keras.layers.Conv2D(8, (3,3), activation='relu', padding='same')(h)
x = tf.keras.layers.UpSampling2D((2,2))(x)
x = tf.keras.layers.Conv2D(8, (3,3), activation='relu', padding='same')(x)
x = tf.keras.layers.UpSampling2D((2,2))(x)
z = tf.keras.layers.Conv2D(1,(3,3), activation='sigmoid', padding='same')(x)
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
plt.title('Difference')
plt.show()

