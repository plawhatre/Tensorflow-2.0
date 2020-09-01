import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255, x_test / 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
#with noise
x_train = x_train + 0.5*np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test = x_test + 0.5*np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train = np.clip(x_train, 0.0, 1.0)
x_test = np.clip(x_test, 0.0, 1.0)

#build model
i = tf.keras.layers.Input(shape=x_train[0].shape)
#Encoder
x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(i)
x = tf.keras.layers.MaxPooling2D((2,2), padding='same')(x)
x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
h = tf.keras.layers.MaxPooling2D((2,2), padding='same')(x)
#Decoder
x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(h)
x = tf.keras.layers.UpSampling2D((2,2))(x)
x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
x = tf.keras.layers.UpSampling2D((2,2))(x)
z = tf.keras.layers.Conv2D(1,(3,3), activation='sigmoid', padding='same')(x)
model = tf.keras.Model(i, z)
#compile model
model.compile(optimizer='adam',
  loss='binary_crossentropy')

#train
r = model.fit(x_train, x_train, validation_data=(x_test, x_test), epochs=100)

x_hat = model.predict(x_test)

i = np.random.randint(x_test.shape[0]) 
plt.figure()
plt.subplot(131)
plt.imshow(x_test[i].reshape(28,28), cmap='gray')
plt.title('Original+Noise')
plt.subplot(132)
plt.imshow(x_hat[i].reshape(28,28), cmap='gray')
plt.title('Reconstructed')
plt.subplot(133)
plt.imshow((x_test[i]-x_hat[i]).reshape(28,28), cmap='gray')
plt.title('Difference')
plt.show()

