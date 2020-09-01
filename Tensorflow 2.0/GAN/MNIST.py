import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf

#load data
data = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = data.load_data()

x_train, x_test = (x_train / 255.0) * 2 - 1, (x_test / 255.0) * 2 - 1

N, H, W = x_train.shape
D = H*W
x_train = x_train.reshape(-1, D)
x_test = x_test.reshape(-1, D)

#latent dimension
latent_dim = 100

#build generator
def build_generator(latent_dim):
	i = tf.keras.layers.Input(shape=(latent_dim,))
	x = tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(alpha=0.2))(i)
	x = tf.keras.layers.BatchNormalization(momentum=0.7)(x)
	x = tf.keras.layers.Dense(512, activation=tf.keras.layers.LeakyReLU(alpha=0.2))(x)
	x = tf.keras.layers.BatchNormalization(momentum=0.7)(x)
	x = tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=0.2))(x)
	x = tf.keras.layers.BatchNormalization(momentum=0.7)(x)
	x = tf.keras.layers.Dense(D, activation='tanh')(x)

	model = tf.keras.Model(i, x)
	return model

#build discriminator
def build_discriminator(img_size):
	i = tf.keras.layers.Input(shape=(img_size, ))
	x = tf.keras.layers.Dense(512, activation=tf.keras.layers.LeakyReLU(alpha=0.2))(i)
	x = tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(alpha=0.2))(x)
	x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
	
	model = tf.keras.Model(i, x)
	return model

#instantiate
generator = build_generator(latent_dim)
discriminator = build_discriminator(D)

#compile discriminator
discriminator.compile(
	optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
	loss='binary_crossentropy',
	metrics=['accuracy']
	)


#Combined Model
z = tf.keras.layers.Input(shape=(latent_dim, ))
img = generator(z)
discriminator.trainable = False
fake_pred = discriminator(img)

combined_model = tf.keras.Model(z, fake_pred)
combined_model.compile(
	optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
	loss='binary_crossentropy'
	)

#train GAN
batch_size = 32
epochs = 20*1000

ones = np.ones(batch_size)
zeros = np.zeros(batch_size)

g_losses = []
d_losses = []

#create a folder to store generated images
if not os.path.exists('Generated_Images'):
	os.makedirs('Generated_Images')

obs_noise = np.random.randn(5*5, latent_dim)
def sample_images(epoch):
	row, col = 5, 5
	# obs_noise = np.random.randn(row*col, latent_dim)
	imgs = generator.predict(obs_noise)

	imgs = 0.5 * imgs + 0.5

	fig, axs = plt.subplots(row, col)
	idx = 0
	for i in range(row):
		for j in range(col):
			axs[i,j].imshow(imgs[idx].reshape(H,W), cmap='gray')
			axs[i,j].axis('off')
			idx += 1

	fig.savefig('Generated_Images/'+"{:05d}".format(epoch)+'.png')
	plt.close()

# Main training loop
for epoch in range(epochs):
	#________Train Discriminator________
	# select some real images
	idx = np.random.randint(0, N, batch_size)
	real_images = x_train[idx]
	# generate fake images
	noise = np.random.randn(batch_size, latent_dim)
	fake_images = generator.predict(noise)

	d_loss_real, d_acc_real = discriminator.train_on_batch(real_images, ones)
	d_loss_fake, d_acc_fake = discriminator.train_on_batch(fake_images, zeros)
	d_loss = 0.5 * (d_loss_real + d_loss_fake)
	d_acc = 0.5*(d_acc_real + d_acc_fake)

	#________Train Generator________
	noise = np.random.randn(batch_size, latent_dim)
	g_loss = combined_model.train_on_batch(noise, ones)

	# to balance the discriminator training
	noise = np.random.randn(batch_size, latent_dim)
	g_loss = combined_model.train_on_batch(noise, ones)

	d_losses.append(d_loss)
	g_losses.append(g_loss)

	if epoch % 200 == 0:
		print(f'Epochs;{epoch}, d_loss:{d_loss}, d_acc:{d_acc}, g_loss:{g_loss}')
		sample_images(epoch)


#plot
plt.plot(g_losses, label='g_loss')
plt.plot(d_losses, label='d_loss')
plt.legend()
plt.show()