import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import datetime
import os
from colorama import init
from termcolor import cprint
init()

class Dense(tf.keras.layers.Layer):
	def __init__(self, M1, M2, f=lambda x: x):
		super(Dense, self).__init__()
		self.W = tf.random.normal(shape=[M1, M2])
		self.b = tf.zeros(shape=[M2], dtype=tf.float32)
		self.f = f

	def forward(self, X):
		return self.f(tf.matmul(X, self.W) + self.b)

class Model(tf.keras.layers.Layer):
	def __init__(self, D, N, hidden_layer_sizes):
		super(Model, self).__init__()
		self.D = D
		self.layers = []

		M1 = self.D
		for M2 in hidden_layer_sizes:
			self.layers.append(Dense(M1, M2))
			M1 = M2
		self.layers.append(Dense(M1, N, f=tf.nn.relu))

	def forward(self, X):
		out = X
		for layer in self.layers:
			out = layer.forward(out)
		return out

	def summary_writer(self):
		#tensorboard logs
		log_dir = "logs\\tf_GradientTape\\" + datetime.datetime.now().strftime("%Y-%m-%d_%Hh-%Mm-%Ss")
		os.makedirs(log_dir)
		summary_writer = tf.summary.create_file_writer(log_dir)
		return summary_writer

	def cost(self, Y, Y_hat):
		cost = tf.math.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=Y_hat))
		return cost

	def accuracy(self, Y, Y_hat):
		num = tf.math.reduce_sum(tf.dtypes.cast(tf.math.logical_and((Y==1), (Y_hat>0.5)), tf.float32))
		den = tf.math.reduce_sum(Y)
		acc = num / den
		return acc

	def gradient_update(self, X, Y, optimizer):
		with tf.GradientTape() as t:
			Y_hat = self.forward(X)
			Loss = self.cost(Y, Y_hat)
		grads = t.gradient(Loss, self.trainable_weights)
		optimizer.apply_gradients(zip(grads, self.trainable_weights))
		Acc = self.accuracy(Y, Y_hat)
		return Loss, Acc

	def fit(self, X, Y, epochs=20, batch_size=1000, lr=0.5):
		N = X.shape[0]
		n_batches = N // batch_size
		print('Train data........')
		optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
		cost_lst = []
		tb_cost = 0
		tb_acc = 0
		summ_writer = self.summary_writer()
		for i in range(epochs):
			np.random.shuffle(X)
			for j in range(n_batches):
				Loss, Acc = self.gradient_update(X[(j*batch_size):((j+1)*batch_size)],
					Y[(j*batch_size):((j+1)*batch_size)], 
					optimizer)
				cost_lst.append(Loss/batch_size)
				tb_cost += Loss
				tb_acc += Acc
				# if j % 10 ==0:
			print(f'Epoch: {i+1}, Loss: {tf.math.reduce_mean(tb_cost)}, Accuracy:{tf.math.reduce_mean(tb_acc)}')

			with summ_writer.as_default():
				tf.summary.scalar('loss', tf.math.reduce_mean(tb_cost), step=i)
				tf.summary.scalar('accuracy', tf.math.reduce_mean(tb_acc), step=i)
		return cost_lst

def load_data():
	# load data
	data = tf.keras.datasets.mnist
	(x_train, y_train), (x_test, y_test) = data.load_data()
	x_train = x_train / 255.0
	x_test = x_test / 255.0
	x_train = x_train.reshape(-1,28**2)
	x_test = x_test.reshape(-1,28**2)
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	y_train = tf.one_hot(y_train, depth=10)
	y_test = tf.one_hot(y_test, depth=10)	
	return x_train, y_train, x_test, y_test

def plots(loss=True, img=False):
	if Loss:
		plt.plot(Loss)
		plt.show()

	if img:
		y_pred = tf.nn.sigmoid(model.forward(x_test))
		y_pred = np.argmax(y_pred, axis=1)
		x_test = x_test.reshape(-1, 28, 28)

		figs, axs = plt.subplots(5,10)
		for i in range(5):
			for j in range(10):
				idx = np.where(y_pred==j)[0][i]
				axs[i,j].imshow(x_test[idx].reshape(28, 28), cmap='gray')

		plt.show()

if __name__ == "__main__":
	cprint('Loading Data..................', 'green')
	x_train, y_train, x_test, y_test = load_data()

	cprint('Build Model..................', 'green')
	model = Model(28**2, 10, [128])

	cprint('Creating Callbacks..................', 'green')
	# tensorboard_callback = tb_callbacks()

	cprint('Training Model..................', 'green')
	Loss = model.fit(x_train, y_train)

	cprint('Plotting..................', 'green')
	plots(loss=True, img=False)

	cprint(f'Open the link in browser', 'green')
	os.system("tensorboard --logdir logs/tf_GradientTape")