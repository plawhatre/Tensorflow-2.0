import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import datetime
import os
from colorama import init
from termcolor import cprint
init()

def load_data():
	# load data
	data = tf.keras.datasets.mnist
	(x_train, y_train), (x_test, y_test) = data.load_data()
	x_train = x_train / 255.0
	x_test = x_test / 255.0
	return x_train, y_train, x_test, y_test

def build_model():
	# build model
	i = tf.keras.layers.Input(shape=x_train[0].shape, name='Images')
	x = tf.keras.layers.Flatten(name='Flatten')(i)
	x = tf.keras.layers.Dense(128, activation='relu', name='Layer_1')(x)
	x = tf.keras.layers.Dropout(0.3, name='Dropout')(x)
	x = tf.keras.layers.Dense(10, activation='sigmoid', name='Layer_2')(x)

	model = tf.keras.Model(i, x)
	return model

def compile_model(model):
	# compile model
	model.compile(optimizer=tf.keras.optimizers.Adam(),
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy'])
	return model

def tb_callbacks():
	#tensorboard logs
	log_dir = "logs\\tf_keras\\" + datetime.datetime.now().strftime("%Y-%m-%d_%Hh-%Mm-%Ss")
	os.makedirs(log_dir)
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
	return tensorboard_callback

def fit_model(model, tensorboard_callback):
	# fit model
	r = model.fit(x_train, y_train, epochs=10, batch_size=1000, callbacks=[tensorboard_callback])
	return r

if __name__ == "__main__":
	cprint('Loading Data..................', 'green')
	x_train, y_train, x_test, y_test = load_data()

	cprint('Build Model..................', 'green')
	model = build_model()

	cprint('Compile Model..................', 'green')
	model = compile_model(model)

	cprint('Creating Callbacks..................', 'green')
	tensorboard_callback = tb_callbacks()

	cprint('Training Model..................', 'green')
	_ = fit_model(model, tensorboard_callback)

	cprint(f'Open the link in browser', 'green')
	os.system("tensorboard --logdir logs/tf_keras")