import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

#load data 
X = np.linspace(-1, 1 , 1000).reshape(-1, 1)
Y = 2 * X +5

#split data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
_, D = np.shape(x_train)

#build model
model = tf.keras.models.Sequential([
	tf.keras.layers.Input(shape=(D,)),
	tf.keras.layers.Dense(1)])

#compile model
model.compile(optimizer=tf.keras.optimizers.SGD(0.01, 0.9),
	loss='mse')

# learning rate scheduling
def schedule(epoch, lr):
	if epoch >= 50:
		return 0.0001
	else:
		return 0.001

scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)

#fit model
r = model.fit(x_train, y_train, epochs=200, callbacks=[scheduler])

#plot
plt.plot(r.history['loss'], label='loss')
plt.show()

#prediction 
y_pred = model.predict(x_test)

print('MSE:', mean_squared_error(y_pred, y_test))
print('R2:', r2_score(y_pred, y_test))

#get weights
print(model.layers[0].get_weights())

#save and load model
model.save('regression.h5')
model = tf.keras.models.load_model('regression.h5')




