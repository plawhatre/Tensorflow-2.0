import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf

N = 1000
x = np.random.random((N, 2)) * 6 - 3
y = np.cos(2*x[:,0]) + 3*x[:,1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33)

#create model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(2,)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(1))

#compile model
model.compile(optimizer='adam',
	loss='mse')

#fit model
r = model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=10)

#plot 
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

y_pred  = model.predict(x_test)
#score
print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

#weights
# print(model.layers[0].get_weights())
# print(model.layers[1].get_weights())