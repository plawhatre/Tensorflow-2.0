import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf

#fabricate dataset
series = np.sin(0.1*np.arange(200)) + np.random.rand(200)*0.1

T = 10
D = 1
X = []
Y = []
for t in range(len(series) - T):
	X.append(series[t:(t+T)])
	Y.append(series[t+T])

X = np.array(X).reshape(-1, T)
X = np.expand_dims(X, -1)
Y = np.array(Y)
N = X.shape[0]

# Train Validation sets
X_train, X_val = X[:int(N/2), :, :], X[int(N/2):, :, :]
Y_train, Y_val = Y[:int(N/2)], Y[int(N/2):]

#buid model
i = tf.keras.layers.Input(shape=X_train[0].shape)
'''
its our choice to return sequqnece to return all the hiden state vector 
and then take the global max instead of taking last hidden state
vector and using global max. Feel free to change the model 
'''
# x = tf.keras.layers.LSTM(5, activation='relu')(i)
# x = tf.keras.layers.Dense(1)(x)
# model = tf.keras.Model(i, x)
 
x = tf.keras.layers.LSTM(5, activation='relu', return_sequences=True)(i)
x = tf.keras.layers.GlobalMaxPooling1D()(x)
x = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(i, x)

#compile model
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
	loss='mse')

#fit the model
r = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=100)

#predict 
Y_test = Y_val.copy()
Y_pred = []
last_x = X[int(N/2), :, :]
while np.size(Y_test)> len(Y_pred):
	p = model.predict(last_x.reshape(1, T, D))[0,0]
	Y_pred.append(p)
	last_x[:-1] = last_x[1:]
	last_x[-1] = p

#plots
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(Y_test, label='True')
plt.plot(Y_pred, label='Predicted')
plt.legend()
plt.show()

