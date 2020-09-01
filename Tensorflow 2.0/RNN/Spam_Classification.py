import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#load data
df = pd.read_csv("data/spam.csv", encoding='ISO-8859-1')
df.drop(df.columns[2:], axis=1, inplace=True)
df.columns = ['labels', 'texts']
df['Y'] = df['labels'].map({'ham': 0, 'spam': 1})
Y = df['Y'].values

#Train test split
x_train, x_test, y_train, y_test = train_test_split(df['texts'], Y, test_size=0.33)

#preprocess the data
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)

x_train = pad_sequences(x_train)

V = len(tokenizer.word_index)	
T = np.array(x_train).shape[1]

x_test = pad_sequences(x_test, maxlen=T)

#dimensionality of the word embeddings
D = 20

#dimensionality of the hidden state vector 
M = 15

#build model 
i = tf.keras.layers.Input(shape=(T,)) # (input is N*T*D)
x = tf.keras.layers.Embedding(V+1, D)(i) 
x = tf.keras.layers.LSTM(M, return_sequences=True)(x) 
x = tf.keras.layers.GlobalMaxPooling1D()(x)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(i, x)

#compile model
model.compile(optimizer='adam',
	loss='binary_crossentropy',
	metrics=['accuracy'])

#fit the model
r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)

#predict
y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5)

#performance
print(classification_report(y_test, y_pred))
plt.figure(figsize=(8,8))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
plt.show()

#plot
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show() 








