
import tensorflow as tf
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

print('________Fine Tunning Required________')

train_path = 'data/train'
test_path = 'data/test'

#number of classes
folders = glob(train_path+'/*')

#load pretrained model
p_model = tf.keras.applications.MobileNetV2( input_shape=(32, 32, 3),
	weights="imagenet",
	include_top=False,
	classes=2)
p_model.trainable = False

x = tf.keras.layers.Flatten()(p_model.output)
x = tf.keras.layers.Dense(2, activation='softmax')(x)

#build model
model = tf.keras.Model(inputs=p_model.input, outputs=x)

#summary
model.summary()

#Data loading and augmentation
gen_train = ImageDataGenerator(
	height_shift_range=0.1,
	width_shift_range=0.1,
	rotation_range=20,
	zoom_range=0.1,
	shear_range=0.1,
	horizontal_flip=True,
	preprocessing_function=preprocess_input
	)

# gen_train = ImageDataGenerator(
# 	preprocessing_function=preprocess_input
# 	)

gen_test = ImageDataGenerator(
	preprocessing_function=preprocess_input
	)
batch_size = 32

train_generator = gen_train.flow_from_directory(
	train_path,
	shuffle=True,
	batch_size=batch_size,
	target_size=(32, 32),
	)
valid_generator = gen_test.flow_from_directory(
	test_path,
	shuffle=True,
	batch_size=batch_size,
	target_size=(32, 32)
	)

#compile model
model.compile(optimizer=tf.keras.optimizers.Adam(lr=5e-4),
	loss='categorical_crossentropy',
	metrics=['accuracy'])

#fit model
r = model.fit_generator(train_generator, 
	validation_data=valid_generator,
	epochs=150,
	steps_per_epoch=3,
  	validation_steps=3)

#plot
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show() 

# #predict
# y_pred = model.predict(x_test)
# y_pred = (y_pred > 0.5)

# #performance
# print(classification_report(y_test, y_pred))
# plt.figure(figsize=(8,8))
# sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
# plt.show()







