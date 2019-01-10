import cv2
import glob
import numpy as np
import random
from sklearn.model_selection import train_test_split
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras import optimizers as opt
from keras.layers import *

def setup_to_finetune(model, n):
    # Setting everything bellow n to be not trainable
    for i, layer in enumerate(model.layers):
        layer.trainable = i > n
    model.compile(
        optimizer=opt.SGD(lr=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def unison_shuffled(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def load_data():
	print("loading data...")
	hardhat_imgs = glob.glob("people_with_hardhat/*.jpg")
	people_imgs = glob.glob("people_no_hardhat/*.jpg")
	crane_imgs = glob.glob("crane/*.jpg")
	all_imgs = np.zeros((2160, 299, 299, 3))
	all_labels = np.zeros((2160, 3))
	for i in range(len(hardhat_imgs)):
		print(i)
		all_imgs[i, :, :, :] = preprocess_input(cv2.imread(hardhat_imgs[i]))
		all_labels[i, 0] = 1
	for i in range(len(people_imgs)):
		print(i)
		all_imgs[i, :, :, :] = preprocess_input(cv2.imread(people_imgs[i]))
		all_labels[i, 1] = 1
	for i in range(len(crane_imgs)):
		print(i)
		all_imgs[i, :, :, :] = preprocess_input(cv2.imread(crane_imgs[i]))
		all_labels[i, 2] = 1
	all_imgs, all_labels = unison_shuffled(all_imgs, all_labels)
	idxs = list(range(len(all_labels)))
	random.shuffle(idxs)
	train_idxs = idxs[0:int(0.8 * len(idxs))]
	test_idxs = idxs[int(0.8 * len(idxs)):]
	X_train = all_imgs[train_idxs, :, :, :]
	y_train = all_labels[train_idxs, :]
	X_test = all_imgs[test_idxs, :, :, :]
	y_test = all_labels[test_idxs, :]
	return X_train, X_test, y_train, y_test

def train_resnet50(X_train, X_test, y_train, y_test):
	print("compiling model...")
	batch_size = 32
	steps_per_epoch = len(X_train) / batch_size
	num_classes = 3
	base_model = ResNet50(weights="imagenet", include_top=False)
	last = base_model.output
	last = GlobalAveragePooling2D()(last)
	pred = Dense(num_classes, activation='softmax')(last)
	model = Model(inputs=base_model.input, outputs=pred)
	train_gen = ImageDataGenerator(
	    rotation_range=30,
	    width_shift_range=0.2,
	    height_shift_range=0.2,
	    shear_range=0.2,
	    zoom_range=0.2,
	    horizontal_flip=True
	)
	model = setup_to_finetune(model, len(base_model.layers)-1)
	print("training model...")
	model.fit_generator(train_gen.flow(X_train, y_train, batch_size=batch_size), steps_per_epoch=steps_per_epoch, epochs=10, validation_data=(X_test, y_test))

if __name__ == '__main__':
	X_train, X_test, y_train, y_test = load_data()
	train_resnet50(X_train, X_test, y_train, y_test)
