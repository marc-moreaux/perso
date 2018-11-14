# Simple CNN model for CIFAR-10
import numpy
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.models import Model, Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
from keras import backend as K

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
img_width = img_height = X_test.shape[1]


def model_base():
	model = Sequential()
	model.add(Convolution2D(96,  3, 3, input_shape=(img_width, img_height, 3), border_mode='same'))# model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Convolution2D(96,  3, 3, border_mode='same'))
	model.add(Activation('relu'))
	model.add(Convolution2D(96,  3, 3, border_mode='same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(192, 3, 3, border_mode='same'))
	model.add(Activation('relu'))
	model.add(Convolution2D(192, 3, 3, border_mode='same'))
	model.add(Activation('relu'))
	model.add(Convolution2D(192, 3, 3, border_mode='same'))
	model.add(Activation('relu'))

	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(192, 3, 3, border_mode='same'))
	model.add(Activation('relu'))
	model.add(Flatten())
	model.add(Dense(192))
	model.add(Activation('sigmoid'))
	model.add(Dense(num_classes, activation='softmax'))

	return model

def learn_and_save(model, batch_size=25, lr=0.01, decay=.01, algo=0, model_type=0):
	# Compile model
	sgd = SGD(lr=lr, momentum=0.9, decay=decay, nesterov=False)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	print(model.summary())
	logs = model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=25, batch_size=batch_size)

	with open("log.txt", "a+") as file:
		file.write("tab[%.5f][%d][%d][%d] = "%(lr,batch_size,algo,model_type))
		file.write(str(logs.history)+"\n\n")


model = model_base()
learn_and_save(model=model, batch_size=8,  lr=0.01*2 , decay=.01)
learn_and_save(model=model, batch_size=8,  lr=0.01*1 , decay=.01)
learn_and_save(model=model, batch_size=8,  lr=0.01/2 , decay=.01)
learn_and_save(model=model, batch_size=8,  lr=0.01/4 , decay=.01)
learn_and_save(model=model, batch_size=8,  lr=0.01/8 , decay=.01)
learn_and_save(model=model, batch_size=8,  lr=0.01/16, decay=.01)

learn_and_save(model=model, batch_size=16, lr=0.01*2 , decay=.01)
learn_and_save(model=model, batch_size=16, lr=0.01*1 , decay=.01)
learn_and_save(model=model, batch_size=16, lr=0.01/2 , decay=.01)
learn_and_save(model=model, batch_size=16, lr=0.01/4 , decay=.01)
learn_and_save(model=model, batch_size=16, lr=0.01/8 , decay=.01)
learn_and_save(model=model, batch_size=16, lr=0.01/16, decay=.01)

learn_and_save(model=model, batch_size=32, lr=0.01*2 , decay=.01)
learn_and_save(model=model, batch_size=32, lr=0.01*1 , decay=.01)
learn_and_save(model=model, batch_size=32, lr=0.01/2 , decay=.01)
learn_and_save(model=model, batch_size=32, lr=0.01/4 , decay=.01)
learn_and_save(model=model, batch_size=32, lr=0.01/8 , decay=.01)
learn_and_save(model=model, batch_size=32, lr=0.01/16, decay=.01)

learn_and_save(model=model, batch_size=64, lr=0.01*2 , decay=.01)
learn_and_save(model=model, batch_size=64, lr=0.01*1 , decay=.01)
learn_and_save(model=model, batch_size=64, lr=0.01/2 , decay=.01)
learn_and_save(model=model, batch_size=64, lr=0.01/4 , decay=.01)
learn_and_save(model=model, batch_size=64, lr=0.01/8 , decay=.01)
learn_and_save(model=model, batch_size=64, lr=0.01/16, decay=.01)

with open("log.txt", "a+") as file:
	file.write("\n\nnew batch of learning\n\n\n")
numpy.random.seed(8)

learn_and_save(model=model, batch_size=8,  lr=0.01*2 , decay=.01)
learn_and_save(model=model, batch_size=8,  lr=0.01*1 , decay=.01)
learn_and_save(model=model, batch_size=8,  lr=0.01/2 , decay=.01)
learn_and_save(model=model, batch_size=8,  lr=0.01/4 , decay=.01)
learn_and_save(model=model, batch_size=8,  lr=0.01/8 , decay=.01)
learn_and_save(model=model, batch_size=8,  lr=0.01/16, decay=.01)

learn_and_save(model=model, batch_size=16, lr=0.01*2 , decay=.01)
learn_and_save(model=model, batch_size=16, lr=0.01*1 , decay=.01)
learn_and_save(model=model, batch_size=16, lr=0.01/2 , decay=.01)
learn_and_save(model=model, batch_size=16, lr=0.01/4 , decay=.01)
learn_and_save(model=model, batch_size=16, lr=0.01/8 , decay=.01)
learn_and_save(model=model, batch_size=16, lr=0.01/16, decay=.01)

learn_and_save(model=model, batch_size=32, lr=0.01*2 , decay=.01)
learn_and_save(model=model, batch_size=32, lr=0.01*1 , decay=.01)
learn_and_save(model=model, batch_size=32, lr=0.01/2 , decay=.01)
learn_and_save(model=model, batch_size=32, lr=0.01/4 , decay=.01)
learn_and_save(model=model, batch_size=32, lr=0.01/8 , decay=.01)
learn_and_save(model=model, batch_size=32, lr=0.01/16, decay=.01)

learn_and_save(model=model, batch_size=64, lr=0.01*2 , decay=.01)
learn_and_save(model=model, batch_size=64, lr=0.01*1 , decay=.01)
learn_and_save(model=model, batch_size=64, lr=0.01/2 , decay=.01)
learn_and_save(model=model, batch_size=64, lr=0.01/4 , decay=.01)
learn_and_save(model=model, batch_size=64, lr=0.01/8 , decay=.01)
learn_and_save(model=model, batch_size=64, lr=0.01/16, decay=.01)


with open("log.txt", "a+") as file:
	file.write("\n\nnew batch of learning\n\n\n")
numpy.random.seed(9)

learn_and_save(model=model, batch_size=8,  lr=0.01*2 , decay=.01)
learn_and_save(model=model, batch_size=8,  lr=0.01*1 , decay=.01)
learn_and_save(model=model, batch_size=8,  lr=0.01/2 , decay=.01)
learn_and_save(model=model, batch_size=8,  lr=0.01/4 , decay=.01)
learn_and_save(model=model, batch_size=8,  lr=0.01/8 , decay=.01)
learn_and_save(model=model, batch_size=8,  lr=0.01/16, decay=.01)

learn_and_save(model=model, batch_size=16, lr=0.01*2 , decay=.01)
learn_and_save(model=model, batch_size=16, lr=0.01*1 , decay=.01)
learn_and_save(model=model, batch_size=16, lr=0.01/2 , decay=.01)
learn_and_save(model=model, batch_size=16, lr=0.01/4 , decay=.01)
learn_and_save(model=model, batch_size=16, lr=0.01/8 , decay=.01)
learn_and_save(model=model, batch_size=16, lr=0.01/16, decay=.01)

learn_and_save(model=model, batch_size=32, lr=0.01*2 , decay=.01)
learn_and_save(model=model, batch_size=32, lr=0.01*1 , decay=.01)
learn_and_save(model=model, batch_size=32, lr=0.01/2 , decay=.01)
learn_and_save(model=model, batch_size=32, lr=0.01/4 , decay=.01)
learn_and_save(model=model, batch_size=32, lr=0.01/8 , decay=.01)
learn_and_save(model=model, batch_size=32, lr=0.01/16, decay=.01)

learn_and_save(model=model, batch_size=64, lr=0.01*2 , decay=.01)
learn_and_save(model=model, batch_size=64, lr=0.01*1 , decay=.01)
learn_and_save(model=model, batch_size=64, lr=0.01/2 , decay=.01)
learn_and_save(model=model, batch_size=64, lr=0.01/4 , decay=.01)
learn_and_save(model=model, batch_size=64, lr=0.01/8 , decay=.01)
learn_and_save(model=model, batch_size=64, lr=0.01/16, decay=.01)


