

from keras.models import Sequential
from keras.layers import Flatten, Dense, ZeroPadding2D, Dropout, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

from keras.regularizers import l2
from keras.models import load_model

class Model:

    def __init__(self):
        self.model = None



    ch, row, col = 3, 80, 320

    def make_basic(self):
        self.model = Sequential()
        self.model.add(Convolution2D(128, 3, 3, activation='relu', input_shape=(80, 320, 3)))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        self.model.add(Flatten(input_shape=(256, 3 ,3)))
        self.model.add(Dense(1000))
        self.model.add(Dense(1000))
        self.model.add(Dense(1))

        self.model.compile(loss='mse', optimizer='adam')

    def make_simple(self):
        self.model = Sequential()
        self.model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", input_shape=(80, 320, 3)))
        self.model.add(ELU())
        self.model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
        self.model.add(ELU())
        self.model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
        self.model.add(Flatten())
        self.model.add(Dropout(.2))
        self.model.add(ELU())
        self.model.add(Dense(512))
        self.model.add(Dropout(.5))
        self.model.add(ELU())
        self.model.add(Dense(1))

        self.model.compile(optimizer="adam", loss="mse")

    def make_simple_2(self):
        self.model = Sequential()
        self.model.add(Convolution2D(64, 64, 9, subsample=(3, 3), border_mode="same", input_shape=(64, 64, 3)))
        self.model.add(ELU())
        self.model.add(Convolution2D(64, 64, 36, subsample=(3, 3), border_mode="same"))
        self.model.add(ELU())
        self.model.add(Convolution2D(64, 64, 128, subsample=(3, 3), border_mode="same"))
        self.model.add(ELU())

        self.model.add(Flatten())
        self.model.add(Dense(1024, W_regularizer=l2(0.001)))
        self.model.add(Dense(512, W_regularizer=l2(0.001)))
        self.model.add(Dense(128, W_regularizer=l2(0.001)))
        self.model.add(Dense(1))

        self.model.compile(optimizer="adam", loss="mse", lr=0.01)

    def VGG_16(self, weights_path=None):
        self.model = Sequential()
        #self.model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
        #self.model.add(ZeroPadding2D((1, 1), input_shape=(160, 320, 3)))
        self.model.add(Convolution2D(64, 3, 3, activation='relu',  input_shape=(64, 64, 3)))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(64, 3, 3, activation='relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(64, 3, 3, activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(64, 3, 3, activation='relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(256, 3, 3, activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(256, 3, 3, activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(256, 3, 3, activation='relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(512, 3, 3, activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(512, 3, 3, activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(512, 3, 3, activation='relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(512, 3, 3, activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(512, 3, 3, activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(512, 3, 3, activation='relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(1))

        self.model.compile(loss='mse', optimizer='adam')

    def save(self, filename):
        ## with open(filename+".json", "w") as json_file:
        ##     json_file.write(self.model.to_json())
        ## self.model.save_weights(filename+".h5")
        self.model.save(filename+".h5")


    def fit(self, X_train, y_train, validation=0.2, shuffle=True, nb_epoch=7):
        self.model.fit(X_train, y_train, validation, shuffle, nb_epoch)


    def fit_generator(self, generator, samples_per_epoch, validation_data, nb_val_samples, nb_epoch=5, verbose=1):
        return self.model.fit_generator(generator,
                                        samples_per_epoch=samples_per_epoch,
                                        validation_data=validation_data,
                                        nb_val_samples=nb_val_samples,
                                        nb_epoch=nb_epoch,
                                        verbose=verbose)


