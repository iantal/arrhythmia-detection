from __future__ import division, print_function

import numpy as np
from config import get_config
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.layers import Input, Conv1D, Dense, add, Dropout, MaxPooling1D, Activation, BatchNormalization, \
    Lambda
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from utils import mkdir_recursive, loaddata, print_results


class EcgModel(object):
    def first_convolution_block(self, inputs, config):
        layer = Conv1D(filters=config.filter_length,
                       kernel_size=config.kernel_size,
                       padding='same',
                       strides=1,
                       kernel_initializer='he_normal')(inputs)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)

        shortcut = MaxPooling1D(pool_size=1,
                                strides=1)(layer)

        layer = Conv1D(filters=config.filter_length,
                       kernel_size=config.kernel_size,
                       padding='same',
                       strides=1,
                       kernel_initializer='he_normal')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Dropout(config.drop_rate)(layer)
        layer = Conv1D(filters=config.filter_length,
                       kernel_size=config.kernel_size,
                       padding='same',
                       strides=1,
                       kernel_initializer='he_normal')(layer)
        return add([shortcut, layer])

    def main_loop_blocks(self, layer, config):
        filter_length = config.filter_length
        n_blocks = 15
        for block_index in range(n_blocks):
            def zeropad(x):
                y = K.zeros_like(x)
                return K.concatenate([x, y], axis=2)

            def zeropad_output_shape(input_shape):
                shape = list(input_shape)
                assert len(shape) == 3
                shape[2] *= 2
                return tuple(shape)

            subsample_length = 2 if block_index % 2 == 0 else 1
            shortcut = MaxPooling1D(pool_size=subsample_length)(layer)

            # 5 is chosen instead of 4 from the original model
            if block_index % 4 == 0 and block_index > 0:
                # double size of the network and match the shapes of both branches
                shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)
                filter_length *= 2

            layer = BatchNormalization()(layer)
            layer = Activation('relu')(layer)
            layer = Conv1D(filters=filter_length,
                           kernel_size=config.kernel_size,
                           padding='same',
                           strides=subsample_length,
                           kernel_initializer='he_normal')(layer)
            layer = BatchNormalization()(layer)
            layer = Activation('relu')(layer)
            layer = Dropout(config.drop_rate)(layer)
            layer = Conv1D(filters=filter_length,
                           kernel_size=config.kernel_size,
                           padding='same',
                           strides=1,
                           kernel_initializer='he_normal')(layer)
            layer = add([shortcut, layer])
        return layer

    def output_block(self, layer, inputs):
        classes = ['N', 'V', '/', 'A', 'F', '~']
        len_classes = len(classes)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        outputs = TimeDistributed(Dense(len_classes, activation='softmax'))(layer)
        model = Model(inputs=inputs, outputs=outputs)

        adam = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(optimizer=adam,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        return model

    def create_network(self, config):
        inputs = Input(shape=(config.input_size, 1), name='input')
        layer = self.first_convolution_block(inputs, config)
        layer = self.main_loop_blocks(layer, config)
        return self.output_block(layer, config, inputs)


class Train(object):
    def __init__(self):
        mkdir_recursive('models')

    @staticmethod
    def run(config, X, y, Xval=None, yval=None):
        classes = ['N', 'V', '/', 'A', 'F', '~']

        x = np.expand_dims(X, axis=2)

        if not config.split:
            x, xv, y, yval = train_test_split(x, y, test_size=0.2, random_state=1)
        else:
            xv = np.expand_dims(Xval, axis=2)
            (m, n) = y.shape
            y = y.reshape((m, 1, n))
            (mvl, nvl) = yval.shape
            yval = yval.reshape((mvl, 1, nvl))

        model = EcgModel().create_network(config)

        callbacks = [
            EarlyStopping(patience=config.patience, verbose=1),
            ReduceLROnPlateau(factor=0.5, patience=3, min_lr=0.01, verbose=1),
            TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_grads=False, write_images=True),
            ModelCheckpoint('models/{}-latest.hdf5'.format(config.feature), monitor='val_loss', save_best_only=False,
                            verbose=1, period=10)
        ]

        model.fit(x, y,
                  validation_data=(xv, yval),
                  epochs=config.epochs,
                  batch_size=config.batch,
                  callbacks=callbacks,
                  initial_epoch=0)
        print_results(config, model, xv, yval, classes, )


if __name__ == "__main__":
    config = get_config()
    print('feature:', config.feature)
    (X, y, Xval, yval) = loaddata(config.input_size, config.feature)
    Train().run(config, X, y, Xval, yval)
