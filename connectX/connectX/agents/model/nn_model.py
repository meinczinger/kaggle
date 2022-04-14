from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, \
    Dropout, Activation, Add, Input, AveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import glorot_uniform
import numpy as np
from agents.bitboard import BitBoard
from tensorflow import function
from pathlib import Path
import logging

# MODEL_FOLDER = Path("/kaggle_simulations/agent/resources/models/")
MODEL_FOLDER = Path("resources/models/")
MAX_EPOCHS = 20


class NNModel:
    def __init__(self, name):
        self._name = name + '.h5'
        self._model = None
        self._history = None
        self._logger = logging.getLogger('agent')
        self._logger.setLevel(logging.DEBUG)
        self._logger.addHandler(logging.StreamHandler())

    def create_model(self):
        raise NotImplementedError

    def train(self, train_x, train_y, max_epochs=MAX_EPOCHS, batch_size=64):
        val_loss = float("inf")
        i = 0
        while i < max_epochs:
            self._history = self._model.\
                fit(train_x, train_y, batch_size=batch_size, epochs=1, shuffle=True, validation_split=0.2)
            if self._history.history['val_loss'][-1] >= val_loss * 0.995:
                break
            val_loss = self._history.history['val_loss'][-1]
            i += 1

    @staticmethod
    def _channels(boards):
        np_boards_channels = np.zeros((len(boards), 6, 7, 2))
        for i in range(len(boards)):
            np_boards = np.array(boards[i])
            np_boards = np_boards.reshape((1, 6, 7))
            np_boards_channels[i, :, :, 0] = np.where(np_boards == 1, 1, 0)
            np_boards_channels[i, :, :, 1] = np.where(np_boards == 2, 1, 0)
        return np_boards_channels

    @function
    def _predict(self, boards):
        return self._model(boards)

    def predict(self, boards):
        np_boards_channels = self._channels(boards)
        # Get the predicted value for each state
        preds = self._predict(np_boards_channels)
        # return np.around(preds.numpy(), decimals=4)
        return [np.around(pred.numpy(), decimals=4) for pred in preds]

    def predict_state(self, bitboard: BitBoard, use_max=True):
        board = bitboard.to_list()
        predictions = self.predict([board])[0]
        actions = bitboard.possible_actions()
        if len(actions) > 0:
            for a in range(7):
                if a not in actions:
                    if use_max:
                        predictions[a] = -float("inf")
                    else:
                        predictions[a] = float("inf")
            if use_max:
                action = np.argmax(predictions)
            else:
                action = np.argmin(predictions)
            return action, predictions[action]
        else:
            return 0, 0

    def predictions(self, bitboard: BitBoard):
        board = bitboard.to_list()
        predictions = self.predict([board])[0]
        return predictions

    def save(self):
        self._model.save(MODEL_FOLDER / self._name, save_format='h5')

    def load(self, name=None, lr=5e-5):
        if name is None:
            model_file = MODEL_FOLDER / self._name
        else:
            model_file = MODEL_FOLDER / name

        self._model = load_model(model_file, custom_objects={'learning_rate': lr})

    def history(self):
        return self._history


class StateValueNNModel(NNModel):
    def __init__(self, name):
        super().__init__(name)

    def create_model(self, lr=1e-3):
        self._model = Sequential([
            Conv2D(256, 4, padding='same', input_shape=(6, 7, 2)),
            BatchNormalization(axis=3),
            Activation('relu'),
            MaxPooling2D(),
            Dropout(0.3),
            Conv2D(128, 3, padding='same'),
            Conv2D(128, 3, padding='same'),
            BatchNormalization(axis=3),
            Activation('relu'),
            MaxPooling2D(),
            Dropout(0.3),
            Conv2D(128, 3, padding='same'),
            Conv2D(128, 3, padding='same'),
            Conv2D(128, 3, padding='same'),
            BatchNormalization(axis=3),
            Activation('relu'),
            Dropout(0.3),
            Flatten(),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dense(1, activation='sigmoid')
        ])

        self._model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['mean_squared_error'])


class PriorsNNModel(NNModel):
    def __init__(self, name):
        super().__init__(name)

    def create_model(self, lr=1e-3):
        self._model = Sequential([
            Conv2D(128, 4, padding='same', input_shape=(6, 7, 2)),
            BatchNormalization(axis=3),
            Activation('relu'),
            MaxPooling2D(),
            Dropout(0.3),
            Conv2D(128, 3, padding='same'),
            Conv2D(128, 3, padding='same'),
            BatchNormalization(axis=3),
            Activation('relu'),
            MaxPooling2D(),
            Dropout(0.3),
            Conv2D(128, 3, padding='same'),
            Conv2D(128, 3, padding='same'),
            Conv2D(128, 3, padding='same'),
            BatchNormalization(axis=3),
            Activation('relu'),
            Dropout(0.3),
            Flatten(),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dense(7, activation='softmax')
        ])
        self._model.compile(loss='kullback_leibler_divergence', optimizer=Adam(learning_rate=lr),
                            metrics=['mean_squared_error'])


class DQNNNModel(NNModel):
    def __init__(self, name):
        super().__init__(name)

    def create_model(self, lr=1e-3):
        self._model = Sequential([
            Conv2D(64, 3, padding='same', input_shape=(6, 7, 2)),
            BatchNormalization(axis=3),
            Activation('relu'),
            MaxPooling2D(),
            Dropout(0.3),
            Conv2D(128, 3, padding='same'),
            Conv2D(128, 3, padding='same'),
            BatchNormalization(axis=3),
            Activation('relu'),
            MaxPooling2D(),
            Dropout(0.3),
            Conv2D(128, 3, padding='same'),
            Conv2D(128, 3, padding='same'),
            Conv2D(128, 3, padding='same'),
            BatchNormalization(axis=3),
            Activation('relu'),
            Dropout(0.3),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(7, activation='tanh')
        ])

        self._model.compile(loss='mse', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])


class DQNResNetNNModel(NNModel):
    def __init__(self, name):
        super().__init__(name)

    @staticmethod
    def _identity_block(input_shape, f, filters, stage, block):
        """
        Implementation of the identity block as defined in Figure 4

        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network

        Returns:
        X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
        """

        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # Retrieve Filters
        F1, F2, F3 = filters

        # Save the input value. You'll need this later to add back to the main path.
        x_shortcut = input_shape

        # First component of main path
        conv_1 = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
                   kernel_initializer=glorot_uniform(seed=0))(input_shape)
        batch_norm_1 = BatchNormalization(axis=3, name=bn_name_base + '2a')(conv_1)
        activation_1 = Activation('relu')(batch_norm_1)

        # Second component of main path
        conv_2 = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
                   kernel_initializer=glorot_uniform(seed=0))(activation_1)
        batch_norm_2 = BatchNormalization(axis=3, name=bn_name_base + '2b')(conv_2)
        activation_2 = Activation('relu')(batch_norm_2)

        # Third component of main path
        conv_3 = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
                   kernel_initializer=glorot_uniform(seed=0))(activation_2)
        batch_norm_3 = BatchNormalization(axis=3, name=bn_name_base + '2c')(conv_3)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
        merge = Add()([x_shortcut, batch_norm_3])

        return Activation('relu')(merge)

    @staticmethod
    def _convolutional_block(input_shape, f, filters, stage, block, s=2):
        """
        Implementation of the convolutional block as defined in Figure 4

        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        s -- Integer, specifying the stride to be used

        Returns:
        X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
        """

        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # Retrieve Filters
        f1, f2, f3 = filters

        # Save the input value
        x_input = input_shape

        ##### MAIN PATH #####
        # First component of main path
        conv_1 = Conv2D(f1, (1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
                   kernel_initializer=glorot_uniform(seed=0))(input_shape)
        batch_norm_1 = BatchNormalization(axis=3, name=bn_name_base + '2a')(conv_1)
        activation_1 = Activation('relu')(batch_norm_1)

        # Second component of main path
        conv_2 = Conv2D(f2, (f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
                   kernel_initializer=glorot_uniform(seed=0))(activation_1)
        batch_norm_2 = BatchNormalization(axis=3, name=bn_name_base + '2b')(conv_2)
        activation_2 = Activation('relu')(batch_norm_2)

        # Third component of main path
        conv_3 = Conv2D(f3, (1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
                   kernel_initializer=glorot_uniform(seed=0))(activation_2)
        batch_norm_3 = BatchNormalization(axis=3, name=bn_name_base + '2c')(conv_3)

        ##### SHORTCUT PATH ####
        shortcut_conv = Conv2D(f3, (1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                            kernel_initializer=glorot_uniform(seed=0))(x_input)
        shortcut_batch_norm = BatchNormalization(axis=3, name=bn_name_base + '1')(shortcut_conv)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
        merge = Add()([batch_norm_3, shortcut_batch_norm])
        return Activation('relu')(merge)

    def create_model(self, lr=1e-3):
        """
        Implementation of the popular ResNet50 the following architecture:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

        Arguments:
        input_shape -- shape of the images of the dataset
        classes -- integer, number of classes

        Returns:
        model -- a Model() instance in Keras
        """

        # Define the input as a tensor with shape input_shape
        X_input = Input((6, 7, 2))

        # Stage 1
        X = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
        X = BatchNormalization(axis=3, name='bn_conv1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(X)

        # Stage 2
        X = self._convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
        X = self._identity_block(X, 3, [64, 64, 256], stage=2, block='b')
        X = self._identity_block(X, 3, [64, 64, 256], stage=2, block='c')

        ### START CODE HERE ###

        # Stage 3 (≈4 lines)
        X = self._convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
        X = self._identity_block(X, 3, [128, 128, 512], stage=3, block='b')
        X = self._identity_block(X, 3, [128, 128, 512], stage=3, block='c')
        X = self._identity_block(X, 3, [128, 128, 512], stage=3, block='d')

        # Stage 4 (≈6 lines)
        # X = self._convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
        # X = self._identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
        # X = self._identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
        # X = self._identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
        # X = self._identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
        # X = self._identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

        # # Stage 5 (≈3 lines)
        # X = self._convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
        # X = self._identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
        # X = self._identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

        # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
        X = AveragePooling2D((2, 2), name='avg_pool')(X)

        ### END CODE HERE ###

        # output layer
        X = Flatten()(X)
        X = Dense(256, activation='relu')(X)
        X = Dense(7, activation='tanh', name='fc' + str(7), kernel_initializer=glorot_uniform(seed=0))(X)

        # Create model
        self._model = Model(inputs=X_input, outputs=X, name='ResNet50')

        self._model.compile(loss='mse', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])

        