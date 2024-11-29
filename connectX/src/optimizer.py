import pandas as pd
from pathlib import Path
import numpy as np

from logger import Logger
from model.cnn import Residual_CNN


class Optimizer:
    def __init__(self, lr: float, batch_size: int, games_folder: Path):
        self._lr = lr
        self._batch_size = batch_size
        self._games_folder = games_folder
        self._logger = Logger.info_logger("Optimizer")

    def optimize(self, train=True):
        self._logger.info("Starting training")
        print("Start train for player 1")
        self.train_model(train, 1)
        print("Start train for player 2")
        self.train_model(train, 2)

    def create_initial_model(self):
        for player in [1, 2]:
            state_value_conv_model = Residual_CNN("best_model_p" + str(player))

            try:
                state_value_conv_model.load(None, self._lr)
            except:
                state_value_conv_model.create_model(self._lr)
                state_value_conv_model.save()

    def train_model(self, train, player):
        train_x_channels = train_y = None
        state_value_conv_model = Residual_CNN("candidate_model_p" + str(player))

        try:
            state_value_conv_model.load(None, self._lr)
        except:
            state_value_conv_model.create_model(self._lr)

        if train:
            # state_values = np.genfromtxt('resources/games/train_state_value' + '_p' + str(player) + '.csv',

            #                              delimiter=',', dtype=np.int64)
            games_file = "train_priors_values" + "_p" + str(player) + ".csv"
            train_values = pd.read_csv(
                self._games_folder / games_file, delimiter=",", header=None
            )

            train_data = train_values.to_numpy(dtype=float)
            train_x = train_data[:, :-8]
            train_x = train_x.reshape((train_x.shape[0], 6, 7, 1))

            # train_x = sample_values[:, :-1]
            # train_x = train_x.reshape(train_x.shape[0], 6, 7)
            train_x_channels = state_value_conv_model._channels(train_x)
            # train_x_channels = np.zeros((train_x.shape[0], 6, 7, 2))
            # train_x_channels[:, :, :, 0] = np.where(train_x == 1, 1, 0)
            # train_x_channels[:, :, :, 1] = np.where(train_x == 2, 1, 0)
            train_y = {
                "value_head": np.array([(row[42] + 1.0) / 2.0 for row in train_data]),
                "policy_head": np.array([row[43:] for row in train_data]),
            }
            # train_y = np.where(train_y == 1, 1, 0)
            # train_y = (train_y + 1.0) / 2.0
            # train_y += 1
            # train_y = to_categorical(train_y, num_classes=3)

        if train:
            state_value_conv_model.train(
                train_x_channels, train_y, 10, self._batch_size
            )
            self._logger.info(
                "Loss of value head network is "
                + str(state_value_conv_model.history().history["value_head_loss"][-1])
            )
            self._logger.info(
                "Loss of policy head network is "
                + str(state_value_conv_model.history().history["policy_head_loss"][-1])
            )

        state_value_conv_model.save()
        return state_value_conv_model
