from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
from tensorflow.keras.models import load_model
import numpy as np
from bitboard import BitBoard
from logger import Logger
from kaggle_environments.utils import Struct
from tensorflow.keras.utils import to_categorical


logger = Logger.info_logger("CE", target="cross_entropy.log")


class Model:
    def __init__(self):
        self._model_p1 = None
        self._model_p2 = None

    @staticmethod
    def _create_model():
        prob_model = Sequential([
            Conv2D(32, 3, padding='same', activation='relu', input_shape=(6, 7, 2)),
            MaxPooling2D(),
            BatchNormalization(),
            Dropout(0.3),
            Conv2D(64, 3, padding='same', activation='relu'),
            Conv2D(64, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            BatchNormalization(),
            Dropout(0.3),
            Conv2D(128, 3, padding='same', activation='relu'),
            Conv2D(128, 3, padding='same', activation='relu'),
            Conv2D(128, 3, padding='same', activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(7, activation='softmax')
        ])
        return prob_model

    def train(self, episodes: list, player: int):
        try:
            model = load_model('resources/models/cross_entropy/prob_model_p' + str(player))
        except:
            model = self._create_model()
        steps = [step['board'] for episode in episodes for step in episode.steps()]
        train_x = np.array(steps).\
            reshape((len(steps), 6, 7, 2))
        train_y = np.array([step['action'] for episode in episodes for step in episode.steps()])
        train_y = to_categorical(train_y, num_classes=7)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        if len(episodes) > 0:
            acc = -float("inf")
            max_epochs = 20
            i = 0
            while i < max_epochs:
                history = model.fit(train_x, train_y, batch_size=64, epochs=1, shuffle=True, validation_split=0.2)
                if history.history['val_accuracy'][-1] <= acc:
                    break
                acc = history.history['val_accuracy'][-1]
                i += 1
        print("Saving model")
        model.save('resources/models/cross_entropy/prob_model_p' + str(player))
        logger.info("Loss of player " + str(player) + " is " + str(history.history['loss'][-1]) + ", " +
                    str(history.history['val_loss'][-1]))
        logger.info("Accuracy of  player " + str(player) + " is " + str(history.history['accuracy'][-1]) + ", " +
                    str(history.history['val_accuracy'][-1]))

    def predict(self, board, possible_actions, player, train=False) -> int:
        if self._model_p1 is None:
            self._model_p1 = load_model('resources/models/cross_entropy/prob_model_p1')
        if self._model_p2 is None:
            self._model_p2 = load_model('resources/models/cross_entropy/prob_model_p2')
        probs = np.around(self._model_p1.predict(board)[0], decimals=4) if player == 1 \
            else np.around(self._model_p2.predict(board)[0], decimals=4)
        for action in range(7):
            if action not in possible_actions:
                probs[action] = 0.
        probs = probs / probs.sum()
        if train:
            action = np.random.choice(range(7), 1, p=probs)[0]
        else:
            action = np.argmax(probs)
        return action


class Episode:
    def __init__(self):
        self._steps = []
        self._reward = 0.
        self._winner = 0

    def add_step(self, cboard, action):
        step = {'board': cboard, 'action': action}
        self._steps.append(step)

    def set_reward(self, reward):
        self._reward = reward

    def set_winner(self, winner):
        self._winner = winner

    def reward(self):
        return self._reward

    def winner(self):
        return self._winner

    def steps(self):
        return self._steps


class Play:
    def __init__(self, model, columns, rows, inarow):
        self._model = model
        self._columns = columns
        self._rows = rows
        self._inarow = inarow

    def play(self) -> Episode:
        board = BitBoard.create_empty_board(self._columns, self._rows, self._inarow, 1)
        episode = Episode()
        reward = 0
        player = 1
        while not board.is_terminal_state():
            cboard = board.channels()
            action = self._model.predict(cboard, board.possible_actions(), player, True)
            board.make_action(action)
            player = (player % 2) + 1
            episode.add_step(cboard, action)
            reward = reward - 0.01

        if not board.is_draw():
            reward += 1
            episode.set_winner(board.last_player())
        else:
            episode.set_winner(0)

        episode.set_reward(reward)
        return episode


class CrossentropyAgent:
    _agent = None

    def __init__(self, configuration):
        self._config = configuration
        self._model = Model()

    def train(self, iteritions, nr_of_episodes):
        for i in range(iteritions):
            self._model = Model()
            player = Play(self._model, self._config.columns, self._config.rows, self._config.inarow)
            episodes = []
            for j in range(nr_of_episodes):
                print('iteration', i, 'episode', j)
                episodes.append(player.play())

            episodes = sorted(episodes, key=lambda e: e.reward())

            episodes_p1 = [episode for episode in episodes if episode.winner() == 1]
            size_p1 = len(episodes_p1)
            episodes_p2 = [episode for episode in episodes if episode.winner() == 2]
            size_p2 = len(episodes_p2)
            episodes_p1 = episodes_p1[int(size_p1*0.1):]
            episodes_p2 = episodes_p2[int(size_p2*0.1):]
            self._model.train(episodes_p1, 1)
            self._model.train(episodes_p2, 2)

    def act(self, observation) -> int:
        board = observation.board

        own_player = observation.mark

        bitboard = BitBoard.create_from_board(self._config.columns, self._config.rows, self._config.inarow,
                                              own_player, board)

        action = self._model.predict(bitboard.channels(), bitboard.possible_actions(), own_player)

        return int(action)

    @staticmethod
    def get_instance(configuration):
        if CrossentropyAgent._agent is None:
            CrossentropyAgent._agent = CrossentropyAgent(configuration)
        return CrossentropyAgent._agent


# config = Struct()
# config.columns = 7
# config.rows = 6
# config.inarow = 4
# config.timeout = 2.0
#
# agent = CrossentropyAgent(config)
# agent.train(10, 1000)
