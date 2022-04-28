from nn_model import DQNNNModel, DQNResNetNNModel
import numpy as np
from bitboard import BitBoard
from kaggle_environments.utils import Struct


EPSILON_START = 1.0
EPSILON_FINAL = 0.05
EPSILON_DECAY_LAST_EPISODE = 10000
BATCH_SIZE = 64
TRAIN_BATCH_SIZE = 8
GAMMA = 0.99
NR_OF_EXECUTIONS = 20000
UPDATE_TARGET_NETWORK = 1000
MIN_BUFFER_SIZE = 1000
BUFFER_SIZE = 10000
MODEL_NAME = 'dqn'
TARGET_MODEL_NAME = 'dqn_target'
LEARNING_RATE = 5e-5
TRAIN_END_POSITIONS = 2000
START_LAST_MOVES = 2
INC_LAST_MOVES = 2


class DQN:
    def __init__(self, configuration):
        self._config = configuration
        self._nn = DQNNNModel(MODEL_NAME)
        self._target_nn = DQNNNModel(TARGET_MODEL_NAME)
        self._epsilon = EPSILON_START
        self._replay_buffer = []

    def play(self, keep_last_moves):
        player = 1
        buffer = []
        board = BitBoard.create_empty_board(self._config.columns, self._config.rows, self._config.inarow, 1)
        while not board.is_terminal_state():
            old_board = board.hash()
            if np.random.random() < self._epsilon:
                action = np.random.choice(board.possible_actions())
            else:
                if player == 1:
                    action, _ = self._nn.predict_state(board)
                else:
                    action, _ = self._nn.predict_state(board, False)
            board.make_action(action)
            player = (player % 2) + 1
            if board.is_terminal_state() and (not board.is_draw()):
                reward = 1 if board.last_player() == 1 else -1
            else:
                reward = 0

            buffer.append([old_board, action, reward, board.hash(), board.is_terminal_state(),
                           board.last_player()])

        #self._replay_buffer.append(buffer[-keep_last_moves:])
        self._replay_buffer += buffer[-keep_last_moves:]
        # Keep only the last portion of the buffer
        self._replay_buffer = self._replay_buffer[-BUFFER_SIZE:]

    def sample(self):
        batch_x = batch_y = None
        sample_indices = np.random.choice(len(self._replay_buffer), BATCH_SIZE)
        for i in sample_indices:
            np_y = self._target_nn.predictions(
                BitBoard.create_from_bitboard(self._config.columns, self._config.rows, self._config.inarow,
                                              1, self._replay_buffer[i][3]))  # , self._replay_buffer[i][5] == 1)
            # _, prediction = self._target_nn.predict_state(
            #     BitBoard.create_from_bitboard(self._config.columns, self._config.rows, self._config.inarow,
            #                                   1, self._replay_buffer[i][3]))#, self._replay_buffer[i][5] == 1)
            np_board = BitBoard.bitboard_to_numpy2d(self._replay_buffer[i][0])
            np_board = BitBoard.board_channels(np_board)
            # np_y = np.zeros((7,))
            # If this is an end-state
            if self._replay_buffer[i][4]:
                np_y = np.zeros((7,))
                np_y[self._replay_buffer[i][1]] = self._replay_buffer[i][2]
            else:
                np_y[self._replay_buffer[i][1]] = self._replay_buffer[i][2] + GAMMA * np_y[self._replay_buffer[i][1]]
            assert not np.any(np.isnan(np_y))
            if batch_x is None:
                batch_x = np_board
                batch_y = np_y
            else:
                batch_x = np.append(batch_x, np_board, axis=0)
                batch_y = np.append(batch_y, np_y, axis=0)

        batch_x = batch_x.reshape((len(sample_indices), 6, 7, 2))
        batch_y = batch_y.reshape((len(sample_indices), 7))
        return batch_x, batch_y

    def sample2(self, keep_last_moves: int):
        batch_x = batch_y = None
        samples = 0
        sample_indices = np.random.choice(len(self._replay_buffer), int(BATCH_SIZE/keep_last_moves))
        for i in sample_indices:
            prev_value = 0
            for j in range(len(self._replay_buffer[i]) - 1, -1, -1):
                np_y = self._target_nn.predictions(
                    BitBoard.create_from_bitboard(self._config.columns, self._config.rows, self._config.inarow,
                                                  1, self._replay_buffer[i][j][3]))  # , self._replay_buffer[i][5] == 1)
                np_board = BitBoard.bitboard_to_numpy2d(self._replay_buffer[i][j][0])
                np_board = BitBoard.board_channels(np_board)
                # If this is an end-state
                if self._replay_buffer[i][j][4]:
                    prev_value = self._replay_buffer[i][j][2]
                    np_y[self._replay_buffer[i][j][1]] = prev_value
                else:
                    prev_value = GAMMA * prev_value
                    np_y[self._replay_buffer[i][j][1]] = prev_value
                samples += 1
                if batch_x is None:
                    batch_x = np_board
                    batch_y = np_y
                else:
                    batch_x = np.append(batch_x, np_board, axis=0)
                    batch_y = np.append(batch_y, np_y, axis=0)

        batch_x = batch_x.reshape((samples, 6, 7, 2))
        batch_y = batch_y.reshape((samples, 7))
        return batch_x, batch_y

    def set_epsilon(self, episode):
        self._epsilon = max(EPSILON_FINAL, EPSILON_START - episode / EPSILON_DECAY_LAST_EPISODE)

    def learn(self):
        self._nn.create_model(LEARNING_RATE)
        self._target_nn.create_model()

        keep_last_moves = START_LAST_MOVES
        for i in range(NR_OF_EXECUTIONS):
            self.set_epsilon(i)
            self.play(keep_last_moves)
            if len(self._replay_buffer) > MIN_BUFFER_SIZE:
                if i % 5 == 0:
                    print("Iteration", i, "epsilon", self._epsilon)
                    batch_x, batch_y = self.sample()
                    self._nn.train(batch_x, batch_y, 1, TRAIN_BATCH_SIZE)
                    if i % UPDATE_TARGET_NETWORK == 0:
                        print("Saving model")
                        self._nn.save()
                        self._target_nn.load(MODEL_NAME)
                        keep_last_moves += INC_LAST_MOVES

        self._nn.save()


# config = Struct()
# config.columns = 7
# config.rows = 6
# config.inarow = 4
# config.timeout = 2.0
#
# dqn = DQN(config)
# dqn.learn()