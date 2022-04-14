import numpy as np
from kaggle_environments.utils import Struct
from agents.bitboard import BitBoard
from agents.logger import Logger
import copy
import pandas as pd
from pathlib import Path


MODEL_FOLDER = Path("resources/models/")
GAMES_FOLDER = Path("resources/games/")


class Simulator:
    def __init__(self, configuration, agent1=None, agent2=None):
        """ Simulates games
        If no agent is given, it does random simulation, otherwise it uses the agent(s)
        """
        self._config = configuration
        self.agents = [agent1, agent2]
        self._logger = Logger.logger('Simulator')

    def self_play(self):
        obs = Struct()
        marks = [1, 2]
        ply = 0
        obs.step = 0
        obs.board = [0] * 42

        bitboard = BitBoard.create_empty_board(self._config.columns, self._config.rows, self._config.inarow, 1)

        # Store all the moves
        history_list = [[], []]

        # Execute moves before we reach a terminal state
        while not bitboard.is_terminal_state():
            obs.board = bitboard.to_list()
            obs.mark = marks[ply]
            # Get a move given by the agent
            action = self.agents[0].act(obs)

            # Make the move
            bitboard.make_action(action)

            # bb = bitboard.hash()
            # history.append([obs.step, bb[0], bb[1]])
            bb_list = bitboard.to_list()
            # mirror = bitboard.mirror_board()
            history_list[ply].append([obs.step] + bb_list)
            # history_list[ply].append([obs.step] + mirror)

            # Increase the number of steps
            obs.step += 1

            # Swap players
            ply = (ply + 1) % 2

        priors = self.agents[0].priors()
        steps = len(priors)
        priors_df = pd.DataFrame([[priors[k]['player']] + [priors[k]['step']] + priors[k]['board'] +
                                  priors[k]['priors'] for k in priors])
        priors_df.iloc[:, 1] = steps - 1 - priors_df.iloc[:, 1]
        priors_df[priors_df[0] == 1].iloc[:, 1:].\
            to_csv(GAMES_FOLDER / 'train_priors_p1.csv', index=False, header=False, mode='a')
        priors_df[priors_df[0] == 2].iloc[:, 1:].\
            to_csv(GAMES_FOLDER / 'train_priors_p2.csv', index=False, header=False, mode='a')

        # Set reward from the first player's point of view
        if bitboard.is_draw():
            reward = 0
        else:
            if bitboard.last_player() == 1:
                reward = 1
            else:
                reward = -1

        steps = len(history_list[0]) + len(history_list[1])

        for ply in range(2):
            # Store results for later training
            h2 = pd.DataFrame(history_list[ply])
            # Reverse the step_nr to make it the distance from the end state
            h2.iloc[:, 0] = steps - 1 - h2.iloc[:, 0]
            h2['value'] = reward if ply == 0 else -reward
            games_file = 'train_state_value_p' + str(ply+1) + '.csv'
            h2.to_csv(GAMES_FOLDER / games_file, index=False, header=False, mode='a')

    def simulate(self, board: BitBoard, to_play: int) -> int:
        """
        Runs a simulation from the given board
        :param board: the current board positionq
        :param to_play: the mark of the player who is next to play
        :return: 0 in case of a tie, 1 if the player wins, who plays next, -1 otherwise
        """
        obs = Struct()
        player = to_play - 1
        obs.step = 0
        obs.mark = 1
        bitboard = copy.copy(board)
        try:
            # Execute moves before we reach a terminal state
            while not bitboard.is_terminal_state():
                obs.board = bitboard.to_list()
                if self.agents[player] is None:
                    # Get a random move
                    action = np.random.choice(bitboard.possible_actions())
                else:
                    # Get a move given by the agent
                    action = self.agents[player].act(obs)
                # Make the move
                bitboard.make_action(action)

                # Swap players
                player = 1 - player
                obs.mark = (obs.mark % 2) + 1
                # Increase the number of steps
                obs.step += 1
            if bitboard.is_draw():
                return 0
            else:
                return bitboard.last_player()
        except Exception as ex:
            self._logger._logger.error(ex, exc_info=True)
