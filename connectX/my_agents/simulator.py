import numpy as np
from kaggle_environments.utils import Struct
from my_agents.bitboard import BitBoard
from my_agents.logger import Logger
import copy
import pandas as pd
from pathlib import Path
import random


MODEL_FOLDER = Path("resources/models/")
GAMES_FOLDER = Path("resources/games/")


class Simulator:
    def __init__(self, configuration, agent1=None, agent2=None):
        """Simulates games
        If no agent is given, it does random simulation, otherwise it uses the agent(s)
        """
        self._config = configuration
        self.agents = [agent1, agent2]
        self._logger = Logger.logger("Simulator")

    def self_play(
        self,
        bitboard=None,
        callback_for_write=None,
        lock=None,
        thread_nr=0,
        nr_of_random_moves=0,
        prob=0.0,
    ):
        obs = Struct()
        marks = [1, 2]

        obs.board = [0] * 42

        if bitboard is None:
            bitboard = BitBoard.create_empty_board(
                self._config.columns, self._config.rows, self._config.inarow, 1
            )

        ply = 0

        step = 1

        last_random_move = 0

        # Execute moves before we reach a terminal state
        while not bitboard.is_terminal_state():
            # print("Thread", thread_nr, "making move", ply)
            obs.board = bitboard.to_list()
            obs.mark = marks[ply]
            if (step <= nr_of_random_moves) and (random.random() < prob):
                # Get a random move
                action = np.random.choice(bitboard.possible_actions())
                last_random_move = step
            else:
                action = self.agents[0].act(obs, step, step == nr_of_random_moves)

            # Make the move
            bitboard.make_action(action)

            # Swap players
            ply = (ply + 1) % 2
            step += 1

        # Set reward from the first player's point of view
        if bitboard.is_draw():
            reward = 0.0
        else:
            reward = 1.0

        priors = self.agents[0].MCTS().priors()

        priors_df = pd.DataFrame(
            [
                [priors[k]["player"]]
                + priors[k]["board"]
                + [reward if priors[k]["player"] == bitboard.last_player() else -reward]
                + priors[k]["priors"]
                for k in priors
            ]
        )

        priors_df = priors_df[last_random_move:]
        
        if callback_for_write is None:
            priors_df[priors_df[0] == 1].iloc[:, 1:].to_csv(
                GAMES_FOLDER / "train_priors_values_p1.csv",
                index=False,
                header=False,
                mode="a",
            )
            priors_df[priors_df[0] == 2].iloc[:, 1:].to_csv(
                GAMES_FOLDER / "train_priors_values_p2.csv",
                index=False,
                header=False,
                mode="a",
            )
        else:
            callback_for_write(
                priors_df[priors_df[0] == 1].iloc[:, 1:],
                "train_priors_values_p1.csv",
                lock,
                thread_nr,
            )
            callback_for_write(
                priors_df[priors_df[0] == 2].iloc[:, 1:],
                "train_priors_values_p2.csv",
                lock,
                thread_nr,
            )

    def simulate(self, board: BitBoard, to_play: int) -> int:
        """
        Runs a simulation from the given board
        :param board: the current board positionq
        :param to_play: the mark of the player who is next to play
        :return: 0 in case of a tie, 1 if the player wins, who plays next, -1 otherwise
        """
        obs = Struct()
        player = to_play - 1
        obs.mark = to_play
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
            if bitboard.is_draw():
                return 0
            else:
                return bitboard.last_player()
        except Exception as ex:
            self._logger._logger.error(ex, exc_info=True)

    def generate_random_position(self, nr_of_moves: int, prob: float) -> BitBoard:
        obs = Struct()
        player = 0
        obs.mark = 1
        bitboard = BitBoard.create_empty_board(
            self._config.columns, self._config.rows, self._config.inarow, 1
        )
        step = 1
        # Execute moves before we reach a terminal state
        for _ in range(nr_of_moves):
            if bitboard.is_terminal_state():
                return None
            obs.board = bitboard.to_list()
            if random.random() < prob:
                # Get a random move
                action = np.random.choice(bitboard.possible_actions())
            else:
                action = self.agents[0].act(obs, step)
                step += 1
            # Make the move
            bitboard.make_action(action)

            # Swap players
            player = 1 - player
            obs.mark = (obs.mark % 2) + 1
        return bitboard
