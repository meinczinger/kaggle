import random
from pathlib import Path
import threading
import concurrent.futures
import time
import gc
import pandas as pd

from game import Simulator, get_config, GameManager
from mcts import NeuralNetworkMonteCarloTreeSearch
from agent import MCTSAgent
from logger import Logger

config = get_config()


class ParallelPlayer:
    def __init__(
        self,
        nr_of_threads: int,
        history_size: int,
        dept_for_random_games: int,
        prob_for_random_move: float,
        time_reduction: float,
        games_folder: Path,
        models_folder: Path,
    ):
        self._nr_of_threads = nr_of_threads
        self._history_size = history_size
        self._dept_for_random_games = dept_for_random_games
        self._prob_for_random_move = prob_for_random_move
        self._time_reduction = time_reduction
        self._games_folder = games_folder
        self._models_folder = models_folder

        self.logger = Logger.info_logger("ParallelPlayer")

    def parallel_self_play(self, iter):
        try:
            self.cut_games_file(1)
            self.cut_games_file(2)

        except:
            print("No files to cut")

        self.logger.info("Starting self play")
        threading.excepthook = self.custom_hook
        lock = threading.Lock()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._nr_of_threads
        ) as executor:
            for i in range(self._nr_of_threads):
                executor.submit(self.self_play, iter, lock, i)
                time.sleep(5)

        print(gc.get_count())
        gc.collect()
        print(gc.get_count())

    def custom_hook(self, args):
        print(f"Thread failed: {args.exc_value}")

    def cut_games_file(self, player):
        # cut states files
        games_file = "train_priors_values" + "_p" + str(player) + ".csv"
        state_values = pd.read_csv(
            self._games_folder / games_file, delimiter=",", header=None
        )
        print("Size of", games_file, "is", len(state_values))
        state_values = state_values[-self._history_size :]
        state_values.to_csv(self._games_folder / games_file, index=False, header=False)

    def self_play(self, iter, lock, thread_nr):
        print("Starting self play", "iter=", iter)
        sim = Simulator(
            config,
            self._games_folder,
            self._models_folder,
            MCTSAgent(
                config,
                NeuralNetworkMonteCarloTreeSearch(
                    config,
                    self_play=True,
                    evaluation=False,
                    use_best_player1=True,
                    use_best_player2=True,
                ),
                self_play=True,
                time_reduction=self._time_reduction,
            ),
        )
        for i in range(iter):
            print("Thread:", thread_nr, "Play:", i)
            sim.self_play(
                None,
                GameManager(self._games_folder),
                lock,
                thread_nr,
                random.randint(0, self._dept_for_random_games),
                self._prob_for_random_move,
            )
