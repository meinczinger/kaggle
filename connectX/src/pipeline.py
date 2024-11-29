import random
from pathlib import Path
from threading import Lock

from game import Simulator, get_config, GameManager
from mcts import NeuralNetworkMonteCarloTreeSearch
from agent import MCTSAgent
from parallel_player import ParallelPlayer

DEPTH_FOR_RANDOM_GAMES_FOR_SELF_PLAY = 10
PROB_FOR_RANDOM_MOVE_SELF_PLAY = 0.1

TIME_REDUCTION = 1.0

GAMES_FOLDER = Path("resources/games/")
MODELS_FOLDER = Path("resources/models/")

config = get_config()

pplayer = ParallelPlayer(
    1,
    10,
    DEPTH_FOR_RANDOM_GAMES_FOR_SELF_PLAY,
    PROB_FOR_RANDOM_MOVE_SELF_PLAY,
    1.0,
    GAMES_FOLDER,
    MODELS_FOLDER,
)

pplayer.parallel_self_play(10)
