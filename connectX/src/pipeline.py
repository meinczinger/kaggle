import random
from pathlib import Path
from threading import Lock

from game import Simulator, get_config, GameManager
from mcts import NeuralNetworkMonteCarloTreeSearch
from agent import MCTSAgent
from parallel_player import ParallelPlayer
from optimizer import Optimizer
from evaluator import Evaluator

DEPTH_FOR_RANDOM_GAMES_FOR_SELF_PLAY = 10
PROB_FOR_RANDOM_MOVE_SELF_PLAY = 0.1

TIME_REDUCTION = 1.0

GAMES_FOLDER = Path("resources/games/")
MODELS_FOLDER = Path("resources/models/")

BUFFER_SIZE = 15000
HISTORY_SIZE = 1
SAMPLE_SIZE = 20000
LEARNING_RATE = 5e-4
TIME_REDUCTION = 1.0
TIME_REDUCTION_EVALUATION = 1.5
Z_STAT_SIGNIFICANT = 1.5
DEPTH_FOR_RANDOM_GAMES_FOR_SELF_PLAY = 10
PROB_FOR_RANDOM_MOVE_SELF_PLAY = 0.1
DEPTH_FOR_RANDOM_GAMES_FOR_EVALUATION = 5
PROB_FOR_RANDOM_MOVE_EVALUATION = 0.3
NR_OF_THREADS_FOR_SELF_PLAY = 20
BATCH_SIZE = 32

NR_OF_ITERATIONS = 200

config = get_config()

pplayer = ParallelPlayer(
    NR_OF_THREADS_FOR_SELF_PLAY,
    HISTORY_SIZE,
    DEPTH_FOR_RANDOM_GAMES_FOR_SELF_PLAY,
    PROB_FOR_RANDOM_MOVE_SELF_PLAY,
    1.0,
    GAMES_FOLDER,
    MODELS_FOLDER,
)

optimizer = Optimizer(LEARNING_RATE, BATCH_SIZE, GAMES_FOLDER)

evaluator = Evaluator(
    TIME_REDUCTION_EVALUATION,
    DEPTH_FOR_RANDOM_GAMES_FOR_EVALUATION,
    PROB_FOR_RANDOM_MOVE_EVALUATION,
    GAMES_FOLDER,
    MODELS_FOLDER,
)

# create model if not exist
# optimizer.create_initial_model()

pplayer.parallel_self_play(NR_OF_ITERATIONS)

optimizer.optimize(True)

evaluator.evaluate()
