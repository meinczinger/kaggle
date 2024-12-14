from pathlib import Path

from game.simulator import Simulator
from game.config import get_config
from agent.mcts_agent import MCTSAgent
from game.board import BitBoard
from game import GameManager
from mcts.classic_mcts import ClassicMonteCarloTreeSearch
from mcts.llm_mcts import LlmMonteCarloTreeSearch


GAMES_FOLDER = Path("resources/games/")


class TestSimulator:
    def test_actions_saved(self):
        sim = Simulator(
            get_config(),
            GAMES_FOLDER,
            "",
            MCTSAgent(get_config(), ClassicMonteCarloTreeSearch(get_config())),
            MCTSAgent(get_config(), ClassicMonteCarloTreeSearch(get_config())),
        )
        sim.self_play(
            BitBoard.create_empty_board(7, 6, 4, 0),
            GameManager(GAMES_FOLDER),
            None,
            0,
            0,
            0,
        )

    def test_llm_mcts(self):
        sim = Simulator(
            get_config(),
            GAMES_FOLDER,
            "",
            MCTSAgent(
                get_config(),
                LlmMonteCarloTreeSearch(get_config(), True, False, True, True),
            ),
            MCTSAgent(
                get_config(),
                LlmMonteCarloTreeSearch(get_config(), True, False, True, True),
            ),
        )
        sim.self_play(
            BitBoard.create_empty_board(7, 6, 4, 0),
            GameManager(GAMES_FOLDER),
            None,
            0,
            0,
            0,
        )
