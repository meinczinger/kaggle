# from dotenv import load_dotenv

# load_dotenv()

from game.board import BitBoard


from game import get_config

from kaggle_environments.utils import Struct


class TestMoves:
    def test_forced_move(self):
        from mcts import NeuralNetworkMonteCarloTreeSearch
        from agent.mcts_agent import MCTSAgent

        board = BitBoard.create_empty_board(7, 6, 4, 0)
        board.make_action(6)
        board.make_action(2)
        board.make_action(6)
        board.make_action(5)
        board.make_action(6)

        agent = MCTSAgent.get_instance(
            get_config(),
            NeuralNetworkMonteCarloTreeSearch.get_instance(get_config()),
            # NeuralNetworkMonteCarloTreeSearch(get_config(), False, False, True, True),
        )
        move = agent.act(Struct(**{"board": board.to_list(), "mark": 1}))

        assert move == 6
