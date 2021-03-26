from mcts import MonteCarloTreeSearch
import logging
import time


class MCTSAgent:
    _agent = None

    def __init__(self, configuration):
        self.logger = logging.getLogger('mcts_agent')
        self.logger.setLevel(logging.ERROR)
        fh = logging.FileHandler('error.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        self._config = configuration
        self._mcts = None

    def act(self, observation):
        deadline = time.time() + self._config.timeout - 0.1
        """ Main method to act on opponents move """
        board = observation.board

        # it seems sometimes the mark is incorrect so
        own_player = observation.mark

        # Starting a new game
        if observation.step <= 1:
            self._mcts = MonteCarloTreeSearch(self._config, board, own_player)

        action = self._mcts.search(board, own_player, deadline, True)

        return int(action)

    @staticmethod
    def get_instance(configuration):
        if MCTSAgent._agent is None:
            MCTSAgent._agent = MCTSAgent(configuration)
        return MCTSAgent._agent


