import logging
from model.nn_model import DQNNNModel
from bitboard import BitBoard


class DQNAgent:
    _agent = None

    def __init__(self, configuration, evaluate=False, simulation=False):
        self.logger = logging.getLogger('nn agent')
        self._config = configuration
        self._mcts = None
        self._evaluate = evaluate
        self._simulation = simulation
        self._model = DQNNNModel('dqn')
        self._model.load()

    def act(self, observation, explore=False):
        """ Main method to act on opponents move """
        board = observation.board

        # it seems sometimes the mark is incorrect so
        own_player = observation.mark

        bitboard = BitBoard.create_from_board(self._config.columns, self._config.rows, self._config.inarow, own_player,
                                              board)
        if own_player == 1:
            action, _ = self._model.predict_state(bitboard)
        else:
            action, _ = self._model.predict_state(bitboard, False)

        return int(action)

    def priors(self):
        return self._mcts.priors()

    @staticmethod
    def get_instance(configuration):
        if DQNAgent._agent is None:
            DQNAgent._agent = DQNAgent(configuration)
        return DQNAgent._agent


