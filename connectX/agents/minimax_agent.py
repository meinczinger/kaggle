from agents.bitboard import BitBoard
from search import MiniMax


class MinimaxAgent:
    _agent = None

    def __init__(self, configuration, depth):
        self.columns = configuration.columns
        self.rows = configuration.rows
        self.inarow = configuration.inarow
        self._depth = depth
        self._search_engine = MiniMax()

    def act(self, observation):
        """ Main method to act on opponents move """
        board = observation.board

        own_player = observation.mark

        state = BitBoard.create_from_board(self.columns, self.rows, self.inarow, own_player, board)

        action = self._search_engine.search(state, own_player, self._depth)

        return action

    @staticmethod
    def get_instance(configuration):
        if MinimaxAgent._agent is None:
            MinimaxAgent._agent = MinimaxAgent(configuration, 3)
        return MinimaxAgent._agent


