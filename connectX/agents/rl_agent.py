from agents.bitboard import BitBoard
from agents.rl_tabular.tabular_base import TabularBase


EPSILON = 0.05
REWARD_WIN = 1.0
REWARD_DRAW = 0.0
REWARD_LOOSE = -1.0
REWARD_STEP = 0.0
MARKS = [1, 2]


class RLAgent:
    def __init__(self, configuration, tabular_base: TabularBase):
        self.columns = configuration.columns
        self.rows = configuration.rows
        self.inarow = configuration.inarow
        self.actTimeout = configuration.actTimeout
        self.my_player = 1
        self._tabular_rl = tabular_base

    def act(self, observation, sim_board=None):
        # Set my player
        self.my_player = observation.mark

        # Create a state from the board
        if sim_board is None:
            board = observation.board
            state = BitBoard.create_from_board(
                self.columns, self.rows, self.inarow, self.my_player, board
            )
        else:
            state = sim_board

        # Get action according to policy
        action = self._tabular_rl.get_action(state)

        self._tabular_rl.post_action(state, action)

        print("Making action", action)
        return int(action)

    @staticmethod
    def get_instance(configuration, tabular_base: TabularBase):
        if RLAgent._agent is None:
            RLAgent._agent = RLAgent(configuration, tabular_base)
        return RLAgent._agent
