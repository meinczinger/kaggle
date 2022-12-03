from my_agents.mcts.base_mcts import BaseMonteCarloTreeSearch
from my_agents.rl_tabular.tabular_base import TabularBase
import numpy as np
from my_agents.simulator import Simulator


class TabularMonteCarloTreeSearch(BaseMonteCarloTreeSearch):
    def __init__(self, configuration, rl_tabular: TabularBase):
        super().__init__(configuration)
        self._explore_factor = 2.0
        self._rl_tabular = rl_tabular
        self._simulator = Simulator(configuration)

    def search(
        self,
        board: list,
        own_player: int,
        step: int,
        deadline: float,
        reuse: bool = False,
    ) -> int:
        action = super().search(board, own_player, step, deadline, reuse)
        node = self._tree.node_from_board(board, own_player)
        self._rl_tabular.post_action(self._tree.board(node), action)
        return action

    def initialize(self, board, own_player):
        super().initialize(board, own_player)
        self._rl_tabular._initialize()

    """ Get the child according UBC """

    def get_ucb_child(self, node, player, expand=True):
        if expand:
            factor = self._explore_factor
        else:
            # If we are not in expansion mode (hence in playing mode), ignore the explore factor
            factor = 0
        children = self._tree.children(node)

        if len(children) == 0:
            if not expand:
                self._logger.error("get_ubc_child - no child found")
            return None
        else:
            # The tree is populated from the perspective of the first player, if have this player, maximize the value,
            # otherwise minimize
            if player == 1:
                child = max(
                    children,
                    key=lambda c: self._tree.value(c) / self._tree.visited(c)
                    + factor
                    * np.sqrt(np.log(self._tree.visited(node)) / self._tree.visited(c))
                    if self._tree.visited(c) > 0
                    else float("inf"),
                )
            else:
                child = min(
                    children,
                    key=lambda c: self._tree.value(c) / self._tree.visited(c)
                    - factor
                    * np.sqrt(np.log(self._tree.visited(node)) / self._tree.visited(c))
                    if self._tree.visited(c) > 0
                    else -float("inf"),
                )
            return child

    def rollout(self, node):
        board = self._tree.board(node)
        value = self._rl_tabular.get_state_value(board)
        if value != 0:
            return value
        if self._tree.leaf(node):
            winner = board.last_player()
        else:
            winner = self._simulator.simulate(board, self._tree.player(node))
        # Tie
        if winner == 0:
            # Draw
            return 0
        if winner == 1:
            return 1
        else:
            return -1
