from my_agents.mcts.base_mcts import BaseMonteCarloTreeSearch
import numpy as np
from my_agents.simulator import Simulator


class ClassicMonteCarloTreeSearch(BaseMonteCarloTreeSearch):
    def __init__(self, configuration):
        super().__init__(configuration)
        self._explore_factor = 2.0
        self._simulator = Simulator(configuration)

    """ Get the child according UBC """

    def get_ucb_child(self, node, player, explore: bool = False):
        if explore:
            factor = self._explore_factor
        else:
            # If we are not in expansion mode (hence in playing mode), ignore the explore factor
            factor = 1.0
        children = self._tree.children(node)

        if len(children) == 0:
            if not explore:
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
        if self._tree.leaf(node):
            winner = board.last_player()
        else:
            winner = self._simulator.simulate(board, self._tree.player(node))
        # Tie
        if winner == 0:
            return 0
        if winner == 1:
            return 1
        else:
            return -1
