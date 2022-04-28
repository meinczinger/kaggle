import numpy as np
import copy
from agents.logger import Logger
import time
from agents.game_tree import GameTree


class BaseMonteCarloTreeSearch:
    def __init__(
        self,
        configuration,
        self_play=False,
        evaluation=False,
        use_best_player1=True,
        use_best_player2=True,
        exploration_phase=0,
    ):
        """
        Monte carlo tree search implementation
        :param configuration: the configuration of the game
        :param board: the current position, where we have to make an action
        :param own_player: which player we are playing with (1 or 2)
        """
        self._logger = Logger.logger("GameTreeControl")
        self._config = configuration
        # Exploration factor for UCB
        self._explore_factor = 1.0
        self._search_batch = 15
        self._tree = None

    def search(
        self,
        board: list,
        own_player: int,
        step: int,
        deadline: float,
        reuse: bool = False,
    ) -> int:
        """performs monte carlo tree search
        :param board: the board acting as the root of the search
        :param own_player: with which layer we are playing with
        :param step: number of step in the game
        :param deadline: by when the search need to finish
        :param reuse: whether the search tree should be reused within one episode.
        :return best action found
        """
        if reuse:
            current_node = self._tree.node_from_board(board, own_player)
            if current_node == -1:
                self.initialize(board, own_player)
            else:
                self._tree.set_current(current_node)
        else:
            self.initialize(board, own_player)
        self.extend_tree(2)
        # Keep extending the tree until we have time
        while time.time() < deadline:
            self.extend_tree(100)
        return self.get_best_action(own_player)

    def initialize(self, board, own_player):
        self._tree = GameTree(self._config, board, own_player)
        self._own_player = own_player

    """ Get the child according UBC """

    def get_ucb_child(self, node, player):
        raise NotImplementedError

    def get_best_action(self, own_player):
        child = self.get_ucb_child(self._tree.current(), own_player, False)
        return self._tree.action(child)

    def build_tree(self, nr_of_iter):
        for i in range(nr_of_iter):
            node = self.descend()
            if self._tree.visited(node) > 0:
                expanded_node = self.expand(node)
            else:
                expanded_node = node
            value = self.rollout(expanded_node)
            self.update(expanded_node, value)

    def extend_tree(self, depth=1):
        self.build_tree(depth)

    def descend(self):
        # If there is no child, echo back the current node
        node = self._tree.current()
        leaf = node
        depth = 0
        player = self._own_player
        while node is not None:
            node = self.get_ucb_child(node, player)
            player = (player % 2) + 1
            if node is not None:
                leaf = node
            depth += 1

        return leaf

    def expand(self, node):
        if self._tree.leaf(node):
            return node

        board = self._tree.board(node)
        actions = board.possible_actions()
        new_nodes = []
        for action in actions:
            new_board = copy.copy(board)
            new_board.make_action(action)
            new_node = self._tree.add_node(
                new_board.is_terminal_state(),
                action,
                new_board,
                new_board.active_player(),
            )
            new_nodes.append(new_node)
            self._tree.set_parent(new_node, node)
            self._tree.add_to_children(node, new_node)
        return np.random.choice(new_nodes)

    def rollout(self, node):
        raise NotImplementedError

    def update(self, node, value):
        traverse_node = node
        while traverse_node is not None:
            self._tree.update_visited(traverse_node, value)
            traverse_node = self._tree.parent(traverse_node)
