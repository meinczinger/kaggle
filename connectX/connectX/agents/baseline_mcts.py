from bitboard import BitBoard
import numpy as np
import copy
from simulator import Simulator
from collections import defaultdict
from logger import Logger
import time


class GameTree:
    def __init__(self, configuration, board, own_player=1):
        """
        Representing the tree for Monte carlo tree search
        :param configuration: configuration of the game
        :param board: the board acting as the root of the search
        :param own_player: with which layer we are playing with
        """
        self._logger = Logger.logger('GameTree')
        self._config = configuration
        # The tree is a dictionary
        self._tree = defaultdict()
        # Nodes are identified by IDs, _last_used_node_id is the last used id
        self._last_used_node_id = -1
        # Exploration factor for UCB
        self._explore_factor = 2.0
        # Create dictionaty to look up node by board
        self._node_lookup = defaultdict()
        # Create the root node
        self._current = \
            self.add_node(False, 0,
                          BitBoard.create_from_board(self._config.columns, self._config.rows, self._config.inarow,
                                                     own_player, board), own_player)

    """ Add a node to the tree """
    def add_node(self, leaf, action, board, player):
        self._last_used_node_id += 1
        self._tree[self._last_used_node_id] = \
            {'action': action, 'nr_of_visits': 0, 'value': 0,
             'leaf': leaf, 'parent': None, 'children': [], 'bitboard': board.hash(), 'player': player}
        # Add the new node to the lookup table
        self._node_lookup[self.bitboard(self._last_used_node_id)] = self._last_used_node_id
        return self._last_used_node_id

    def node_from_board(self, board, player):
        bitboard = \
            BitBoard.create_from_board(self._config.columns, self._config.rows, self._config.inarow, player, board)
        try:
            return self._node_lookup[bitboard.hash()]
        except:
            return -1

    """ The node contains the bitboards, create a BitBoard class from these """
    def board(self, node):
        return BitBoard.create_from_bitboard(self._config.columns, self._config.rows, self._config.inarow,
                                             self.player(node), [self.bitboard(node)[0], self.bitboard(node)[1]])

    """ Returns the bitboards for a node """
    def bitboard(self, node):
        return self._tree[node]['bitboard']

    """ Returns the current node """
    def current(self):
        return self._current

    """ Set the current node """
    def set_current(self, node):
        self._current = node

    """ Get the player of the node """
    def player(self, node):
        return self._tree[node]['player']

    """ Get the action which the brought the previous node to the current one """
    def action(self, node):
        return self._tree[node]['action']

    """ Is this a leaf node? """
    def leaf(self, node):
        return self._tree[node]['leaf']

    """ Get the node's parent """
    def parent(self, node):
        return self._tree[node]['parent']

    """ The number of times a node was visited """
    def visited(self, node):
        return self._tree[node]['nr_of_visits']

    """ Increase the number of times visited """
    def inc_visited(self, node):
        self._tree[node]['nr_of_visits'] += 1

    """ Increase the value of a node """
    def inc_value(self, node, value):
        self._tree[node]['value'] += value

    """ How often there was a win from this node """
    def value(self, node):
        return self._tree[node]['value']

    """ Returns the children of the node """
    def children(self, node):
        return self._tree[node]['children']

    """ Set the node's parent """
    def set_parent(self, node, parent):
        self._tree[node]['parent'] = parent

    """ Add a child """
    def add_to_children(self, node, child):
        self._tree[node]['children'].append(child)

    """ Update the number of visits """
    def update_visited(self, node, value):
        self.inc_visited(node)
        self.inc_value(node, value)

    """ Get the child according UBC """
    def get_ubc_child(self, node, player, expand=True):
        if expand:
            factor = self._explore_factor
        else:
            # If we are not in expansion mode (hence in playing mode), ignore the explore factor
            factor = 0
        children = self.children(node)

        if len(children) == 0:
            if not expand:
                self._logger.debug("get_ubc_child - no child found")
            return None
        else:
            # The tree is populated from the perspective of the first player, if have this player, maximize the value,
            # otherwise minimize
            if player == 1:
                child = max(children,
                            key=lambda c: self.value(c) / self.visited(c) + factor *
                                          np.sqrt(np.log(self.visited(node)) / self.visited(c))
                            if self.visited(c) > 0 else float("inf"))
            else:
                child = min(children,
                            key=lambda c: self.value(c) / self.visited(c) - factor *
                                          np.sqrt(np.log(self.visited(node)) / self.visited(c))
                            if self.visited(c) > 0 else -float("inf"))
            return child

    def get_best_action(self, player):
        child = self.get_ubc_child(self.current(), player, False)
        return self.action(child)


class MonteCarloTreeSearch:
    def __init__(self, configuration, board, own_player):
        """
        Monte carlo tree search implementation
        :param configuration: the configuration of the game
        :param board: the current position, where we have to make an action
        :param own_player: which player we are playing with (1 or 2)
        """
        self._logger = Logger.logger('GameTreeControl')
        self._config = configuration
        self._initialize(board, own_player)
        self._simulator = Simulator(configuration)

    def search(self, board: list, own_player: int, deadline: float, reuse: bool=False) -> int:
        """ performs monte carlo tree search
        :param board: the board acting as the root of the search
        :param own_player: with which layer we are playing with
        :param deadline: by when the search need to finish
        :param reuse: whether the search tree should be reused within one episode.
        :return best action found
        """
        if reuse:
            current_node = self._tree.node_from_board(board, own_player)
            if current_node == -1:
                self._initialize(board, own_player)
            else:
                self._tree.set_current(current_node)
        else:
            self._initialize(board, own_player)
        self.extend_tree(2)
        # Keep extending the tree until we have time
        while time.time() < deadline:
            self.extend_tree(100)
        return self.get_best_action()

    def _initialize(self, board, own_player):
        self._tree = GameTree(self._config, board, own_player)
        self._step_count = 0
        self._epoch_step_count = 0
        self._own_player = own_player

    def get_best_action(self):
        return self._tree.get_best_action(self._own_player)

    def build_tree(self, iter):
        for i in range(iter):
            node = self.descend()
            if self._tree.visited(node) > 0:
                expanded_node = self.expand(node)
            else:
                expanded_node = node
            value = self.simulate(expanded_node)
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
            node = self._tree.get_ubc_child(node, player)
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
            new_node = self._tree.add_node(new_board.is_terminal_state(), action, new_board,
                                           new_board.active_player())
            new_nodes.append(new_node)
            self._tree.set_parent(new_node, node)
            self._tree.add_to_children(node, new_node)
        final_node = np.random.choice(new_nodes)
        return final_node

    def simulate(self, node):
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

    def update(self, node, value):
        traverse_node = node
        while traverse_node is not None:
            self._tree.update_visited(traverse_node, value)
            traverse_node = self._tree.parent(traverse_node)

