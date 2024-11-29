from game.board.bitboard import BitBoard
from collections import defaultdict


class GameTree:
    def __init__(self, configuration, board, own_player=1):
        """
        Representing the tree for Monte carlo tree search
        :param configuration: configuration of the game
        :param board: the board acting as the root of the search
        :param own_player: with which layer we are playing with
        """
        self._config = configuration
        self._config = configuration
        # The tree is a dictionary
        self._tree = defaultdict()
        # Nodes are identified by IDs, _last_used_node_id is the last used id
        self._last_used_node_id = -1
        # Create dictionaty to look up node by board
        self._node_lookup = defaultdict()
        # Create the root node
        self._root = self.add_node(
            False,
            0,
            BitBoard.create_from_board(
                self._config.columns,
                self._config.rows,
                self._config.inarow,
                own_player,
                board,
            ),
            own_player,
        )

    """ Add a node to the tree """

    def add_node(self, leaf, action, board, player):
        self._last_used_node_id += 1
        self._tree[self._last_used_node_id] = {
            "action": action,
            "nr_of_visits": 0,
            "value": 0,
            "prior": 0,
            "leaf": leaf,
            "parent": None,
            "children": [],
            "bitboard": board.hash(),
            "board_list": board.bitboard_to_numpy(),
            "board": board,
            "player": player,
            "prior_set": False,
            "value_set": False,
        }
        # Add the new node to the lookup table
        self._node_lookup[self.bitboard(self._last_used_node_id)] = (
            self._last_used_node_id
        )
        return self._last_used_node_id

    def node_from_board(self, board, player):
        bitboard = BitBoard.create_from_board(
            self._config.columns, self._config.rows, self._config.inarow, player, board
        )
        try:
            return self._node_lookup[bitboard.hash()]
        except:
            return -1

    """ The node contains the bitboards, create a BitBoard class from these """

    def board(self, node):
        # return BitBoard.create_from_bitboard(self._config.columns, self._config.rows, self._config.inarow,
        #                                      self.player(node), [self.bitboard(node)[0], self.bitboard(node)[1]])
        return self._tree[node]["board"]

    """ Returns the bitboards for a node """

    def bitboard(self, node):
        return self._tree[node]["bitboard"]

    def board_list(self, node):
        return self._tree[node]["board_list"]

    """ Returns the root node """

    def root(self):
        return self._root

    """ Set the root node """

    def set_root(self, node):
        self._root = node
        # And cut it's parent
        self._tree[node]["parent"] = None

    """ Get the player of the node """

    def player(self, node):
        return self._tree[node]["player"]

    """ Get the action which the brought the previous node to the current one """

    def action(self, node) -> int:
        return self._tree[node]["action"]

    """ Is this a leaf node? """

    def leaf(self, node):
        return self._tree[node]["leaf"]

    """ Get the node's parent """

    def parent(self, node):
        return self._tree[node]["parent"]

    """ The number of times a node was visited """

    def visited(self, node):
        return self._tree[node]["nr_of_visits"]

    """ Increase the number of times visited """

    def inc_visited(self, node):
        self._tree[node]["nr_of_visits"] += 1

    """ Increase the value of a node """

    def inc_value(self, node, value):
        self._tree[node]["value"] += value

    """ How often there was a win from this node """

    def value(self, node):
        return self._tree[node]["value"]

    """ Set the value of a node """

    def set_value(self, node, value):
        self._tree[node]["value"] = value

    def prior(self, node):
        return self._tree[node]["prior"]

    def set_prior(self, node, prior):
        self._tree[node]["prior"] = prior

    """ Returns the children of the node """

    def children(self, node):
        return self._tree[node]["children"]

    """ Set the node's parent """

    def set_parent(self, node, parent):
        self._tree[node]["parent"] = parent

    """ Add a child """

    def add_to_children(self, node, child):
        self._tree[node]["children"].append(child)

    """ Update the number of visits """

    def update_visited(self, node, value):
        self.inc_visited(node)
        self.inc_value(node, value)
