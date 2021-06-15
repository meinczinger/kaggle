from base_mcts import BaseMonteCarloTreeSearch
import numpy as np
from collections import defaultdict
from bitboard import BitBoard
from model.nn_model import StateValueNNModel, PriorsNNModel


class NeuralNetworkMonteCarloTreeSearch(BaseMonteCarloTreeSearch):
    def __init__(self, configuration, self_play=False, use_best_player1=True, use_best_player2=True,
                 exploration_phase=0):
        super().__init__(configuration)
        self._self_play = self_play
        self._exploration_phase = exploration_phase
        self._priors_history = defaultdict()
        self._explore_factor = 1.0
        if use_best_player1:
            value_model1 = 'best_state_value_model'
            prior_model1 = 'best_priors_model'
        else:
            value_model1 = 'candidate_state_value_model'
            prior_model1 = 'candidate_priors_model'
        if use_best_player2:
            value_model2 = 'best_state_value_model'
            prior_model2 = 'best_priors_model'
        else:
            value_model2 = 'candidate_state_value_model'
            prior_model2 = 'candidate_priors_model'

        self._state_value_model = [StateValueNNModel(value_model1 + '_p1'), StateValueNNModel(value_model2 + '_p2')]
        self._state_value_model[0].load()
        self._state_value_model[1].load()

        self._priors_model = [PriorsNNModel(prior_model1 + '_p1'), PriorsNNModel(prior_model2 + '_p2')]
        self._priors_model[0].load()
        self._priors_model[1].load()
        self._move = 0

    def search(self, board: list, own_player: int, step: int, deadline: float, reuse: bool = False) -> int:
        action = super().search(board, own_player, step, deadline, reuse)

        if self._self_play:
            self._add_to_priors_history(board, step)
        return action

    def get_best_action(self, own_player, step):
        children = self._tree.children(self._tree.current())
        if len(children) == 0:
            return None
        else:
            sum_of_visits = sum([self._tree.visited(c) for c in children])
            probs = [self._tree.visited(c) / sum_of_visits for c in children]
            if step < self._exploration_phase:
                child = np.random.choice(children, 1, p=probs)[0]
            else:
                child = children[np.argmax(probs)]
        return self._tree.action(child)

    def _add_to_priors_history(self, board, step):
        children = self._tree.children(self._tree.current())
        sum_of_visits = sum([self._tree.visited(child) for child in children])
        priors = [0.] * 7
        for child in children:
            priors[self._tree.action(child)] = self._tree.visited(child) / sum_of_visits
        self._priors_history[self._move] = \
            {'player': self._tree.player(self._tree.current()), 'step': step, 'board': board, 'priors': priors}
        self._move = self._move + 1

    def priors(self):
        return self._priors_history

    def _leaf_value(self, node):
        board = self._tree.board(node)
        winner = board.last_player()
        return 0 if winner == 0 else 1 #if winner == 1 else -1

    def _get_value_predictions(self, node, player):
        # Get boards for all the children which just got expanded
        boards = []
        for child in self._tree.children(node):
            boards.append(BitBoard.from_bitboard_to_list(
                self._config.columns, self._config.rows, self._tree.bitboard(child)))
        # Get the predicted value for each state
        pred = self._state_value_model[player-1].predict(boards)
        pred = pred * 2 - 1.0
        return pred

    def _get_priors_probabilities(self, node, player):
        np_board = [BitBoard.from_bitboard_to_list(
            self._config.columns, self._config.rows, self._tree.bitboard(node))]
        return np.around(self._priors_model[player-1].predict(np_board)[0], decimals=4)

    def expand(self, node):
        expanded_node = super().expand(node)

        # Get the predicted priors for each state
        priors = self._get_priors_probabilities(node, self._tree.player(node))
        # Store the values in the child nodes
        children = self._tree.children(node)
        for child in children:
            self._tree.set_prior(child, priors[self._tree.action(child)])

        return expanded_node

    # def expand(self, node):
    #     expanded = super().expand(node)
    #
    #     if len(self._tree.children(node)) > 0:
    #         values = self._get_value_predictions(node, self._tree.player(node))
    #         # Store the values in the child nodes
    #         for child, value in zip(self._tree.children(node), values):
    #             if self._tree.leaf(child):
    #                 val = self._leaf_value(child)
    #                 # Leaf node found, select as expanded to get the reward
    #                 expanded = child
    #             else:
    #                 val = value[0]
    #             self._tree.update_visited(child, val)
    #
    #         # Get the predicted priors for each state
    #         priors = self._get_priors_probabilities(node, self._tree.player(node))
    #         # Store the values in the child nodes
    #         children = self._tree.children(node)
    #         for child in children:
    #             self._tree.set_prior(child, priors[self._tree.action(child)])
    #
    #     return expanded

    def get_priors(self):
        return self._priors_history

    def initialize(self, board, own_player):
        super().initialize(board, own_player)
        self._priors_history = defaultdict()
        self._move = 0

    """ Get the child according UBC """
    def get_ucb_child(self, node, player):
        children = self._tree.children(node)
        if len(children) == 0:
            return None
        else:
            # The tree is populated from the perspective of the first player, if have this player, maximize the value,
            # otherwise minimize
            # shuffled = copy.copy(children)
            # random.shuffle(shuffled)
            if self._self_play:
                noises = np.random.dirichlet([0.03] * 7)
                child = np.argmax([
                    (self._tree.value(c) / self._tree.visited(c) if self._tree.visited(c) > 0 else 0.) +
                    self._explore_factor * (0.75 * self._tree.prior(c) + 0.25 * noises[self._tree.action(c)]) *
                    np.sqrt(self._tree.visited(node)) / (self._tree.visited(c) + 1) for c in children])
            else:
                child = np.argmax([
                    (self._tree.value(c) / self._tree.visited(c) if self._tree.visited(c) > 0 else 0.) +
                    self._explore_factor * self._tree.prior(c) *
                    np.sqrt(self._tree.visited(node)) / (self._tree.visited(c) + 1) for c in children])
            return children[child]

    # def build_tree(self, iter, own_player):
    #     for i in range(iter):
    #         node = self.descend(own_player)
    #         self.expand(node)
    #         if self._tree.leaf(node):
    #             self.update(node, 1)
    #         else:
    #             for child in self._tree.children(node):
    #                 value = self.rollout(child)
    #                 self.update(child, value)

    def rollout(self, node):
        if self._tree.leaf(node):
            return self._leaf_value(node)

        # # Set prior based on predicted value
        # parent = self._tree.parent(node)
        # if parent is not None:
        #     np_board = [BitBoard.from_bitboard_to_list(
        #         self._config.columns, self._config.rows, self._tree.bitboard(parent))]
        #     prior_pred = self._priors_model[self._tree.player(node)-1].predict(np_board)[0]
        #     self._tree.set_prior(node, prior_pred[self._tree.action(node)])

        # Get the predicted state value
        np_board = [BitBoard.from_bitboard_to_list(
            self._config.columns, self._config.rows, self._tree.bitboard(node))]
        pred = self._state_value_model[(self._tree.player(node) % 2)].predict(np_board)[0][0]
        return pred * 2 - 1.0


    def update(self, node, value):
        # Set the value of the node
        #self._tree.update_visited(node, 0)
        traverse_node = node
        factor = 1.0
        while traverse_node is not None:
            self._tree.update_visited(traverse_node, factor * value)
            factor *= -1.0
            traverse_node = self._tree.parent(traverse_node)
