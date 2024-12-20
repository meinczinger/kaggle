from mcts.base_mcts import BaseMonteCarloTreeSearch
import numpy as np
from collections import defaultdict
from game.board.bitboard import BitBoard
from model.cnn import Residual_CNN
import random
import logging

RANDOM_MOVE_PROB = 0.05
NUMBER_OF_MOVES_WITH_TEMPRETURE_ONE = 16


class NeuralNetworkMonteCarloTreeSearch(BaseMonteCarloTreeSearch):
    _nn_mcts = None

    def __init__(
        self,
        configuration,
        self_play=False,
        evaluation=False,
        use_best_player1=True,
        use_best_player2=True,
    ):
        super().__init__(configuration)
        self._self_play = self_play
        self._evaluation = evaluation
        self._priors_history = defaultdict()
        self._node_value_cache = defaultdict()
        self._explore_factor = 1.0
        if use_best_player1:
            value_model1 = "best_model"
        else:
            value_model1 = "candidate_model"
        if use_best_player2:
            value_model2 = "best_model"
        else:
            value_model2 = "candidate_model"

        self._model = [
            Residual_CNN(value_model1 + "_p1"),
            Residual_CNN(value_model2 + "_p2"),
        ]
        self._model[0].load()
        self._model[1].load()

        self._move = 0
        self._logger = logging.getLogger("agent")
        self._logger.setLevel(logging.DEBUG)
        self._logger.addHandler(logging.StreamHandler())

    def search(
        self,
        board: list,
        own_player: int,
        step: int,
        deadline: float,
        reuse: bool = False,
    ) -> int:
        action = super().search(board, own_player, step, deadline, reuse)

        if self._self_play:
            self._add_to_priors_history(board, step)
        return action

    def get_best_action(self, own_player):
        children = self._tree.children(self._tree.root())
        if len(children) == 0:
            return None
        else:
            sum_of_visits = sum([self._tree.visited(c) for c in children])
            probs = [self._tree.visited(c) / sum_of_visits for c in children]
            if self._self_play and (self._step < NUMBER_OF_MOVES_WITH_TEMPRETURE_ONE):
                child = children[np.random.choice(np.arange(len(children)), p=probs)]
            else:
                child = children[np.argmax(probs)]
        return self._tree.action(child)

    def _add_to_priors_history(self, board, step):
        children = self._tree.children(self._tree.root())
        sum_of_visits = sum([self._tree.visited(child) for child in children])
        priors = [0.0] * 7
        for child in children:
            priors[self._tree.action(child)] = self._tree.visited(child) / sum_of_visits
        self._priors_history[self._move] = {
            # player is the player who has played before this action
            "player": (self._tree.player(self._tree.root()) % 2) + 1,
            "step": step,
            "board": board,
            "priors": priors,
        }
        self._move = self._move + 1

    def priors(self):
        return self._priors_history

    def _leaf_value(self, node):
        board = self._tree.board(node)
        # winner = board.last_player()
        draw = board.is_end_state() and board.is_draw()
        # return 0 if draw else -1 if winner == 0 else 1
        return 0 if draw else 1

    def get_priors(self):
        return self._priors_history

    def initialize(self, board, own_player):
        super().initialize(board, own_player)
        self._priors_history = defaultdict()
        self._move = 0
        self._node_value_cache.clear()

    """ Get the child according UBC """

    def get_ucb_child(self, node, player, explore: bool = False):
        children = self._tree.children(node)
        if len(children) == 0:
            return None
        else:
            if explore:
                explore_factor = self._explore_factor
            else:
                explore_factor = 1.0
            # The tree is populated from the perspective of the first player, if have this player, maximize the value,
            # otherwise minimize
            if self._self_play and (node == self._tree.root()):
                noise = np.random.dirichlet([2] * 7)
                ucb = [
                    (
                        self._tree.value(c) / self._tree.visited(c)
                        if self._tree.visited(c) > 0
                        else 0.0
                    )
                    + explore_factor
                    * (0.75 * self._tree.prior(c) + 0.25 * noise[self._tree.action(c)])
                    * np.sqrt(self._tree.visited(node))
                    / (self._tree.visited(c) + 1)
                    for c in children
                ]
                child = np.argmax(ucb)
            else:
                child = np.argmax(
                    [
                        (
                            self._tree.value(c) / self._tree.visited(c)
                            if self._tree.visited(c) > 0
                            else 0.0
                        )
                        + explore_factor
                        * self._tree.prior(c)
                        * np.sqrt(self._tree.visited(node))
                        / (self._tree.visited(c) + 1)
                        for c in children
                    ]
                )
            return children[child]

    def build_tree(self, nr_of_iter):
        for i in range(nr_of_iter):
            node = self.descend()
            _ = self.expand(node)
            value = self.rollout(node)
            self.update(node, value)

            # node = self.descend()
            # expanded_node, value = self.expand(node)
            # self.update(expanded_node, value)

    def rollout(self, node):
        if self._tree.leaf(node):
            return self._leaf_value(node)

        children = self._tree.children(node)
        if len(children) > 0:
            # If the next player is 1 (2), use the second (first) model
            value_predictions, priors_prediction = self._model[
                self._tree.player(node) % 2
            ].predict([self._tree.board_list(node)])
            for child in children:
                if self._tree.leaf(child):
                    self._tree.set_prior(child, 1)
                else:
                    self._tree.set_prior(
                        child, priors_prediction[0][self._tree.action(child)]
                    )
        return 2 * value_predictions[0][0] - 1.0
        # value_predictions, priors_prediction = self._model[
        #     1 - (self._tree.player(node) % 2)
        # ].predict([self._tree.board_list(node)])
        # # Store the priors

        # return 2 * value_predictions[0][0] - 1.0

    def update(self, node, value):
        # Set the value of the node
        traverse_node = node
        factor = 1.0
        while traverse_node is not None:
            self._tree.update_visited(traverse_node, factor * value)
            factor *= -1.0
            traverse_node = self._tree.parent(traverse_node)

    @staticmethod
    def get_instance(configuration):
        if NeuralNetworkMonteCarloTreeSearch._nn_mcts is None:
            NeuralNetworkMonteCarloTreeSearch._nn_mcts = (
                NeuralNetworkMonteCarloTreeSearch(configuration)
            )
        return NeuralNetworkMonteCarloTreeSearch._nn_mcts
