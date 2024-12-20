from mcts.base_mcts import BaseMonteCarloTreeSearch
import numpy as np
from game.simulator import Simulator
from game.board.bitboard import BitBoard
from kaggle_environments.utils import Struct
import copy


class ClassicMonteCarloTreeSearch(BaseMonteCarloTreeSearch):
    _mcts = None

    def __init__(self, configuration):
        super().__init__(configuration)
        self._explore_factor = 2.0

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
                    key=lambda c: (
                        self._tree.value(c) / self._tree.visited(c)
                        + factor
                        * np.sqrt(
                            np.log(self._tree.visited(node)) / self._tree.visited(c)
                        )
                        if self._tree.visited(c) > 0
                        else float("inf")
                    ),
                )
            else:
                child = min(
                    children,
                    key=lambda c: (
                        self._tree.value(c) / self._tree.visited(c)
                        - factor
                        * np.sqrt(
                            np.log(self._tree.visited(node)) / self._tree.visited(c)
                        )
                        if self._tree.visited(c) > 0
                        else -float("inf")
                    ),
                )
            return child

    def rollout(self, node):
        board = self._tree.board(node)
        if self._tree.leaf(node):
            winner = board.last_player()
        else:
            winner = self.simulate(board, self._tree.player(node))
        # Tie
        if winner == 0:
            return 0
        if winner == 1:
            return 1
        else:
            return -1

    def simulate(self, board: BitBoard, to_play: int) -> int:
        """
        Runs a simulation from the given board
        :param board: the current board positionq
        :param to_play: the mark of the player who is next to play
        :return: 0 in case of a tie, 1 if the player wins, who plays next, -1 otherwise
        """
        obs = Struct()
        player = to_play - 1
        obs.mark = to_play
        bitboard = copy.copy(board)
        try:
            # Execute moves before we reach a terminal state
            while not bitboard.is_terminal_state():
                obs.board = bitboard.to_list()
                if self.agents[player] is None:
                    # Get a random move
                    action = np.random.choice(bitboard.possible_actions())
                else:
                    # Get a move given by the agent
                    action = self.agents[player].act(obs)
                # Make the move
                bitboard.make_action(action)

                # Swap players
                player = 1 - player
                obs.mark = (obs.mark % 2) + 1
            if bitboard.is_draw():
                return 0
            else:
                return bitboard.last_player()
        except Exception as ex:
            # self._logger._logger.error(ex, exc_info=True)
            pass

    @staticmethod
    def get_instance(configuration):
        if ClassicMonteCarloTreeSearch._mcts is None:
            ClassicMonteCarloTreeSearch._mcts = ClassicMonteCarloTreeSearch(
                configuration
            )
        return ClassicMonteCarloTreeSearch._mcts
