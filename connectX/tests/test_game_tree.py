import unittest
from base_mcts import BaseMonteCarloTreeSearch
import copy
from kaggle_environments.utils import Struct


class TestGameTree(unittest.TestCase):
    def test_game_tree(self):
        config = Struct()
        config.columns = 7
        config.rows = 7
        config.inarow = 4
        tree = BaseMonteCarloTreeSearch(config)
        tree.build_tree()
        self.assertEqual(1, tree.get_best_action())


if __name__ == '__main__':
    unittest.main()