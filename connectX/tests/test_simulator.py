import unittest
from bitboard import BitBoard
from simulator import Simulator
import copy
from kaggle_environments.utils import Struct


class TestSimulator(unittest.TestCase):
    def test_simulation(self):
        board = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0]
        config = Struct()
        config.columns = 7
        config.rows = 7
        config.inarow = 4
        winner = Simulator.simulate(config, board, 1)
        self.assertGreaterEqual(2, winner)
        self.assertLessEqual(0, winner)


if __name__ == '__main__':
    unittest.main()