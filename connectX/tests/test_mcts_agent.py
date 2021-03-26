import unittest
from kaggle_environments.utils import Struct
from mcts_agent import MCTSAgent


class TestMCTSAgent(unittest.TestCase):
    config = Struct()
    config.columns = 7
    config.rows = 6
    config.inarow = 4
    config.timeout = 2.0

    def test_action1(self):
        obs = Struct()
        obs.board = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 2, 0, 0, 0, 2, 0, 1, 2, 0, 0, 0, 1, 0, 1, 2,
                     2, 0,
                     0, 1, 0, 2, 1, 2, 0]
        obs.mark = 1
        mcts = MCTSAgent(self.config)
        action = mcts.act(obs)
        self.assertEqual(3, action)

    def test_action2(self):
        obs = Struct()
        obs.board = [0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 1, 0,
                     0, 1, 0, 0, 1, 2, 0,
                     0, 1, 2, 1, 2, 2, 2,
                     2, 1, 2, 2, 1, 1, 1]
        obs.mark = 2
        mcts = MCTSAgent(self.config)
        action = mcts.act(obs)
        self.assertEqual(1, action)

    def test_action3(self):
        obs = Struct()
        obs.board = [0, 0, 0, 0, 0, 0, 0,
                     0, 0, 1, 0, 0, 0, 0,
                     0, 0, 2, 2, 0, 0, 0,
                     0, 0, 2, 1, 0, 0, 0,
                     0, 2, 1, 2, 1, 0, 0,
                     1, 2, 1, 1, 2, 0, 0]
        obs.mark = 1
        mcts = MCTSAgent(self.config)
        action = mcts.act(obs)
        self.assertNotEqual(1, action)

    def test_action4(self):
        obs = Struct()
        obs.board = [0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 1, 0, 0, 0,
                     0, 0, 0, 1, 1, 0, 0,
                     2, 0, 0, 2, 1, 0, 0,
                     2, 2, 1, 2, 1, 0, 0]
        obs.mark = 2
        mcts = MCTSAgent(self.config)
        action = mcts.act(obs)
        self.assertEqual(4, action)


if __name__ == '__main__':
    unittest.main()