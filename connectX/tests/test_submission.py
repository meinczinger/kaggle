import unittest
import main
from kaggle_environments.utils import Struct
import random


class TestAgent(unittest.TestCase):
    obs = Struct()
    obs.step = 1
    obs.lastOpponentAction = 1
    config = Struct()
    config.columns = 7
    config.rows = 7
    config.inarow = 4

    def test_submission_function_call(self):
        self.assertIn(main.act(TestAgent.obs, TestAgent.config), {0, 1, 2}, "Action must be 0, 1 or 2")

    def test_play(self):
        for i in range(1000):
            TestAgent.obs.step = i
            TestAgent.obs.lastOpponentAction = random.randint(0, TestAgent.config.signs - 1)
            self.assertIn(main.act(TestAgent.obs, TestAgent.config), {0, 1, 2},
                          "Action must be 0, 1 or 2")


if __name__ == '__main__':
    unittest.main()
