import unittest
import submission
from kaggle_environments.utils import Struct


class TestAgent(unittest.TestCase):
    def test_submission_function_call(self):
        obs = Struct()
        obs.step = 1
        obs.lastOpponentAction = 1
        config = Struct()
        config.episodeSteps = 10
        config.agentTimeout = 60
        config.actTimeout = 1
        config.runTimeout = 1200
        config.isProduction = False
        config.signs = 3

        self.assertIn(submission.AgentFactory.get_agent('Random', config).run(obs), {0, 1, 2}, "Action must be 0, 1 or 2")


if __name__ == '__main__':
    unittest.main()