import unittest
from agents.monte_carlo_agent import StateAction


class TestStateAction(unittest.TestCase):
    def test_state_action_add(self):
        sa = StateAction(0.1, 7)
        sa.add("s1", 0, 0)
        sa.add("s1", 1, 1)
        sa.update()
        #self.assertEqual(1, sa.get_count("s1", 0))


if __name__ == '__main__':
    unittest.main()