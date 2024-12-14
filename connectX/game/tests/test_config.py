from kaggle_environments.utils import Struct
from game.config import get_config


class TestConfig:
    def test_struct(self):
        config = get_config()
        assert config.columns == 7

