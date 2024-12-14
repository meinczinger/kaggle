import yaml

class ModelParameters:
    def __init__(self, file_path: str):
        self.params = self._load_params(file_path)

    def _load_params(self, file_path: str) -> dict:
        config = {}
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)

        return config

        