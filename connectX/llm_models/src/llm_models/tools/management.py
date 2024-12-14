import torch


class Singleton(object):
    def __new__(cls, *args, **kwargs):
        # see if the instance is already in existence. If not, make a new one.
        if not hasattr(cls, "_singleton_instance"):
            cls._singleton_instance = super(Singleton, cls).__new__(cls)
        return cls._singleton_instance


class ModelManagement:
    def __init__(self):
        if hasattr(self, "_instantiated"):
            return
        self._instantiated = True

    def load_model(
        self,
        model: torch.nn,
        model_path: str,
        device: str,
        optimizer: torch.optim = None,
    ):
        checkpoint = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return model, optimizer
