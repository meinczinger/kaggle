import pandas as pd
from threading import Lock
from pathlib import Path


class GameManager:
    def __init__(self, folder: Path) -> None:
        self.folder = folder

    def save(self, df: pd.DataFrame, name: str, lock: Lock):
        with lock:
            df.to_csv(self.folder / name, index=False, header=False, mode="a")
