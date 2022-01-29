from typing import Any
from pathlib import Path
import pickle


class FilePersistent:
    @staticmethod
    def save_to(obj: Any, path: str) -> None:
        p = Path(path)
        with p.open('wb') as f:
            pickle.dump(obj, f)
    
    @staticmethod
    def load_from(path: str) -> Any:
        p = Path(path)
        with p.open('rb') as f:
            obj = pickle.load(f)
            return obj

