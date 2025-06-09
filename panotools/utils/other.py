from typing import *

K = TypeVar("K")
V = TypeVar("V")


def get_first(d: Dict[K, V], keys: List[K], default: V | None = None) -> V:
    for key in keys:
        if key in d:
            return d[key]
    return default
