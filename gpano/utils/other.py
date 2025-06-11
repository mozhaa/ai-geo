from collections import defaultdict
from typing import *


def get_first[K, V](d: Dict[K, V], keys: List[K], default: Optional[V] = None) -> V:
    for key in keys:
        if key in d:
            return d[key]
    return default


def safe_index(obj: Any, indices: List[int], raise_on_error: bool = False) -> Optional[Any]:
    pobj = obj
    for idx in indices:
        if not isinstance(pobj, list) or idx >= len(pobj):
            if raise_on_error:
                raise ValueError(f"invalid indices {indices} on {obj}")
            return None
        pobj = pobj[idx]
    return pobj


def batchedby[T](iterable: Iterable[T], key: Callable[[T], Any], n: int) -> Iterator[List[T]]:
    groups = defaultdict(list)
    for x in iter(iterable):
        k = key(x)
        groups[k].append(x)
        if len(groups[k]) == n:
            yield groups.pop(k)
    yield from groups.values()
