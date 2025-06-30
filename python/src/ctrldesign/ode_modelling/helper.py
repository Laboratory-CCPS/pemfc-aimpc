from typing import Iterable, Optional


def find_element_idx(v: Iterable[str], signal: str) -> Optional[int]:
    for i, el in enumerate(v):
        if el == signal:
            return i

    return None
