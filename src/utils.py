from typing import List


def symmetric_filters(known_positions: List, n_filters: int):
    kc = [known_positions for _ in range(n_filters)]
    return kc