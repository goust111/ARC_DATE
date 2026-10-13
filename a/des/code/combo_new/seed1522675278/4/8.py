import random

import numpy as np


BLACK = 0
GREY = 5
YELLOW = 4
BULB = 9
SRCS = [1, 2, 3, 6, 7, 8]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each source color through the grey wire tree, split it at yellow junctions, and color reached bulbs by topmost-source priority.

    Input structure:
    - Grey orthogonal wires form an acyclic network.
    - Colored sources and maroon bulbs sit at wire endpoints.
    - Yellow nodes are three-way splitters.

    Rule:
    - Start from each source and traverse the wire network.
    - At a splitter, continue along every outgoing branch except the incoming edge.
    - Any reached bulb takes the color of the lexicographically earliest source among those that can reach it.

    Key insight:
    - The network is a tree, so signal propagation is a simple branch traversal with deterministic tie-breaking at bulbs.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    cells = {(r, c) for r in range(g.shape[0]) for c in range(g.shape[1]) if g[r, c] != BLACK}
    nbrs = {p: [(p[0] + dr, p[1] + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)) if (p[0] + dr, p[1] + dc) in cells] for p in cells}
    hits = {}
    sources = sorted([(r, c, int(g[r, c])) for r in range(g.shape[0]) for c in range(g.shape[1]) if g[r, c] not in (BLACK, GREY, YELLOW, BULB)], key=lambda x: (x[0], x[1]))
    for sr, sc, color in sources:
        start = next((n for n in nbrs[(sr, sc)] if g[n] in (GREY, YELLOW, BULB)), None)
        stack = [(start, (sr, sc))] if start else []
        seen = set()
        while stack:
            cur, prev = stack.pop()
            if (cur, prev) in seen:
                continue
            seen.add((cur, prev))
            if g[cur] == BULB:
                hits.setdefault(cur, (sr, sc, color))
                continue
            for nxt in nbrs[cur]:
                if nxt != prev:
                    stack.append((nxt, cur))
    for (r, c), (_, _, color) in hits.items():
        out[r, c] = color
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((18, 18), dtype=int)
    g[3, 3:9] = GREY
    g[3:10, 8] = GREY
    g[8, 8:15] = GREY
    g[3, 8] = YELLOW
    g[8, 8] = YELLOW
    g[3, 2] = random.choice(SRCS)
    g[10, 8] = random.choice([c for c in SRCS if c != g[3, 2]])
    g[8, 15] = BULB
    g[3, 9] = BULB
    return g
