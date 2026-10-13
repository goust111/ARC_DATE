from __future__ import annotations

import collections
import random

import numpy as np
from arc_utils import neighbors4


BG = 0
PIPE = 5


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate seed colors through a grey pipe network using shortest-path ownership.

    Input structure:
    - Black background with a one-cell-wide grey orthogonal pipe network.
    - Some pipe endpoints are replaced by colored seed pixels.
    - The network may branch or cross.

    Rule:
    - Treat grey pipe cells and colored endpoint seeds as one traversable graph.
    - Expand each seed color along the graph by shortest 4-connected path distance.
    - A grey cell takes the unique closest seed color; if two colors reach it at the same distance, it stays grey.
    - Non-pipe background remains black.

    Key insight:
    - The final color of a pipe cell is a graph-distance Voronoi result over the pipe network, not a flood by drawing order.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    traversable = (g == PIPE) | ((g != BG) & (g != PIPE))
    seeds = [(int(r), int(c), int(g[r, c])) for r, c in np.argwhere((g != BG) & (g != PIPE))]
    dist = np.full(g.shape, 10**9, dtype=int)
    owner = np.full(g.shape, -2, dtype=int)
    q = collections.deque()
    for r, c, color in seeds:
        dist[r, c] = 0
        owner[r, c] = color
        q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and traversable[nr, nc]:
                nd = dist[r, c] + 1
                if nd < dist[nr, nc]:
                    dist[nr, nc] = nd
                    owner[nr, nc] = owner[r, c]
                    q.append((nr, nc))
                elif nd == dist[nr, nc] and owner[nr, nc] != owner[r, c]:
                    owner[nr, nc] = -1
    pipe_cells = np.argwhere(g == PIPE)
    for r, c in pipe_cells:
        if owner[r, c] >= 0:
            out[r, c] = owner[r, c]
    return out


def generate_input() -> np.ndarray:
    h = random.randint(15, 25)
    w = random.randint(15, 25)
    g = np.zeros((h, w), dtype=int)
    hub_r = random.randint(4, h - 5)
    hub_c = random.randint(4, w - 5)
    g[hub_r, hub_c] = PIPE
    arms = random.sample(["up", "down", "left", "right"], random.randint(2, 4))
    endpoints = []
    for arm in arms:
        if arm == "up":
            r = random.randint(1, hub_r - 2)
            g[r : hub_r + 1, hub_c] = PIPE
            endpoints.append((r, hub_c))
        elif arm == "down":
            r = random.randint(hub_r + 2, h - 2)
            g[hub_r : r + 1, hub_c] = PIPE
            endpoints.append((r, hub_c))
        elif arm == "left":
            c = random.randint(1, hub_c - 2)
            g[hub_r, c : hub_c + 1] = PIPE
            endpoints.append((hub_r, c))
        else:
            c = random.randint(hub_c + 2, w - 2)
            g[hub_r, hub_c : c + 1] = PIPE
            endpoints.append((hub_r, c))
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], len(endpoints))
    for (r, c), color in zip(endpoints, colors, strict=False):
        g[r, c] = color
    return g
