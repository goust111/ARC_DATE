import random
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


BG = 0
GREY = 5
SOURCE_COLORS = (1, 2, 3, 4)
MIX_COLORS = (6, 7, 8, 9)
GEN_ATTEMPTS = 300


Cell = Tuple[int, int]


def _neighbors4(r: int, c: int, h: int, w: int) -> Iterable[Cell]:
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _parse_legend(row: np.ndarray) -> Dict[frozenset[int], int]:
    legend: Dict[frozenset[int], int] = {}
    c = 0
    while c + 2 < row.size:
        left, mid, right = int(row[c]), int(row[c + 1]), int(row[c + 2])
        if left != BG and mid != BG and right != BG:
            legend[frozenset((left, right))] = mid
            c += 3
        else:
            c += 1
    return legend


def _grey_graph(g: np.ndarray) -> Dict[Cell, List[Cell]]:
    h, w = g.shape
    nodes = set(map(tuple, np.argwhere(g[:-1, :] == GREY)))
    graph: Dict[Cell, List[Cell]] = {}
    for r, c in nodes:
        graph[(int(r), int(c))] = [
            (nr, nc)
            for nr, nc in _neighbors4(int(r), int(c), h - 1, w)
            if (nr, nc) in nodes
        ]
    return graph


def _source_endpoints(g: np.ndarray, graph: Dict[Cell, List[Cell]]) -> Dict[Cell, int]:
    h, w = g.shape
    used: set[Cell] = set()
    out: Dict[Cell, int] = {}
    for r in range(h - 2):
        for c in range(w - 1):
            cells = {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}
            if cells & used:
                continue
            color = int(g[r, c])
            if color in (BG, GREY):
                continue
            if all(int(g[rr, cc]) == color for rr, cc in cells):
                adjacent = set()
                for rr, cc in cells:
                    for nr, nc in _neighbors4(rr, cc, h - 1, w):
                        if (nr, nc) in graph and len(graph[(nr, nc)]) == 1:
                            adjacent.add((nr, nc))
                if len(adjacent) == 1:
                    out[adjacent.pop()] = color
                    used |= cells
    return out


def _mix(a: int, b: int, legend: Dict[frozenset[int], int]) -> int:
    if a == b:
        return a
    key = frozenset((a, b))
    assert key in legend, f"missing mix legend for {a}, {b}"
    return legend[key]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    out[-1, :] = BG

    legend = _parse_legend(g[-1, :])
    graph = _grey_graph(g)
    if not graph:
        return out

    sources = _source_endpoints(g, graph)
    endpoints = [cell for cell, nbrs in graph.items() if len(nbrs) == 1]
    roots = [cell for cell in endpoints if cell not in sources]
    root = max(roots or endpoints, key=lambda p: (p[0], p[1]))

    node_color: Dict[Cell, Optional[int]] = {}

    def dfs(cell: Cell, parent: Optional[Cell]) -> Optional[int]:
        colors: List[int] = []
        for nbr in sorted(graph[cell]):
            if nbr == parent:
                continue
            child_color = dfs(nbr, cell)
            if child_color is not None:
                colors.append(child_color)
        if cell in sources:
            colors.append(sources[cell])

        color: Optional[int] = None
        for item in colors:
            color = item if color is None else _mix(color, item, legend)
        node_color[cell] = color
        return color

    dfs(root, None)

    for (r, c), color in node_color.items():
        if color is not None:
            out[r, c] = color
    return out


def _place_source_block(grid: np.ndarray, endpoint: Cell, color: int, side: str) -> bool:
    r, c = endpoint
    if side == "top":
        cells = [(r - 2, c - 1), (r - 2, c), (r - 1, c - 1), (r - 1, c)]
    elif side == "left":
        cells = [(r - 1, c - 2), (r, c - 2), (r - 1, c - 1), (r, c - 1)]
    elif side == "right":
        cells = [(r - 1, c + 1), (r, c + 1), (r - 1, c + 2), (r, c + 2)]
    else:
        return False
    h, w = grid.shape
    if any(rr < 0 or cc < 0 or rr >= h - 1 or cc >= w for rr, cc in cells):
        return False
    if any(grid[rr, cc] != BG for rr, cc in cells):
        return False
    for rr, cc in cells:
        grid[rr, cc] = color
    return True


def _gen_attempt() -> Optional[np.ndarray]:
    h = random.randint(18, 26)
    w = random.randint(18, 26)
    trunk_c = random.randint(7, w - 8)
    top_r = 2
    bottom_r = h - 3

    branch_count = random.randint(2, 3)
    possible_rows = list(range(top_r + 3, bottom_r - 2, 3))
    if len(possible_rows) < branch_count:
        return None
    merge_rows = sorted(random.sample(possible_rows, branch_count))

    colors = random.sample(SOURCE_COLORS, branch_count + 1)
    mix_pool = list(MIX_COLORS)
    random.shuffle(mix_pool)
    if len(mix_pool) < branch_count:
        return None

    grid = np.full((h, w), BG, dtype=int)
    grid[top_r : bottom_r + 1, trunk_c] = GREY
    if not _place_source_block(grid, (top_r, trunk_c), colors[0], "top"):
        return None

    current = colors[0]
    legend_triples: List[Tuple[int, int, int]] = []

    for idx, row in enumerate(merge_rows):
        side = random.choice(("left", "right"))
        length = random.randint(4, 6)
        if side == "left":
            ep_c = trunk_c - length
            if ep_c < 2:
                return None
            grid[row, ep_c : trunk_c + 1] = GREY
            if not _place_source_block(grid, (row, ep_c), colors[idx + 1], "left"):
                return None
        else:
            ep_c = trunk_c + length
            if ep_c > w - 3:
                return None
            grid[row, trunk_c : ep_c + 1] = GREY
            if not _place_source_block(grid, (row, ep_c), colors[idx + 1], "right"):
                return None

        mixed = mix_pool[idx]
        legend_triples.append((current, mixed, colors[idx + 1]))
        current = mixed

    col = 0
    for a, m, b in legend_triples:
        if col + 2 >= w:
            return None
        grid[-1, col : col + 3] = [a, m, b]
        col += 4

    try:
        solved = main(grid)
    except Exception:
        return None
    if np.array_equal(solved, grid):
        return None
    if np.any(solved[:-1, :] == GREY):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe-mixing input")
