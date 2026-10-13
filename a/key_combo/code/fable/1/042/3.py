from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_ROOT = Path("/home/lixinyu/snap/new")
if str(_ARC_ROOT) not in sys.path:
    sys.path.append(str(_ARC_ROOT))

from solvers.arc_utils import Color, find_components, neighbors4


BG = int(Color.BLACK)
BAR = int(Color.BLUE)
RED = int(Color.RED)
GREY = int(Color.GREY)
GEN_RETRY_LIMIT = 400


def _bar_length(grid: np.ndarray) -> int:
    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=True, colors=[BAR])
    assert len(comps) == 1, "expected exactly one blue counter bar"
    comp = comps[0]
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h == 1 or w == 1, "counter bar must be straight"
    assert int(comp["area"]) == max(h, w), "counter bar must be 1-pixel wide"
    return int(comp["area"])


def _snake_path(grid: np.ndarray) -> list[tuple[int, int]]:
    comps = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=False,
        colors=[RED, GREY],
    )
    assert len(comps) == 1, "expected exactly one red/grey snake"
    coords = [tuple(map(int, rc)) for rc in comps[0]["coords"]]
    coord_set = set(coords)

    red_cells = [cell for cell in coords if int(grid[cell]) == RED]
    assert len(red_cells) == 1, "expected exactly one red endpoint"
    start = red_cells[0]

    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for cell in coords:
        nbrs = [nbr for nbr in neighbors4(*cell) if nbr in coord_set]
        adjacency[cell] = nbrs

    endpoints = [cell for cell, nbrs in adjacency.items() if len(nbrs) == 1]
    assert len(endpoints) == 2, "snake must be a simple path"
    assert start in endpoints, "red endpoint must be at one end of the snake"
    for cell, nbrs in adjacency.items():
        deg = len(nbrs)
        if cell == start:
            assert deg == 1
            assert int(grid[cell]) == RED
        else:
            assert int(grid[cell]) == GREY
            assert deg in {1, 2}, "snake cannot branch or self-intersect"

    path = [start]
    prev = None
    cur = start
    while True:
        options = [nbr for nbr in adjacency[cur] if nbr != prev]
        if not options:
            break
        assert len(options) == 1, "snake path order must be unique"
        nxt = options[0]
        path.append(nxt)
        prev, cur = cur, nxt

    assert len(path) == len(coords), "snake traversal must cover every snake pixel"
    return path


def _fall_distance(rest: np.ndarray, tail: list[tuple[int, int]]) -> int:
    height = int(rest.shape[0])
    delta = 0
    while True:
        for r, c in tail:
            nr = r + delta + 1
            if nr >= height or int(rest[nr, c]) != BG:
                return delta
        delta += 1


def _solve_with_info(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {BG, BAR, RED, GREY}, "unexpected colors"

    n = _bar_length(g)
    path = _snake_path(g)
    assert 2 <= n < len(path), "counter bar length must cut inside the snake"

    cut = path[n - 1]
    tail = path[n:]
    assert tail, "cut must leave a non-empty tail"

    rest = g.copy()
    rest[cut] = BG
    for r, c in tail:
        rest[r, c] = BG

    delta = _fall_distance(rest, tail)
    reason_bottom = any(r + delta + 1 >= g.shape[0] for r, c in tail)
    reason_blocker = any(
        r + delta + 1 < g.shape[0] and int(rest[r + delta + 1, c]) != BG for r, c in tail
    )

    out = rest.copy()
    for r, c in tail:
        out[r + delta, c] = int(g[r, c])

    info = {
        "n": n,
        "path": path,
        "cut": cut,
        "tail": tail,
        "delta": delta,
        "reason_bottom": reason_bottom,
        "reason_blocker": reason_blocker,
    }
    return out, info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cut the grey snake at the address encoded by the blue bar, then let the detached tail fall straight down.

    Input structure:
    - The background is black.
    - One foreground object is a 1-pixel-wide non-self-intersecting grey 4-connected snake with one endpoint recolored red.
    - A separate blue straight line segment sits in a corner and serves as a counter bar.

    Rule:
    - Let n be the blue bar length.
    - Starting from the red endpoint and following the snake order, the nth snake cell is the cut point.
    - Turn that cut cell to background, keep the red-headed front segment fixed, and treat the remaining tail segment as one rigid piece.
    - Drop the tail vertically until another one-step fall would leave the grid or collide with any remaining foreground pixel; the blue bar stays in place.

    Key insight:
    - The blue bar is an address, not a decoration: it chooses a single cut position on the snake, and only the detached tail participates in the gravity phase.
    """
    return _solve_with_info(grid)[0]


def _path_size(path: list[tuple[int, int]]) -> tuple[int, int]:
    max_r = max(r for r, _ in path)
    max_c = max(c for _, c in path)
    return max_r + 1, max_c + 1


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]], r0: int, c0: int) -> None:
    for idx, (r, c) in enumerate(path):
        grid[r0 + r, c0 + c] = RED if idx == 0 else GREY


def _mirror_lr(grid: np.ndarray) -> np.ndarray:
    return np.fliplr(grid).copy()


def _build_bottom_h() -> np.ndarray:
    n = random.randint(2, 6)
    seg1 = n + random.randint(1, 3)
    v1 = random.randint(2, 5)
    h2 = random.randint(2, 4)
    v2 = random.randint(0, 3)

    path = [(0, c) for c in range(seg1)]
    path.extend((r, seg1 - 1) for r in range(1, v1 + 1))
    path.extend((v1, c) for c in range(seg1, seg1 + h2))
    if v2:
        path.extend((r, seg1 + h2 - 1) for r in range(v1 + 1, v1 + v2 + 1))

    bar_orientation = random.choice(["h", "v"])
    if bar_orientation == "h":
        r0 = random.randint(2, 4)
        c0 = n + random.randint(2, 5)
    else:
        r0 = n + random.randint(2, 4)
        c0 = random.randint(2, 5)

    path_h, path_w = _path_size(path)
    fall = random.randint(2, 6)
    H = r0 + path_h + fall
    W = max(c0 + path_w + random.randint(2, 4), n + 2)

    grid = np.full((H, W), BG, dtype=int)
    if bar_orientation == "h":
        grid[0, :n] = BAR
    else:
        grid[:n, 0] = BAR
    _paint_path(grid, path, r0, c0)
    if random.random() < 0.5:
        grid = _mirror_lr(grid)
    return grid


def _build_bottom_v() -> np.ndarray:
    n = random.randint(2, 6)
    seg1 = n + random.randint(1, 3)
    h1 = random.randint(2, 4)
    v2 = random.randint(1, 3)
    h2 = random.randint(0, 3)

    path = [(r, 0) for r in range(seg1)]
    path.extend((seg1 - 1, c) for c in range(1, h1 + 1))
    path.extend((r, h1) for r in range(seg1, seg1 + v2))
    if h2:
        path.extend((seg1 + v2 - 1, c) for c in range(h1 + 1, h1 + h2 + 1))

    bar_orientation = random.choice(["h", "v"])
    if bar_orientation == "h":
        r0 = random.randint(2, 4)
        c0 = n + random.randint(2, 5)
    else:
        r0 = n + random.randint(2, 4)
        c0 = random.randint(2, 5)

    path_h, path_w = _path_size(path)
    fall = random.randint(2, 6)
    H = r0 + path_h + fall
    W = max(c0 + path_w + random.randint(2, 4), n + 2)

    grid = np.full((H, W), BG, dtype=int)
    if bar_orientation == "h":
        grid[0, :n] = BAR
    else:
        grid[:n, 0] = BAR
    _paint_path(grid, path, r0, c0)
    if random.random() < 0.5:
        grid = _mirror_lr(grid)
    return grid


def _build_bar_support() -> np.ndarray:
    n = random.randint(2, 6)
    v1 = random.randint(2, 5)
    top_pad = random.randint(1, 4)
    fall = random.randint(2, 6)
    right_pad = random.randint(3, 8)

    path = [(0, c) for c in range(1, n + 1)]
    path.extend((r, n) for r in range(1, v1 + 1))
    path.extend((v1, c) for c in range(n - 1, -1, -1))

    path_h, path_w = _path_size(path)
    H = top_pad + path_h + fall + 1
    W = path_w + right_pad

    grid = np.full((H, W), BG, dtype=int)
    grid[H - 1, :n] = BAR
    _paint_path(grid, path, top_pad, 0)
    if random.random() < 0.5:
        grid = _mirror_lr(grid)
    return grid


def _validate_generated(grid: np.ndarray, expected: str) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    colors = set(map(int, np.unique(grid)))
    if not colors <= {BG, BAR, RED, GREY}:
        return False

    out, info = _solve_with_info(grid)
    if out.shape != grid.shape or info["delta"] <= 0:
        return False

    if expected == "bottom":
        if not info["reason_bottom"] or info["reason_blocker"]:
            return False
    elif expected == "blocker":
        if not info["reason_blocker"] or info["reason_bottom"]:
            return False
    else:
        return False

    blue_cells = np.argwhere(grid == BAR)
    if blue_cells.size == 0:
        return False
    rs = blue_cells[:, 0]
    cs = blue_cells[:, 1]
    touches_top = int(rs.min()) == 0
    touches_bottom = int(rs.max()) == grid.shape[0] - 1
    touches_left = int(cs.min()) == 0
    touches_right = int(cs.max()) == grid.shape[1] - 1
    if not ((touches_top or touches_bottom) and (touches_left or touches_right)):
        return False

    snake_plus_red = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=False,
        colors=[RED, GREY],
    )
    blue_bar = find_components(grid, bg=BG, connectivity=4, monochromatic=True, colors=[BAR])
    return len(snake_plus_red) == 1 and len(blue_bar) == 1


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        roll = random.random()
        if roll < 0.25:
            grid = _build_bottom_h()
            expected = "bottom"
        elif roll < 0.50:
            grid = _build_bottom_v()
            expected = "bottom"
        else:
            grid = _build_bar_support()
            expected = "blocker"

        if _validate_generated(grid, expected):
            return grid
    raise RuntimeError("failed to generate a valid input")
