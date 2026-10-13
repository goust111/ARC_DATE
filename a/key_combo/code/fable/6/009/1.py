from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    arc_utils_dir = str(_ARC_UTILS_DIR)
    if arc_utils_dir not in sys.path:
        sys.path.append(arc_utils_dir)

try:
    from arc_utils import Color, in_bounds, neighbors4
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        YELLOW = 4
        GREY = 5

    def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
        h, w = shape
        return 0 <= r < h and 0 <= c < w

    def neighbors4(r: int, c: int):
        yield r - 1, c
        yield r + 1, c
        yield r, c - 1
        yield r, c + 1


BG = Color.BLACK
BLUE = Color.BLUE
RED = Color.RED
YELLOW = Color.YELLOW
WALL = Color.GREY

ACTUAL = "actual"
SWAPPED = "swapped"
SOLID = "solid"
OPEN = "open"

GEN_RETRIES = 300


def _single_pos(grid: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(grid == color)
    if coords.shape[0] != 1:
        raise ValueError(f"expected exactly one pixel of color {color}, found {coords.shape[0]}")
    return int(coords[0, 0]), int(coords[0, 1])


def _is_blocked(grid: np.ndarray, r: int, c: int, source: str, mode: str) -> bool:
    if int(grid[r, c]) != WALL:
        return False
    if mode == SOLID:
        return True
    if mode == OPEN:
        return False
    parity = (r + c) & 1
    if mode == ACTUAL:
        blocked_parity = 0 if source == "red" else 1
    elif mode == SWAPPED:
        blocked_parity = 1 if source == "red" else 0
    else:
        raise ValueError(f"unknown mode: {mode}")
    return parity == blocked_parity


def _flood_reach(grid: np.ndarray, start: tuple[int, int], source: str, mode: str) -> np.ndarray:
    h, w = grid.shape
    reach = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque([start])
    reach[start] = True
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if not in_bounds((h, w), nr, nc):
                continue
            if reach[nr, nc] or _is_blocked(grid, nr, nc, source, mode):
                continue
            reach[nr, nc] = True
            q.append((nr, nc))
    return reach


def _simulate_output(grid: np.ndarray, mode: str = ACTUAL) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    red_pos = _single_pos(grid, RED)
    blue_pos = _single_pos(grid, BLUE)
    red_reach = _flood_reach(grid, red_pos, "red", mode)
    blue_reach = _flood_reach(grid, blue_pos, "blue", mode)

    out = np.asarray(grid, dtype=int).copy()
    empty = out == BG
    out[empty & red_reach & ~blue_reach] = RED
    out[empty & blue_reach & ~red_reach] = BLUE
    out[empty & red_reach & blue_reach] = YELLOW
    return out, red_reach, blue_reach


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood-fill from the two opposite-corner markers, treating each gray wall parity as opaque to only one source.

    Input structure:
    - Black is the background and gray pixels form wall segments.
    - There is exactly one red marker in a corner and one blue marker in the opposite corner.
    - All other non-marker, non-wall cells are black empty cells.

    Rule:
    - For a gray wall cell, even parity `(row + col)` blocks the red flood but is transparent to the blue flood.
    - Odd parity gray wall cells do the opposite: they block the blue flood but are transparent to the red flood.
    - Run separate 4-connected floods from the red and blue corner markers through every cell that is not blocked for that source.
    - Recolor each black cell by reachability: red if only red reaches it, blue if only blue reaches it, yellow if both reach it, and black if neither reaches it. Keep all walls gray and both corner markers unchanged.

    Key insight:
    - The single visible wall network is really two interleaved barrier systems indexed by each wall cell's own coordinate parity, so the answer comes from comparing two complementary reach maps rather than from a standard single flood-fill.
    """
    out, _red_reach, _blue_reach = _simulate_output(np.asarray(grid, dtype=int), ACTUAL)
    return out


def _draw_pod(
    grid: np.ndarray,
    arena: tuple[int, int, int, int],
    side: str,
    outer_h: int,
    outer_w: int,
    corridor_len: int,
    category: str,
    rng: random.Random,
) -> dict | None:
    ar0, ac0, ar1, ac1 = arena
    h, w = grid.shape

    if side == "top":
        if outer_w > (ac1 - ac0 + 1) or outer_h + corridor_len > ar0:
            return None
        c0 = rng.randint(ac0, ac1 - outer_w + 1)
        r0 = ar0 - corridor_len - outer_h
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        door_c = rng.randint(c0 + 2, c1 - 2)
        tunnel = [(r, door_c) for r in range(r1 + 1, ar0)]
        slab = (r1 + 1, door_c - 2, ar0 - 1, door_c + 2)
        carve = [(r, door_c) for r in range(r1 - 1, r1 + 1)]
        interior = (r0 + 2, c0 + 2, r1 - 2, c1 - 2)
    elif side == "bottom":
        if outer_w > (ac1 - ac0 + 1) or outer_h + corridor_len > (h - 1 - ar1):
            return None
        c0 = rng.randint(ac0, ac1 - outer_w + 1)
        r0 = ar1 + corridor_len + 1
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        door_c = rng.randint(c0 + 2, c1 - 2)
        tunnel = [(r, door_c) for r in range(ar1 + 1, r0)]
        slab = (ar1 + 1, door_c - 2, r0 - 1, door_c + 2)
        carve = [(r, door_c) for r in range(r0, r0 + 2)]
        interior = (r0 + 2, c0 + 2, r1 - 2, c1 - 2)
    elif side == "left":
        if outer_h > (ar1 - ar0 + 1) or outer_w + corridor_len > ac0:
            return None
        r0 = rng.randint(ar0, ar1 - outer_h + 1)
        c0 = ac0 - corridor_len - outer_w
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        door_r = rng.randint(r0 + 2, r1 - 2)
        tunnel = [(door_r, c) for c in range(c1 + 1, ac0)]
        slab = (door_r - 2, c1 + 1, door_r + 2, ac0 - 1)
        carve = [(door_r, c) for c in range(c1 - 1, c1 + 1)]
        interior = (r0 + 2, c0 + 2, r1 - 2, c1 - 2)
    elif side == "right":
        if outer_h > (ar1 - ar0 + 1) or outer_w + corridor_len > (w - 1 - ac1):
            return None
        r0 = rng.randint(ar0, ar1 - outer_h + 1)
        c0 = ac1 + corridor_len + 1
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        door_r = rng.randint(r0 + 2, r1 - 2)
        tunnel = [(door_r, c) for c in range(ac1 + 1, c0)]
        slab = (door_r - 2, ac1 + 1, door_r + 2, c0 - 1)
        carve = [(door_r, c) for c in range(c0, c0 + 2)]
        interior = (r0 + 2, c0 + 2, r1 - 2, c1 - 2)
    else:
        raise ValueError(f"unknown side: {side}")

    ir0, ic0, ir1, ic1 = interior
    if ir0 > ir1 or ic0 > ic1:
        return None

    grid[r0 : r1 + 1, c0 : c1 + 1] = WALL
    grid[ir0 : ir1 + 1, ic0 : ic1 + 1] = BG

    sr0, sc0, sr1, sc1 = slab
    if sr0 <= sr1 and sc0 <= sc1:
        grid[sr0 : sr1 + 1, sc0 : sc1 + 1] = WALL

    for r, c in carve:
        grid[r, c] = BG
    for r, c in tunnel:
        grid[r, c] = BG

    if category == "both":
        gate_cells: list[tuple[int, int]] = []
    elif category in {"red_only", "blue_only"}:
        wanted_parity = 1 if category == "red_only" else 0
        candidates = [(r, c) for r, c in tunnel if ((r + c) & 1) == wanted_parity]
        if not candidates:
            return None
        gate_cells = [rng.choice(candidates)]
    elif category == "none":
        if len(tunnel) < 2:
            return None
        pair_starts = [i for i in range(len(tunnel) - 1)]
        start = rng.choice(pair_starts)
        gate_cells = [tunnel[start], tunnel[start + 1]]
    else:
        raise ValueError(f"unknown category: {category}")

    for r, c in gate_cells:
        grid[r, c] = WALL

    return {
        "side": side,
        "category": category,
        "interior": interior,
        "tunnel": tunnel,
        "gate_cells": gate_cells,
        "outer_bbox": (r0, c0, r1, c1),
    }


def _expected_color(category: str) -> int:
    if category == "red_only":
        return RED
    if category == "blue_only":
        return BLUE
    if category == "both":
        return YELLOW
    if category == "none":
        return BG
    raise ValueError(f"unknown category: {category}")


def _check_chambers(grid: np.ndarray, out: np.ndarray, pods: list[dict]) -> bool:
    for pod in pods:
        r0, c0, r1, c1 = pod["interior"]
        expected = _expected_color(pod["category"])
        room = out[r0 : r1 + 1, c0 : c1 + 1]
        if room.size == 0 or not np.all(room == expected):
            return False
    return True


def _semantic_ok(grid: np.ndarray, pods: list[dict]) -> bool:
    out, red_reach, blue_reach = _simulate_output(grid, ACTUAL)
    if not _check_chambers(grid, out, pods):
        return False

    empty = grid == BG
    counts = {
        BG: int(np.sum(empty & ~red_reach & ~blue_reach)),
        RED: int(np.sum(empty & red_reach & ~blue_reach)),
        BLUE: int(np.sum(empty & blue_reach & ~red_reach)),
        YELLOW: int(np.sum(empty & red_reach & blue_reach)),
    }
    if any(v == 0 for v in counts.values()):
        return False

    solid, _, _ = _simulate_output(grid, SOLID)
    open_out, _, _ = _simulate_output(grid, OPEN)
    swapped, _, _ = _simulate_output(grid, SWAPPED)
    if np.array_equal(out, solid) or np.array_equal(out, open_out) or np.array_equal(out, swapped):
        return False

    if np.array_equal(red_reach, blue_reach):
        return False

    walls = np.argwhere(grid == WALL)
    if walls.size == 0:
        return False
    wall_parities = {(int(r + c) & 1) for r, c in walls}
    if wall_parities != {0, 1}:
        return False

    return True


def _place_markers(grid: np.ndarray, rng: random.Random) -> None:
    h, w = grid.shape
    if rng.random() < 0.5:
        red_pos, blue_pos = (0, 0), (h - 1, w - 1)
    else:
        red_pos, blue_pos = (0, w - 1), (h - 1, 0)
    grid[red_pos] = RED
    grid[blue_pos] = BLUE


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(22, 29)
    w = rng.randint(22, 29)
    grid = np.zeros((h, w), dtype=int)

    arena_h = rng.randint(7, 8)
    arena_w = rng.randint(7, 8)
    ar0 = (h - arena_h) // 2
    ac0 = (w - arena_w) // 2
    ar1 = ar0 + arena_h - 1
    ac1 = ac0 + arena_w - 1
    arena = (ar0, ac0, ar1, ac1)

    _place_markers(grid, rng)

    sides = ["top", "right", "bottom", "left"]
    categories = ["red_only", "blue_only", "both", "none"]
    rng.shuffle(categories)

    pods: list[dict] = []
    for side, category in zip(sides, categories):
        if side in {"top", "bottom"}:
            outer_h = rng.randint(6, 8)
            outer_w = rng.randint(6, arena_w)
        else:
            outer_h = rng.randint(6, arena_h)
            outer_w = rng.randint(6, 8)
        corridor_len = rng.randint(2, 4)
        pod = _draw_pod(grid, arena, side, outer_h, outer_w, corridor_len, category, rng)
        if pod is None:
            return None
        pods.append(pod)

    if np.any(grid[[0, 0, h - 1, h - 1], [0, w - 1, 0, w - 1]] == WALL):
        return None

    if not _semantic_ok(grid, pods):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
