from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_arc_utils = SourceFileLoader("arc_utils", str(_ARC_UTILS_PATH)).load_module()

Color = _arc_utils.Color
coords_where = _arc_utils.coords_where
get_bg = _arc_utils.get_bg


RNG = np.random.default_rng()

_DIRS: tuple[tuple[int, int], ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _initial_direction(r: int, c: int, h: int, w: int) -> int:
    assert (0 <= r < h and 0 <= c < w), "start must be within the grid"
    is_top = int(r) == 0
    is_bottom = int(r) == h - 1
    is_left = int(c) == 0
    is_right = int(c) == w - 1

    assert is_top or is_bottom or is_left or is_right, "start pixel must lie on the boundary"
    assert not ((is_top or is_bottom) and (is_left or is_right)), "start pixel must not be on a corner"
    if r == 0:
        return 2  # down
    if r == h - 1:
        return 0  # up
    if c == 0:
        return 1  # right
    return 3  # left


def _simulate_to_exit(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: int,
    *,
    max_steps: int,
) -> tuple[np.ndarray, dict[tuple[int, int], int], int]:
    h, w = grid.shape
    working = np.array(grid, dtype=int, copy=True)
    out = np.array(grid, dtype=int, copy=True)
    sr, sc = map(int, start)
    out[sr, sc] = Color.GREEN

    ctrl_visits: dict[tuple[int, int], int] = {}
    visited_triple = 0
    moves = 0
    r, c = sr, sc
    d = int(direction)

    for _ in range(max_steps):
        dr, dc = _DIRS[d]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return out, ctrl_visits, moves

        cell = int(working[nr, nc])
        if cell == Color.PINK:
            d = (d + 1) % 4  # right turn (clockwise)
            working[nr, nc] = Color.TEAL
            out[nr, nc] = Color.TEAL
            ctrl_visits[(nr, nc)] = ctrl_visits.get((nr, nc), 0) + 1
        elif cell == Color.TEAL:
            d = (d - 1) % 4  # left turn (counter-clockwise)
            working[nr, nc] = Color.PINK
            out[nr, nc] = Color.PINK
            ctrl_visits[(nr, nc)] = ctrl_visits.get((nr, nc), 0) + 1
        else:
            r, c = nr, nc
            out[r, c] = Color.GREEN
            moves += 1

    # generation inputs are assumed to be solvable by finite exit;
    # guard here to avoid an accidental infinite simulation loop.
    raise AssertionError("simulation exceeded max_steps; input does not terminate")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a green path from a boundary start pixel while pink and teal controls turn direction and toggle on each visit.

    Input structure:
    - The grid has exactly one green boundary start pixel (not in a corner), one black background, and interior pink/teal control pixels.
    - The green start pixel points one cell inward from the boundary.
    - Pink pixels cause right turns, teal pixels cause left turns.

    Rule:
    - Repeatedly inspect the cell immediately ahead in current direction.
    - If the next cell is pink, do not move, turn 90° right, recolor that control teal, and continue.
    - If the next cell is teal, do not move, turn 90° left, recolor that control pink, and continue.
    - If the next cell is not a control, move into it and paint it green.
    - Stop when the next move would leave the grid.

    Key insight:
    - Control pixels are stateful switches; revisits must use the control's current color, so revisiting the same control changes both trajectory and final control colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h >= 3 and w >= 3, "expected a nontrivial grid"

    bg = get_bg(g)
    colors = set(map(int, np.unique(g)))
    assert Color.GREEN in colors, "expected exactly one green start pixel"
    assert colors <= {int(bg), Color.GREEN, Color.PINK, Color.TEAL}, (
        "expected only background, green, pink and teal pixels"
    )

    start_mask = coords_where(g == Color.GREEN)
    assert start_mask.shape[0] == 1, "expected exactly one green start pixel"
    start = (int(start_mask[0, 0]), int(start_mask[0, 1]))
    sr, sc = start
    assert not (0 < sr < h - 1 and 0 < sc < w - 1), "start must be on the boundary"

    control_mask = (g == Color.PINK) | (g == Color.TEAL)
    if np.any(control_mask):
        ctrl_coords = np.argwhere(control_mask)
        assert np.all(
            (0 < ctrl_coords[:, 0]) & (ctrl_coords[:, 0] < h - 1) & (0 < ctrl_coords[:, 1]) & (ctrl_coords[:, 1] < w - 1)
        ), "control pixels must be strictly interior"
    assert np.any(control_mask), "expected at least one control pixel"

    start_dir = _initial_direction(sr, sc, h, w)

    out, _, _ = _simulate_to_exit(g, start, start_dir, max_steps=h * w * 12)
    return out


def _random_start(rng: np.random.Generator, h: int, w: int) -> tuple[tuple[int, int], int]:
    side = int(rng.integers(4))
    if side == 0:
        r = 0
        c = int(rng.integers(1, w - 1))
        return (r, c), 2
    if side == 1:
        r = h - 1
        c = int(rng.integers(1, w - 1))
        return (r, c), 0
    if side == 2:
        r = int(rng.integers(1, h - 1))
        c = 0
        return (r, c), 1
    r = int(rng.integers(1, h - 1))
    c = w - 1
    return (r, c), 3


def _pick_control_positions(
    rng: np.random.Generator,
    h: int,
    w: int,
    start: tuple[int, int],
    k: int,
) -> list[tuple[int, int]]:
    if k == 0:
        return []
    sr, sc = start
    candidates = [
        (r, c)
        for r in range(1, h - 1)
        for c in range(1, w - 1)
        if not (r == sr and c == sc)
    ]
    if not candidates:
        raise RuntimeError("grid too small for control placement")

    chosen: list[tuple[int, int]] = []
    attempts = 0
    while len(chosen) < k and attempts < 4000:
        attempts += 1
        r, c = candidates[int(rng.integers(len(candidates)))]
        if (r, c) in chosen:
            continue
        if any(abs(r - rr) + abs(c - cc) <= 1 for rr, cc in chosen):
            continue
        chosen.append((r, c))
    if len(chosen) < k:
        # fallback: allow denser placement to avoid dead-end retries
        rng.shuffle(candidates)
        chosen = candidates[:k]
    return chosen


def _probe_grid(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: int,
) -> tuple[bool, dict[tuple[int, int], int], int, int]:
    h, w = grid.shape
    try:
        _, ctrl_visits, moves = _simulate_to_exit(
            grid,
            start,
            direction,
            max_steps=h * w * 8,
        )
    except AssertionError:
        return False, {}, 0, 0

    revisit_max = max(ctrl_visits.values(), default=0)
    return True, ctrl_visits, moves, revisit_max


def _valid_candidate(
    h: int,
    w: int,
    require_revisit: bool,
) -> np.ndarray | None:
    bg = Color.BLACK
    start, direction = _random_start(RNG, h, w)
    sr, sc = start

    control_count = int(RNG.integers(2, min(11, (h - 2) * (w - 2) // 2)))
    control_positions = _pick_control_positions(RNG, h, w, start, control_count)
    if not control_positions:
        return None

    colors = [int(c) for c in RNG.choice([Color.PINK, Color.TEAL], size=len(control_positions))]
    if len(set(colors)) == 1 and len(colors) > 0:
        colors[0] = Color.PINK if colors[0] == Color.TEAL else Color.TEAL

    grid = np.full((h, w), int(bg), dtype=int)
    grid[sr, sc] = Color.GREEN
    for (r, c), col in zip(control_positions, colors):
        grid[r, c] = int(col)

    ok, ctrl_visits, moves, revisit_max = _probe_grid(grid, start, direction)
    if not ok:
        return None
    if moves <= 0:
        return None
    if require_revisit and revisit_max < 2:
        return None
    if not require_revisit and revisit_max >= 2:
        return None

    # ensure generator actually reaches the task's semantics, not an accidental artifact
    try:
        expected = main(grid)
    except AssertionError:
        return None
    if expected.shape != grid.shape:
        return None
    if not isinstance(expected, np.ndarray):
        return None

    seen_control_colors = set(map(int, expected[expected != int(bg)]))
    if Color.GREEN not in seen_control_colors and len(colors) > 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    attempts = 600
    for _ in range(attempts):
        h = int(RNG.integers(8, 27))
        w = int(RNG.integers(8, 27))
        h = min(h, 28)
        w = min(w, 28)
        if h < 3 or w < 3:
            continue
        require_revisit = bool(RNG.random() < 0.5)
        grid = _valid_candidate(h, w, require_revisit=require_revisit)
        if grid is None:
            continue
        try:
            _ = main(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
