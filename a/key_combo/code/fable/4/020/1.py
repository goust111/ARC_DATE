from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import Color, get_bg

MAX_GENERATION_ATTEMPTS = 600
ARC_BAR_PALETTE = [
    Color.BLUE,
    Color.RED,
    Color.YELLOW,
    Color.GREY,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
]


def main(grid: np.ndarray) -> np.ndarray:
    """Trace a gray-starting ray from the green seed with wall-gated side reflections.

    Input structure:
    - The background is black.
    - Left and right border columns may contain colored vertical bar segments; border gaps are black.
    - The interior grid has exactly one green start pixel and all other cells are black.

    Rule:
    - Start from the green pixel and move one cell up with current horizontal direction (initially right).
    - Paint each visited interior cell with current color; keep the start pixel green.
    - If the next cell is top row, paint it and terminate.
    - If the next step would enter a side border cell:
      - a non-black border cell triggers a bounce: do not enter it, reverse horizontal direction, and inherit that border color.
      - a black border cell triggers leakage: paint that border cell with current color and terminate.

    Key insight:
    - Only border contacts change trajectory; each bounce inherits the struck bar color, so path color history records the last struck bar, while gaps terminate immediately.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be 2D"
    bg = get_bg(g)
    assert bg == Color.BLACK, "background must be black"

    starts = np.argwhere(g == Color.GREEN)
    assert starts.shape[0] == 1, "input must contain exactly one green start pixel"
    r, c = map(int, starts[0])
    H, W = g.shape
    assert 0 < c < W - 1, "start must be in interior column"
    assert 0 <= r < H, "start row out of bounds"

    out = g.copy()
    direction = 1  # 1 => right, -1 => left
    current = int(Color.GREY)

    while True:
        if not (r == starts[0][0] and c == starts[0][1]):
            out[r, c] = current
        else:
            out[r, c] = Color.GREEN

        if r == 0:
            break

        next_r = r - 1
        next_c = c + direction

        if next_c == 0 or next_c == W - 1:
            wall_color = int(g[next_r, next_c])
            if wall_color == int(bg):
                out[next_r, next_c] = current
                break
            current = wall_color
            direction *= -1
            r = next_r
        else:
            r, c = next_r, next_c

    return out


def _simulate_outcome(grid: np.ndarray, start: tuple[int, int]) -> tuple[str, int]:
    """Return (outcome, bounce_count) for a given grid and start."""
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    r, c = map(int, start)
    direction = 1
    current = Color.GREY  # kept for parity with main, not used for control
    del current
    bounces = 0

    while True:
        if r == 0:
            return "top", bounces
        next_r = r - 1
        next_c = c + direction
        if next_c == 0 or next_c == W - 1:
            wall_color = int(g[next_r, next_c])
            if wall_color == Color.BLACK:
                return "gap", bounces
            direction *= -1
            bounces += 1
            r = next_r
        else:
            r, c = next_r, next_c


def _predict_bounce_collisions(H: int, W: int, start_r: int, start_c: int) -> list[tuple[int, int]]:
    """Simulate a perfect bounce trajectory and return wall-contact cells."""
    r, c = start_r, start_c
    direction = 1
    contacts: list[tuple[int, int]] = []

    while r > 0:
        next_r = r - 1
        next_c = c + direction
        if next_c == 0 or next_c == W - 1:
            contacts.append((next_r, next_c))
            direction *= -1
            r = next_r
        else:
            r, c = next_r, next_c
    return contacts


def _paint_interval(column: np.ndarray, r0: int, r1: int, color: int) -> None:
    """Paint [r0, r1] inclusive in a wall column."""
    if r0 > r1:
        return
    r0 = max(0, int(r0))
    r1 = min(len(column) - 1, int(r1))
    if r0 > r1:
        return
    column[r0 : r1 + 1] = int(color)


def _add_random_segments(
    column: np.ndarray,
    *,
    min_segments: int = 0,
    max_segments: int = 2,
    reserve: set[int] | None = None,
) -> None:
    """Add 0..2 random short contiguous wall segments, avoiding reserved rows."""
    if max_segments < min_segments:
        max_segments = min_segments
    H = len(column)
    used = np.zeros(H, dtype=bool)
    if reserve is not None:
        for row in reserve:
            if 0 <= row < H:
                used[row] = True

    n_segments = random.randint(min_segments, max_segments)
    for _ in range(n_segments):
        if used.all():
            return
        seg_len = random.randint(1, min(4, H))
        for _ in range(30):
            start = random.randint(0, max(0, H - seg_len))
            if used[start : start + seg_len].any():
                continue
            color = random.choice(ARC_BAR_PALETTE)
            column[start : start + seg_len] = color
            used[start : start + seg_len] = True
            break


def _build_grid(
    mode: str,
    H: int,
    W: int,
    start_r: int,
    start_c: int,
) -> np.ndarray | None:
    left = np.full(H, Color.BLACK, dtype=int)
    right = np.full(H, Color.BLACK, dtype=int)

    if mode == "top":
        # No wall contacts before reaching top
        _add_random_segments(left, min_segments=1, max_segments=3)
        _add_random_segments(right, min_segments=1, max_segments=3)
        left[0:1] = Color.BLACK

    elif mode == "gap":
        # Force first right-wall encounter to be a black gap
        step_to_right = W - 2 - start_c
        if start_r <= step_to_right:
            return None
        gap_row = start_r - step_to_right - 1
        if not (0 <= gap_row < H):
            return None
        right[gap_row] = Color.BLACK

        # Add extra bars for visual richness, avoiding the gap row
        _add_random_segments(left, min_segments=1, max_segments=2)
        _add_random_segments(right, min_segments=1, max_segments=2, reserve={gap_row})

    elif mode == "bounce":
        contacts = _predict_bounce_collisions(H, W, start_r, start_c)
        if not contacts:
            return None

        left_rows = [r for r, cc in contacts if cc == 0]
        right_rows = [r for r, cc in contacts if cc == W - 1]

        if not left_rows and not right_rows:
            return None

        if left_rows:
            color = random.choice(ARC_BAR_PALETTE)
            _paint_interval(left, min(left_rows), max(left_rows), color)
        if right_rows:
            color = random.choice(ARC_BAR_PALETTE)
            if left_rows and color in {left[min(left_rows)], left[max(left_rows)]}:
                color = random.choice([c for c in ARC_BAR_PALETTE if c != int(left[0])])
            _paint_interval(right, min(right_rows), max(right_rows), color)

        # Fill a little extra so side structures are not too minimal
        if left_rows:
            _add_random_segments(left, min_segments=0, max_segments=1, reserve=set(left_rows))
        else:
            _add_random_segments(left, min_segments=1, max_segments=2)

        if right_rows:
            _add_random_segments(right, min_segments=0, max_segments=1, reserve=set(right_rows))
        else:
            _add_random_segments(right, min_segments=1, max_segments=2)

    else:
        raise ValueError(f"unknown mode: {mode}")

    g = np.full((H, W), Color.BLACK, dtype=int)
    g[:, 0] = left
    g[:, -1] = right
    g[start_r, start_c] = Color.GREEN
    return g


def _is_valid_input(grid: np.ndarray, mode: str) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        return False
    H, W = g.shape
    if H == 0 or W == 0 or H > 30 or W > 30:
        return False
    if get_bg(g) != Color.BLACK:
        return False

    starts = np.argwhere(g == Color.GREEN)
    if starts.shape[0] != 1:
        return False
    sr, sc = map(int, starts[0])
    if not (0 < sr < H and 0 < sc < W - 1):
        return False

    interior = g.copy()
    interior[:, 0] = Color.BLACK
    interior[:, -1] = Color.BLACK
    interior[sr, sc] = Color.BLACK
    if np.any(interior != Color.BLACK):
        return False

    outcome, bounces = _simulate_outcome(g, (sr, sc))
    if mode == "top":
        return outcome == "top" and bounces == 0
    if mode == "gap":
        return outcome == "gap"
    if mode == "bounce":
        return outcome == "top" and bounces >= 1
    return False


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        mode = random.choices(["top", "gap", "bounce"], weights=[0.28, 0.36, 0.36], k=1)[0]
        H = random.randint(8, 30)
        W = random.randint(10, 30)

        if mode == "top":
            # Ensure no side-wall contact within this run.
            start_c = random.randint(2, max(2, W - 3))
            max_safe_row = max(0, W - 2 - start_c)
            if max_safe_row < 1:
                continue
            start_r = random.randint(1, min(H - 1, max_safe_row))
        elif mode == "gap":
            start_c = random.randint(2, max(2, W - 3))
            min_row_for_gap = W - 1 - start_c
            if min_row_for_gap >= H:
                continue
            start_r = random.randint(min_row_for_gap, H - 1)
        else:  # bounce
            start_c = random.randint(2, max(2, W - 3))
            min_row_for_bounce = W - 1 - start_c
            if min_row_for_bounce >= H:
                continue
            start_r = random.randint(min_row_for_bounce, H - 1)

        grid = _build_grid(mode, H, W, start_r, start_c)
        if grid is None:
            continue
        if _is_valid_input(grid, mode):
            return grid

    raise RuntimeError("failed to generate a valid 020/1 input")
