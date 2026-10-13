from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import Color, neighbors4  # type: ignore
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        ORANGE = 7

    def neighbors4(r: int, c: int):
        yield r - 1, c
        yield r + 1, c
        yield r, c - 1
        yield r, c + 1


BG = int(Color.BLACK)
SEED = int(Color.ORANGE)
COLOR_TO_DIR = {
    int(Color.RED): (0, 1),
    int(Color.BLUE): (1, 0),
    int(Color.GREEN): (0, -1),
    int(Color.YELLOW): (-1, 0),
}
SOURCE_COLORS = set(COLOR_TO_DIR)

GRID_MIN = 16
GRID_MAX = 26
BOX_MIN = 7
BOX_MAX = 13
MAX_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _find_seed(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == SEED)
    assert coords.shape[0] == 1, "expected exactly one orange seed"
    return tuple(map(int, coords[0]))


def _trace_walls(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    source_positions = np.argwhere(np.isin(g, list(SOURCE_COLORS)))

    rays: list[dict[str, tuple[int, int] | bool]] = []
    for r, c in source_positions:
        rr = int(r)
        cc = int(c)
        rays.append(
            {
                "pos": (rr, cc),
                "dir": COLOR_TO_DIR[int(g[rr, cc])],
                "active": True,
            }
        )

    wall = np.zeros((h, w), dtype=bool)
    blocked = g != BG
    active = list(range(len(rays)))

    while active:
        proposals: dict[tuple[int, int], list[int]] = {}
        for idx in active:
            r, c = rays[idx]["pos"]  # type: ignore[index]
            dr, dc = rays[idx]["dir"]  # type: ignore[index]
            nr, nc = int(r) + int(dr), int(c) + int(dc)
            if not _in_bounds((h, w), nr, nc):
                continue
            if blocked[nr, nc] or wall[nr, nc]:
                continue
            proposals.setdefault((nr, nc), []).append(idx)

        if not proposals:
            break

        next_active: list[int] = []
        for (nr, nc), owners in proposals.items():
            if len(owners) != 1:
                continue
            idx = owners[0]
            wall[nr, nc] = True
            rays[idx]["pos"] = (nr, nc)
            next_active.append(idx)
        active = next_active

    return wall


def _reachable_background(grid: np.ndarray, wall: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    reachable = np.zeros((h, w), dtype=bool)
    seen = {seed}
    q = deque([seed])

    while q:
        r, c = q.popleft()
        for rr, cc in neighbors4(int(r), int(c)):
            if not _in_bounds((h, w), int(rr), int(cc)):
                continue
            pos = (int(rr), int(cc))
            if pos in seen:
                continue
            if wall[pos]:
                continue
            if int(g[pos]) != BG:
                continue
            seen.add(pos)
            reachable[pos] = True
            q.append(pos)

    return reachable


def _boundary_reached(mask: np.ndarray) -> bool:
    return bool(
        mask[0, :].any()
        or mask[-1, :].any()
        or mask[:, 0].any()
        or mask[:, -1].any()
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast invisible direction-coded walls, then flood-fill the seed's reachable background.

    Input structure:
    - The background is black.
    - Red, blue, green, and yellow single pixels encode rays pointing right, down,
      left, and up respectively.
    - Exactly one orange seed pixel is present; any non-black pixel blocks rays and fill.

    Rule:
    - Advance every direction source one cell at a time in its encoded direction.
    - A ray stops before leaving the grid, before entering any original non-background
      cell, or when another ray would claim the same next cell in the same step; only
      uniquely claimed traversed cells become invisible walls.
    - Treat invisible walls, all sources, and the seed as impassable. Starting from the
      seed, flood-fill the 4-connected reachable black cells with orange.

    Key insight:
    - The visible colored pixels are only ray emitters. The real enclosure is the hidden
      union of their stepwise wall segments, and mutual truncation can leave crucial gaps.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    seed = _find_seed(g)
    wall = _trace_walls(g)
    reachable = _reachable_background(g, wall, seed)

    out = g.copy()
    out[reachable] = SEED
    return out


def _rand_with_parity(lo: int, hi: int, parity: int) -> int:
    vals = [v for v in range(lo, hi + 1) if v % 2 == parity]
    return random.choice(vals)


def _background_cells(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if int(grid[r, c]) == BG:
                cells.append((r, c))
    return cells


def _place_horizontal_template(
    grid: np.ndarray, top: int, left: int, bottom: int, right: int, variant: int
) -> None:
    grid[top, left] = int(Color.RED)
    grid[top, right] = int(Color.GREEN)
    grid[bottom, left] = int(Color.RED)
    grid[bottom, right] = int(Color.GREEN)
    if variant == 0:
        grid[top + 1, left] = int(Color.BLUE)
        grid[bottom - 1, right] = int(Color.YELLOW)
    else:
        grid[bottom - 1, left] = int(Color.YELLOW)
        grid[top + 1, right] = int(Color.BLUE)


def _place_vertical_template(
    grid: np.ndarray, top: int, left: int, bottom: int, right: int, variant: int
) -> None:
    grid[top, left] = int(Color.BLUE)
    grid[bottom, left] = int(Color.YELLOW)
    grid[top, right] = int(Color.BLUE)
    grid[bottom, right] = int(Color.YELLOW)
    if variant == 0:
        grid[top, left + 1] = int(Color.RED)
        grid[bottom, right - 1] = int(Color.GREEN)
    else:
        grid[top, right - 1] = int(Color.GREEN)
        grid[bottom, left + 1] = int(Color.RED)


def _place_extras(
    grid: np.ndarray, top: int, left: int, bottom: int, right: int
) -> None:
    h, w = grid.shape
    candidates: list[tuple[list[tuple[int, int]], int]] = []

    if top >= 3:
        cells = [(r, c) for r in range(1, top) for c in range(w) if int(grid[r, c]) == BG]
        if cells:
            candidates.append((cells, int(Color.YELLOW)))
    if bottom <= h - 4:
        cells = [(r, c) for r in range(bottom + 1, h - 1) for c in range(w) if int(grid[r, c]) == BG]
        if cells:
            candidates.append((cells, int(Color.BLUE)))
    if left >= 3:
        cells = [(r, c) for r in range(h) for c in range(1, left) if int(grid[r, c]) == BG]
        if cells:
            candidates.append((cells, int(Color.GREEN)))
    if right <= w - 4:
        cells = [(r, c) for r in range(h) for c in range(right + 1, w - 1) if int(grid[r, c]) == BG]
        if cells:
            candidates.append((cells, int(Color.RED)))

    random.shuffle(candidates)
    for cells, color in candidates[: random.randint(0, min(3, len(candidates)))]:
        r, c = random.choice(cells)
        grid[r, c] = color


def _build_attempt() -> tuple[np.ndarray, bool, tuple[int, int, int, int]] | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    leak = random.random() < 0.5
    template = random.choice(["horizontal", "vertical"])

    if template == "horizontal":
        box_h = random.randint(BOX_MIN, BOX_MAX)
        box_w = _rand_with_parity(BOX_MIN, BOX_MAX, 1 if leak else 0)
    else:
        box_h = _rand_with_parity(BOX_MIN, BOX_MAX, 1 if leak else 0)
        box_w = random.randint(BOX_MIN, BOX_MAX)

    if box_h + 4 > h or box_w + 4 > w:
        return None

    top = random.randint(2, h - box_h - 2)
    left = random.randint(2, w - box_w - 2)
    bottom = top + box_h - 1
    right = left + box_w - 1

    grid = np.full((h, w), BG, dtype=int)
    variant = random.randint(0, 1)
    if template == "horizontal":
        _place_horizontal_template(grid, top, left, bottom, right, variant)
    else:
        _place_vertical_template(grid, top, left, bottom, right, variant)

    seed_choices = _background_cells(grid, top + 1, left + 1, bottom - 1, right - 1)
    if not seed_choices:
        return None
    sr, sc = random.choice(seed_choices)
    grid[sr, sc] = SEED

    _place_extras(grid, top, left, bottom, right)
    return grid, leak, (top, left, bottom, right)


def _valid_attempt(grid: np.ndarray, leak: bool, box: tuple[int, int, int, int]) -> bool:
    top, left, bottom, right = box
    seed = _find_seed(grid)
    wall = _trace_walls(grid)
    reachable = _reachable_background(grid, wall, seed)
    out = main(grid)

    if not np.array_equal(out[grid != BG], grid[grid != BG]):
        return False

    if not np.array_equal(out[reachable], np.full(reachable.sum(), SEED, dtype=int)):
        return False

    background_count = int((grid == BG).sum())
    filled_count = int(reachable.sum())
    if filled_count < 10:
        return False
    if int(wall.sum()) < 8:
        return False

    boundary = _boundary_reached(reachable)
    if boundary != leak:
        return False

    if leak:
        if filled_count <= background_count // 2:
            return False
    else:
        expected_inside = (bottom - top - 1) * (right - left - 1)
        if filled_count != expected_inside:
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        built = _build_attempt()
        if built is None:
            continue
        grid, leak, box = built
        if _valid_attempt(grid, leak, box):
            return grid
    raise RuntimeError("failed to generate a valid invisible-ray-wall input")
