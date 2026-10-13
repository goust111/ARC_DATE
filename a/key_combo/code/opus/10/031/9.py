import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
TARGET_WIDTH = 12
GEN_RETRIES = 500


def _is_overlap_period(grid: np.ndarray, period: int) -> bool:
    period = int(period)
    _, width = grid.shape
    if period <= 0 or period >= width:
        return False
    return bool(np.array_equal(grid[:, : width - period], grid[:, period:]))


def _infer_min_period(grid: np.ndarray) -> int:
    width = int(grid.shape[1])
    for period in range(1, width):
        if _is_overlap_period(grid, period):
            return int(period)
    return width


def _tile_and_recolor(base: np.ndarray, period: int) -> np.ndarray:
    reps = (TARGET_WIDTH + int(period) - 1) // int(period)
    tiled = np.tile(base, (1, reps))[:, :TARGET_WIDTH].copy()
    for c in range(TARGET_WIDTH):
        mask = tiled[:, c] != BLACK
        if not np.any(mask):
            continue
        tiled[mask, c] = RED if (c % int(period)) % 2 == 0 else GREEN
    return tiled


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer a horizontal repeat period, complete the pattern to width 12, and recolor by column phase parity.

    Input structure:
    - The grid has black background and a blue foreground pattern formed by horizontally repeating one base slice.
    - The input width is less than 12, and at least two full periods are visible.
    - The repetition starts at the left edge, so the leftmost period-width block is the base slice.

    Rule:
    - Find the smallest positive horizontal translation period P whose overlap matches everywhere it is visible.
    - Take the leftmost P columns as the base slice.
    - Repeat that slice horizontally until the output is 12 columns wide, truncating the last copy if needed.
    - Recolor each non-black output pixel by its phase `c mod P`: even phases become red, odd phases become green.

    Key insight:
    - The same inferred period P controls both the geometric completion and the red/green phase coloring, so a wrong P breaks both parts at once.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    period = _infer_min_period(g)
    base = g[:, :period].copy()
    return _tile_and_recolor(base, period)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _mask_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if cells.size == 0:
        return False
    start = tuple(map(int, cells[0]))
    seen = {start}
    q = deque([start])
    height, width = mask.shape
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen) == int(mask.sum())


def _random_connected_mask(height: int, width: int, cells: int, rng: random.Random) -> np.ndarray:
    start = (rng.randrange(height), rng.randrange(width))
    shape = {start}
    while len(shape) < cells:
        frontier = []
        for r, c in shape:
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in shape:
                    frontier.append((rr, cc))
        if not frontier:
            break
        shape.add(rng.choice(frontier))
    mask = np.zeros((height, width), dtype=bool)
    for r, c in shape:
        mask[r, c] = True
    return mask


def _mask_is_irregular(mask: np.ndarray) -> bool:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if int(rows.sum()) < 2 or int(cols.sum()) < 2:
        return False
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    bbox_area = int(r1 - r0 + 1) * int(c1 - c0 + 1)
    return bbox_area > int(mask.sum())


def _phase_columns(mask: np.ndarray) -> set[int]:
    return {int(c) for c in np.where(np.any(mask, axis=0))[0]}


def _make_base_slice(height: int, period: int, rng: random.Random) -> np.ndarray | None:
    total = height * period
    min_cells = max(4, (total + 4) // 5)
    max_cells = min(total - 1, max(min_cells, (total * 9) // 20))
    if min_cells > max_cells:
        return None

    for _ in range(200):
        cells = rng.randint(min_cells, max_cells)
        mask = _random_connected_mask(height, period, cells, rng)
        if int(mask.sum()) != cells:
            continue
        if not _mask_connected(mask):
            continue
        if not _mask_is_irregular(mask):
            continue
        cols = _phase_columns(mask)
        if len(cols) < 2:
            continue
        if not any(c % 2 == 0 for c in cols):
            continue
        if not any(c % 2 == 1 for c in cols):
            continue
        if period > 1 and all(np.array_equal(mask[:, c], mask[:, 0]) for c in range(1, period)):
            continue
        if period % 2 == 0:
            half = period // 2
            if half > 0 and all(np.array_equal(mask[:, c], mask[:, c + half]) for c in range(period - half)):
                continue
        base = np.zeros((height, period), dtype=int)
        base[mask] = BLUE
        return base
    return None


def _sample_input_width(period: int, rng: random.Random) -> int:
    exact = 2 * period
    partials = [w for w in range(exact + 1, min(9, exact + period - 1) + 1)]
    if partials and rng.random() < 0.5:
        return int(rng.choice(partials))
    return int(exact)


def _oracle_output(base: np.ndarray, period: int) -> np.ndarray:
    return _tile_and_recolor(base, period)


def _build_input(base: np.ndarray, width: int) -> np.ndarray:
    period = int(base.shape[1])
    reps = (int(width) + period - 1) // period
    return np.tile(base, (1, reps))[:, :width].copy()


def _semantic_check(inp: np.ndarray, base: np.ndarray, period: int) -> bool:
    height, width = inp.shape
    if not (5 <= height <= 7 and 6 <= width <= 9 and width < TARGET_WIDTH):
        return False
    if set(map(int, np.unique(inp))) - {BLACK, BLUE}:
        return False
    if width < 2 * period:
        return False
    if not np.array_equal(inp, _build_input(base, width)):
        return False
    if _infer_min_period(inp) != int(period):
        return False
    expected = _oracle_output(base, period)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False
    colors = set(map(int, np.unique(expected)))
    if RED not in colors or GREEN not in colors:
        return False
    for alt_period in range(1, width + 1):
        if alt_period == int(period):
            continue
        alt_base = inp[:, :alt_period]
        alt_output = _oracle_output(alt_base, alt_period)
        if np.array_equal(alt_output, expected):
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    period = rng.choice([3, 4])
    height = rng.randint(5, 7)
    base = _make_base_slice(height, period, rng)
    if base is None:
        return None
    width = _sample_input_width(period, rng)
    inp = _build_input(base, width)
    if not _semantic_check(inp, base, period):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
