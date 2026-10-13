from __future__ import annotations

import hashlib
import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
PINK = 6
ORANGE = 7

SHAPE_COLORS = (RED, GREEN, YELLOW)
COUNT_PAIRS = ((2, 3), (2, 4), (3, 2), (3, 4), (4, 2), (4, 3))
ORIENTATIONS = ("H", "V")
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_RETRIES = 600

_SEEN_OUTPUTS: set[str] = set()
_GEN_SUCCESS_COUNT = 0


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return count


def _signature(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(tuple(grid.shape)).encode()).hexdigest()


def _parse_scene(grid: np.ndarray) -> tuple[str, np.ndarray, int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)

    full_rows = np.flatnonzero(np.all(g == BLUE, axis=1)).tolist()
    full_cols = np.flatnonzero(np.all(g == BLUE, axis=0)).tolist()
    has_horizontal = len(full_rows) == 2
    has_vertical = len(full_cols) == 2
    assert has_horizontal ^ has_vertical, "expected exactly one guide-bar orientation"

    if has_horizontal:
        top, bottom = map(int, full_rows)
        strip = g[top + 1 : bottom, :]
        shape_mask = np.isin(strip, SHAPE_COLORS)
        box = _bbox(shape_mask)
        assert box is not None, "expected a shape between the guide bars"
        outside_rows = np.ones(h, dtype=bool)
        outside_rows[top : bottom + 1] = False
        n_cols = int(np.count_nonzero((g[:, 0] == ORANGE) & outside_rows))
        n_rows = int(np.count_nonzero((g[:, w - 1] == PINK) & outside_rows))
        return "H", _crop_bbox(strip, box), n_rows, n_cols

    left, right = map(int, full_cols)
    strip = g[:, left + 1 : right]
    shape_mask = np.isin(strip, SHAPE_COLORS)
    box = _bbox(shape_mask)
    assert box is not None, "expected a shape between the guide bars"
    outside_cols = np.ones(w, dtype=bool)
    outside_cols[left : right + 1] = False
    n_cols = int(np.count_nonzero((g[0, :] == ORANGE) & outside_cols))
    n_rows = int(np.count_nonzero((g[h - 1, :] == PINK) & outside_cols))
    return "V", _crop_bbox(strip, box), n_rows, n_cols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the interior shape between the blue guide bars and tile it by border-encoded repeat counts.

    Input structure:
    - The grid has a black background with exactly two full blue guide bars, either two rows or two columns.
    - A small 4-connected multi-color shape made from red, green, and yellow lies strictly between those bars.
    - Outside the guide-bar span, orange border pixels encode the repeat-column count and pink border pixels encode the repeat-row count.

    Rule:
    - Detect whether the blue guide bars are horizontal or vertical.
    - Crop the shape between the bars to its tight bounding box, keeping black holes inside that crop.
    - Count orange pixels on the orientation-specific first border side and pink pixels on the opposite side to obtain the tiling width and height.
    - Tile the cropped block into that many columns and rows with no gaps, and output only the tiled matrix.

    Key insight:
    - The guide bars say which pair of borders carries the code, while the border-pixel counts themselves determine the tiling dimensions and the cropped interior shape supplies the repeated content.
    """
    _orientation, base, n_rows, n_cols = _parse_scene(grid)
    assert 2 <= n_rows <= 4 and 2 <= n_cols <= 4, "repeat counts must stay in the encoded range"
    return np.tile(base, (n_rows, n_cols))


def _random_shape_block() -> np.ndarray | None:
    for _ in range(200):
        area = random.randint(3, 6)
        cells = {(2, 2)}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < 5 and 0 <= nc < 5 and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue

        rows, cols = zip(*cells)
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 == r0 or c1 == c0:
            continue
        if (r1 - r0 + 1) > 4 or (c1 - c0 + 1) > 4:
            continue

        mask = np.zeros((5, 5), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        box = _bbox(mask)
        assert box is not None
        tight = mask[box[0] : box[2] + 1, box[1] : box[3] + 1]
        if not (tight[0].any() and tight[-1].any() and tight[:, 0].any() and tight[:, -1].any()):
            continue

        palette = random.sample(SHAPE_COLORS, random.choice((2, 3 if area >= 4 else 2)))
        coords = [tuple(map(int, rc)) for rc in np.argwhere(tight)]
        random.shuffle(coords)
        block = np.zeros(tight.shape, dtype=int)
        for idx, (r, c) in enumerate(coords):
            block[r, c] = palette[idx % len(palette)]
        if len(set(int(block[r, c]) for r, c in coords)) < 2:
            continue
        return block
    return None


def _place_horizontal(
    grid: np.ndarray, block: np.ndarray, n_cols: int, n_rows: int
) -> bool:
    h, w = grid.shape
    bh, bw = block.shape
    gap_min = max(4, bh + 3)
    gap_max = min(8, h - 5)
    if gap_min > gap_max:
        return False

    gap = random.randint(gap_min, gap_max)
    candidates = []
    for top in range(1, h - gap - 1):
        bottom = top + gap
        outside = top + (h - 1 - bottom)
        if outside >= max(n_cols, n_rows):
            candidates.append((top, bottom))
    if not candidates:
        return False

    top, bottom = random.choice(candidates)
    grid[top, :] = BLUE
    grid[bottom, :] = BLUE

    row_lo = top + 2
    row_hi = bottom - bh - 1
    if row_lo > row_hi:
        return False
    col_lo = 2
    col_hi = w - bw - 3
    if col_lo > col_hi:
        return False

    r0 = random.randint(row_lo, row_hi)
    c0 = random.randint(col_lo, col_hi)
    mask = block != BLACK
    grid[r0 : r0 + bh, c0 : c0 + bw][mask] = block[mask]

    row_pool = list(range(0, top)) + list(range(bottom + 1, h))
    orange_rows = random.sample(row_pool, n_cols)
    pink_rows = random.sample(row_pool, n_rows)
    grid[orange_rows, 0] = ORANGE
    grid[pink_rows, w - 1] = PINK
    return True


def _place_vertical(
    grid: np.ndarray, block: np.ndarray, n_cols: int, n_rows: int
) -> bool:
    h, w = grid.shape
    bh, bw = block.shape
    gap_min = max(4, bw + 3)
    gap_max = min(8, w - 5)
    if gap_min > gap_max:
        return False

    gap = random.randint(gap_min, gap_max)
    candidates = []
    for left in range(1, w - gap - 1):
        right = left + gap
        outside = left + (w - 1 - right)
        if outside >= max(n_cols, n_rows):
            candidates.append((left, right))
    if not candidates:
        return False

    left, right = random.choice(candidates)
    grid[:, left] = BLUE
    grid[:, right] = BLUE

    col_lo = left + 2
    col_hi = right - bw - 1
    if col_lo > col_hi:
        return False
    row_lo = 2
    row_hi = h - bh - 3
    if row_lo > row_hi:
        return False

    r0 = random.randint(row_lo, row_hi)
    c0 = random.randint(col_lo, col_hi)
    mask = block != BLACK
    grid[r0 : r0 + bh, c0 : c0 + bw][mask] = block[mask]

    col_pool = list(range(0, left)) + list(range(right + 1, w))
    orange_cols = random.sample(col_pool, n_cols)
    pink_cols = random.sample(col_pool, n_rows)
    grid[0, orange_cols] = ORANGE
    grid[h - 1, pink_cols] = PINK
    return True


def _validate_generated_scene(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    assert g.ndim == 2 and max(g.shape) <= 30
    assert set(map(int, np.unique(g))) <= {BLACK, BLUE, RED, GREEN, YELLOW, PINK, ORANGE}

    full_rows = np.flatnonzero(np.all(g == BLUE, axis=1)).tolist()
    full_cols = np.flatnonzero(np.all(g == BLUE, axis=0)).tolist()
    assert (len(full_rows) == 2) ^ (len(full_cols) == 2)
    assert np.count_nonzero(g == BLUE) == (2 * w if len(full_rows) == 2 else 2 * h)

    shape_mask = np.isin(g, SHAPE_COLORS)
    assert 3 <= int(np.count_nonzero(shape_mask)) <= 6
    assert _component_count(shape_mask) == 1
    assert 2 <= len(set(map(int, g[shape_mask].tolist()))) <= 3

    orientation, block, n_rows, n_cols = _parse_scene(g)
    assert (n_rows, n_cols) in COUNT_PAIRS
    assert block.shape[0] >= 2 and block.shape[1] >= 2
    assert int(np.count_nonzero(np.isin(block, SHAPE_COLORS))) == int(np.count_nonzero(shape_mask))

    if orientation == "H":
        top, bottom = map(int, full_rows)
        assert 4 <= bottom - top <= 8
        orange_coords = np.argwhere(g == ORANGE)
        pink_coords = np.argwhere(g == PINK)
        assert len(orange_coords) == n_cols
        assert len(pink_coords) == n_rows
        assert all(int(c) == 0 and (int(r) < top or int(r) > bottom) for r, c in orange_coords)
        assert all(int(c) == w - 1 and (int(r) < top or int(r) > bottom) for r, c in pink_coords)
        shape_rows, shape_cols = np.where(shape_mask)
        assert int(shape_rows.min()) >= top + 2 and int(shape_rows.max()) <= bottom - 2
        assert int(shape_cols.min()) >= 2 and int(shape_cols.max()) <= w - 3
    else:
        left, right = map(int, full_cols)
        assert 4 <= right - left <= 8
        orange_coords = np.argwhere(g == ORANGE)
        pink_coords = np.argwhere(g == PINK)
        assert len(orange_coords) == n_cols
        assert len(pink_coords) == n_rows
        assert all(int(r) == 0 and (int(c) < left or int(c) > right) for r, c in orange_coords)
        assert all(int(r) == h - 1 and (int(c) < left or int(c) > right) for r, c in pink_coords)
        shape_rows, shape_cols = np.where(shape_mask)
        assert int(shape_cols.min()) >= left + 2 and int(shape_cols.max()) <= right - 2
        assert int(shape_rows.min()) >= 2 and int(shape_rows.max()) <= h - 3

    non_guide_mask = (g != BLACK) & (g != BLUE)
    non_guide_box = _bbox(non_guide_mask)
    shape_box = _bbox(shape_mask)
    assert non_guide_box is not None and shape_box is not None
    assert non_guide_box != shape_box, "boundary code must prevent a simpler global crop shortcut"

    out = main(g)
    assert out.shape == (block.shape[0] * n_rows, block.shape[1] * n_cols)
    assert np.array_equal(out, np.tile(block, (n_rows, n_cols)))
    alt_cols = 2 if n_cols != 2 else 3
    alt_rows = 2 if n_rows != 2 else 3
    assert not np.array_equal(out, np.tile(block, (n_rows, alt_cols)))
    assert not np.array_equal(out, np.tile(block, (alt_rows, n_cols)))
    return out


def _gen_attempt(success_index: int) -> np.ndarray | None:
    block = _random_shape_block()
    if block is None:
        return None

    orientation = ORIENTATIONS[success_index % len(ORIENTATIONS)]
    n_cols, n_rows = COUNT_PAIRS[success_index % len(COUNT_PAIRS)]

    h = random.randint(16, 22)
    w = random.randint(16, 22)
    grid = np.zeros((h, w), dtype=int)
    ok = (
        _place_horizontal(grid, block, n_cols, n_rows)
        if orientation == "H"
        else _place_vertical(grid, block, n_cols, n_rows)
    )
    if not ok:
        return None

    out = _validate_generated_scene(grid)
    out_sig = _signature(out)
    if out_sig in _SEEN_OUTPUTS:
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_SUCCESS_COUNT

    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(_GEN_SUCCESS_COUNT)
        if grid is None:
            continue
        _SEEN_OUTPUTS.add(_signature(main(grid)))
        _GEN_SUCCESS_COUNT += 1
        return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
