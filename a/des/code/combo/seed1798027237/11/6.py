from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5

GEN_MAX_ATTEMPTS = 300
GEN_OUTPUT_SIZE_MIN = 4
GEN_OUTPUT_SIZE_MAX = 8
GEN_BAND_SIZE_MIN = 1
GEN_BAND_SIZE_MAX = 3


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only rows with a red marker in column 0 and columns with a blue marker in row 0,
    then compact the remaining subgrid and erase the markers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    keep_rows = np.flatnonzero(g[:, 0] == RED)
    keep_cols = np.flatnonzero(g[0, :] == BLUE)
    assert keep_rows.size > 0 and keep_cols.size > 0

    out = g[np.ix_(keep_rows, keep_cols)].copy()
    out[(out == RED) | (out == BLUE)] = BLACK
    return out


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    nbrs: list[tuple[int, int]] = []
    if r > 0:
        nbrs.append((r - 1, c))
    if r + 1 < h:
        nbrs.append((r + 1, c))
    if c > 0:
        nbrs.append((r, c - 1))
    if c + 1 < w:
        nbrs.append((r, c + 1))
    return nbrs


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    seen = {start}
    queue: deque[tuple[int, int]] = deque([start])

    while queue:
        r, c = queue.popleft()
        for rr, cc in _neighbors4(r, c, h, w):
            if mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                queue.append((rr, cc))
    return len(seen) == int(mask.sum())


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    outside = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for rr, cc in _neighbors4(r, c, h, w):
            if not mask[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                queue.append((rr, cc))

    filled = mask.copy()
    filled[~mask & ~outside] = True
    return filled


def _shape_bbox(mask: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _valid_shape(mask: np.ndarray) -> bool:
    h, w = mask.shape
    area = int(mask.sum())
    total = h * w
    if area == 0 or area == total:
        return False
    if area < max(4, total // 5):
        return False
    if area > int(total * 0.85):
        return False
    if not _is_connected(mask):
        return False

    span_h, span_w = _shape_bbox(mask)
    if span_h < max(2, h // 2):
        return False
    if span_w < max(2, w // 2):
        return False
    return True


def _make_blob(h: int, w: int) -> np.ndarray:
    target = random.randint(max(4, (h * w) // 3), max(4, (2 * h * w) // 3))
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    frontier = _neighbors4(start[0], start[1], h, w)

    while len(cells) < target:
        if not frontier:
            break
        idx = random.randrange(len(frontier))
        r, c = frontier.pop(idx)
        if (r, c) in cells:
            continue
        cells.add((r, c))
        for rr, cc in _neighbors4(r, c, h, w):
            if (rr, cc) not in cells:
                frontier.append((rr, cc))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    if random.random() < 0.6:
        mask = _fill_holes(mask)
    return mask


def _make_bars(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    horizontal_backbone = random.random() < 0.5

    if horizontal_backbone:
        r = random.randrange(h)
        c0 = random.randrange(max(1, w // 3))
        c1 = random.randrange(max(c0 + 2, w // 2), w)
        mask[r, c0 : c1 + 1] = True

        leg_count = random.randint(1, min(3, c1 - c0 + 1))
        cols = random.sample(list(range(c0, c1 + 1)), leg_count)
        for c in cols:
            r0 = random.randint(0, r)
            r1 = random.randint(r, h - 1)
            mask[r0 : r1 + 1, c] = True

        if random.random() < 0.5:
            c = random.choice(cols)
            r2 = random.randrange(h)
            left = random.randint(0, c)
            right = random.randint(c, w - 1)
            mask[r2, left : right + 1] = True
    else:
        c = random.randrange(w)
        r0 = random.randrange(max(1, h // 3))
        r1 = random.randrange(max(r0 + 2, h // 2), h)
        mask[r0 : r1 + 1, c] = True

        arm_count = random.randint(1, min(3, r1 - r0 + 1))
        rows = random.sample(list(range(r0, r1 + 1)), arm_count)
        for r in rows:
            c0 = random.randint(0, c)
            c1 = random.randint(c, w - 1)
            mask[r, c0 : c1 + 1] = True

        if random.random() < 0.5:
            r = random.choice(rows)
            c2 = random.randrange(w)
            top = random.randint(0, r)
            bottom = random.randint(r, h - 1)
            mask[top : bottom + 1, c2] = True

    return mask


def _make_frame(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    r0 = random.randint(0, max(0, h // 4))
    c0 = random.randint(0, max(0, w // 4))
    r1 = random.randint(max(r0 + 2, h // 2), h - 1)
    c1 = random.randint(max(c0 + 2, w // 2), w - 1)

    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True

    if random.random() < 0.5:
        if random.random() < 0.5:
            rr = random.randint(r0, r1)
            mask[rr, c0 : c1 + 1] = True
        else:
            cc = random.randint(c0, c1)
            mask[r0 : r1 + 1, cc] = True

    return mask


def _generate_shape(h: int, w: int) -> np.ndarray:
    builders = (_make_blob, _make_bars, _make_frame)
    for _ in range(100):
        mask = random.choice(builders)(h, w)
        if _valid_shape(mask):
            return mask
    raise RuntimeError("failed to generate a valid grey shape")


def _band_sizes(count: int) -> list[int]:
    sizes = [random.randint(GEN_BAND_SIZE_MIN, GEN_BAND_SIZE_MAX) for _ in range(count)]
    if all(size == 1 for size in sizes):
        sizes[random.randrange(count)] = random.randint(2, GEN_BAND_SIZE_MAX)
    return sizes


def _band_layout(sizes: list[int]) -> tuple[list[tuple[int, int]], list[int]]:
    bands: list[tuple[int, int]] = []
    keep_positions: list[int] = []
    cursor = 1
    for size in sizes:
        bands.append((cursor, cursor + size))
        keep_positions.append(cursor + random.randrange(size))
        cursor += size
    return bands, keep_positions


def _render_expanded(mask: np.ndarray, row_sizes: list[int], col_sizes: list[int]) -> tuple[np.ndarray, np.ndarray]:
    row_bands, keep_rows = _band_layout(row_sizes)
    col_bands, keep_cols = _band_layout(col_sizes)

    h = 1 + sum(row_sizes)
    w = 1 + sum(col_sizes)
    grid = np.full((h, w), BLACK, dtype=int)

    for i, (r0, r1) in enumerate(row_bands):
        for j, (c0, c1) in enumerate(col_bands):
            if mask[i, j]:
                grid[r0:r1, c0:c1] = GREY

    for r in keep_rows:
        grid[r, 0] = RED
    for c in keep_cols:
        grid[0, c] = BLUE

    expected = np.where(mask, GREY, BLACK).astype(int)
    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    out_h = random.randint(GEN_OUTPUT_SIZE_MIN, GEN_OUTPUT_SIZE_MAX)
    out_w = random.randint(GEN_OUTPUT_SIZE_MIN, GEN_OUTPUT_SIZE_MAX)
    mask = _generate_shape(out_h, out_w)

    row_sizes = _band_sizes(out_h)
    col_sizes = _band_sizes(out_w)
    grid, expected = _render_expanded(mask, row_sizes, col_sizes)

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
