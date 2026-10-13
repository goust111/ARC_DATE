from __future__ import annotations

import random
from collections import deque

import numpy as np


ROW_PROBE_COLOR = 7
COL_PROBE_COLOR = 8
GEN_RETRIES = 300


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _mosaic_bbox(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    comps = _components(grid != bg)
    assert comps, "expected a non-background mosaic"

    mosaic = max(comps, key=len)
    r_vals = [r for r, _ in mosaic]
    c_vals = [c for _, c in mosaic]
    r0, r1 = min(r_vals), max(r_vals)
    c0, c1 = min(c_vals), max(c_vals)
    side_r = r1 - r0 + 1
    side_c = c1 - c0 + 1

    assert side_r == side_c, "mosaic must be square"
    assert len(mosaic) == side_r * side_c, "mosaic must fully fill its bounding box"
    assert 4 <= side_r <= 8, "mosaic side must be within the intended range"

    for comp in comps:
        if comp is mosaic:
            continue
        assert len(comp) == 1, "all non-mosaic foreground objects must be singleton probes"

    return r0, c0, r1, c1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sample a color mosaic using row probes on the right and column probes below.

    Input structure:
    - bg is the most common color.
    - One large non-background square component is a dense NxN mosaic.
    - Singleton probes to the right mark sampled rows.
    - Singleton probes below mark sampled columns.

    Rule:
    - Sort right-side probes by row; each probe selects the mosaic row at the same offset.
    - Sort bottom probes by column; each probe selects the mosaic column at the same offset.
    - Output the submatrix formed by the row/column intersections.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    r0, c0, r1, c1 = _mosaic_bbox(g, bg)
    mosaic = g[r0 : r1 + 1, c0 : c1 + 1]

    row_probes: list[tuple[int, int]] = []
    col_probes: list[tuple[int, int]] = []

    for r, c in np.argwhere(g != bg):
        rr = int(r)
        cc = int(c)
        inside_mosaic = r0 <= rr <= r1 and c0 <= cc <= c1
        if inside_mosaic:
            continue
        if r0 <= rr <= r1 and cc > c1:
            row_probes.append((rr, cc))
        elif c0 <= cc <= c1 and rr > r1:
            col_probes.append((rr, cc))
        else:
            raise AssertionError("foreground singleton outside the expected probe zones")

    assert row_probes, "expected at least one row probe"
    assert col_probes, "expected at least one column probe"

    row_probes.sort(key=lambda p: (p[0], p[1]))
    col_probes.sort(key=lambda p: (p[1], p[0]))

    row_idxs = [r - r0 for r, _ in row_probes]
    col_idxs = [c - c0 for _, c in col_probes]

    assert len(set(row_idxs)) == len(row_idxs), "row probes must select distinct mosaic rows"
    assert len(set(col_idxs)) == len(col_idxs), "column probes must select distinct mosaic columns"

    out = np.empty((len(row_idxs), len(col_idxs)), dtype=int)
    for i, mr in enumerate(row_idxs):
        for j, mc in enumerate(col_idxs):
            out[i, j] = int(mosaic[mr, mc])
    return out


def _random_mosaic(side: int, palette: list[int]) -> np.ndarray:
    total = side * side
    n_colors = random.randint(3, min(6, len(palette), total))
    used = random.sample(palette, n_colors)
    cells = used[:]
    while len(cells) < total:
        cells.append(random.choice(used))
    random.shuffle(cells)
    return np.array(cells, dtype=int).reshape(side, side)


def _expected_output(mosaic: np.ndarray, row_offsets: list[int], col_offsets: list[int]) -> np.ndarray:
    out = np.empty((len(row_offsets), len(col_offsets)), dtype=int)
    for i, r in enumerate(row_offsets):
        for j, c in enumerate(col_offsets):
            out[i, j] = int(mosaic[r, c])
    return out


def _generate_once() -> np.ndarray | None:
    side = random.randint(4, 8)
    row_count = random.randint(1, side)
    col_count = random.randint(1, side)

    top_margin = random.randint(0, 3)
    left_margin = random.randint(0, 3)
    right_margin = random.randint(3, 8)
    bottom_margin = random.randint(3, 8)

    height = top_margin + side + bottom_margin
    width = left_margin + side + right_margin
    r0 = top_margin
    c0 = left_margin
    r1 = r0 + side - 1
    c1 = c0 + side - 1

    if width - (c1 + 2) < 1 or height - (r1 + 2) < 1:
        return None

    colors = list(range(10))
    bg_choices = [c for c in colors if c not in (ROW_PROBE_COLOR, COL_PROBE_COLOR)]
    bg = random.choice(bg_choices)
    palette = [c for c in colors if c not in (bg, ROW_PROBE_COLOR, COL_PROBE_COLOR)]
    if len(palette) < 3:
        return None

    grid = np.full((height, width), bg, dtype=int)
    mosaic = _random_mosaic(side, palette)
    grid[r0 : r1 + 1, c0 : c1 + 1] = mosaic

    row_offsets = sorted(random.sample(range(side), row_count))
    col_offsets = sorted(random.sample(range(side), col_count))

    row_probe_cols = [random.randint(c1 + 2, width - 1) for _ in row_offsets]
    col_probe_rows = [random.randint(r1 + 2, height - 1) for _ in col_offsets]

    for row_offset, probe_c in zip(row_offsets, row_probe_cols, strict=False):
        grid[r0 + row_offset, probe_c] = ROW_PROBE_COLOR
    for col_offset, probe_r in zip(col_offsets, col_probe_rows, strict=False):
        grid[probe_r, c0 + col_offset] = COL_PROBE_COLOR

    expected = _expected_output(mosaic, row_offsets, col_offsets)

    counts = np.bincount(grid.ravel(), minlength=10)
    if int(np.argmax(counts)) != bg:
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid probe-sampling mosaic")
