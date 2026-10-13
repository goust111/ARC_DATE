from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

INPUT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
PANEL_TYPES = ("V", "H", "B", "N")
MAX_GENERATION_ATTEMPTS = 300


def _bands_from_separator_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    n = int(mask.size)
    pos = 0
    while pos < n:
        if bool(mask[pos]):
            return []
        start = pos
        while pos < n and not bool(mask[pos]):
            pos += 1
        if start == pos:
            return []
        bands.append((start, pos))
        while pos < n and bool(mask[pos]):
            pos += 1
        if pos == n and bool(mask[n - 1]):
            return []
    return bands


def _try_partition(grid: np.ndarray, separator: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    row_sep = np.all(grid == separator, axis=1)
    col_sep = np.all(grid == separator, axis=0)
    if not bool(row_sep.any()) or not bool(col_sep.any()):
        return None

    row_bands = _bands_from_separator_mask(row_sep)
    col_bands = _bands_from_separator_mask(col_sep)
    if len(row_bands) < 2 or len(col_bands) < 2:
        return None

    heights = {r1 - r0 for r0, r1 in row_bands}
    widths = {c1 - c0 for c0, c1 in col_bands}
    if len(heights) != 1 or heights != widths:
        return None

    return row_bands, col_bands


def _parse_panels(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    bg = BLACK
    candidates: list[tuple[int, list[tuple[int, int]], list[tuple[int, int]]]] = []
    for color in np.unique(grid):
        sep = int(color)
        if sep == bg:
            continue
        partition = _try_partition(grid, sep)
        if partition is not None:
            row_bands, col_bands = partition
            score = int(np.count_nonzero(np.all(grid == sep, axis=1))) + int(
                np.count_nonzero(np.all(grid == sep, axis=0))
            )
            candidates.append((score, row_bands, col_bands))

    assert candidates, "failed to infer separator color and panel partition"
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, row_bands, col_bands = candidates[0]
    return bg, row_bands, col_bands


def _classify_mask(mask: np.ndarray) -> str:
    vertical = bool(np.array_equal(mask, np.fliplr(mask)))
    horizontal = bool(np.array_equal(mask, np.flipud(mask)))
    if vertical and horizontal:
        return "B"
    if vertical:
        return "V"
    if horizontal:
        return "H"
    return "N"


def _paint_symbol(out: np.ndarray, panel_r: int, panel_c: int, kind: str) -> None:
    r0 = panel_r * 3
    c0 = panel_c * 3
    if kind == "V":
        out[r0 : r0 + 3, c0 + 1] = RED
    elif kind == "H":
        out[r0 + 1, c0 : c0 + 3] = BLUE
    elif kind == "B":
        out[r0 : r0 + 3, c0 + 1] = GREEN
        out[r0 + 1, c0 : c0 + 3] = GREEN
    else:
        out[r0 + 1, c0 + 1] = YELLOW


def main(grid: np.ndarray) -> np.ndarray:
    """
    Encode each panel's reflection-axis type as a 3x3 symbol.

    Input structure
    The canvas has black background pixels and grey separator rows and columns
    that partition it into equal square panels. Each panel contains a
    single-color foreground pattern on black.

    Rule
    For each panel, inspect the foreground mask inside the full panel bounds:
    - vertical symmetry only -> draw a red vertical line
    - horizontal symmetry only -> draw a blue horizontal line
    - both symmetries -> draw a green cross
    - neither symmetry -> draw a yellow center dot

    Output
    Each input panel becomes one 3x3 output cell, with no separator lines.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg, row_bands, col_bands = _parse_panels(g)
    out = np.full((len(row_bands) * 3, len(col_bands) * 3), BLACK, dtype=int)

    for pr, (r0, r1) in enumerate(row_bands):
        for pc, (c0, c1) in enumerate(col_bands):
            panel = g[r0:r1, c0:c1]
            mask = panel != bg
            assert bool(mask.any()), "each panel must contain a foreground pattern"
            kind = _classify_mask(mask)
            _paint_symbol(out, pr, pc, kind)

    return out


def _sample_cluster(rows: int, cols: int, count: int) -> set[tuple[int, int]]:
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    start = random.choice(cells)
    points = {start}
    while len(points) < count:
        if random.random() < 0.75:
            base_r, base_c = random.choice(tuple(points))
            neighbors = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = base_r + dr
                nc = base_c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbors.append((nr, nc))
            if neighbors:
                points.add(random.choice(neighbors))
                continue
        points.add(random.choice(cells))
    return points


def _symmetrized_mask(size: int, reps: set[tuple[int, int]], vertical: bool, horizontal: bool) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    for r, c in reps:
        row_targets = {r, size - 1 - r} if horizontal else {r}
        col_targets = {c, size - 1 - c} if vertical else {c}
        for rr in row_targets:
            for cc in col_targets:
                mask[int(rr), int(cc)] = True
    return mask


def _good_mask(mask: np.ndarray) -> bool:
    count = int(mask.sum())
    size = int(mask.shape[0])
    area = size * size
    if count < 3 or count > int(area * 0.6):
        return False
    pts = np.argwhere(mask)
    height = int(pts[:, 0].max() - pts[:, 0].min() + 1)
    width = int(pts[:, 1].max() - pts[:, 1].min() + 1)
    return height >= 2 and width >= 2


def _make_mask(size: int, kind: str) -> np.ndarray:
    row_reps_both = (size + 1) // 2
    col_reps_both = (size + 1) // 2
    row_reps_v = size
    col_reps_v = (size + 1) // 2
    row_reps_h = (size + 1) // 2
    col_reps_h = size

    for _ in range(200):
        if kind == "B":
            count = random.randint(2, max(2, row_reps_both + col_reps_both - 1))
            reps = _sample_cluster(row_reps_both, col_reps_both, count)
            mask = _symmetrized_mask(size, reps, vertical=True, horizontal=True)
        elif kind == "V":
            count = random.randint(2, max(2, size))
            reps = _sample_cluster(row_reps_v, col_reps_v, count)
            mask = _symmetrized_mask(size, reps, vertical=True, horizontal=False)
        elif kind == "H":
            count = random.randint(2, max(2, size))
            reps = _sample_cluster(row_reps_h, col_reps_h, count)
            mask = _symmetrized_mask(size, reps, vertical=False, horizontal=True)
        else:
            count = random.randint(3, max(3, size + 1))
            reps = _sample_cluster(size, size, count)
            mask = np.zeros((size, size), dtype=bool)
            for r, c in reps:
                mask[r, c] = True

        if _classify_mask(mask) == kind and _good_mask(mask):
            return mask

    raise RuntimeError(f"failed to generate a {kind} mask")


def _panel_origin(panel_r: int, panel_c: int, panel_size: int) -> tuple[int, int]:
    return panel_r * (panel_size + 1), panel_c * (panel_size + 1)


def _gen_attempt() -> np.ndarray:
    panel_rows = random.randint(2, 4)
    panel_cols = random.randint(2, 4)
    panel_size = random.randint(4, 7)

    height = panel_rows * panel_size + (panel_rows - 1)
    width = panel_cols * panel_size + (panel_cols - 1)
    grid = np.full((height, width), BLACK, dtype=int)

    for split_r in range(1, panel_rows):
        grid[split_r * panel_size + (split_r - 1), :] = GREY
    for split_c in range(1, panel_cols):
        grid[:, split_c * panel_size + (split_c - 1)] = GREY

    total_panels = panel_rows * panel_cols
    kinds = [PANEL_TYPES[idx % len(PANEL_TYPES)] for idx in range(total_panels)]
    random.shuffle(kinds)

    for idx, kind in enumerate(kinds):
        pr = idx // panel_cols
        pc = idx % panel_cols
        color = random.choice(INPUT_COLORS)
        mask = _make_mask(panel_size, kind)
        r0, c0 = _panel_origin(pr, pc, panel_size)
        panel = grid[r0 : r0 + panel_size, c0 : c0 + panel_size]
        panel[mask] = color

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        try:
            out = main(grid)
        except AssertionError:
            continue
        if out.shape[0] == 0 or out.shape[1] == 0:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
