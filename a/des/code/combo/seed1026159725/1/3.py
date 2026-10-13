from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


def _bands(indices: np.ndarray) -> list[tuple[int, int]]:
    assert indices.size > 0, "expected at least one non-separator band"
    vals = [int(x) for x in indices]
    out: list[tuple[int, int]] = []
    start = prev = vals[0]
    for x in vals[1:]:
        if x == prev + 1:
            prev = x
        else:
            out.append((start, prev))
            start = prev = x
    out.append((start, prev))
    return out


def _separator_color(g: np.ndarray) -> int:
    candidates = []
    for color in np.unique(g):
        same = int(color)
        has_full_row = bool(np.any(np.all(g == same, axis=1)))
        has_full_col = bool(np.any(np.all(g == same, axis=0)))
        if has_full_row and has_full_col:
            candidates.append(same)
    assert len(candidates) == 1, f"expected one separator color, got {candidates}"
    return candidates[0]


def _background_color(g: np.ndarray, sep: int) -> int:
    payload = g[g != int(sep)]
    colors, counts = np.unique(payload, return_counts=True)
    max_count = int(counts.max())
    winners = colors[counts == max_count]
    assert len(winners) == 1, "background color must be uniquely dominant"
    return int(winners[0])


def _shape_signature(cell: np.ndarray, bg: int, sep: int) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere((cell != int(bg)) & (cell != int(sep)))
    assert coords.size > 0, "every cell must contain a non-empty pattern"
    coords = coords - coords.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in coords))


def main(grid: np.ndarray) -> np.ndarray:
    """
    In a board partitioned by separator lines, find the only pattern shape that
    appears in exactly two cells and crop the rectangle spanning those cells.

    Each cell's pattern is matched by occupied shape only: non-background,
    non-separator pixels are translation-normalized and compared without using
    their colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    sep = _separator_color(g)
    bg = _background_color(g, sep)
    row_bands = _bands(np.where(~np.all(g == sep, axis=1))[0])
    col_bands = _bands(np.where(~np.all(g == sep, axis=0))[0])

    row_heights = {int(r1 - r0 + 1) for r0, r1 in row_bands}
    col_widths = {int(c1 - c0 + 1) for c0, c1 in col_bands}
    assert len(row_heights) == 1 and len(col_widths) == 1, "cells must be equally sized"

    by_shape: dict[tuple[tuple[int, int], ...], list[tuple[int, int]]] = defaultdict(list)
    for rr, (r0, r1) in enumerate(row_bands):
        for cc, (c0, c1) in enumerate(col_bands):
            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            by_shape[_shape_signature(cell, bg, sep)].append((rr, cc))

    matches = [cells for cells in by_shape.values() if len(cells) == 2]
    assert len(matches) == 1, "expected exactly one duplicated normalized shape"
    pair = matches[0]

    min_r = min(r for r, _ in pair)
    max_r = max(r for r, _ in pair)
    min_c = min(c for _, c in pair)
    max_c = max(c for _, c in pair)

    out_r0 = row_bands[min_r][0]
    out_r1 = row_bands[max_r][1]
    out_c0 = col_bands[min_c][0]
    out_c1 = col_bands[max_c][1]
    return g[out_r0 : out_r1 + 1, out_c0 : out_c1 + 1].copy()


# --------------- generator ---------------

MAX_GRID_SIDE = 30
ROW_COUNT_RANGE = (2, 4)
COL_COUNT_RANGE = (2, 4)
CELL_H_RANGE = (5, 8)
CELL_W_RANGE = (5, 8)
SEP_THICKNESS_CHOICES = (1, 2)
OUTER_BORDER_PROB = 0.5
CELL_MARGIN = 1
MAX_SHAPE_AREA = 8
GENERATION_ATTEMPTS = 400
SHAPE_ATTEMPTS = 200


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _normalize_coords(coords: set[tuple[int, int]]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    norm = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    return norm, int(h), int(w)


def _shape_dims(sig: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return (
        max(r for r, _ in sig) + 1,
        max(c for _, c in sig) + 1,
    )


def _random_shape(max_h: int, max_w: int) -> tuple[tuple[int, int], ...] | None:
    area_hi = min(MAX_SHAPE_AREA, max_h * max_w - 1)
    if area_hi < 3:
        return None
    area = random.randint(3, area_hi)

    start = (random.randrange(max_h), random.randrange(max_w))
    cells = {start}
    frontier = set(_neighbors4(start[0], start[1], max_h, max_w))
    while len(cells) < area:
        if not frontier:
            return None
        nxt = random.choice(tuple(frontier))
        frontier.remove(nxt)
        cells.add(nxt)
        for nb in _neighbors4(nxt[0], nxt[1], max_h, max_w):
            if nb not in cells:
                frontier.add(nb)

    sig, h, w = _normalize_coords(cells)
    if h == 1 and w == 1:
        return None
    return sig


def _valid_offsets(cell_h: int, cell_w: int, shape_h: int, shape_w: int) -> list[tuple[int, int]]:
    return [
        (r0, c0)
        for r0 in range(CELL_MARGIN, cell_h - shape_h - CELL_MARGIN + 1)
        for c0 in range(CELL_MARGIN, cell_w - shape_w - CELL_MARGIN + 1)
    ]


def _paint_shape(
    sig: tuple[tuple[int, int], ...],
    cell_h: int,
    cell_w: int,
    bg: int,
    palette: list[int],
    *,
    offset: tuple[int, int] | None = None,
) -> np.ndarray:
    patch = np.full((cell_h, cell_w), int(bg), dtype=int)
    shape_h, shape_w = _shape_dims(sig)
    offsets = _valid_offsets(cell_h, cell_w, shape_h, shape_w)
    assert offsets, "shape must fit with a clear margin"
    base_r, base_c = random.choice(offsets) if offset is None else offset

    n_colors = 1 if len(sig) == 1 or random.random() < 0.55 else 2
    colors = random.sample(palette, n_colors)
    if n_colors == 1:
        assignments = [colors[0]] * len(sig)
    else:
        assignments = [random.choice(colors) for _ in sig]
        if len(set(assignments)) == 1:
            assignments[0] = colors[0]
            assignments[-1] = colors[1]

    for (dr, dc), color in zip(sig, assignments):
        patch[base_r + dr, base_c + dc] = int(color)
    return patch


def _choose_layout() -> tuple[int, int, int, int, int, bool] | None:
    rows = random.randint(*ROW_COUNT_RANGE)
    cols = random.randint(*COL_COUNT_RANGE)
    sep = random.choice(SEP_THICKNESS_CHOICES)
    outer_border = random.random() < OUTER_BORDER_PROB

    row_sep_count = rows - 1 + (2 if outer_border else 0)
    col_sep_count = cols - 1 + (2 if outer_border else 0)
    max_cell_h = min(CELL_H_RANGE[1], (MAX_GRID_SIDE - row_sep_count * sep) // rows)
    max_cell_w = min(CELL_W_RANGE[1], (MAX_GRID_SIDE - col_sep_count * sep) // cols)
    if max_cell_h < CELL_H_RANGE[0] or max_cell_w < CELL_W_RANGE[0]:
        return None

    cell_h = random.randint(CELL_H_RANGE[0], max_cell_h)
    cell_w = random.randint(CELL_W_RANGE[0], max_cell_w)
    return rows, cols, cell_h, cell_w, sep, outer_border


def _build_grid(
    rows: int,
    cols: int,
    cell_h: int,
    cell_w: int,
    sep: int,
    outer_border: bool,
    bg: int,
    sep_color: int,
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    total_h = rows * cell_h + sep * (rows - 1 + (2 if outer_border else 0))
    total_w = cols * cell_w + sep * (cols - 1 + (2 if outer_border else 0))
    grid = np.full((total_h, total_w), int(bg), dtype=int)

    row_bands: list[tuple[int, int]] = []
    col_bands: list[tuple[int, int]] = []

    r = 0
    if outer_border:
        grid[r : r + sep, :] = sep_color
        r += sep
    for idx in range(rows):
        row_bands.append((r, r + cell_h - 1))
        r += cell_h
        if idx != rows - 1:
            grid[r : r + sep, :] = sep_color
            r += sep
    if outer_border:
        grid[r : r + sep, :] = sep_color

    c = 0
    if outer_border:
        grid[:, c : c + sep] = sep_color
        c += sep
    for idx in range(cols):
        col_bands.append((c, c + cell_w - 1))
        c += cell_w
        if idx != cols - 1:
            grid[:, c : c + sep] = sep_color
            c += sep
    if outer_border:
        grid[:, c : c + sep] = sep_color

    return grid, row_bands, col_bands


def _choose_pair_positions(rows: int, cols: int) -> tuple[tuple[int, int], tuple[int, int]]:
    branch = random.choice(("same_row", "same_col", "diag"))
    if branch == "same_row":
        r = random.randrange(rows)
        c0 = random.randrange(cols - 1)
        c1 = random.randrange(c0 + 1, cols)
        return (r, c0), (r, c1)
    if branch == "same_col":
        c = random.randrange(cols)
        r0 = random.randrange(rows - 1)
        r1 = random.randrange(r0 + 1, rows)
        return (r0, c), (r1, c)

    r0 = random.randrange(rows - 1)
    c0 = random.randrange(cols - 1)
    r1 = random.randrange(r0 + 1, rows)
    c1 = random.randrange(c0 + 1, cols)
    return (r0, c0), (r1, c1)


def _sample_unique_shapes(count: int, usable_h: int, usable_w: int) -> list[tuple[tuple[int, int], ...]] | None:
    shapes: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for _ in range(SHAPE_ATTEMPTS * count):
        sig = _random_shape(usable_h, usable_w)
        if sig is None or sig in seen:
            continue
        seen.add(sig)
        shapes.append(sig)
        if len(shapes) == count:
            return shapes
    return None


def _crop_expected(
    grid: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    pair: tuple[tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    (r0, c0), (r1, c1) = pair
    rr0 = row_bands[min(r0, r1)][0]
    rr1 = row_bands[max(r0, r1)][1]
    cc0 = col_bands[min(c0, c1)][0]
    cc1 = col_bands[max(c0, c1)][1]
    return grid[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    layout = _choose_layout()
    if layout is None:
        raise AssertionError("layout selection failed")
    rows, cols, cell_h, cell_w, sep, outer_border = layout

    colors = random.sample(range(10), 2)
    bg, sep_color = colors[0], colors[1]
    palette = [color for color in range(10) if color not in {bg, sep_color}]

    grid, row_bands, col_bands = _build_grid(
        rows, cols, cell_h, cell_w, sep, outer_border, bg, sep_color
    )
    pair = _choose_pair_positions(rows, cols)

    usable_h = cell_h - 2 * CELL_MARGIN
    usable_w = cell_w - 2 * CELL_MARGIN
    shapes = _sample_unique_shapes(rows * cols - 1, usable_h, usable_w)
    if shapes is None:
        raise AssertionError("shape sampling failed")

    pair_shape = shapes[0]
    other_shapes = shapes[1:]
    pair_same_colors = random.random() < 0.5
    pair_shape_h, pair_shape_w = _shape_dims(pair_shape)
    pair_offsets = _valid_offsets(cell_h, cell_w, pair_shape_h, pair_shape_w)
    offset_a = random.choice(pair_offsets)
    if len(pair_offsets) >= 2 and random.random() < 0.7:
        offset_b_choices = [off for off in pair_offsets if off != offset_a]
        offset_b = random.choice(offset_b_choices) if offset_b_choices else offset_a
    else:
        offset_b = random.choice(pair_offsets)

    shape_iter = iter(other_shapes)
    first_pair, second_pair = pair
    first_pair_patch: np.ndarray | None = None
    for rr, (r0, r1) in enumerate(row_bands):
        for cc, (c0, c1) in enumerate(col_bands):
            if (rr, cc) == first_pair:
                patch = _paint_shape(
                    pair_shape, cell_h, cell_w, bg, palette, offset=offset_a
                )
                first_pair_patch = patch.copy()
            elif (rr, cc) == second_pair:
                patch = _paint_shape(
                    pair_shape, cell_h, cell_w, bg, palette, offset=offset_b
                )
                if not pair_same_colors:
                    attempts = 0
                    while (
                        attempts < 20
                        and first_pair_patch is not None
                        and np.array_equal(patch, first_pair_patch)
                    ):
                        patch = _paint_shape(
                            pair_shape, cell_h, cell_w, bg, palette, offset=offset_b
                        )
                        attempts += 1
            else:
                patch = _paint_shape(next(shape_iter), cell_h, cell_w, bg, palette)
            grid[r0 : r1 + 1, c0 : c1 + 1] = patch

    expected = _crop_expected(grid, row_bands, col_bands, pair)
    solved = main(grid)
    assert np.array_equal(solved, expected)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        try:
            inp, _ = _generate_case()
            return inp
        except AssertionError:
            continue
    raise AssertionError("failed to generate a valid input")
