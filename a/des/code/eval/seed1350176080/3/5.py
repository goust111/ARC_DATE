from __future__ import annotations

import numpy as np


BG = 0
GREY = 5
AZURE = 8
SEPARATOR = 9

_RNG = np.random.default_rng()
_SEED_COLORS = np.array([1, 2, 3, 4, 6, 7], dtype=int)
_DIRECTIONS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
_MAX_GENERATION_ATTEMPTS = 300


def _separator_positions(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    height, width = map(int, grid.shape)
    candidates: list[tuple[int, list[int], list[int]]] = []
    for color in map(int, np.unique(grid)):
        rows = [r for r in range(height) if np.all(grid[r, :] == color)]
        cols = [c for c in range(width) if np.all(grid[:, c] == color)]
        if rows or cols:
            candidates.append((color, rows, cols))

    both = [cand for cand in candidates if cand[1] and cand[2]]
    if len(both) == 1:
        return both[0]
    if both:
        return max(both, key=lambda item: (len(item[1]) + len(item[2]), item[0] == SEPARATOR))
    assert candidates, "expected at least one separator-line candidate"
    return max(candidates, key=lambda item: (len(item[1]) + len(item[2]), item[0] == SEPARATOR))


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for sep in sorted(int(x) for x in separators):
        if start < sep:
            spans.append((start, sep))
        start = sep + 1
    if start < length:
        spans.append((start, length))
    return spans


def _seed_info(region: np.ndarray, separator_color: int) -> tuple[int, int, int, int, int]:
    seed_candidates: list[tuple[int, int, int, int, int]] = []
    blocked = {BG, GREY, separator_color}
    for color in map(int, np.unique(region)):
        if color in blocked:
            continue
        coords = np.argwhere(region == color)
        if int(coords.shape[0]) != 3:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if int(r1 - r0) != 1 or int(c1 - c0) != 1:
            continue
        local = {tuple(map(int, cell - np.array([r0, c0], dtype=int))) for cell in coords}
        missing = {(0, 0), (0, 1), (1, 0), (1, 1)} - local
        if len(missing) != 1:
            continue
        missing_r, missing_c = next(iter(missing))
        dr = 1 if missing_r == 1 else -1
        dc = 1 if missing_c == 1 else -1
        start_r = int(r0 if dr == 1 else r1)
        start_c = int(c0 if dc == 1 else c1)
        seed_candidates.append((color, dr, dc, start_r, start_c))
    assert len(seed_candidates) == 1, "expected exactly one L-shaped seed per region"
    return seed_candidates[0]


def _fill_region(region: np.ndarray, separator_color: int) -> np.ndarray:
    out = region.copy()
    seed_color, dr, dc, start_r, start_c = _seed_info(region, separator_color)
    height, width = map(int, region.shape)
    row_iter = range(start_r, height, 1) if dr == 1 else range(start_r, -1, -1)
    col_template = list(range(start_c, width, 1) if dc == 1 else range(start_c, -1, -1))

    for r in row_iter:
        downgraded = False
        for c in col_template:
            value = int(region[r, c])
            if value == GREY:
                if downgraded:
                    break
                downgraded = True
                continue
            if value == BG:
                out[r, c] = AZURE if downgraded else seed_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each separator-delimited region from its L seed, downgrading past grey blockers."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    separator_color, sep_rows, sep_cols = _separator_positions(g)
    row_spans = _spans(int(g.shape[0]), sep_rows)
    col_spans = _spans(int(g.shape[1]), sep_cols)

    out = g.copy()
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            region = g[r0:r1, c0:c1]
            out[r0:r1, c0:c1] = _fill_region(region, separator_color)
    return out


def _cycle_choices(values: list, count: int) -> list:
    order = list(values)
    _RNG.shuffle(order)
    out = order[:]
    while len(out) < count:
        out.append(values[int(_RNG.integers(len(values)))])
    _RNG.shuffle(out)
    return out[:count]


def _place_seed(region: np.ndarray, color: int, dr: int, dc: int) -> None:
    height, width = map(int, region.shape)
    if dr == 1 and dc == 1:
        cells = ((0, 0), (0, 1), (1, 0))
    elif dr == 1 and dc == -1:
        cells = ((0, width - 2), (0, width - 1), (1, width - 1))
    elif dr == -1 and dc == 1:
        cells = ((height - 2, 0), (height - 1, 0), (height - 1, 1))
    else:
        cells = ((height - 2, width - 1), (height - 1, width - 2), (height - 1, width - 1))
    for r, c in cells:
        region[r, c] = color


def _place_obstacles(region: np.ndarray, dr: int, dc: int, branch: int) -> None:
    if branch == 0:
        return

    height, width = map(int, region.shape)
    row_choices = list(range(2, height)) if dr == 1 else list(range(0, height - 2))
    assert row_choices, "region must leave at least one obstacle row beyond the seed"
    row = int(row_choices[int(_RNG.integers(len(row_choices)))])
    order = list(range(width)) if dc == 1 else list(range(width - 1, -1, -1))

    if branch == 1:
        idx1 = int(_RNG.integers(1, width - 1))
        region[row, order[idx1]] = GREY
        return

    idx1 = int(_RNG.integers(1, width - 3))
    idx2 = int(_RNG.integers(idx1 + 2, width - 1))
    region[row, order[idx1]] = GREY
    region[row, order[idx2]] = GREY


def _build_grid(
    row_heights: list[int],
    col_widths: list[int],
    directions: list[tuple[int, int]],
    branches: list[int],
    colors: np.ndarray,
) -> np.ndarray:
    height = int(sum(row_heights) + len(row_heights) - 1)
    width = int(sum(col_widths) + len(col_widths) - 1)
    grid = np.full((height, width), BG, dtype=int)

    row_spans: list[tuple[int, int]] = []
    cursor = 0
    for i, size in enumerate(row_heights):
        row_spans.append((cursor, cursor + int(size)))
        cursor += int(size)
        if i + 1 < len(row_heights):
            grid[cursor, :] = SEPARATOR
            cursor += 1

    col_spans: list[tuple[int, int]] = []
    cursor = 0
    for i, size in enumerate(col_widths):
        col_spans.append((cursor, cursor + int(size)))
        cursor += int(size)
        if i + 1 < len(col_widths):
            grid[:, cursor] = SEPARATOR
            cursor += 1

    idx = 0
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            region = np.full((r1 - r0, c1 - c0), BG, dtype=int)
            dr, dc = directions[idx]
            _place_seed(region, int(colors[idx]), dr, dc)
            _place_obstacles(region, dr, dc, int(branches[idx]))
            grid[r0:r1, c0:c1] = region
            idx += 1
    return grid


def _region_output_checks(inp: np.ndarray, out: np.ndarray, branches: list[int]) -> bool:
    separator_color, sep_rows, sep_cols = _separator_positions(inp)
    if separator_color != SEPARATOR:
        return False
    if not np.all(out[sep_rows, :] == SEPARATOR):
        return False
    if not np.all(out[:, sep_cols] == SEPARATOR):
        return False

    row_spans = _spans(int(inp.shape[0]), sep_rows)
    col_spans = _spans(int(inp.shape[1]), sep_cols)
    idx = 0
    saw_azure = False
    saw_tail_bg = False
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            region_in = inp[r0:r1, c0:c1]
            region_out = out[r0:r1, c0:c1]
            _, dr, dc, start_r, start_c = _seed_info(region_in, separator_color)
            row_iter = range(start_r, region_in.shape[0], 1) if dr == 1 else range(start_r, -1, -1)
            col_order = list(range(start_c, region_in.shape[1], 1) if dc == 1 else range(start_c, -1, -1))
            branch = int(branches[idx])
            if branch > 0 and np.any(region_out == AZURE):
                saw_azure = True
            if branch == 2:
                for rr in row_iter:
                    seen = 0
                    after_second = []
                    for cc in col_order:
                        if int(region_in[rr, cc]) == GREY:
                            seen += 1
                            continue
                        if seen >= 2:
                            after_second.append((rr, cc))
                    if after_second and all(int(region_out[rr, cc]) == BG for rr, cc in after_second):
                        saw_tail_bg = True
                        break
            idx += 1
    return saw_azure and saw_tail_bg


def _generate_attempt() -> np.ndarray | None:
    row_count = int(_RNG.integers(2, 4))
    col_count = int(_RNG.integers(2, 4))
    row_heights = [int(x) for x in _RNG.integers(5, 10, size=row_count)]
    col_widths = [int(x) for x in _RNG.integers(5, 10, size=col_count)]

    region_count = row_count * col_count
    directions = _cycle_choices(_DIRECTIONS, region_count)
    branches = _cycle_choices([0, 1, 2], region_count)
    colors = _RNG.choice(_SEED_COLORS, size=region_count, replace=True)

    grid = _build_grid(row_heights, col_widths, directions, branches, colors)
    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    if not _region_output_checks(grid, out, branches):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
