from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5
SOURCE_COLORS = (BLUE, RED)

GRID_H_RANGE = (13, 19)
GRID_W_RANGE = (13, 19)
SEPARATOR_MARGIN = 5
SPRITE_HEIGHTS = (3, 4, 5)
SPRITE_WIDTHS = (3, 4, 5)
SPRITE_PIXELS = (5, 6, 7, 8)
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int, shape: tuple[int, int]):
    h, w = shape
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(cols) == 1, "expected exactly one full grey separator column"
    return int(cols[0])


def _yellow_pairs(grid: np.ndarray, axis: int) -> list[tuple[int, int, int]]:
    rows: list[tuple[int, int, int]] = []
    left = np.argwhere(grid[:, :axis] == YELLOW)
    right = np.argwhere(grid[:, axis + 1 :] == YELLOW)

    left_by_row: dict[int, list[int]] = {}
    right_by_row: dict[int, list[int]] = {}
    for r, c in left:
        left_by_row.setdefault(int(r), []).append(int(c))
    for r, c in right:
        right_by_row.setdefault(int(r), []).append(int(c) + axis + 1)

    assert set(left_by_row) == set(right_by_row), "yellow endpoints must pair by row"
    for row in sorted(left_by_row):
        assert len(left_by_row[row]) == 1 and len(right_by_row[row]) == 1
        lc = left_by_row[row][0]
        rc = right_by_row[row][0]
        assert lc < axis < rc
        rows.append((row, lc, rc))

    assert 2 <= len(rows) <= 3, "expected two or three yellow bridge rows"
    return rows


def _source_coords(grid: np.ndarray, axis: int) -> list[tuple[int, int]]:
    mask = np.isin(grid[:, :axis], SOURCE_COLORS)
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    assert 5 <= len(coords) <= 8, "expected a small two-color source pattern"
    assert set(map(int, grid[:, axis + 1 :].flatten())).isdisjoint(SOURCE_COLORS)
    return coords


def _swap(color: int) -> int:
    if color == BLUE:
        return RED
    if color == RED:
        return BLUE
    raise AssertionError("unexpected source color")


def _apply_rule(grid: np.ndarray, swap_all_rows: bool = False, swap_no_rows: bool = False) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    axis = _separator_col(g)
    pairs = _yellow_pairs(g, axis)
    source = _source_coords(g, axis)
    activated_rows = {row for row, _lc, _rc in pairs}

    out = g.copy()
    for row, left_c, right_c in pairs:
        for c in range(left_c, right_c + 1):
            if c == axis or int(out[row, c]) in (BG, YELLOW):
                out[row, c] = YELLOW

    for r, c in source:
        source_color = int(g[r, c])
        if swap_no_rows:
            new_color = source_color
        elif swap_all_rows or r in activated_rows:
            new_color = _swap(source_color)
        else:
            new_color = source_color
        out[r, c] = new_color

    for r, c in source:
        mirror_c = 2 * axis - c
        assert 0 <= mirror_c < g.shape[1]
        assert int(g[r, mirror_c]) == BG, "mirrored target must be background in the input"
        assert int(out[r, mirror_c]) == BG, "yellow bridges must not overlap the mirrored copy"
        out[r, mirror_c] = int(out[r, c])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Bridge paired yellow pixels across the mirror axis, then swap source colors only on the activated rows before mirroring.

    Input structure:
    - A black background grid contains exactly one full grey vertical separator column.
    - On the left side there is one irregular source pattern made only of blue and red pixels.
    - Two or three yellow single pixels appear on each side of the separator, paired by sharing a row across the separator.

    Rule:
    - For every cross-separator yellow pair, draw a horizontal yellow bridge between the two endpoints, turning the separator cell on that row yellow.
    - The separator rows turned yellow are the activated rows.
    - Swap blue and red only for source-pattern pixels that lie on activated rows, leaving other source pixels unchanged.
    - Copy the resulting left-side pattern to the right by reflection across the separator; keep the yellow bridges and the non-activated grey separator cells unchanged.

    Key insight:
    - The yellow pairs are row selectors: their bridges activate specific separator rows, and only those rows participate in the color swap before the mirrored copy is created.
    """
    return _apply_rule(grid)


def _random_mask(height: int, width: int, count: int) -> np.ndarray | None:
    mask = np.zeros((height, width), dtype=bool)
    seed = (random.randrange(height), random.randrange(width))
    mask[seed] = True

    while int(mask.sum()) < count:
        frontier = []
        for r, c in np.argwhere(mask):
            for rr, cc in _neighbors4(int(r), int(c), mask.shape):
                if not mask[rr, cc]:
                    frontier.append((rr, cc))
        if not frontier:
            return None
        mask[random.choice(frontier)] = True

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) < 3 or len(cols) < 2:
        return None
    if mask.all():
        return None
    if np.array_equal(mask, np.fliplr(mask)) and np.array_equal(mask, np.flipud(mask)):
        return None
    return mask


def _color_sprite(mask: np.ndarray) -> np.ndarray | None:
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    if len(coords) < 2:
        return None
    sprite = np.zeros(mask.shape, dtype=int)
    blue_count = random.randint(1, len(coords) - 1)
    blue_cells = set(random.sample(coords, blue_count))
    for r, c in coords:
        sprite[r, c] = BLUE if (r, c) in blue_cells else RED
    if set(map(int, sprite[mask])) != set(SOURCE_COLORS):
        return None
    return sprite


def _make_sprite() -> np.ndarray | None:
    height = random.choice(SPRITE_HEIGHTS)
    width = random.choice(SPRITE_WIDTHS)
    count = random.choice(SPRITE_PIXELS)
    if count >= height * width:
        return None
    mask = _random_mask(height, width, count)
    if mask is None:
        return None
    return _color_sprite(mask)


def _source_row_cols(grid: np.ndarray, axis: int) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for r, c in _source_coords(grid, axis):
        out.setdefault(r, []).append(c)
    return out


def _validate_semantics(inp: np.ndarray, out: np.ndarray) -> bool:
    axis = _separator_col(inp)
    pairs = _yellow_pairs(inp, axis)
    activated_rows = {row for row, _lc, _rc in pairs}
    source = _source_coords(inp, axis)

    changed = 0
    unchanged = 0
    mirror_targets = set()
    for r, c in source:
        expected = _swap(int(inp[r, c])) if r in activated_rows else int(inp[r, c])
        if int(out[r, c]) != expected:
            return False
        mirror_c = 2 * axis - c
        mirror_targets.add((r, mirror_c))
        if int(out[r, mirror_c]) != expected:
            return False
        if expected != int(inp[r, c]):
            changed += 1
        else:
            unchanged += 1

    if changed == 0 or unchanged == 0:
        return False

    for r in range(inp.shape[0]):
        if r in activated_rows:
            if int(out[r, axis]) != YELLOW:
                return False
        elif int(out[r, axis]) != GREY:
            return False

    for row, left_c, right_c in pairs:
        for c in range(left_c, right_c + 1):
            if c == axis or int(inp[row, c]) in (BG, YELLOW):
                if int(out[row, c]) != YELLOW:
                    return False

    source_rows = {r for r, _c in source}
    if not (source_rows & activated_rows):
        return False
    if source_rows <= activated_rows:
        return False

    for r in range(inp.shape[0]):
        for c in range(axis + 1, inp.shape[1]):
            if (r, c) in mirror_targets:
                continue
            if int(inp[r, c]) == YELLOW and int(out[r, c]) != YELLOW:
                return False
    return True


def _attempt_input() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    if width < 2 * SEPARATOR_MARGIN + 1:
        return None

    axis = random.randint(SEPARATOR_MARGIN, width - SEPARATOR_MARGIN - 1)
    sprite = _make_sprite()
    if sprite is None:
        return None
    sh, sw = sprite.shape

    min_c0 = max(1, 2 * axis - (width - 1))
    max_c0 = axis - sw - 1
    if min_c0 > max_c0:
        return None
    c0 = random.randint(min_c0, max_c0)
    if height - sh - 1 < 1:
        return None
    r0 = random.randint(1, height - sh - 1)

    grid = np.full((height, width), BG, dtype=int)
    grid[:, axis] = GREY
    mask = sprite != BG
    grid[r0 : r0 + sh, c0 : c0 + sw][mask] = sprite[mask]

    row_cols = _source_row_cols(grid, axis)
    source_rows = sorted(row_cols)
    if len(source_rows) < 3:
        return None

    max_pairs = min(3, len(source_rows) - 1)
    if max_pairs < 2:
        return None
    pair_count = random.choice([n for n in (2, 3) if n <= max_pairs])
    active_rows = sorted(random.sample(source_rows, pair_count))

    for row in active_rows:
        row_max = max(row_cols[row])
        mirror_min = 2 * axis - row_max
        left_choices = list(range(row_max + 1, axis))
        right_choices = list(range(axis + 1, mirror_min))
        if not left_choices or not right_choices:
            return None
        grid[row, random.choice(left_choices)] = YELLOW
        grid[row, random.choice(right_choices)] = YELLOW

    try:
        out = main(grid)
    except Exception:
        return None

    if not _validate_semantics(grid, out):
        return None

    no_swap = _apply_rule(grid, swap_no_rows=True)
    all_swap = _apply_rule(grid, swap_all_rows=True)
    if np.array_equal(out, no_swap) or np.array_equal(out, all_swap):
        return None

    if np.array_equal(out[:, axis + 1 :], 0):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-bridge ARC input")
