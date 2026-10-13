from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox, crop_bbox, get_bg, paste


def _regions_between(size: int, separator_lines: list[int]) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    start = 0
    for line in sorted(separator_lines):
        assert start < line < size - 1, "separator lines must split nonempty regions"
        regions.append((start, line - 1))
        start = line + 1
    assert start < size, "expected a trailing region after separator lines"
    regions.append((start, size - 1))
    return regions


def _extract_cells(
    grid: np.ndarray, bg: int
) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], list[np.ndarray]]:
    h, w = map(int, grid.shape)
    separator_candidates: list[tuple[int, list[int], list[int]]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        row_lines = [r for r in range(h) if np.all(grid[r, :] == color)]
        col_lines = [c for c in range(w) if np.all(grid[:, c] == color)]
        if row_lines or col_lines:
            separator_candidates.append((color, row_lines, col_lines))

    assert len(separator_candidates) == 1, "expected one separator color"
    separator, row_lines, col_lines = separator_candidates[0]

    row_regions = _regions_between(h, row_lines) if row_lines else [(0, h - 1)]
    col_regions = _regions_between(w, col_lines) if col_lines else [(0, w - 1)]
    assert len(row_regions) * len(col_regions) >= 3, "expected at least three cells"
    assert len({r1 - r0 + 1 for r0, r1 in row_regions}) == 1, "cell heights must match"
    assert len({c1 - c0 + 1 for c0, c1 in col_regions}) == 1, "cell widths must match"

    cells: list[np.ndarray] = []
    for r0, r1 in row_regions:
        for c0, c1 in col_regions:
            block = crop_bbox(grid, (r0, c0, r1, c1))
            assert not np.any(block == separator), "separator color must stay on separator lines"
            cells.append(block)

    return separator, row_regions, col_regions, cells


def _pairwise_conflict_count(a: np.ndarray, b: np.ndarray, bg: int) -> int:
    mask = (a != bg) & (b != bg) & (a != b)
    return int(np.count_nonzero(mask))


def _find_impostor_index(cells: list[np.ndarray], bg: int) -> int:
    n = len(cells)
    assert n >= 3
    counts = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            cnt = _pairwise_conflict_count(cells[i], cells[j], bg)
            counts[i, j] = cnt
            counts[j, i] = cnt

    conflict_degree = np.count_nonzero(counts > 0, axis=1)
    best = int(conflict_degree.max())
    assert best >= 2, "expected one cell to conflict with multiple peers"
    candidates = np.flatnonzero(conflict_degree == best)
    assert len(candidates) == 1, "expected a unique impostor cell"
    return int(candidates[0])


def _union_consistent_cells(cells: list[np.ndarray], bg: int) -> np.ndarray:
    h, w = map(int, cells[0].shape)
    out = np.full((h, w), bg, dtype=int)
    for r in range(h):
        for c in range(w):
            colors = {int(cell[r, c]) for cell in cells if int(cell[r, c]) != bg}
            assert len(colors) <= 1, "surviving cells must agree where they overlap"
            if colors:
                out[r, c] = colors.pop()
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the one conflicting cell from a separator grid and merge the remaining partial copies into the repaired single-cell pattern.

    Input structure:
    - Solid separator rows and/or columns divide the canvas into equal-sized rectangular cells.
    - All but one cells contain incomplete copies of the same pattern on aligned local coordinates.
    - One cell is an impostor pattern whose visible non-background pixels disagree in color with other cells at some matching positions.

    Rule:
    - Detect the separator color and split the canvas into its aligned cells.
    - Identify the unique cell that conflicts with multiple other cells wherever both show non-background colors, and discard that whole cell.
    - For each remaining local coordinate, keep the unique non-background color visible in any survivor; if all survivors show background there, leave it background.

    Key insight:
    - No single cell is complete, so the answer comes from cross-cell agreement: the good partial copies are mutually consistent, while the impostor is the only tile involved in repeated color conflicts.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = int(get_bg(g))

    _separator, _row_regions, _col_regions, cells = _extract_cells(g, bg)
    impostor_idx = _find_impostor_index(cells, bg)
    survivors = [cell for i, cell in enumerate(cells) if i != impostor_idx]
    assert len(survivors) >= 2

    out = _union_consistent_cells(survivors, bg)
    assert np.any(out != bg), "the repaired pattern must contain foreground"
    assert all(not np.array_equal(cell, out) for cell in survivors), "no surviving cell should be complete"
    return out


# --------------- generator ---------------

GEN_ATTEMPTS = 400
GEN_LAYOUTS = ((1, 3), (1, 4), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3), (4, 1))
GEN_CELL_MIN = 4
GEN_CELL_MAX = 8
GEN_PATTERN_MIN_PIXELS = 6
GEN_PATTERN_MAX_PIXELS = 12
GEN_PATTERN_MIN_COLORS = 1
GEN_PATTERN_MAX_COLORS = 3
GEN_NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _random_colors(count: int, *, exclude: set[int]) -> list[int]:
    pool = [c for c in range(10) if c not in exclude]
    random.shuffle(pool)
    assert len(pool) >= count
    return pool[:count]


def _grow_connected_coords(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    start = (random.randrange(h), random.randrange(w))
    coords = {start}
    while len(coords) < count:
        frontier = set()
        for r, c in coords:
            for dr, dc in GEN_NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in coords:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        coords.add(random.choice(tuple(frontier)))
    return list(coords)


def _make_full_pattern(cell_h: int, cell_w: int, bg: int, palette: list[int], good_count: int) -> np.ndarray | None:
    sprite_h = random.randint(3, cell_h)
    sprite_w = random.randint(3, cell_w)
    min_pixels = max(GEN_PATTERN_MIN_PIXELS, good_count + 4)
    max_pixels = min(GEN_PATTERN_MAX_PIXELS, sprite_h * sprite_w - 1, cell_h * cell_w - 1)
    if max_pixels < min_pixels:
        return None

    pixel_count = random.randint(min_pixels, max_pixels)
    coords = _grow_connected_coords(sprite_h, sprite_w, pixel_count)
    if coords is None:
        return None

    sprite = np.full((sprite_h, sprite_w), bg, dtype=int)
    n_colors = random.randint(GEN_PATTERN_MIN_COLORS, min(GEN_PATTERN_MAX_COLORS, len(palette)))
    colors = random.sample(palette, n_colors)
    random.shuffle(coords)
    for i, (r, c) in enumerate(coords):
        sprite[r, c] = colors[i] if i < n_colors else random.choice(colors)

    b = bbox(sprite != bg)
    assert b is not None
    cropped = crop_bbox(sprite, b)
    pattern = np.full((cell_h, cell_w), bg, dtype=int)
    r0 = random.randint(0, cell_h - cropped.shape[0])
    c0 = random.randint(0, cell_w - cropped.shape[1])
    paste(pattern, cropped, r0, c0, transparent=bg)
    return pattern


def _make_good_cells(
    pattern: np.ndarray, bg: int, good_count: int
) -> tuple[list[np.ndarray], list[int], list[tuple[int, int]]]:
    coords = [(int(r), int(c)) for r, c in np.argwhere(pattern != bg)]
    pixel_count = len(coords)
    assert pixel_count >= good_count + 2

    all_indices = list(range(pixel_count))
    random.shuffle(all_indices)
    max_conflicts = min(4, pixel_count - good_count)
    conflict_count = random.randint(2, max_conflicts)
    conflict_indices = all_indices[:conflict_count]
    non_conflict = all_indices[conflict_count:]
    random.shuffle(non_conflict)
    exclusive = non_conflict[:good_count]
    remaining = non_conflict[good_count:]

    visible = np.zeros((good_count, pixel_count), dtype=bool)
    visible[:, conflict_indices] = True
    for cell_idx, pixel_idx in enumerate(exclusive):
        visible[cell_idx, pixel_idx] = True
    for pixel_idx in remaining:
        owner_count = random.randint(1, good_count)
        owners = random.sample(range(good_count), owner_count)
        visible[owners, pixel_idx] = True

    cells: list[np.ndarray] = []
    for cell_idx in range(good_count):
        cell = np.full_like(pattern, bg)
        for pixel_idx, (r, c) in enumerate(coords):
            if visible[cell_idx, pixel_idx]:
                cell[r, c] = pattern[r, c]
        cells.append(cell)

    assert all(np.any(cell != bg) for cell in cells)
    assert all(not np.array_equal(cell, pattern) for cell in cells)
    return cells, conflict_indices, coords


def _make_impostor(
    pattern: np.ndarray, bg: int, palette: list[int], coords: list[tuple[int, int]], conflict_indices: list[int]
) -> np.ndarray:
    impostor = np.full_like(pattern, bg)
    conflict_set = set(conflict_indices)
    for pixel_idx, (r, c) in enumerate(coords):
        if pixel_idx in conflict_set:
            choices = [color for color in palette if color != int(pattern[r, c])]
            impostor[r, c] = random.choice(choices)
            continue
        if random.random() < 0.55:
            if random.random() < 0.35:
                choices = [color for color in palette if color != int(pattern[r, c])]
                impostor[r, c] = random.choice(choices)
            else:
                impostor[r, c] = int(pattern[r, c])

    if np.count_nonzero(impostor != bg) < len(conflict_indices):
        for pixel_idx in conflict_indices:
            r, c = coords[pixel_idx]
            choices = [color for color in palette if color != int(pattern[r, c])]
            impostor[r, c] = random.choice(choices)
    assert np.any(impostor != bg)
    assert not np.array_equal(impostor, pattern)
    return impostor


def _render_grid(
    cell_rows: int, cell_cols: int, cell_h: int, cell_w: int, bg: int, separator: int, cells: list[np.ndarray]
) -> np.ndarray:
    h = cell_rows * cell_h + (cell_rows - 1)
    w = cell_cols * cell_w + (cell_cols - 1)
    grid = np.full((h, w), bg, dtype=int)
    for rr in range(1, cell_rows):
        grid[rr * cell_h + rr - 1, :] = separator
    for cc in range(1, cell_cols):
        grid[:, cc * cell_w + cc - 1] = separator

    idx = 0
    for rr in range(cell_rows):
        for cc in range(cell_cols):
            r0 = rr * (cell_h + 1)
            c0 = cc * (cell_w + 1)
            paste(grid, cells[idx], r0, c0)
            idx += 1
    return grid


def _gen_attempt() -> np.ndarray | None:
    cell_rows, cell_cols = random.choice(GEN_LAYOUTS)
    max_h = min(GEN_CELL_MAX, (30 - (cell_rows - 1)) // cell_rows)
    max_w = min(GEN_CELL_MAX, (30 - (cell_cols - 1)) // cell_cols)
    if max_h < GEN_CELL_MIN or max_w < GEN_CELL_MIN:
        return None

    cell_h = random.randint(GEN_CELL_MIN, max_h)
    cell_w = random.randint(GEN_CELL_MIN, max_w)
    total_cells = cell_rows * cell_cols
    good_count = total_cells - 1

    bg, separator = random.sample(range(10), 2)
    content_palette = _random_colors(random.randint(3, 5), exclude={bg, separator})
    pattern = _make_full_pattern(cell_h, cell_w, bg, content_palette, good_count)
    if pattern is None:
        return None

    good_cells, conflict_indices, coords = _make_good_cells(pattern, bg, good_count)
    impostor = _make_impostor(pattern, bg, content_palette, coords, conflict_indices)

    cells = good_cells + [impostor]
    random.shuffle(cells)
    grid = _render_grid(cell_rows, cell_cols, cell_h, cell_w, bg, separator, cells)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, pattern):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid impostor-repair input")
