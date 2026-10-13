from __future__ import annotations

import random

import numpy as np


BG = 0
SEP = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
]
TILE_SIZE_CHOICES = [5, 7]
TILE_ROWS_RANGE = (3, 5)
TILE_COLS_RANGE = (3, 5)
TARGET_COUNT_RANGE = (2, 6)
MAX_GENERATION_ATTEMPTS = 400


def _sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _separator_spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    assert separators, "expected separator lines"
    spans: list[tuple[int, int]] = []
    start = 0
    for sep_idx in separators:
        assert start < sep_idx, "separator lines must leave a nonempty region"
        spans.append((start, sep_idx - 1))
        start = sep_idx + 1
    assert start < length, "separator lines must leave a final nonempty region"
    spans.append((start, length - 1))
    return spans


def _find_separator_grid(g: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = map(int, g.shape)
    candidates: list[tuple[int, list[tuple[int, int]], list[tuple[int, int]]]] = []

    for color in map(int, np.unique(g)):
        rows = [r for r in range(h) if np.all(g[r, :] == color)]
        cols = [c for c in range(w) if np.all(g[:, c] == color)]
        if not rows or not cols:
            continue
        try:
            row_spans = _separator_spans(h, rows)
            col_spans = _separator_spans(w, cols)
        except AssertionError:
            continue
        row_sizes = {r1 - r0 + 1 for r0, r1 in row_spans}
        col_sizes = {c1 - c0 + 1 for c0, c1 in col_spans}
        if len(row_sizes) == 1 and len(col_sizes) == 1 and len(row_spans) >= 2 and len(col_spans) >= 2:
            candidates.append((color, row_spans, col_spans))

    assert len(candidates) == 1, "expected exactly one separator grid"
    return candidates[0]


def _background_color(g: np.ndarray, separator: int) -> int:
    values, counts = np.unique(g[g != separator], return_counts=True)
    assert values.size > 0, "expected non-separator pixels"
    return int(values[np.argmax(counts)])


def _direction_from_pointer(r: int, c: int, tile_h: int, tile_w: int) -> tuple[int, int]:
    dr = _sign(2 * int(r) - (int(tile_h) - 1))
    dc = _sign(2 * int(c) - (int(tile_w) - 1))
    assert (dr, dc) != (0, 0), "pointer must be off-center"
    return dr, dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the adjacent source pattern into each pointer-marked region.

    The pointer pixel's offset from the region center encodes one of the eight
    neighbor directions. The adjacent region in that direction is the source.
    Copy the source pattern's occupied mask to the target region and recolor it
    to the pointer pixel's color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    separator, row_spans, col_spans = _find_separator_grid(g)
    bg = _background_color(g, separator)
    tile_h = row_spans[0][1] - row_spans[0][0] + 1
    tile_w = col_spans[0][1] - col_spans[0][0] + 1

    out = g.copy()
    for tile_r, (r0, r1) in enumerate(row_spans):
        for tile_c, (c0, c1) in enumerate(col_spans):
            tile = g[r0 : r1 + 1, c0 : c1 + 1]
            occupied = np.argwhere(tile != bg)
            if occupied.shape[0] != 1:
                continue

            pr, pc = map(int, occupied[0])
            direction = _direction_from_pointer(pr, pc, tile_h, tile_w)
            src_r = tile_r + direction[0]
            src_c = tile_c + direction[1]
            assert 0 <= src_r < len(row_spans) and 0 <= src_c < len(col_spans), "pointer leaves grid"

            sr0, sr1 = row_spans[src_r]
            sc0, sc1 = col_spans[src_c]
            source = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
            source_mask = source != bg
            assert int(source_mask.sum()) > 1, "adjacent source region must contain a pattern"
            assert bool(source_mask[pr, pc]), "pointer location must be covered by the copied pattern"

            color = int(tile[pr, pc])
            target = out[r0 : r1 + 1, c0 : c1 + 1]
            target[:] = bg
            target[source_mask] = color

    return out


def _direction_cells(tile_h: int, tile_w: int, direction: tuple[int, int]) -> list[tuple[int, int]]:
    center_r = (tile_h - 1) // 2
    center_c = (tile_w - 1) // 2
    cells: list[tuple[int, int]] = []
    for r in range(tile_h):
        for c in range(tile_w):
            if (r, c) == (center_r, center_c):
                continue
            if _direction_from_pointer(r, c, tile_h, tile_w) == direction:
                cells.append((r, c))
    return cells


def _pick_direction_cell(tile_h: int, tile_w: int, direction: tuple[int, int]) -> tuple[int, int]:
    cells = _direction_cells(tile_h, tile_w, direction)
    center_r = (tile_h - 1) // 2
    center_c = (tile_w - 1) // 2
    weights = [1 + abs(r - center_r) + abs(c - center_c) for r, c in cells]
    return random.choices(cells, weights=weights, k=1)[0]


def _neighbors8(r: int, c: int, tile_h: int, tile_w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr = r + dr
            nc = c + dc
            if 0 <= nr < tile_h and 0 <= nc < tile_w:
                out.append((nr, nc))
    return out


def _build_source_shape(
    tile_h: int, tile_w: int, required_dirs: set[tuple[int, int]]
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], tuple[int, int]]]:
    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    shape: set[tuple[int, int]] = set()

    for direction in random.sample(list(required_dirs), len(required_dirs)):
        coord = _pick_direction_cell(tile_h, tile_w, direction)
        anchors[direction] = coord
        shape.add(coord)

    min_size = max(2, len(shape))
    max_size = min(tile_h * tile_w - 1, len(shape) + random.randint(2, 5))
    target_size = random.randint(min_size, max_size)
    all_cells = [(r, c) for r in range(tile_h) for c in range(tile_w)]

    while len(shape) < target_size:
        frontier = [
            cell
            for cell in all_cells
            if cell not in shape and any(nei in shape for nei in _neighbors8(cell[0], cell[1], tile_h, tile_w))
        ]
        pool = frontier if frontier and random.random() < 0.8 else [cell for cell in all_cells if cell not in shape]
        shape.add(random.choice(pool))

    return shape, anchors


def _choose_relations(
    tile_rows: int, tile_cols: int
) -> tuple[dict[tuple[int, int], set[tuple[int, int]]], dict[tuple[int, int], tuple[int, int]]] | None:
    max_targets = min(TARGET_COUNT_RANGE[1], tile_rows * tile_cols - 1)
    min_targets = min(TARGET_COUNT_RANGE[0], max_targets)
    if max_targets < 1:
        return None

    desired_targets = random.randint(max(1, min_targets), max_targets)
    cells = [(r, c) for r in range(tile_rows) for c in range(tile_cols)]

    for _ in range(200):
        state = np.zeros((tile_rows, tile_cols), dtype=int)
        sources_to_dirs: dict[tuple[int, int], set[tuple[int, int]]] = {}
        target_to_dir: dict[tuple[int, int], tuple[int, int]] = {}

        for _ in range(desired_targets * 50):
            if len(target_to_dir) == desired_targets:
                break
            blank_cells = [(r, c) for r, c in cells if state[r, c] == 0]
            if not blank_cells:
                break
            tr, tc = random.choice(blank_cells)
            dr, dc = random.choice(DIRS)
            sr = tr + dr
            sc = tc + dc
            if not (0 <= sr < tile_rows and 0 <= sc < tile_cols):
                continue
            if state[sr, sc] == 2:
                continue

            state[tr, tc] = 2
            if state[sr, sc] == 0:
                state[sr, sc] = 1
                sources_to_dirs[(sr, sc)] = set()
            sources_to_dirs[(sr, sc)].add((dr, dc))
            target_to_dir[(tr, tc)] = (dr, dc)

        used_dirs = set(target_to_dir.values())
        if len(target_to_dir) == desired_targets and len(used_dirs) >= min(2, desired_targets):
            return sources_to_dirs, target_to_dir

    return None


def _render_grid(
    tile_rows: int,
    tile_cols: int,
    tile_h: int,
    tile_w: int,
    source_shapes: dict[tuple[int, int], set[tuple[int, int]]],
    source_colors: dict[tuple[int, int], int],
    target_specs: dict[tuple[int, int], tuple[tuple[int, int], int]],
) -> np.ndarray:
    height = tile_rows * tile_h + (tile_rows - 1)
    width = tile_cols * tile_w + (tile_cols - 1)
    grid = np.full((height, width), BG, dtype=int)
    grid[tile_h :: tile_h + 1, :] = SEP
    grid[:, tile_w :: tile_w + 1] = SEP

    def tile_origin(rr: int, cc: int) -> tuple[int, int]:
        return rr * (tile_h + 1), cc * (tile_w + 1)

    for (sr, sc), shape in source_shapes.items():
        r0, c0 = tile_origin(sr, sc)
        color = source_colors[(sr, sc)]
        for rr, cc in shape:
            grid[r0 + rr, c0 + cc] = color

    for (tr, tc), (pointer, color) in target_specs.items():
        r0, c0 = tile_origin(tr, tc)
        grid[r0 + pointer[0], c0 + pointer[1]] = color

    return grid


def _generate_attempt() -> np.ndarray | None:
    tile_rows = random.randint(*TILE_ROWS_RANGE)
    tile_cols = random.randint(*TILE_COLS_RANGE)
    tile_h = random.choice(TILE_SIZE_CHOICES)
    tile_w = random.choice(TILE_SIZE_CHOICES)

    relations = _choose_relations(tile_rows, tile_cols)
    if relations is None:
        return None
    sources_to_dirs, target_to_dir = relations

    source_shapes: dict[tuple[int, int], set[tuple[int, int]]] = {}
    source_anchors: dict[tuple[int, int], dict[tuple[int, int], tuple[int, int]]] = {}
    for source_cell, required_dirs in sources_to_dirs.items():
        shape, anchors = _build_source_shape(tile_h, tile_w, required_dirs)
        source_shapes[source_cell] = shape
        source_anchors[source_cell] = anchors

    source_colors = {cell: random.choice(PALETTE) for cell in source_shapes}
    target_specs: dict[tuple[int, int], tuple[tuple[int, int], int]] = {}
    for (tr, tc), direction in target_to_dir.items():
        source_cell = (tr + direction[0], tc + direction[1])
        pointer = source_anchors[source_cell][direction]
        target_specs[(tr, tc)] = (pointer, random.choice(PALETTE))

    grid = _render_grid(tile_rows, tile_cols, tile_h, tile_w, source_shapes, source_colors, target_specs)

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
