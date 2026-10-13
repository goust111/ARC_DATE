import random

import numpy as np


BLACK = 0
GREEN = 3
GREY = 5
MAROON = 9
SEED_COLORS = (1, 2, 4, 6, 7, 8)

CARDINAL_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIAGONAL_DIRS = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
ALL_DIRS = CARDINAL_DIRS + DIAGONAL_DIRS

LINE_WIDTH = 1
LAYOUT_OPTIONS = ((2, 2), (2, 3), (3, 2))
TILE_SIZE_OPTIONS = ((5, 5), (5, 7), (7, 5), (7, 7), (7, 9), (9, 7), (9, 9))
GENERATION_RETRIES = 300


def _background_color(grid: np.ndarray, separator: int) -> int:
    cells = grid[grid != separator]
    values, counts = np.unique(cells, return_counts=True)
    return int(values[np.argmax(counts)])


def _interior_spans(separators: list[int], limit: int) -> list[tuple[int, int]]:
    seps = sorted(set(int(x) for x in separators))
    assert len(seps) >= 3, "need at least two tile rows/cols"
    assert seps[0] == 0 and seps[-1] == limit - 1, "separator lines must frame the grid"

    gaps = [seps[i + 1] - seps[i] for i in range(len(seps) - 1)]
    assert len(set(gaps)) == 1, "separator lines must be evenly spaced"
    assert gaps[0] >= 4, "tile interior must fit the 3x3 template"

    spans = []
    for left, right in zip(seps, seps[1:]):
        assert right - left >= 2
        spans.append((left + 1, right - 1))
    return spans


def _partition_grid(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = grid.shape
    candidates = []
    for color in np.unique(grid):
        rows = [r for r in range(h) if np.all(grid[r, :] == color)]
        cols = [c for c in range(w) if np.all(grid[:, c] == color)]
        if len(rows) < 3 or len(cols) < 3:
            continue
        try:
            row_spans = _interior_spans(rows, h)
            col_spans = _interior_spans(cols, w)
        except AssertionError:
            continue
        candidates.append((int(color), row_spans, col_spans, len(rows) + len(cols)))

    assert candidates, "failed to find the green partition lines"
    green_candidates = [cand for cand in candidates if cand[0] == GREEN]
    color, row_spans, col_spans, _ = max(green_candidates or candidates, key=lambda item: item[3])
    return color, row_spans, col_spans


def _shift_mask(mask: np.ndarray, dr: int, dc: int) -> np.ndarray:
    shifted = np.zeros_like(mask)
    src_r0 = max(0, -dr)
    src_r1 = mask.shape[0] - max(0, dr)
    src_c0 = max(0, -dc)
    src_c1 = mask.shape[1] - max(0, dc)
    dst_r0 = max(0, dr)
    dst_r1 = mask.shape[0] - max(0, -dr)
    dst_c0 = max(0, dc)
    dst_c1 = mask.shape[1] - max(0, -dc)
    if src_r0 >= src_r1 or src_c0 >= src_c1:
        return shifted
    shifted[dst_r0:dst_r1, dst_c0:dst_c1] = mask[src_r0:src_r1, src_c0:src_c1]
    return shifted


def _grow_mask(shape: tuple[int, int], seed: tuple[int, int], directions: list[tuple[int, int]], steps: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[seed] = True
    for _ in range(steps):
        expanded = mask.copy()
        for dr, dc in directions:
            expanded |= _shift_mask(mask, dr, dc)
        mask = expanded
    return mask


def _template_directions(tile: np.ndarray, background: int) -> list[tuple[int, int]]:
    centers = np.argwhere(tile == GREY)
    assert len(centers) == 1, "template tile must contain exactly one grey center"
    cr, cc = map(int, centers[0])
    assert 1 <= cr < tile.shape[0] - 1 and 1 <= cc < tile.shape[1] - 1, "template center must have 8 neighbors"

    directions = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr = cr + dr
            nc = cc + dc
            value = int(tile[nr, nc])
            if value == MAROON:
                directions.append((dr, dc))
            else:
                assert value == background, "template neighbors must be maroon or background"
    assert directions, "template must activate at least one direction"
    return directions


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    separator, row_spans, col_spans = _partition_grid(g)
    background = _background_color(g, separator)
    template_r0, template_r1 = row_spans[0]
    template_c0, template_c1 = col_spans[0]
    template_tile = g[template_r0 : template_r1 + 1, template_c0 : template_c1 + 1]
    directions = _template_directions(template_tile, background)

    out = g.copy()
    for row_index, (r0, r1) in enumerate(row_spans):
        for col_index, (c0, c1) in enumerate(col_spans):
            if row_index == 0 and col_index == 0:
                continue

            tile = g[r0 : r1 + 1, c0 : c1 + 1]
            seed_locs = np.argwhere((tile != background) & (tile != separator))
            assert len(seed_locs) == 1, "each non-template region must contain exactly one seed pixel"
            sr, sc = map(int, seed_locs[0])
            seed_color = int(tile[sr, sc])
            assert seed_color not in {GREY, MAROON}, "seed colors must differ from template colors"

            steps = min(sr, tile.shape[0] - 1 - sr, sc, tile.shape[1] - 1 - sc)
            grown = _grow_mask(tile.shape, (sr, sc), directions, steps)
            patch = out[r0 : r1 + 1, c0 : c1 + 1]
            patch[grown] = seed_color

    return out


def _choose_directions(rng: random.Random) -> list[tuple[int, int]]:
    branch = rng.choice(("cardinal", "diagonal", "mixed"))
    if branch == "cardinal":
        count = rng.randint(1, len(CARDINAL_DIRS))
        return rng.sample(CARDINAL_DIRS, count)
    if branch == "diagonal":
        count = rng.randint(1, len(DIAGONAL_DIRS))
        return rng.sample(DIAGONAL_DIRS, count)

    count = rng.randint(2, len(ALL_DIRS))
    must_have = [rng.choice(CARDINAL_DIRS), rng.choice(DIAGONAL_DIRS)]
    remaining = [direction for direction in ALL_DIRS if direction not in must_have]
    chosen = must_have + rng.sample(remaining, count - 2)
    rng.shuffle(chosen)
    return chosen


def _tile_origin(index: int, tile_size: int) -> int:
    return LINE_WIDTH + index * (tile_size + LINE_WIDTH)


def _make_input(rows: int, cols: int, tile_h: int, tile_w: int, directions: list[tuple[int, int]], rng: random.Random) -> np.ndarray:
    height = rows * tile_h + (rows + 1) * LINE_WIDTH
    width = cols * tile_w + (cols + 1) * LINE_WIDTH
    grid = np.full((height, width), GREEN, dtype=int)

    for row in range(rows):
        for col in range(cols):
            r0 = _tile_origin(row, tile_h)
            c0 = _tile_origin(col, tile_w)
            grid[r0 : r0 + tile_h, c0 : c0 + tile_w] = BLACK

    template_r0 = _tile_origin(0, tile_h)
    template_c0 = _tile_origin(0, tile_w)
    center_r = template_r0 + tile_h // 2
    center_c = template_c0 + tile_w // 2
    grid[center_r, center_c] = GREY
    for dr, dc in directions:
        grid[center_r + dr, center_c + dc] = MAROON

    needed_colors = rows * cols - 1
    seed_colors = rng.sample(SEED_COLORS, needed_colors)
    color_index = 0
    for row in range(rows):
        for col in range(cols):
            if row == 0 and col == 0:
                continue
            r0 = _tile_origin(row, tile_h)
            c0 = _tile_origin(col, tile_w)
            grid[r0 + tile_h // 2, c0 + tile_w // 2] = seed_colors[color_index]
            color_index += 1

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    rows, cols = rng.choice(LAYOUT_OPTIONS)
    tile_h, tile_w = rng.choice(TILE_SIZE_OPTIONS)
    directions = _choose_directions(rng)
    grid = _make_input(rows, cols, tile_h, tile_w, directions, rng)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        result = _gen_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate input after {GENERATION_RETRIES} attempts")
