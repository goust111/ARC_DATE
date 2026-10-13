from __future__ import annotations

import random

import numpy as np


BG = 0
SEPARATOR_WIDTH = 1
CELL_SIZES = (2, 3)
REGION_CELL_COUNTS = {2: (4, 5), 3: (4,)}
BRANCHES = ("h_only", "v_only", "both", "none")
COLOR_POOL = tuple(range(1, 10))
GEN_RETRIES = 500


def _mode(values: np.ndarray) -> int:
    palette, counts = np.unique(values, return_counts=True)
    return int(palette[np.argmax(counts)])


def _regular_gap(indices: list[int], length: int) -> int | None:
    if not indices:
        return None
    if indices[0] == 0 or indices[-1] == length - 1:
        return None
    gaps = [indices[0]]
    gaps.extend(b - a - 1 for a, b in zip(indices, indices[1:]))
    gaps.append(length - 1 - indices[-1])
    if min(gaps) <= 0 or len(set(gaps)) != 1:
        return None
    return int(gaps[0])


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds = [-1, *separators, length]
    spans: list[tuple[int, int]] = []
    for left, right in zip(bounds, bounds[1:]):
        if right - left > 1:
            spans.append((left + 1, right))
    return spans


def _separator_layout(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = grid.shape
    candidates: list[tuple[int, list[int], list[int]]] = []
    for color in map(int, np.unique(grid)):
        row_lines = [int(r) for r in np.where(np.all(grid == color, axis=1))[0]]
        col_lines = [int(c) for c in np.where(np.all(grid == color, axis=0))[0]]
        if _regular_gap(row_lines, h) is None or _regular_gap(col_lines, w) is None:
            continue
        candidates.append((color, row_lines, col_lines))

    assert len(candidates) == 1, "expected one regular separator color"
    separator, row_lines, col_lines = candidates[0]
    row_spans = _spans(h, row_lines)
    col_spans = _spans(w, col_lines)
    assert row_spans and col_spans, "separator lines must split the grid"

    heights = {r1 - r0 for r0, r1 in row_spans}
    widths = {c1 - c0 for c0, c1 in col_spans}
    assert len(heights) == 1 and len(widths) == 1, "regions must be equal-sized"
    return int(separator), row_spans, col_spans


def _sprite_bbox(region: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    coords = np.argwhere(region != bg)
    assert len(coords) > 0, "each region needs one seed sprite"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    mask = region != bg
    outside = mask.copy()
    outside[r0:r1, c0:c1] = False
    assert not np.any(outside), "seed pixels must stay inside one implicit cell"
    return int(r0), int(c0), int(r1), int(c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill blank implicit cells inside each separator-bounded region according to the seed sprite's mirror symmetries.

    Input structure:
    - Full rows and columns of one separator color split the grid into equal rectangular regions.
    - Inside each region, the background is black and the pixels form an implicit square cell lattice, typically 4x4 or 5x5 cells.
    - Exactly one cell in each region contains a 2x2 or 3x3 non-background seed sprite; every other cell in that region is plain background.

    Rule:
    - Detect the separator color, then cut the grid into equal regions.
    - In each region, locate the unique seed cell and test whether its sprite is symmetric under horizontal reflection, vertical reflection, both, or neither.
    - If only horizontal symmetry holds, copy the sprite into the other cells of the seed's row. If only vertical symmetry holds, copy it into the other cells of the seed's column.
    - If both symmetries hold, copy the sprite into every cell of the region. If neither symmetry holds, leave that region unchanged.

    Key insight:
    - The seed sprite's own symmetry class is the control signal that decides whether propagation happens across one row, one column, the whole region, or nowhere.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    separator, row_spans, col_spans = _separator_layout(g)
    payload = g[g != separator]
    bg = _mode(payload)

    out = g.copy()
    for rr0, rr1 in row_spans:
        for cc0, cc1 in col_spans:
            region = g[rr0:rr1, cc0:cc1]
            assert not np.any(region == separator), "separator pixels cannot appear inside a region"

            r0, c0, r1, c1 = _sprite_bbox(region, bg)
            cell_h = r1 - r0
            cell_w = c1 - c0
            assert cell_h == cell_w and cell_h in CELL_SIZES, "seed sprite must occupy one 2x2 or 3x3 cell"
            cell_size = int(cell_h)
            assert rr1 - rr0 > cell_size and cc1 - cc0 > cell_size
            assert (rr1 - rr0) % cell_size == 0 and (cc1 - cc0) % cell_size == 0
            assert r0 % cell_size == 0 and c0 % cell_size == 0

            sprite = region[r0:r1, c0:c1].copy()
            cell_row = r0 // cell_size
            cell_col = c0 // cell_size
            cell_rows = (rr1 - rr0) // cell_size
            cell_cols = (cc1 - cc0) // cell_size

            horizontal = np.array_equal(sprite, np.flipud(sprite))
            vertical = np.array_equal(sprite, np.fliplr(sprite))

            target = out[rr0:rr1, cc0:cc1]
            if horizontal and vertical:
                for tr in range(cell_rows):
                    for tc in range(cell_cols):
                        rs = tr * cell_size
                        cs = tc * cell_size
                        target[rs : rs + cell_size, cs : cs + cell_size] = sprite
            elif horizontal:
                rs = cell_row * cell_size
                for tc in range(cell_cols):
                    cs = tc * cell_size
                    target[rs : rs + cell_size, cs : cs + cell_size] = sprite
            elif vertical:
                cs = cell_col * cell_size
                for tr in range(cell_rows):
                    rs = tr * cell_size
                    target[rs : rs + cell_size, cs : cs + cell_size] = sprite

    return out


def _branch_of(sprite: np.ndarray) -> str:
    horizontal = np.array_equal(sprite, np.flipud(sprite))
    vertical = np.array_equal(sprite, np.fliplr(sprite))
    if horizontal and vertical:
        return "both"
    if horizontal:
        return "h_only"
    if vertical:
        return "v_only"
    return "none"


def _layout_options() -> list[tuple[int, int, int, int]]:
    options: list[tuple[int, int, int, int]] = []
    for cell_size in CELL_SIZES:
        for cells_per_region in REGION_CELL_COUNTS[cell_size]:
            region_size = cell_size * cells_per_region
            for region_rows in range(2, 4):
                for region_cols in range(2, 4):
                    height = region_rows * region_size + (region_rows - 1) * SEPARATOR_WIDTH
                    width = region_cols * region_size + (region_cols - 1) * SEPARATOR_WIDTH
                    if height <= 30 and width <= 30 and region_rows * region_cols >= len(BRANCHES):
                        options.append((cell_size, cells_per_region, region_rows, region_cols))
    return options


LAYOUT_OPTIONS = _layout_options()


def _sample_colors(k: int, exclude: set[int]) -> list[int]:
    pool = [color for color in COLOR_POOL if color not in exclude]
    assert pool, "no drawable colors available"
    if len(pool) >= k:
        return random.sample(pool, k)
    return [random.choice(pool) for _ in range(k)]


def _make_both_sprite(size: int, separator: int) -> np.ndarray:
    colors = _sample_colors(4, {separator})
    if size == 2:
        return np.full((2, 2), colors[0], dtype=int)

    while True:
        a, b, c, d = _sample_colors(4, {separator})
        sprite = np.array(
            [
                [a, b, a],
                [c, d, c],
                [a, b, a],
            ],
            dtype=int,
        )
        if len(set(map(int, sprite.flat))) >= 2:
            return sprite


def _make_h_only_sprite(size: int, separator: int) -> np.ndarray:
    if size == 2:
        a, b = _sample_colors(2, {separator})
        return np.array([[a, b], [a, b]], dtype=int)

    while True:
        row0 = _sample_colors(3, {separator})
        row1 = _sample_colors(3, {separator})
        sprite = np.array([row0, row1, row0], dtype=int)
        if _branch_of(sprite) == "h_only":
            return sprite


def _make_v_only_sprite(size: int, separator: int) -> np.ndarray:
    if size == 2:
        a, b = _sample_colors(2, {separator})
        return np.array([[a, a], [b, b]], dtype=int)

    while True:
        left = _sample_colors(3, {separator})
        middle = _sample_colors(3, {separator})
        sprite = np.array(
            [
                [left[0], middle[0], left[0]],
                [left[1], middle[1], left[1]],
                [left[2], middle[2], left[2]],
            ],
            dtype=int,
        )
        if _branch_of(sprite) == "v_only":
            return sprite


def _make_none_sprite(size: int, separator: int) -> np.ndarray:
    while True:
        colors = _sample_colors(size * size, {separator})
        sprite = np.array(colors, dtype=int).reshape(size, size)
        if _branch_of(sprite) == "none":
            return sprite


def _make_sprite(size: int, branch: str, separator: int) -> np.ndarray:
    if branch == "both":
        sprite = _make_both_sprite(size, separator)
    elif branch == "h_only":
        sprite = _make_h_only_sprite(size, separator)
    elif branch == "v_only":
        sprite = _make_v_only_sprite(size, separator)
    else:
        assert branch == "none"
        sprite = _make_none_sprite(size, separator)
    assert _branch_of(sprite) == branch
    return sprite


def _region_origin(region_row: int, region_col: int, region_size: int) -> tuple[int, int]:
    return (
        region_row * region_size + region_row * SEPARATOR_WIDTH,
        region_col * region_size + region_col * SEPARATOR_WIDTH,
    )


def _base_grid(region_rows: int, region_cols: int, region_size: int, separator: int) -> np.ndarray:
    height = region_rows * region_size + (region_rows - 1) * SEPARATOR_WIDTH
    width = region_cols * region_size + (region_cols - 1) * SEPARATOR_WIDTH
    grid = np.full((height, width), BG, dtype=int)

    for cut in range(region_rows - 1):
        idx = (cut + 1) * region_size + cut * SEPARATOR_WIDTH
        grid[idx : idx + SEPARATOR_WIDTH, :] = separator
    for cut in range(region_cols - 1):
        idx = (cut + 1) * region_size + cut * SEPARATOR_WIDTH
        grid[:, idx : idx + SEPARATOR_WIDTH] = separator
    return grid


def _render_region(region_size: int, cell_size: int, cells_per_region: int, sprite: np.ndarray, seed_row: int, seed_col: int) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((region_size, region_size), BG, dtype=int)
    out = inp.copy()

    rs = seed_row * cell_size
    cs = seed_col * cell_size
    inp[rs : rs + cell_size, cs : cs + cell_size] = sprite
    out[rs : rs + cell_size, cs : cs + cell_size] = sprite

    branch = _branch_of(sprite)
    if branch == "both":
        targets = [(r, c) for r in range(cells_per_region) for c in range(cells_per_region)]
    elif branch == "h_only":
        targets = [(seed_row, c) for c in range(cells_per_region)]
    elif branch == "v_only":
        targets = [(r, seed_col) for r in range(cells_per_region)]
    else:
        targets = [(seed_row, seed_col)]

    for tr, tc in targets:
        rr = tr * cell_size
        cc = tc * cell_size
        out[rr : rr + cell_size, cc : cc + cell_size] = sprite

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    cell_size, cells_per_region, region_rows, region_cols = random.choice(LAYOUT_OPTIONS)
    region_size = cell_size * cells_per_region
    separator = random.choice(COLOR_POOL)

    inp = _base_grid(region_rows, region_cols, region_size, separator)
    expected = inp.copy()

    branches = list(BRANCHES)
    region_count = region_rows * region_cols
    while len(branches) < region_count:
        branches.append(random.choice(BRANCHES))
    random.shuffle(branches)

    idx = 0
    for region_row in range(region_rows):
        for region_col in range(region_cols):
            branch = branches[idx]
            idx += 1
            sprite = _make_sprite(cell_size, branch, separator)
            seed_row = random.randrange(cells_per_region)
            seed_col = random.randrange(cells_per_region)
            region_in, region_out = _render_region(region_size, cell_size, cells_per_region, sprite, seed_row, seed_col)

            r0, c0 = _region_origin(region_row, region_col, region_size)
            inp[r0 : r0 + region_size, c0 : c0 + region_size] = region_in
            expected[r0 : r0 + region_size, c0 : c0 + region_size] = region_out

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
