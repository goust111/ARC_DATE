import random

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
MAX_GRID_SIZE = 30
MAX_RETRIES = 400

NO_OVERLAP_LAYOUTS = (
    (2, 2),
    (2, 3),
    (3, 2),
    (2, 4),
    (4, 2),
    (3, 3),
    (3, 4),
    (4, 3),
)
OVERLAP_LAYOUTS = (
    (3, 3),
    (3, 5),
    (5, 3),
)
PATTERN_STYLES = ("stripes", "blocks", "formula")


def _bands(mask: np.ndarray) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            bands.append((start, idx))
            start = None
    if start is not None:
        bands.append((start, len(mask)))
    return bands


def _extract_tiles(grid: np.ndarray) -> tuple[list[np.ndarray], int, int]:
    row_bands = _bands(np.any(grid != BG, axis=1))
    col_bands = _bands(np.any(grid != BG, axis=0))
    assert row_bands and col_bands, "expected at least one non-background tile"

    tile_heights = {r1 - r0 for r0, r1 in row_bands}
    tile_widths = {c1 - c0 for c0, c1 in col_bands}
    assert len(tile_heights) == 1, "all tile rows must share one height"
    assert len(tile_widths) == 1, "all tile columns must share one width"

    tile_h = next(iter(tile_heights))
    tile_w = next(iter(tile_widths))
    tiles: list[np.ndarray] = []
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            tile = np.asarray(grid[r0:r1, c0:c1], dtype=int)
            assert tile.shape == (tile_h, tile_w)
            tiles.append(tile)
    return tiles, tile_h, tile_w


def _majority_template(tiles: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(tiles, axis=0)
    n_tiles, tile_h, tile_w = stack.shape
    assert n_tiles >= 2, "need multiple copies for voting"

    template = np.zeros((tile_h, tile_w), dtype=int)
    for r in range(tile_h):
        for c in range(tile_w):
            counts = np.bincount(stack[:, r, c], minlength=10)
            winners = np.flatnonzero(counts == counts.max())
            template[r, c] = int(winners[0])
    return template


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the clean template from repeated near-identical copies and merge all altered pixels onto it.

    Input structure:
    - The background is black.
    - Several equal-size non-black rectangles are arranged on a regular row/column lattice with black gaps.
    - Each rectangle is one copy of the same multi-color template.
    - Every copy differs from the clean template in only a few pixels.

    Rule:
    - Use the occupied row bands and occupied column bands to segment the repeated rectangles.
    - For each pixel position inside a rectangle, take the majority color across all copies to recover the clean template.
    - Start from that clean template and, in reading order from left to right then top to bottom, repaint every pixel where a copy differs from the template.
    - If several copies alter the same position, the later copy in reading order overwrites the earlier ones.

    Key insight:
    - The repeated rectangles form a voting code: agreement reveals the canonical template, while disagreements encode the pixels to merge.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    tiles, tile_h, tile_w = _extract_tiles(g)
    template = _majority_template(tiles)

    out = template.copy()
    for tile in tiles:
        assert tile.shape == (tile_h, tile_w)
        diff = tile != template
        out[diff] = tile[diff]
    return out


def _make_template(tile_h: int, tile_w: int, rng: random.Random) -> np.ndarray:
    palette_size = rng.randint(2, min(5, len(COLOR_POOL)))
    palette = rng.sample(COLOR_POOL, palette_size)
    template = np.full((tile_h, tile_w), palette[0], dtype=int)
    style = rng.choice(PATTERN_STYLES)

    if style == "stripes":
        for _ in range(rng.randint(1, max(1, tile_h // 2 + 1))):
            row = rng.randrange(tile_h)
            template[row, :] = rng.choice(palette[1:] or palette)
        for _ in range(rng.randint(1, max(1, tile_w // 2 + 1))):
            col = rng.randrange(tile_w)
            template[:, col] = rng.choice(palette[1:] or palette)
        if tile_h >= 3 and tile_w >= 3:
            r0 = rng.randrange(tile_h - 1)
            r1 = rng.randrange(r0 + 1, tile_h + 1)
            c0 = rng.randrange(tile_w - 1)
            c1 = rng.randrange(c0 + 1, tile_w + 1)
            template[r0:r1, c0:c1] = rng.choice(palette[1:] or palette)
    elif style == "blocks":
        for _ in range(rng.randint(2, 5)):
            color = rng.choice(palette)
            r0 = rng.randrange(tile_h)
            r1 = rng.randrange(r0, tile_h)
            c0 = rng.randrange(tile_w)
            c1 = rng.randrange(c0, tile_w)
            template[r0 : r1 + 1, c0 : c1 + 1] = color
        accent = rng.choice(palette[1:] or palette)
        if rng.random() < 0.5:
            for idx in range(min(tile_h, tile_w)):
                template[idx, idx] = accent
        else:
            for idx in range(min(tile_h, tile_w)):
                template[idx, tile_w - 1 - idx] = accent
    else:
        step_r = rng.randint(1, palette_size)
        step_c = rng.randint(1, palette_size)
        offset = rng.randrange(palette_size)
        alt = rng.randint(0, palette_size - 1)
        for r in range(tile_h):
            for c in range(tile_w):
                idx = offset + step_r * r + step_c * c + alt * ((r + c) % 2)
                template[r, c] = palette[idx % palette_size]
        if rng.random() < 0.5:
            row = rng.randrange(tile_h)
            template[row, :] = rng.choice(palette)
        if rng.random() < 0.5:
            col = rng.randrange(tile_w)
            template[:, col] = rng.choice(palette)

    if len(np.unique(template)) == 1:
        r = rng.randrange(tile_h)
        c = rng.randrange(tile_w)
        choices = [color for color in COLOR_POOL if color != int(template[r, c])]
        template[r, c] = rng.choice(choices)
    return template


def _sample_geometry(
    rows_count: int, cols_count: int, area_floor: int, rng: random.Random
) -> tuple[int, int, int, int, int, int, int, int] | None:
    for _ in range(120):
        gap_r = rng.randint(1, 3)
        gap_c = rng.randint(1, 3)
        top = rng.randint(0, 2)
        bottom = rng.randint(0, 2)
        left = rng.randint(0, 2)
        right = rng.randint(0, 2)

        max_tile_h = (MAX_GRID_SIZE - top - bottom - (rows_count - 1) * gap_r) // rows_count
        max_tile_w = (MAX_GRID_SIZE - left - right - (cols_count - 1) * gap_c) // cols_count
        max_tile_h = min(max_tile_h, 8)
        max_tile_w = min(max_tile_w, 8)
        if max_tile_h < 3 or max_tile_w < 3:
            continue

        feasible = [
            (tile_h, tile_w)
            for tile_h in range(3, max_tile_h + 1)
            for tile_w in range(3, max_tile_w + 1)
            if tile_h * tile_w >= area_floor
        ]
        if not feasible:
            continue

        tile_h, tile_w = rng.choice(feasible)
        total_h = top + rows_count * tile_h + (rows_count - 1) * gap_r + bottom
        total_w = left + cols_count * tile_w + (cols_count - 1) * gap_c + right
        if total_h <= MAX_GRID_SIZE and total_w <= MAX_GRID_SIZE:
            return tile_h, tile_w, gap_r, gap_c, top, bottom, left, right
    return None


def _mutation_positions(
    copy_count: int,
    tile_h: int,
    tile_w: int,
    want_overlap: bool,
    rng: random.Random,
) -> list[set[int]]:
    area = tile_h * tile_w
    counts = [1] * copy_count
    shared_copies: list[int] = []
    shared_pos = None

    if want_overlap:
        max_shared = min(4, (copy_count - 1) // 2)
        shared_size = rng.randint(2, max_shared)
        shared_copies = sorted(rng.sample(range(copy_count), shared_size))
        shared_pos = rng.randrange(area)
        capacity = area + shared_size - 1
    else:
        shared_size = 0
        capacity = area

    extra_budget = rng.randint(0, min(2 * copy_count, capacity - copy_count))
    for _ in range(extra_budget):
        candidates = [idx for idx, count in enumerate(counts) if count < 3]
        counts[rng.choice(candidates)] += 1

    positions = [set() for _ in range(copy_count)]
    if shared_pos is not None:
        for idx in shared_copies:
            positions[idx].add(shared_pos)

    owners: list[int] = []
    for idx in range(copy_count):
        owners.extend([idx] * (counts[idx] - len(positions[idx])))

    free_positions = [pos for pos in range(area) if pos != shared_pos]
    rng.shuffle(owners)
    rng.shuffle(free_positions)
    assert len(owners) <= len(free_positions)

    for owner, pos in zip(owners, free_positions):
        positions[owner].add(pos)

    for idx, count in enumerate(counts):
        assert len(positions[idx]) == count
    return positions


def _render_grid(
    tiles: list[np.ndarray],
    rows_count: int,
    cols_count: int,
    tile_h: int,
    tile_w: int,
    gap_r: int,
    gap_c: int,
    top: int,
    left: int,
) -> np.ndarray:
    total_h = top + rows_count * tile_h + (rows_count - 1) * gap_r
    total_w = left + cols_count * tile_w + (cols_count - 1) * gap_c
    grid = np.zeros((total_h, total_w), dtype=int)
    idx = 0
    for rr in range(rows_count):
        r0 = top + rr * (tile_h + gap_r)
        for cc in range(cols_count):
            c0 = left + cc * (tile_w + gap_c)
            grid[r0 : r0 + tile_h, c0 : c0 + tile_w] = tiles[idx]
            idx += 1
    return grid


def _expected_output(template: np.ndarray, tiles: list[np.ndarray]) -> np.ndarray:
    out = template.copy()
    for tile in tiles:
        diff = tile != template
        out[diff] = tile[diff]
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    want_overlap = rng.random() < 0.5
    rows_count, cols_count = rng.choice(OVERLAP_LAYOUTS if want_overlap else NO_OVERLAP_LAYOUTS)
    copy_count = rows_count * cols_count

    geom = _sample_geometry(rows_count, cols_count, copy_count, rng)
    if geom is None:
        return None
    tile_h, tile_w, gap_r, gap_c, top, _bottom, left, _right = geom

    template = _make_template(tile_h, tile_w, rng)
    tiles = [template.copy() for _ in range(copy_count)]
    mutation_sets = _mutation_positions(copy_count, tile_h, tile_w, want_overlap, rng)

    last_color_by_pos: dict[int, int] = {}
    for idx, tile in enumerate(tiles):
        for pos in sorted(mutation_sets[idx]):
            r = pos // tile_w
            c = pos % tile_w
            options = [color for color in COLOR_POOL if color != int(template[r, c])]
            previous = last_color_by_pos.get(pos)
            if previous in options and len(options) > 1 and rng.random() < 0.75:
                options = [color for color in options if color != previous]
            tile[r, c] = rng.choice(options)
            last_color_by_pos[pos] = int(tile[r, c])

    grid = _render_grid(tiles, rows_count, cols_count, tile_h, tile_w, gap_r, gap_c, top, left)
    expected = _expected_output(template, tiles)

    row_bands = _bands(np.any(grid != BG, axis=1))
    col_bands = _bands(np.any(grid != BG, axis=0))
    if len(row_bands) != rows_count or len(col_bands) != cols_count:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
