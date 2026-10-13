from __future__ import annotations

import random

import numpy as np


SEPARATOR = 5
MISSING = 6
VISIBLE_COLORS = [1, 2, 3, 4, 7, 8, 9]
MAX_SIDE = 30
MAX_GENERATION_ATTEMPTS = 400


def _content_runs(length: int, separator_indices: set[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx in range(length):
        if idx in separator_indices:
            if start is not None:
                runs.append((start, idx))
                start = None
        elif start is None:
            start = idx
    if start is not None:
        runs.append((start, length))
    return runs


def _compress_blocks(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    sep_rows = {r for r in range(g.shape[0]) if np.all(g[r, :] == SEPARATOR)}
    sep_cols = {c for c in range(g.shape[1]) if np.all(g[:, c] == SEPARATOR)}
    assert sep_rows and sep_cols, "expected full separator rows and columns"

    row_runs = _content_runs(g.shape[0], sep_rows)
    col_runs = _content_runs(g.shape[1], sep_cols)
    assert row_runs and col_runs, "separator lines must leave at least one content block"

    compressed = np.empty((len(row_runs), len(col_runs)), dtype=int)
    for rr, (r0, r1) in enumerate(row_runs):
        for cc, (c0, c1) in enumerate(col_runs):
            block = g[r0:r1, c0:c1]
            colors = np.unique(block)
            assert len(colors) == 1, "each block must be monochrome"
            color = int(colors[0])
            assert color != SEPARATOR, "separator color cannot appear inside a block"
            compressed[rr, cc] = color
    return compressed, row_runs, col_runs


def _consistent_tile(grid: np.ndarray, row_period: int, col_period: int, missing: int | None) -> np.ndarray | None:
    tile = np.full((row_period, col_period), -1, dtype=int)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            value = int(grid[r, c])
            if missing is not None and value == missing:
                continue
            rr = r % row_period
            cc = c % col_period
            seen = int(tile[rr, cc])
            if seen == -1:
                tile[rr, cc] = value
            elif seen != value:
                return None
    if np.any(tile == -1):
        return None
    return tile


def _infer_unique_minimal_period(grid: np.ndarray, missing: int | None) -> tuple[int, int, np.ndarray]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int, np.ndarray]] = []
    for row_period in range(1, h + 1):
        for col_period in range(1, w + 1):
            tile = _consistent_tile(grid, row_period, col_period, missing)
            if tile is not None:
                candidates.append((row_period * col_period, row_period, col_period, tile))

    assert candidates, "grid must admit at least one periodic explanation"
    min_area = min(area for area, _rp, _cp, _tile in candidates)
    minimal = [(rp, cp, tile) for area, rp, cp, tile in candidates if area == min_area]
    assert len(minimal) == 1, "minimum-area period must be unique"
    return minimal[0]


def _expand_blocks(
    blocks: np.ndarray,
    row_sizes: list[int],
    col_sizes: list[int],
    row_separators: list[int],
    col_separators: list[int],
) -> np.ndarray:
    height = sum(row_sizes) + sum(row_separators)
    width = sum(col_sizes) + sum(col_separators)
    grid = np.full((height, width), SEPARATOR, dtype=int)

    r = 0
    for rr, row_size in enumerate(row_sizes):
        c = 0
        for cc, col_size in enumerate(col_sizes):
            grid[r : r + row_size, c : c + col_size] = int(blocks[rr, cc])
            c += col_size
            if cc < len(col_separators):
                c += col_separators[cc]
        r += row_size
        if rr < len(row_separators):
            r += row_separators[rr]

    return grid


def _tile_has_required_dependency(tile: np.ndarray) -> bool:
    h, w = tile.shape
    if len(np.unique(tile)) < 2:
        return False
    if h > 1 and len({tuple(row.tolist()) for row in tile}) < 2:
        return False
    if w > 1 and len({tuple(tile[:, c].tolist()) for c in range(w)}) < 2:
        return False
    return True


def _informative_residues(tile: np.ndarray) -> set[tuple[int, int]]:
    info: set[tuple[int, int]] = set()
    for r in range(tile.shape[0]):
        row_diverse = len(set(tile[r, :].tolist())) > 1
        for c in range(tile.shape[1]):
            col_diverse = len(set(tile[:, c].tolist())) > 1
            if row_diverse and col_diverse:
                info.add((r, c))
    return info


def _sample_tile(row_period: int, col_period: int) -> np.ndarray:
    area = row_period * col_period
    max_colors = min(4, area, len(VISIBLE_COLORS))
    for _ in range(300):
        n_colors = random.randint(2, max_colors)
        palette = random.sample(VISIBLE_COLORS, n_colors)
        tile = np.array(
            [random.choice(palette) for _ in range(area)],
            dtype=int,
        ).reshape(row_period, col_period)
        if len(np.unique(tile)) != n_colors:
            continue
        if not _tile_has_required_dependency(tile):
            continue
        try:
            rp, cp, _ = _infer_unique_minimal_period(tile, missing=None)
        except AssertionError:
            continue
        if (rp, cp) != (row_period, col_period):
            continue
        if row_period > 1 and col_period > 1 and not _informative_residues(tile):
            continue
        return tile
    raise RuntimeError("failed to sample a valid periodic tile")


def _sample_axis_sizes(n_bands: int) -> tuple[list[int], list[int]]:
    for _ in range(300):
        band_sizes = [random.randint(1, 4) for _ in range(n_bands)]
        if len(set(band_sizes)) < 2:
            continue
        separators = [random.randint(1, 2) for _ in range(n_bands - 1)]
        if sum(band_sizes) + sum(separators) <= MAX_SIDE:
            return band_sizes, separators
    raise RuntimeError("failed to sample non-uniform block sizes")


def _choose_missing_blocks(blocks: np.ndarray, tile: np.ndarray) -> np.ndarray | None:
    h, w = blocks.shape
    row_period, col_period = tile.shape
    all_coords = [(r, c) for r in range(h) for c in range(w)]

    preserve: set[tuple[int, int]] = set()
    for rr in range(row_period):
        for cc in range(col_period):
            residue_coords = [
                (r, c)
                for r in range(rr, h, row_period)
                for c in range(cc, w, col_period)
            ]
            preserve.add(random.choice(residue_coords))

    informative = _informative_residues(tile)
    eligible = [coord for coord in all_coords if coord not in preserve]
    if len(eligible) < 2:
        return None

    total = h * w
    min_missing = max(2, int(np.ceil(total * 0.18)))
    max_missing = min(len(eligible), max(min_missing, int(np.floor(total * 0.35))))
    if min_missing > max_missing:
        return None

    for _ in range(250):
        count = random.randint(min_missing, max_missing)
        chosen = random.sample(eligible, count)
        row_residues = {r % row_period for r, _c in chosen}
        col_residues = {c % col_period for _r, c in chosen}
        residue_pairs = {(r % row_period, c % col_period) for r, c in chosen}
        color_set = {int(blocks[r, c]) for r, c in chosen}
        if len(residue_pairs) < min(2, row_period * col_period):
            continue
        if len(color_set) < min(2, len(np.unique(tile))):
            continue
        if row_period > 1 and len(row_residues) < 2:
            continue
        if col_period > 1 and len(col_residues) < 2:
            continue
        if informative and not any(((r % row_period), (c % col_period)) in informative for r, c in chosen):
            continue

        masked = blocks.copy()
        for r, c in chosen:
            masked[r, c] = MISSING
        try:
            rp, cp, inferred = _infer_unique_minimal_period(masked, missing=MISSING)
        except AssertionError:
            continue
        if (rp, cp) != tile.shape:
            continue
        if not np.array_equal(inferred, tile):
            continue
        return masked
    return None


def _has_smaller_pixel_period(grid: np.ndarray) -> bool:
    h, w = grid.shape
    full_area = h * w
    for row_period in range(1, h + 1):
        for col_period in range(1, w + 1):
            area = row_period * col_period
            if area >= full_area:
                continue
            if _consistent_tile(grid, row_period, col_period, missing=None) is not None:
                return True
    return False


def _sample_periods() -> tuple[int, int]:
    return random.choice([(1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3)])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress separator-cut color blocks into block coordinates, then restore pink-covered blocks from the unique block-level period.

    Input structure:
    - Full grey separator rows and columns split the grid into a rectangular array of content blocks.
    - Every content block is a solid color, while some entire blocks are replaced by pink.
    - Block heights and widths vary, so geometric sizes do not encode the period directly.

    Rule:
    - Remove the grey separator rows and columns conceptually, treating each solid block as one compressed cell.
    - In that compressed block grid, treat pink as a missing value and infer the unique minimum-area repeating tile.
    - Repaint each pink block in the original grid with the inferred block color, leaving all separators and visible blocks unchanged.

    Key insight:
    - The periodic pattern exists only after coordinate compression: unequal block sizes break any direct pixel-level periodic interpretation.
    """
    compressed, row_runs, col_runs = _compress_blocks(np.asarray(grid, dtype=int))
    row_period, col_period, tile = _infer_unique_minimal_period(compressed, missing=MISSING)

    out = np.asarray(grid, dtype=int).copy()
    for rr in range(compressed.shape[0]):
        for cc in range(compressed.shape[1]):
            if int(compressed[rr, cc]) != MISSING:
                continue
            color = int(tile[rr % row_period, cc % col_period])
            r0, r1 = row_runs[rr]
            c0, c1 = col_runs[cc]
            out[r0:r1, c0:c1] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    row_period, col_period = _sample_periods()
    block_rows = random.choice([2 * row_period, 2 * row_period + 1])
    block_cols = random.choice([2 * col_period, 2 * col_period + 1])

    row_sizes, row_separators = _sample_axis_sizes(block_rows)
    col_sizes, col_separators = _sample_axis_sizes(block_cols)

    tile = _sample_tile(row_period, col_period)
    rr = np.arange(block_rows)[:, None] % row_period
    cc = np.arange(block_cols)[None, :] % col_period
    output_blocks = tile[rr, cc].astype(int)

    input_blocks = _choose_missing_blocks(output_blocks, tile)
    if input_blocks is None:
        return None

    inp = _expand_blocks(input_blocks, row_sizes, col_sizes, row_separators, col_separators)
    out = _expand_blocks(output_blocks, row_sizes, col_sizes, row_separators, col_separators)

    if max(inp.shape) > MAX_SIDE or max(out.shape) > MAX_SIDE:
        return None
    if not _has_smaller_pixel_period(out):
        try:
            compressed, _row_runs, _col_runs = _compress_blocks(inp)
            rp, cp, inferred = _infer_unique_minimal_period(compressed, missing=MISSING)
        except AssertionError:
            return None
        if (rp, cp) != tile.shape or not np.array_equal(inferred, tile):
            return None
        solved = main(inp)
        if not np.array_equal(solved, out):
            return None
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid block-coordinate-period puzzle")
