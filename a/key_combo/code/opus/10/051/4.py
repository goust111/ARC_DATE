from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
INPUT_FOREGROUND = 5
RANK_COLORS = tuple(range(1, 10))
GEN_TILE_SIZES = (4, 5)
GEN_COUNTS = tuple(range(2, 15))
GEN_PATTERN_MODES = ("cluster", "dual_cluster", "serpentine", "anchor_bias")
GEN_MAX_ATTEMPTS = 300


def _separator_positions(mask: np.ndarray) -> list[int]:
    idx = np.flatnonzero(mask)
    groups: list[list[int]] = []
    for pos in map(int, idx):
        if groups and pos == groups[-1][-1] + 1:
            groups[-1].append(pos)
        else:
            groups.append([pos])
    assert len(groups) == 2, "expected exactly two separator lines"
    assert all(len(group) == 1 for group in groups), "separator lines must be one cell wide"
    return [group[0] for group in groups]


def _tile_spans(size: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for sep in separators:
        spans.append((start, sep))
        start = sep + 1
    spans.append((start, size))
    assert len(spans) == 3
    lengths = {end - start for start, end in spans}
    assert len(lengths) == 1 and next(iter(lengths)) > 0, "tiles must be equal-sized"
    return spans


def _parse_tiles(grid: np.ndarray) -> tuple[list[tuple[int, int, int, int]], int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    row_seps = _separator_positions(np.all(g == BACKGROUND, axis=1))
    col_seps = _separator_positions(np.all(g == BACKGROUND, axis=0))
    assert 0 not in row_seps and g.shape[0] - 1 not in row_seps
    assert 0 not in col_seps and g.shape[1] - 1 not in col_seps

    row_spans = _tile_spans(g.shape[0], row_seps)
    col_spans = _tile_spans(g.shape[1], col_seps)
    tile_h = row_spans[0][1] - row_spans[0][0]
    tile_w = col_spans[0][1] - col_spans[0][0]

    tiles: list[tuple[int, int, int, int]] = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            patch = g[r0:r1, c0:c1]
            assert np.all(np.isin(patch, [BACKGROUND, INPUT_FOREGROUND])), "unexpected tile colors"
            tiles.append((r0, r1, c0, c1))
    assert len(tiles) == 9
    return tiles, tile_h, tile_w


def main(grid: np.ndarray) -> np.ndarray:
    """
    Invert each tile and color its new foreground by the tile's original grey-density rank.

    Input structure:
    - The grid is split by one-cell-wide black separator rows and columns into a 3x3 array of equal tiles.
    - Inside each tile, cells are only black background or grey foreground.
    - The nine tiles have pairwise distinct counts of grey cells.

    Rule:
    - Count the grey cells in every tile and rank the tiles from fewest grey cells to most.
    - Keep the separator rows and columns black.
    - Within each tile, invert occupancy: old grey cells become black and old black cells become foreground.
    - Color every new foreground cell in that tile with the color assigned to its original rank: rank 1->1, rank 2->2, ..., rank 9->9.

    Key insight:
    - The ranking is computed before inversion, but the ranked color is applied only to the pixels created by the inversion, so visible output area runs opposite to the ranking source.
    """
    g = np.asarray(grid, dtype=int)
    tiles, _tile_h, _tile_w = _parse_tiles(g)

    counts = []
    for r0, r1, c0, c1 in tiles:
        patch = g[r0:r1, c0:c1]
        counts.append(int(np.count_nonzero(patch == INPUT_FOREGROUND)))

    assert len(set(counts)) == 9, "tile grey counts must be unique"

    ranks: dict[int, int] = {}
    for rank, tile_idx in enumerate(np.argsort(counts), start=1):
        ranks[int(tile_idx)] = int(RANK_COLORS[rank - 1])

    out = np.zeros_like(g)
    for tile_idx, (r0, r1, c0, c1) in enumerate(tiles):
        patch = g[r0:r1, c0:c1]
        out[r0:r1, c0:c1][patch == BACKGROUND] = ranks[tile_idx]

    return out


def _grow_cells(tile_h: int, tile_w: int, count: int, seeds: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells = set(seeds)
    frontier = set(seeds)
    while len(cells) < count:
        if not frontier:
            frontier = set(cells)
        base_r, base_c = random.choice(tuple(frontier))
        frontier.discard((base_r, base_c))
        nbrs = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr = base_r + dr
            cc = base_c + dc
            if 0 <= rr < tile_h and 0 <= cc < tile_w and (rr, cc) not in cells:
                nbrs.append((rr, cc))
        if not nbrs:
            continue
        pick = random.choice(nbrs)
        cells.add(pick)
        frontier.add(pick)
    return cells


def _ordered_cells(tile_h: int, tile_w: int, mode: str) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(tile_h) for c in range(tile_w)]
    if mode == "serpentine":
        rows = list(range(tile_h))
        cols = list(range(tile_w))
        random.shuffle(rows)
        random.shuffle(cols)
        order: list[tuple[int, int]] = []
        for i, r in enumerate(rows):
            row_cols = cols if i % 2 == 0 else list(reversed(cols))
            for c in row_cols:
                order.append((r, c))
        return order

    anchor = (random.randrange(tile_h), random.randrange(tile_w))
    return sorted(
        cells,
        key=lambda rc: (
            abs(rc[0] - anchor[0]) + abs(rc[1] - anchor[1]),
            (rc[0] - anchor[0]) ** 2 + (rc[1] - anchor[1]) ** 2,
            random.random(),
        ),
    )


def _make_tile_mask(tile_h: int, tile_w: int, count: int) -> np.ndarray:
    area = tile_h * tile_w
    assert 2 <= count <= min(14, area - 2)

    mode = random.choice(GEN_PATTERN_MODES)
    if mode == "cluster":
        seed = (random.randrange(tile_h), random.randrange(tile_w))
        cells = _grow_cells(tile_h, tile_w, count, [seed])
    elif mode == "dual_cluster":
        all_cells = [(r, c) for r in range(tile_h) for c in range(tile_w)]
        seed_a, seed_b = random.sample(all_cells, 2)
        cells = _grow_cells(tile_h, tile_w, count, [seed_a, seed_b])
    else:
        order = _ordered_cells(tile_h, tile_w, mode)
        cells = set(order[:count])

    mask = np.zeros((tile_h, tile_w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _wrong_no_inversion(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    tiles, _tile_h, _tile_w = _parse_tiles(g)
    counts = [int(np.count_nonzero(g[r0:r1, c0:c1] == INPUT_FOREGROUND)) for r0, r1, c0, c1 in tiles]
    out = np.zeros_like(g)
    for rank, tile_idx in enumerate(np.argsort(counts), start=1):
        r0, r1, c0, c1 = tiles[int(tile_idx)]
        patch = g[r0:r1, c0:c1]
        out[r0:r1, c0:c1][patch == INPUT_FOREGROUND] = rank
    return out


def _wrong_rank_after_inversion(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    tiles, tile_h, tile_w = _parse_tiles(g)
    area = tile_h * tile_w
    counts = [area - int(np.count_nonzero(g[r0:r1, c0:c1] == INPUT_FOREGROUND)) for r0, r1, c0, c1 in tiles]
    out = np.zeros_like(g)
    for rank, tile_idx in enumerate(np.argsort(counts), start=1):
        r0, r1, c0, c1 = tiles[int(tile_idx)]
        patch = g[r0:r1, c0:c1]
        out[r0:r1, c0:c1][patch == BACKGROUND] = rank
    return out


def _sample_counts(tile_h: int, tile_w: int) -> list[int]:
    area = tile_h * tile_w
    max_count = min(14, area - 2)
    assert max_count >= 10
    middle = random.sample(list(range(3, max_count)), 7)
    counts = [2, max_count, *middle]
    random.shuffle(counts)
    return counts


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
        tiles, tile_h, tile_w = _parse_tiles(grid)
    except AssertionError:
        return False

    if np.array_equal(out, _wrong_no_inversion(grid)):
        return False
    if np.array_equal(out, _wrong_rank_after_inversion(grid)):
        return False

    counts = [int(np.count_nonzero(grid[r0:r1, c0:c1] == INPUT_FOREGROUND)) for r0, r1, c0, c1 in tiles]
    if len(set(counts)) != 9:
        return False

    area = tile_h * tile_w
    ranked = sorted((count, idx) for idx, count in enumerate(counts))
    for rank, (count, idx) in enumerate(ranked, start=1):
        r0, r1, c0, c1 = tiles[idx]
        patch = out[r0:r1, c0:c1]
        color_cells = int(np.count_nonzero(patch == rank))
        nonzero_cells = int(np.count_nonzero(patch))
        if color_cells != area - count or nonzero_cells != area - count:
            return False
        if np.any((grid[r0:r1, c0:c1] == INPUT_FOREGROUND) & (patch != BACKGROUND)):
            return False

    row_seps = np.flatnonzero(np.all(grid == BACKGROUND, axis=1))
    col_seps = np.flatnonzero(np.all(grid == BACKGROUND, axis=0))
    return len(row_seps) == 2 and len(col_seps) == 2


def _build_candidate() -> np.ndarray:
    tile_h = random.choice(GEN_TILE_SIZES)
    tile_w = random.choice(GEN_TILE_SIZES)
    grid_h = tile_h * 3 + 2
    grid_w = tile_w * 3 + 2
    grid = np.zeros((grid_h, grid_w), dtype=int)

    counts = _sample_counts(tile_h, tile_w)
    for idx, count in enumerate(counts):
        tile_r = idx // 3
        tile_c = idx % 3
        r0 = tile_r * (tile_h + 1)
        c0 = tile_c * (tile_w + 1)
        mask = _make_tile_mask(tile_h, tile_w, count)
        grid[r0 : r0 + tile_h, c0 : c0 + tile_w][mask] = INPUT_FOREGROUND
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _build_candidate()
        if _semantic_checks(grid):
            return grid
    raise RuntimeError("failed to generate a valid tile-inversion density-ranking grid")
