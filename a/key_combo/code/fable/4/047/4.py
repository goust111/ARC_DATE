from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND = 0
NON_BACKGROUND_COLORS = tuple(range(1, 10))
MODE_COLOR_MAP = "color_map"
MODE_PERMUTE = "permute"
MAX_GENERATION_ATTEMPTS = 200


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(mask.tolist() + [False]):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append((start, idx - 1))
            start = None
    return runs


def _all_unique(tile: np.ndarray) -> bool:
    flat = tile.reshape(-1).tolist()
    return len(set(map(int, flat))) == len(flat)


def _canonical_pattern(tile: np.ndarray) -> tuple[int, ...]:
    mapping: dict[int, int] = {}
    next_id = 0
    signature: list[int] = []
    for value in tile.reshape(-1):
        color = int(value)
        if color not in mapping:
            mapping[color] = next_id
            next_id += 1
        signature.append(mapping[color])
    return tuple(signature)


def _infer_color_map(src: np.ndarray, dst: np.ndarray) -> dict[int, int] | None:
    if src.shape != dst.shape or np.array_equal(src, dst):
        return None
    if _all_unique(src) or _all_unique(dst):
        return None

    forward: dict[int, int] = {}
    reverse: dict[int, int] = {}
    for left, right in zip(src.reshape(-1), dst.reshape(-1)):
        a = int(left)
        b = int(right)
        if a in forward and forward[a] != b:
            return None
        if b in reverse and reverse[b] != a:
            return None
        forward[a] = b
        reverse[b] = a
    return forward


def _apply_color_map(tile: np.ndarray, color_map: dict[int, int]) -> np.ndarray | None:
    out = np.empty_like(tile)
    for color in np.unique(tile):
        key = int(color)
        if key not in color_map:
            return None
        out[tile == color] = color_map[key]
    return out


def _infer_permutation(src: np.ndarray, dst: np.ndarray) -> list[int] | None:
    if src.shape != dst.shape or np.array_equal(src, dst):
        return None
    flat_src = [int(v) for v in src.reshape(-1)]
    flat_dst = [int(v) for v in dst.reshape(-1)]
    if len(set(flat_src)) != len(flat_src) or len(set(flat_dst)) != len(flat_dst):
        return None
    if Counter(flat_src) != Counter(flat_dst):
        return None

    dst_positions = {color: idx for idx, color in enumerate(flat_dst)}
    permutation = [dst_positions[color] for color in flat_src]
    if all(idx == target for idx, target in enumerate(permutation)):
        return None
    return permutation


def _apply_permutation(tile: np.ndarray, permutation: list[int]) -> np.ndarray:
    flat = tile.reshape(-1)
    out = np.empty_like(flat)
    for src_idx, dst_idx in enumerate(permutation):
        out[dst_idx] = flat[src_idx]
    return out.reshape(tile.shape)


def _parse_bands(grid: np.ndarray, bg: int) -> list[tuple[tuple[int, int], list[tuple[int, int]], list[np.ndarray]]]:
    bands: list[tuple[tuple[int, int], list[tuple[int, int]], list[np.ndarray]]] = []
    row_runs = _segments(np.any(grid != bg, axis=1))
    assert row_runs, "expected at least one occupied horizontal band"
    for r0, r1 in row_runs:
        band = grid[r0 : r1 + 1]
        col_runs = _segments(np.any(band != bg, axis=0))
        assert len(col_runs) == 3, "each band must contain exactly three tiles"
        tiles = [band[:, c0 : c1 + 1].copy() for c0, c1 in col_runs]
        shapes = {tile.shape for tile in tiles}
        assert len(shapes) == 1, "all tiles in a band must have the same shape"
        bands.append(((r0, r1), col_runs, tiles))
    return bands


def _solve_band(tiles: list[np.ndarray]) -> tuple[int, np.ndarray]:
    candidates: list[tuple[str, int, np.ndarray]] = []
    triplets = ((0, 1, 2), (0, 2, 1), (1, 2, 0))
    for src_idx, dst_idx, target_idx in triplets:
        src = tiles[src_idx]
        dst = tiles[dst_idx]
        target = tiles[target_idx]

        color_map = _infer_color_map(src, dst)
        if color_map is not None:
            mapped = _apply_color_map(target, color_map)
            if mapped is not None:
                candidates.append((MODE_COLOR_MAP, target_idx, mapped))

        permutation = _infer_permutation(src, dst)
        if permutation is not None:
            candidates.append((MODE_PERMUTE, target_idx, _apply_permutation(target, permutation)))

    assert len(candidates) == 1, "band must admit exactly one example-pair interpretation"
    _mode, target_idx, transformed = candidates[0]
    return target_idx, transformed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace the third tile in each horizontal band by applying the hidden example transformation.

    Input structure:
    - The background rows split the grid into independent horizontal bands.
    - Each band contains exactly three same-sized rectangular tiles separated by background columns.
    - In every band, two tiles form a left-to-right example pair and the remaining tile is the query tile.

    Rule:
    - Test each left-to-right tile pair in the band as the candidate example pair.
    - If the pair preserves the cellwise pattern but changes repeated colors, infer the color bijection and recolor the third tile.
    - If the pair preserves a unique-color multiset but changes positions, infer the cell permutation and rearrange the third tile.
    - Keep the full layout unchanged and replace only the query tile with its transformed version.

    Key insight:
    - The hidden variable is the transformation mode itself: repeated-color example pairs teach recoloring, while unique-color conserved pairs teach a positional permutation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _most_common_color(g)
    out = g.copy()
    for (r0, r1), col_runs, tiles in _parse_bands(g, bg):
        target_idx, transformed = _solve_band(tiles)
        c0, c1 = col_runs[target_idx]
        out[r0 : r1 + 1, c0 : c1 + 1] = transformed
    return out


def _random_shape(cells_min: int, cells_max: int) -> tuple[int, int]:
    choices = []
    for h in range(2, 5):
        for w in range(2, 5):
            cells = h * w
            if cells_min <= cells <= cells_max:
                choices.append((h, w))
    return random.choice(choices)


def _fill_with_palette(shape: tuple[int, int], palette: list[int]) -> np.ndarray:
    h, w = shape
    flat = [random.choice(palette) for _ in range(h * w)]
    for idx, color in enumerate(palette):
        flat[idx] = color
    random.shuffle(flat)
    return np.array(flat, dtype=int).reshape(shape)


def _distinct_pattern_tile(shape: tuple[int, int], palette: list[int], forbidden_sig: tuple[int, ...]) -> np.ndarray:
    while True:
        tile = _fill_with_palette(shape, palette)
        if len(set(tile.reshape(-1).tolist())) == 1:
            continue
        if _canonical_pattern(tile) != forbidden_sig:
            return tile


def _color_map_band() -> np.ndarray:
    shape = _random_shape(4, 12)
    cells = shape[0] * shape[1]
    domain_size = random.randint(2, min(4, cells - 1, len(NON_BACKGROUND_COLORS) // 2))
    domain = random.sample(NON_BACKGROUND_COLORS, domain_size)
    codomain_pool = [c for c in NON_BACKGROUND_COLORS if c not in domain]
    codomain = random.sample(codomain_pool, domain_size)
    color_map = dict(zip(domain, codomain))

    example_src = _fill_with_palette(shape, domain)
    while _all_unique(example_src):
        example_src = _fill_with_palette(shape, domain)
    example_dst = np.vectorize(color_map.get)(example_src).astype(int)

    query_palette = random.sample(domain, random.randint(2, len(domain))) if len(domain) > 1 else domain
    query = _distinct_pattern_tile(shape, query_palette, _canonical_pattern(example_src))

    tiles = [None, None, None]
    query_idx = random.randrange(3)
    example_positions = [idx for idx in range(3) if idx != query_idx]
    tiles[example_positions[0]] = example_src
    tiles[example_positions[1]] = example_dst
    tiles[query_idx] = query
    return _render_band(tiles)


def _random_non_identity_permutation(n: int) -> list[int]:
    perm = list(range(n))
    while True:
        random.shuffle(perm)
        if any(idx != val for idx, val in enumerate(perm)):
            return perm


def _permute_flat(tile: np.ndarray, permutation: list[int]) -> np.ndarray:
    flat = tile.reshape(-1)
    out = np.empty_like(flat)
    for src_idx, dst_idx in enumerate(permutation):
        out[dst_idx] = flat[src_idx]
    return out.reshape(tile.shape)


def _permutation_band() -> np.ndarray:
    shape = _random_shape(4, 9)
    cell_count = shape[0] * shape[1]
    colors = random.sample(NON_BACKGROUND_COLORS, cell_count)
    random.shuffle(colors)
    example_src = np.array(colors, dtype=int).reshape(shape)
    permutation = _random_non_identity_permutation(cell_count)
    example_dst = _permute_flat(example_src, permutation)

    query_palette = random.sample(NON_BACKGROUND_COLORS, random.randint(2, min(4, cell_count - 1)))
    while True:
        query = _fill_with_palette(shape, query_palette)
        if _apply_permutation(query, permutation).tolist() != query.tolist():
            break

    tiles = [None, None, None]
    query_idx = random.randrange(3)
    example_positions = [idx for idx in range(3) if idx != query_idx]
    tiles[example_positions[0]] = example_src
    tiles[example_positions[1]] = example_dst
    tiles[query_idx] = query
    return _render_band(tiles)


def _render_band(tiles: list[np.ndarray | None]) -> np.ndarray:
    assert all(tile is not None for tile in tiles)
    typed_tiles = [tile for tile in tiles if tile is not None]
    h, w = typed_tiles[0].shape
    gap = random.randint(1, 2)
    band = np.full((h, 3 * w + 2 * gap), BACKGROUND, dtype=int)
    for idx, tile in enumerate(typed_tiles):
        c0 = idx * (w + gap)
        band[:, c0 : c0 + w] = tile
    return band


def _build_grid(bands: list[np.ndarray]) -> np.ndarray:
    top_margin = random.randint(0, 2)
    bottom_margin = random.randint(0, 2)
    band_gaps = [random.randint(1, 2) for _ in range(len(bands) - 1)]
    left_offsets = [random.randint(0, 3) for _ in bands]
    right_margin = random.randint(0, 3)

    height = top_margin + bottom_margin + sum(band.shape[0] for band in bands) + sum(band_gaps)
    width = max(offset + band.shape[1] for offset, band in zip(left_offsets, bands)) + right_margin
    assert height <= 30 and width <= 30

    grid = np.full((height, width), BACKGROUND, dtype=int)
    row = top_margin
    for idx, band in enumerate(bands):
        offset = left_offsets[idx]
        h, w = band.shape
        grid[row : row + h, offset : offset + w] = band
        row += h
        if idx < len(bands) - 1:
            row += band_gaps[idx]
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        num_bands = random.randint(2, 4)
        modes = [MODE_COLOR_MAP, MODE_PERMUTE]
        while len(modes) < num_bands:
            modes.append(random.choice((MODE_COLOR_MAP, MODE_PERMUTE)))
        random.shuffle(modes)

        bands = [_color_map_band() if mode == MODE_COLOR_MAP else _permutation_band() for mode in modes]
        grid = _build_grid(bands)
        solved = main(grid)
        if solved.shape == grid.shape and np.any(solved != grid):
            return grid
    raise RuntimeError("failed to generate a valid input grid")
