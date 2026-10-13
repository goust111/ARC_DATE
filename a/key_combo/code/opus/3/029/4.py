from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


BG = 0
DIVIDER = 5
MAGENTA = 6
GROUP_A = (1, 2, 3)
GROUP_B = (4, 7, 8)
BAR_HEIGHTS = (2, 3, 4)
TARGET_COUNTS = tuple(range(9))
TILE_SIZES = (7, 8, 9)

POOL_PER_COUNT = 24
POOL_MAX_ATTEMPTS = 60000
SELECTION_ATTEMPTS = 500
GENERATION_ATTEMPTS = 300

_POOL_CACHE: dict[int, dict[int, list[dict[str, object]]]] = {}


def _segments(length: int, dividers: list[int]) -> list[tuple[int, int]]:
    segments = []
    start = 0
    for divider in dividers + [length]:
        segments.append((start, divider))
        start = divider + 1
    return segments


def _outward_offset(row: int, r0: int, r1: int, bar_height: int) -> int:
    if row < r0:
        return ((r0 - 1 - row) // bar_height) + 1
    if row > r1:
        return ((row - r1 - 1) // bar_height) + 1
    return 0


def _vertical_bar_pair(tile: np.ndarray, color: int) -> tuple[int, int, int, int]:
    coords = np.argwhere(tile == int(color))
    assert coords.size > 0, "expected a visible bar color"

    cols = sorted(set(map(int, coords[:, 1])))
    assert len(cols) == 2, "each tile color must form exactly two vertical bars"

    bars: list[tuple[int, int, int]] = []
    for col in cols:
        rows = sorted(map(int, coords[coords[:, 1] == col, 0]))
        assert rows == list(range(rows[0], rows[-1] + 1)), "bar pixels must be contiguous"
        bars.append((rows[0], rows[-1], col))

    left, right = bars
    assert left[0] == right[0] and left[1] == right[1], "paired bars must share one row interval"
    return left[0], left[1], left[2], right[2]


def _expand_tile(tile: np.ndarray) -> np.ndarray:
    tile = np.asarray(tile, dtype=int)
    height, width = tile.shape
    source_colors = sorted(set(map(int, tile.ravel())) - {BG})
    assert len(source_colors) == 2, "each tile must contain exactly two source colors"
    assert MAGENTA not in source_colors and DIVIDER not in source_colors

    claims: dict[tuple[int, int], set[int]] = defaultdict(set)
    for color in source_colors:
        r0, r1, left_col, right_col = _vertical_bar_pair(tile, color)
        bar_height = r1 - r0 + 1
        for row in range(height):
            offset = _outward_offset(row, r0, r1, bar_height)
            for col, direction in ((left_col, -1), (right_col, 1)):
                stepped_col = col + direction * offset
                if 0 <= stepped_col < width:
                    claims[(row, stepped_col)].add(int(color))

    out = np.full(tile.shape, BG, dtype=int)
    for (row, col), colors in claims.items():
        out[row, col] = next(iter(colors)) if len(colors) == 1 else MAGENTA
    return out


def _find_layout(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    grid = np.asarray(grid, dtype=int)
    height, width = grid.shape
    candidates: list[tuple[int, list[tuple[int, int]], list[tuple[int, int]]]] = []
    for color in map(int, np.unique(grid)):
        rows = [r for r in range(height) if np.all(grid[r, :] == color)]
        cols = [c for c in range(width) if np.all(grid[:, c] == color)]
        if len(rows) != 2 or len(cols) != 2:
            continue
        row_segments = _segments(height, rows)
        col_segments = _segments(width, cols)
        if len(row_segments) != 3 or len(col_segments) != 3:
            continue
        row_sizes = [r1 - r0 for r0, r1 in row_segments]
        col_sizes = [c1 - c0 for c0, c1 in col_segments]
        if len(set(row_sizes)) == 1 and row_sizes == col_sizes:
            candidates.append((color, row_segments, col_segments))

    assert len(candidates) == 1, "expected one 3x3 divider layout"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Expand each tile's two bar pairs, count magenta overlaps, and sort the tiles by that count.

    Input structure:
    - A grey full-row/full-column divider splits the grid into a 3x3 array of equal square tile slots.
    - Each tile contains exactly two colors on a black background.
    - Each tile color appears as one pair of same-height vertical bars that share a row interval.

    Rule:
    - Inside each tile, extend every bar pair using the stepped outward rule: within the original row interval the bar stays in place, and every full bar-height above or below that interval shifts the left bar one column left and the right bar one column right, clipped to the tile boundary.
    - When the two tile colors claim the same pixel after expansion, recolor that pixel magenta; otherwise keep the claiming color.
    - Count each tile's magenta pixels, then sort the nine expanded tiles by that count in increasing order and place them back row-major from the top-left while leaving the grey divider lines unchanged.

    Key insight:
    - The bars are not the final objects to rank; their heights and positions only matter because the stepped expansion creates a hidden overlap count, and that overlap count is the sorting key.
    """
    g = np.asarray(grid, dtype=int)
    divider_color, row_segments, col_segments = _find_layout(g)

    tiles: list[tuple[int, np.ndarray]] = []
    for r0, r1 in row_segments:
        for c0, c1 in col_segments:
            tile = g[r0:r1, c0:c1]
            expanded = _expand_tile(tile)
            magenta_count = int(np.count_nonzero(expanded == MAGENTA))
            tiles.append((magenta_count, expanded))

    counts = [count for count, _tile in tiles]
    assert len(set(counts)) == len(counts), "magenta counts must be unique"

    out = np.full_like(g, int(divider_color))
    ordered_tiles = [tile for _count, tile in sorted(tiles, key=lambda item: item[0])]
    for tile, (r0, r1), (c0, c1) in zip(
        ordered_tiles,
        [seg for seg in row_segments for _ in col_segments],
        col_segments * len(row_segments),
        strict=False,
    ):
        out[r0:r1, c0:c1] = tile
    return out


def _expand_from_specs(
    tile_size: int,
    pair_specs: tuple[tuple[int, int, int, int], tuple[int, int, int, int]],
    colors: tuple[int, int],
) -> tuple[np.ndarray, int]:
    claims: dict[tuple[int, int], set[int]] = defaultdict(set)
    for color, (bar_height, r0, left_col, right_col) in zip(colors, pair_specs, strict=False):
        r1 = r0 + bar_height - 1
        for row in range(tile_size):
            offset = _outward_offset(row, r0, r1, bar_height)
            for col, direction in ((left_col, -1), (right_col, 1)):
                stepped_col = col + direction * offset
                if 0 <= stepped_col < tile_size:
                    claims[(row, stepped_col)].add(int(color))

    tile = np.full((tile_size, tile_size), BG, dtype=int)
    overlap = 0
    for (row, col), claim_colors in claims.items():
        if len(claim_colors) == 1:
            tile[row, col] = next(iter(claim_colors))
        else:
            tile[row, col] = MAGENTA
            overlap += 1
    return tile, overlap


def _render_input_tile(
    tile_size: int,
    pair_specs: tuple[tuple[int, int, int, int], tuple[int, int, int, int]],
    colors: tuple[int, int],
) -> np.ndarray:
    tile = np.full((tile_size, tile_size), BG, dtype=int)
    for color, (bar_height, r0, left_col, right_col) in zip(colors, pair_specs, strict=False):
        tile[r0 : r0 + bar_height, left_col] = int(color)
        tile[r0 : r0 + bar_height, right_col] = int(color)
    return tile


def _interval_overlap_len(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def _candidate_meta(pair_specs: tuple[tuple[int, int, int, int], tuple[int, int, int, int]]) -> dict[str, object]:
    (h1, r01, lc1, rc1), (h2, r02, lc2, rc2) = pair_specs
    return {
        "height_sig": tuple(sorted((h1, h2))),
        "gap_sig": tuple(sorted((rc1 - lc1, rc2 - lc2))),
        "row_sig": _interval_overlap_len(r01, r01 + h1 - 1, r02, r02 + h2 - 1),
    }


def _sample_pair_specs(tile_size: int, rng: random.Random) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    h1, h2 = rng.sample(BAR_HEIGHTS, 2)
    r01 = rng.randrange(tile_size - h1 + 1)
    r02 = rng.randrange(tile_size - h2 + 1)
    cols = sorted(rng.sample(range(tile_size), 4))
    pairings = (
        ((cols[0], cols[1]), (cols[2], cols[3])),
        ((cols[0], cols[2]), (cols[1], cols[3])),
        ((cols[0], cols[3]), (cols[1], cols[2])),
    )
    (lc1, rc1), (lc2, rc2) = rng.choice(pairings)
    if rng.random() < 0.5:
        return ((h1, r01, lc1, rc1), (h2, r02, lc2, rc2))
    return ((h2, r02, lc2, rc2), (h1, r01, lc1, rc1))


def _candidate_pool(tile_size: int) -> dict[int, list[dict[str, object]]]:
    if tile_size in _POOL_CACHE:
        return _POOL_CACHE[tile_size]

    rng = random.Random(1000 + tile_size)
    pools: dict[int, list[dict[str, object]]] = {count: [] for count in TARGET_COUNTS}
    seen: set[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = set()

    attempts = 0
    while attempts < POOL_MAX_ATTEMPTS and any(len(pools[count]) < POOL_PER_COUNT for count in TARGET_COUNTS):
        attempts += 1
        pair_specs = _sample_pair_specs(tile_size, rng)
        if pair_specs in seen:
            continue
        seen.add(pair_specs)

        _tile, overlap = _expand_from_specs(tile_size, pair_specs, (1, 4))
        if overlap not in pools:
            continue

        candidate = {"pairs": pair_specs, "count": overlap}
        candidate.update(_candidate_meta(pair_specs))
        pools[overlap].append(candidate)

    assert all(len(pools[count]) >= POOL_PER_COUNT for count in TARGET_COUNTS), "failed to build candidate pool"
    _POOL_CACHE[tile_size] = pools
    return pools


def _has_duplicate_signature(
    candidates: list[dict[str, object]],
    key_fn,
) -> bool:
    seen: dict[object, int] = {}
    for candidate in candidates:
        key = key_fn(candidate)
        count = int(candidate["count"])
        if key in seen and seen[key] != count:
            return True
        seen[key] = count
    return False


def _selection_valid(candidates: list[dict[str, object]]) -> bool:
    return (
        _has_duplicate_signature(candidates, lambda item: item["height_sig"])
        and _has_duplicate_signature(candidates, lambda item: item["gap_sig"])
        and _has_duplicate_signature(candidates, lambda item: item["row_sig"])
        and _has_duplicate_signature(candidates, lambda item: (item["height_sig"], item["gap_sig"]))
        and _has_duplicate_signature(candidates, lambda item: (item["height_sig"], item["row_sig"]))
    )


def _pick_candidate_set(tile_size: int, rng: random.Random) -> list[dict[str, object]]:
    pools = _candidate_pool(tile_size)
    for _ in range(SELECTION_ATTEMPTS):
        chosen = [rng.choice(pools[count]) for count in TARGET_COUNTS]
        if _selection_valid(chosen):
            return chosen
    raise RuntimeError("failed to choose a non-degenerate candidate set")


def _tile_features(tile: np.ndarray) -> dict[str, object]:
    colors = sorted(set(map(int, tile.ravel())) - {BG})
    assert len(colors) == 2
    assert colors[0] in GROUP_A and colors[1] in GROUP_B

    pair_a = _vertical_bar_pair(tile, colors[0])
    pair_b = _vertical_bar_pair(tile, colors[1])
    h1 = pair_a[1] - pair_a[0] + 1
    h2 = pair_b[1] - pair_b[0] + 1
    return {
        "height_sig": tuple(sorted((h1, h2))),
        "gap_sig": tuple(sorted((pair_a[3] - pair_a[2], pair_b[3] - pair_b[2]))),
        "row_sig": _interval_overlap_len(pair_a[0], pair_a[1], pair_b[0], pair_b[1]),
        "pixel_count": 2 * (h1 + h2),
    }


def _validate_instance(grid: np.ndarray) -> None:
    divider_color, row_segments, col_segments = _find_layout(grid)
    assert divider_color == DIVIDER

    input_tiles = [grid[r0:r1, c0:c1] for r0, r1 in row_segments for c0, c1 in col_segments]
    expanded_tiles = [_expand_tile(tile) for tile in input_tiles]
    counts = [int(np.count_nonzero(tile == MAGENTA)) for tile in expanded_tiles]
    assert sorted(counts) == list(TARGET_COUNTS)
    assert counts != sorted(counts), "input should not already be in output order"

    features = [_tile_features(tile) for tile in input_tiles]
    assert _has_duplicate_signature(
        [{"count": count, **feature} for count, feature in zip(counts, features, strict=False)],
        lambda item: item["pixel_count"],
    )
    assert _has_duplicate_signature(
        [{"count": count, **feature} for count, feature in zip(counts, features, strict=False)],
        lambda item: item["height_sig"],
    )
    assert _has_duplicate_signature(
        [{"count": count, **feature} for count, feature in zip(counts, features, strict=False)],
        lambda item: (item["height_sig"], item["gap_sig"]),
    )

    out = main(grid)
    out_divider, out_row_segments, out_col_segments = _find_layout(out)
    assert out_divider == divider_color
    assert out_row_segments == row_segments and out_col_segments == col_segments

    ordered_tiles = [tile for _count, tile in sorted(zip(counts, expanded_tiles, strict=False), key=lambda item: item[0])]
    out_tiles = [out[r0:r1, c0:c1] for r0, r1 in row_segments for c0, c1 in col_segments]
    for expected, actual in zip(ordered_tiles, out_tiles, strict=False):
        assert np.array_equal(expected, actual)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        tile_size = rng.choice(TILE_SIZES)
        selected = _pick_candidate_set(tile_size, rng)
        by_count = {int(candidate["count"]): candidate for candidate in selected}

        count_order = list(TARGET_COUNTS)
        rng.shuffle(count_order)
        if count_order == list(TARGET_COUNTS):
            count_order = count_order[1:] + count_order[:1]

        grid_size = 3 * tile_size + 2
        grid = np.full((grid_size, grid_size), BG, dtype=int)
        separator_rows = [tile_size, 2 * tile_size + 1]
        separator_cols = [tile_size, 2 * tile_size + 1]
        grid[separator_rows, :] = DIVIDER
        grid[:, separator_cols] = DIVIDER
        row_segments = _segments(grid_size, separator_rows)
        col_segments = _segments(grid_size, separator_cols)

        for slot_index, count in enumerate(count_order):
            candidate = by_count[count]
            colors = (rng.choice(GROUP_A), rng.choice(GROUP_B))
            tile = _render_input_tile(tile_size, candidate["pairs"], colors)
            r0, r1 = row_segments[slot_index // 3]
            c0, c1 = col_segments[slot_index % 3]
            grid[r0:r1, c0:c1] = tile

        try:
            _validate_instance(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid tile-overlap sorting input")
