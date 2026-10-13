from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
FOREGROUND = 2
GRID_SHAPE = (11, 11)
TILE_SIZE = 3
SEPARATORS = (3, 7)
SLOT_ORIGINS = (0, 4, 8)
RETRY_LIMIT = 400
ORDER_DISTRIBUTIONS = (
    (2, 3, 4),
    (2, 4, 3),
    (3, 2, 4),
    (3, 3, 3),
    (4, 2, 3),
    (1, 4, 4),
    (1, 5, 3),
    (2, 2, 5),
    (2, 5, 2),
    (3, 4, 2),
    (4, 3, 2),
)


def _mask_key(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in mask.astype(int).ravel())


def _render_mask(mask: np.ndarray) -> np.ndarray:
    tile = np.zeros((TILE_SIZE, TILE_SIZE), dtype=int)
    tile[mask] = FOREGROUND
    return tile


def _rotation_order(mask: np.ndarray) -> int:
    if np.array_equal(mask, np.rot90(mask, -1)):
        return 4
    if np.array_equal(mask, np.rot90(mask, 2)):
        return 2
    return 1


def _enumerate_pattern_pool() -> dict[tuple[int, int], list[np.ndarray]]:
    buckets: dict[tuple[int, int], list[np.ndarray]] = {}
    for bits in range(1, (1 << (TILE_SIZE * TILE_SIZE)) - 1):
        mask = np.array([(bits >> i) & 1 for i in range(TILE_SIZE * TILE_SIZE)], dtype=bool).reshape(TILE_SIZE, TILE_SIZE)
        count = int(mask.sum())
        if not 1 <= count <= 8:
            continue
        order = _rotation_order(mask)
        buckets.setdefault((order, count), []).append(mask)
    return buckets


PATTERN_POOL = _enumerate_pattern_pool()
AVAILABLE_COUNTS = {
    order: sorted(count for level, count in PATTERN_POOL if level == order)
    for order in (4, 2, 1)
}


def _extract_tiles(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.shape == GRID_SHAPE, f"expected {GRID_SHAPE}, got {g.shape}"
    for sep in SEPARATORS:
        assert np.all(g[sep, :] == BACKGROUND), "separator rows must be black"
        assert np.all(g[:, sep] == BACKGROUND), "separator columns must be black"

    tiles = []
    for r0 in SLOT_ORIGINS:
        for c0 in SLOT_ORIGINS:
            tile = g[r0 : r0 + TILE_SIZE, c0 : c0 + TILE_SIZE]
            assert tile.shape == (TILE_SIZE, TILE_SIZE)
            assert set(int(x) for x in tile.ravel()) <= {BACKGROUND, FOREGROUND}, "tiles must use only black and red"
            tiles.append(tile.copy())
    return tiles


def _annotate_tiles(tiles: list[np.ndarray]) -> list[tuple[int, int, tuple[int, ...], np.ndarray]]:
    annotated = []
    for tile in tiles:
        mask = tile == FOREGROUND
        annotated.append((_rotation_order(mask), int(mask.sum()), _mask_key(mask), tile.copy()))
    return annotated


def _build_grid(tiles: list[np.ndarray]) -> np.ndarray:
    assert len(tiles) == 9
    grid = np.zeros(GRID_SHAPE, dtype=int)
    idx = 0
    for r0 in SLOT_ORIGINS:
        for c0 in SLOT_ORIGINS:
            grid[r0 : r0 + TILE_SIZE, c0 : c0 + TILE_SIZE] = tiles[idx]
            idx += 1
    return grid


def _sorted_sequence(annotated: list[tuple[int, int, tuple[int, ...], np.ndarray]]) -> list[tuple[int, ...]]:
    return [key for _order, _count, key, _tile in sorted(annotated, key=lambda item: (-item[0], item[1]))]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder the 3x3 tiles by rotation-symmetry order, breaking ties by foreground count.

    Input structure:
    - The input is a fixed 11x11 grid split by black separator rows 3 and 7 and black separator columns 3 and 7.
    - These separators create a 3x3 array of 3x3 tiles.
    - Each tile is a red-on-black pattern, and all nine tile patterns are distinct.

    Rule:
    - For each tile, measure its rotational symmetry order: 4 if a 90-degree rotation preserves it, 2 if only a 180-degree rotation preserves it, and 1 otherwise.
    - Sort tiles by symmetry order from high to low; among tiles with the same order, sort by red-pixel count from low to high.
    - Place the sorted tiles back into the nine slots in ordinary row-major order, without changing any tile internally.

    Key insight:
    - The decisive property is a derived geometric invariant found by mentally rotating each tile; pixel count matters only as the tie-breaker within a symmetry class.
    """
    tiles = _extract_tiles(grid)
    annotated = _annotate_tiles(tiles)

    assert len({key for _order, _count, key, _tile in annotated}) == 9, "all nine tile patterns must be distinct"
    order_counts: dict[int, list[int]] = {4: [], 2: [], 1: []}
    for order, count, _key, _tile in annotated:
        order_counts[order].append(count)
    assert all(order_counts[order] for order in (4, 2, 1)), "expected at least one tile of each symmetry order"
    assert all(len(counts) == len(set(counts)) for counts in order_counts.values()), "counts must be unique within each order"

    ordered_tiles = [tile for _order, _count, _key, tile in sorted(annotated, key=lambda item: (-item[0], item[1]))]
    return _build_grid(ordered_tiles)


def _sample_tile_specs() -> list[tuple[int, int, np.ndarray]]:
    n4, n2, n1 = random.choice(ORDER_DISTRIBUTIONS)
    chosen_specs: list[tuple[int, int, np.ndarray]] = []
    used_keys: set[tuple[int, ...]] = set()
    for order, needed in ((4, n4), (2, n2), (1, n1)):
        counts = random.sample(AVAILABLE_COUNTS[order], needed)
        for count in counts:
            bucket = PATTERN_POOL[(order, count)]
            candidates = [mask for mask in bucket if _mask_key(mask) not in used_keys]
            mask = random.choice(candidates).copy()
            used_keys.add(_mask_key(mask))
            chosen_specs.append((order, count, mask))
    assert len(chosen_specs) == 9
    return chosen_specs


def _tiles_from_specs(specs: list[tuple[int, int, np.ndarray]]) -> list[np.ndarray]:
    return [_render_mask(mask) for _order, _count, mask in specs]


def _validate_scene(inp: np.ndarray) -> bool:
    try:
        tiles = _extract_tiles(inp)
        annotated = _annotate_tiles(tiles)
        out = main(inp)
    except Exception:
        return False

    output_tiles = _extract_tiles(out)
    correct_sequence = _sorted_sequence(annotated)
    input_sequence = [key for _order, _count, key, _tile in annotated]
    count_only_sequence = [key for _order, _count, key, _tile in sorted(annotated, key=lambda item: item[1])]
    order_only_sequence = [key for _order, _count, key, _tile in sorted(annotated, key=lambda item: -item[0])]

    order_counts: dict[int, list[int]] = {4: [], 2: [], 1: []}
    for order, count, _key, _tile in annotated:
        order_counts[order].append(count)

    return all(
        [
            len({key for _order, _count, key, _tile in annotated}) == 9,
            all(order_counts[order] for order in (4, 2, 1)),
            any(len(order_counts[order]) >= 2 for order in (4, 2, 1)),
            all(len(counts) == len(set(counts)) for counts in order_counts.values()),
            input_sequence != correct_sequence,
            count_only_sequence != correct_sequence,
            order_only_sequence != correct_sequence,
            [ _mask_key(tile == FOREGROUND) for tile in output_tiles ] == correct_sequence,
        ]
    )


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        specs = _sample_tile_specs()
        random.shuffle(specs)
        grid = _build_grid(_tiles_from_specs(specs))
        if _validate_scene(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
