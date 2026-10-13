from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

GEN_TILE_SIZE_OPTIONS = (6, 7)
GEN_COUNTS = tuple(range(9))
GEN_MAX_ATTEMPTS = 300
GEN_TILE_ATTEMPTS = 500


def _segments(length: int, dividers: list[int]) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start = 0
    for divider in sorted(dividers) + [int(length)]:
        segments.append((start, int(divider)))
        start = int(divider) + 1
    return segments


def _tile_layout(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_dividers = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == BLACK)]
    col_dividers = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == BLACK)]
    assert len(row_dividers) == 2
    assert len(col_dividers) == 2

    row_segments = _segments(grid.shape[0], row_dividers)
    col_segments = _segments(grid.shape[1], col_dividers)
    assert len(row_segments) == 3
    assert len(col_segments) == 3

    heights = [r1 - r0 for r0, r1 in row_segments]
    widths = [c1 - c0 for c0, c1 in col_segments]
    assert len(set(heights)) == 1
    assert len(set(widths)) == 1
    assert heights[0] == widths[0]
    assert heights[0] >= 2
    return row_segments, col_segments


def _count_greedy_full_blocks(tile: np.ndarray) -> int:
    used = np.zeros(tile.shape, dtype=bool)
    count = 0
    for r in range(tile.shape[0] - 1):
        for c in range(tile.shape[1] - 1):
            if not np.all(tile[r : r + 2, c : c + 2] == BLUE):
                continue
            if np.any(used[r : r + 2, c : c + 2]):
                continue
            used[r : r + 2, c : c + 2] = True
            count += 1
    return count


def _recolor_blue_pixels(tile: np.ndarray, block_count: int) -> np.ndarray:
    out = tile.copy()
    if block_count == 0:
        color = RED
    elif block_count == 1:
        color = GREEN
    else:
        color = YELLOW
    out[out == BLUE] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort a 3x3 array of separated gray/blue tiles by each tile's greedy 2x2 full-blue packing count, then recolor by that count.

    Input structure:
    - Two full black separator rows and two full black separator columns split the canvas into a 3x3 array of equal square tiles.
    - Each tile contains only gray background pixels and blue foreground pixels.
    - Separator lines are black and are not part of any tile.

    Rule:
    - For each tile, scan from top-left to bottom-right and greedily place every non-overlapping 2x2 block whose four cells are all blue.
    - Use the number of placed 2x2 blocks as that tile's sorting key.
    - Sort the nine tiles by increasing key and place them into the output slots from bottom-left to top-right, left-to-right within each row.
    - Recolor all blue pixels in each moved tile by its key: 0 -> red, 1 -> green, 2 or more -> yellow. Keep gray backgrounds and black separator lines unchanged.

    Key insight:
    - The ordering key is a geometric packing count, not a raw blue-pixel count: tiles with similar foreground mass can rank differently if their blue pixels do or do not form complete 2x2 blocks.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_segments, col_segments = _tile_layout(g)
    tiles: list[tuple[int, np.ndarray]] = []
    for r0, r1 in row_segments:
        for c0, c1 in col_segments:
            tile = g[r0:r1, c0:c1]
            assert set(map(int, np.unique(tile))) <= {BLUE, GREY}
            tiles.append((_count_greedy_full_blocks(tile), tile.copy()))

    counts = [count for count, _tile in tiles]
    assert len(tiles) == 9
    assert len(set(counts)) == len(counts)

    ordered_tiles = sorted(tiles, key=lambda item: item[0])
    destination_slots = [
        (r0, r1, c0, c1)
        for r0, r1 in reversed(row_segments)
        for c0, c1 in col_segments
    ]

    out = np.full_like(g, BLACK)
    for (count, tile), (r0, r1, c0, c1) in zip(ordered_tiles, destination_slots):
        out[r0:r1, c0:c1] = _recolor_blue_pixels(tile, count)
    return out


def _tile_anchor_positions(tile_size: int) -> list[tuple[int, int]]:
    anchors = list(range(0, tile_size - 1, 2))
    assert len(anchors) >= 3
    return [(r, c) for r in anchors[:3] for c in anchors[:3]]


def _blue_total(tile: np.ndarray) -> int:
    return int(np.count_nonzero(tile == BLUE))


def _target_blue_totals(tile_size: int, rng: random.Random) -> dict[int, int]:
    shared = rng.choice([8, 9] if tile_size == 6 else [8, 9, 10])
    return {
        0: rng.choice([5, 6, 7]),
        1: shared,
        2: shared,
        3: rng.choice([12, 13, 14]),
        4: rng.choice([16, 17, 18]),
        5: rng.choice([20, 21, 22]),
        6: rng.choice([24, 25, 26]),
        7: rng.choice([28, 29] if tile_size == 6 else [28, 29, 30]),
        8: 32 if tile_size == 6 else rng.choice([32, 33]),
    }


def _build_tile(
    tile_size: int,
    block_count: int,
    target_blue_total: int,
    rng: random.Random,
) -> np.ndarray | None:
    anchors = _tile_anchor_positions(tile_size)
    all_cells = [(r, c) for r in range(tile_size) for c in range(tile_size)]

    for _ in range(GEN_TILE_ATTEMPTS):
        tile = np.full((tile_size, tile_size), GREY, dtype=int)
        chosen_anchors = rng.sample(anchors, block_count)
        occupied: set[tuple[int, int]] = set()
        for r0, c0 in chosen_anchors:
            tile[r0 : r0 + 2, c0 : c0 + 2] = BLUE
            for dr in range(2):
                for dc in range(2):
                    occupied.add((r0 + dr, c0 + dc))

        if _blue_total(tile) > target_blue_total:
            continue

        candidates = [cell for cell in all_cells if cell not in occupied]
        rng.shuffle(candidates)
        remaining = target_blue_total - _blue_total(tile)
        for r, c in candidates:
            if remaining == 0:
                break
            tile[r, c] = BLUE
            if _count_greedy_full_blocks(tile) != block_count:
                tile[r, c] = GREY
                continue
            remaining -= 1

        if remaining != 0:
            continue
        if _count_greedy_full_blocks(tile) != block_count:
            continue
        if block_count == 0 and _blue_total(tile) < 2:
            continue
        return tile
    return None


def _semantic_guard(tiles_by_count: dict[int, np.ndarray]) -> bool:
    counts = sorted(tiles_by_count)
    blue_totals = {count: _blue_total(tile) for count, tile in tiles_by_count.items()}

    if len(set(counts)) != 9:
        return False
    if 0 not in counts or 1 not in counts or max(counts) < 2:
        return False
    if blue_totals[1] != blue_totals[2]:
        return False
    if any(_count_greedy_full_blocks(tile) != count for count, tile in tiles_by_count.items()):
        return False
    return True


def _compose_grid(tiles: list[np.ndarray], tile_size: int) -> np.ndarray:
    side = tile_size * 3 + 2
    grid = np.full((side, side), BLACK, dtype=int)
    starts = [0, tile_size + 1, 2 * tile_size + 2]
    for idx, tile in enumerate(tiles):
        row = idx // 3
        col = idx % 3
        r0 = starts[row]
        c0 = starts[col]
        grid[r0 : r0 + tile_size, c0 : c0 + tile_size] = tile
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    tile_size = rng.choice(GEN_TILE_SIZE_OPTIONS)
    targets = _target_blue_totals(tile_size, rng)

    tiles_by_count: dict[int, np.ndarray] = {}
    for count in GEN_COUNTS:
        tile = _build_tile(tile_size, count, targets[count], rng)
        if tile is None:
            return None
        tiles_by_count[count] = tile

    if not _semantic_guard(tiles_by_count):
        return None

    pairs = [(count, tiles_by_count[count]) for count in GEN_COUNTS]
    rng.shuffle(pairs)
    index_by_count = {count: idx for idx, (count, _tile) in enumerate(pairs)}
    if index_by_count[1] < index_by_count[2]:
        i = index_by_count[1]
        j = index_by_count[2]
        pairs[i], pairs[j] = pairs[j], pairs[i]
    shuffled_tiles = [tile for _count, tile in pairs]
    return _compose_grid(shuffled_tiles, tile_size)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid tiled-packing sorting input")
