from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
ARRAY_SIZE = 3
TILE_SIZE = 4
SEP = 1
GRID_SIDE = ARRAY_SIZE * TILE_SIZE + (ARRAY_SIZE - 1) * SEP
PALETTE = tuple(range(1, 10))
SECONDARY_COUNTS = (1, 2, 3)
GEN_ATTEMPTS = 400


def _blocks_from_separators(length: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds = [-1, *separators, length]
    blocks: list[tuple[int, int]] = []
    for left, right in zip(bounds, bounds[1:]):
        if left + 1 <= right - 1:
            blocks.append((left + 1, right - 1))
    return blocks


def _parse_layout(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert (h, w) == (GRID_SIDE, GRID_SIDE), "expected a 14x14 separated tile grid"

    sep_rows = [r for r in range(h) if np.all(g[r] == BG)]
    sep_cols = [c for c in range(w) if np.all(g[:, c] == BG)]
    assert len(sep_rows) == ARRAY_SIZE - 1 and len(sep_cols) == ARRAY_SIZE - 1

    row_blocks = _blocks_from_separators(h, sep_rows)
    col_blocks = _blocks_from_separators(w, sep_cols)
    assert len(row_blocks) == ARRAY_SIZE and len(col_blocks) == ARRAY_SIZE
    assert all(r1 - r0 + 1 == TILE_SIZE for r0, r1 in row_blocks)
    assert all(c1 - c0 + 1 == TILE_SIZE for c0, c1 in col_blocks)
    return row_blocks, col_blocks


def _major_and_minor_colors(grid: np.ndarray) -> tuple[int, int]:
    counts = Counter(int(v) for v in grid.flat if int(v) != BG)
    assert len(counts) == 2, "expected exactly two non-background tile colors"
    ordered = counts.most_common()
    assert ordered[0][1] > ordered[1][1], "main color must be structurally dominant"
    return int(ordered[0][0]), int(ordered[1][0])


def _tile_infos(
    grid: np.ndarray,
) -> tuple[
    list[tuple[tuple[int, int], tuple[int, int], tuple[int, int], np.ndarray]],
    int,
    int,
]:
    g = np.asarray(grid, dtype=int)
    row_blocks, col_blocks = _parse_layout(g)
    main_color, secondary_color = _major_and_minor_colors(g)

    infos: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int], np.ndarray]] = []
    seen_holes: set[tuple[int, int]] = set()

    for tile_r, (r0, r1) in enumerate(row_blocks):
        for tile_c, (c0, c1) in enumerate(col_blocks):
            tile = g[r0 : r1 + 1, c0 : c1 + 1]
            zeros = np.argwhere(tile == BG)
            assert zeros.shape[0] == 1, "each tile must contain exactly one black hole"
            hole_r, hole_c = map(int, zeros[0])
            assert 0 <= hole_r < ARRAY_SIZE and 0 <= hole_c < ARRAY_SIZE

            colors = set(map(int, np.unique(tile)))
            assert colors == {BG, main_color, secondary_color}

            secondary_count = int(np.sum(tile == secondary_color))
            assert secondary_count in SECONDARY_COUNTS
            assert int(np.sum(tile == main_color)) > secondary_count

            hole = (hole_r, hole_c)
            seen_holes.add(hole)
            infos.append(((r0, r1), (c0, c1), hole, tile.copy()))

    expected_holes = {(r, c) for r in range(ARRAY_SIZE) for c in range(ARRAY_SIZE)}
    assert seen_holes == expected_holes, "hole coordinates must encode the full 3x3 array"
    return infos, main_color, secondary_color


def _apply_rule(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    infos, main_color, secondary_color = _tile_infos(g)
    out = g.copy()

    for (r0, r1), (c0, c1), hole, tile in infos:
        if (hole[0] + hole[1]) % 2 == 0:
            continue
        transformed = tile.copy()
        transformed[tile == main_color] = BG
        transformed[tile == secondary_color] = main_color
        out[r0 : r1 + 1, c0 : c1 + 1] = transformed

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Selectively invert separated position-coded tiles based on the parity of each tile's hole coordinate.

    Input structure:
    - A 14x14 black-background grid is partitioned by single black separator rows and columns into a 3x3 array of 4x4 tiles.
    - Every tile contains one dominant non-black color, one minority distractor color, and exactly one black hole.
    - The hole's local coordinate inside a tile is the tile's encoded array coordinate.

    Rule:
    - Keep the separator rows and columns black.
    - Read each tile's unique hole position `(lr, lc)` and compute `lr + lc`.
    - If the sum is even, leave the tile unchanged; if it is odd, turn the dominant color to black and turn the minority color into the dominant color, while the hole stays black.

    Key insight:
    - The hole is not just a label for where the tile belongs: its encoded coordinate parity decides whether that tile undergoes the color inversion.
    """
    return _apply_rule(np.asarray(grid, dtype=int))


def _sample_secondary_counts() -> list[int]:
    odd_tiles = [idx for idx in range(ARRAY_SIZE * ARRAY_SIZE) if sum(divmod(idx, ARRAY_SIZE)) % 2 == 1]
    even_tiles = [idx for idx in range(ARRAY_SIZE * ARRAY_SIZE) if idx not in odd_tiles]
    target = set(SECONDARY_COUNTS)

    for _ in range(200):
        counts = [random.choice(SECONDARY_COUNTS) for _ in range(ARRAY_SIZE * ARRAY_SIZE)]
        if {counts[i] for i in odd_tiles} == target and {counts[i] for i in even_tiles} == target:
            return counts
    raise RuntimeError("failed to sample parity-balanced secondary counts")


def _secondary_positions(hole: tuple[int, int], count: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(TILE_SIZE) for c in range(TILE_SIZE) if (r, c) != hole]
    style = random.choice(("random", "near", "far"))

    if style == "random":
        return random.sample(cells, count)

    ranked = sorted(
        cells,
        key=lambda pos: abs(pos[0] - hole[0]) + abs(pos[1] - hole[1]),
    )
    if style == "far":
        ranked = list(reversed(ranked))
    pool = ranked[: min(len(ranked), count + 5)]
    return random.sample(pool, count)


def _build_tile(
    main_color: int,
    secondary_color: int,
    hole: tuple[int, int],
    secondary_count: int,
) -> np.ndarray:
    tile = np.full((TILE_SIZE, TILE_SIZE), int(main_color), dtype=int)
    tile[hole] = BG
    for r, c in _secondary_positions(hole, secondary_count):
        tile[r, c] = int(secondary_color)
    return tile


def _render_input(
    main_color: int,
    secondary_color: int,
    secondary_counts: list[int],
) -> np.ndarray:
    grid = np.full((GRID_SIDE, GRID_SIDE), BG, dtype=int)
    for tile_r in range(ARRAY_SIZE):
        for tile_c in range(ARRAY_SIZE):
            idx = tile_r * ARRAY_SIZE + tile_c
            hole = (tile_r, tile_c)
            tile = _build_tile(main_color, secondary_color, hole, secondary_counts[idx])
            r0 = tile_r * (TILE_SIZE + SEP)
            c0 = tile_c * (TILE_SIZE + SEP)
            grid[r0 : r0 + TILE_SIZE, c0 : c0 + TILE_SIZE] = tile
    return grid


def _semantic_checks(inp: np.ndarray) -> bool:
    try:
        infos, main_color, secondary_color = _tile_infos(inp)
        out = _apply_rule(inp)
    except Exception:
        return False

    odd_counts: set[int] = set()
    even_counts: set[int] = set()
    odd_changed = 0
    even_unchanged = 0

    for (r0, r1), (c0, c1), hole, tile in infos:
        secondary_count = int(np.sum(tile == secondary_color))
        parity = (hole[0] + hole[1]) % 2
        if parity == 1:
            odd_counts.add(secondary_count)
            odd_changed += int(np.sum(out[r0 : r1 + 1, c0 : c1 + 1] != tile))
        else:
            even_counts.add(secondary_count)
            even_unchanged += int(np.array_equal(out[r0 : r1 + 1, c0 : c1 + 1], tile))

    if odd_counts != set(SECONDARY_COUNTS) or even_counts != set(SECONDARY_COUNTS):
        return False
    if odd_changed == 0 or even_unchanged != 5:
        return False

    expected = inp.copy()
    for (r0, r1), (c0, c1), hole, tile in infos:
        if (hole[0] + hole[1]) % 2 == 0:
            continue
        transformed = tile.copy()
        transformed[tile == main_color] = BG
        transformed[tile == secondary_color] = main_color
        expected[r0 : r1 + 1, c0 : c1 + 1] = transformed

    return bool(np.array_equal(out, expected) and not np.array_equal(out, inp))


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        main_color, secondary_color = random.sample(PALETTE, 2)
        counts = _sample_secondary_counts()
        inp = _render_input(main_color, secondary_color, counts)
        if _semantic_checks(inp):
            return inp
    raise RuntimeError("failed to generate a valid gated-tile-parity ARC input")
