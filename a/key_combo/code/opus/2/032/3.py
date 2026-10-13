from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
SEP = 7
DAMAGE = 9
PATTERN_COLORS = [1, 2, 3, 4, 5, 6, 8]

MAX_GEN_RETRIES = 400
POLYOMINO_RETRIES = 80


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_divider_color(grid: np.ndarray) -> int:
    bg = get_bg(grid)
    candidates: list[int] = []
    for r in range(grid.shape[0]):
        row = grid[r]
        if int(row[0]) != bg and bool(np.all(row == row[0])):
            candidates.append(int(row[0]))
    for c in range(grid.shape[1]):
        col = grid[:, c]
        if int(col[0]) != bg and bool(np.all(col == col[0])):
            candidates.append(int(col[0]))
    if not candidates:
        raise AssertionError("expected full divider rows/cols")
    return int(Counter(candidates).most_common(1)[0][0])


def _split_spans(size: int, dividers: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for stop in dividers + [int(size)]:
        if stop > start:
            spans.append((int(start), int(stop)))
        start = int(stop + 1)
    return spans


def _extract_tiles(grid: np.ndarray) -> list[np.ndarray]:
    sep = _find_divider_color(grid)
    row_dividers = [r for r in range(grid.shape[0]) if bool(np.all(grid[r] == sep))]
    col_dividers = [c for c in range(grid.shape[1]) if bool(np.all(grid[:, c] == sep))]
    row_spans = _split_spans(grid.shape[0], row_dividers)
    col_spans = _split_spans(grid.shape[1], col_dividers)
    assert len(row_spans) >= 2 and len(col_spans) >= 2, "expected tiled layout"

    tiles = [grid[r0:r1, c0:c1].copy() for r0, r1 in row_spans for c0, c1 in col_spans]
    shapes = {tile.shape for tile in tiles}
    assert len(shapes) == 1, "all tiles must share one shape"
    return tiles


def _majority_tile(tiles: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(tiles, axis=0)
    n, h, w = stack.shape
    out = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            out[r, c] = int(Counter(map(int, stack[:, r, c])).most_common(1)[0][0])
    assert out.shape == (h, w) and n >= 2
    return out


def _consensus_halo_output(consensus: np.ndarray, halo_color: int, distance: int) -> np.ndarray:
    core_mask = consensus != get_bg(consensus)
    box = bbox(core_mask)
    if box is None:
        raise AssertionError("consensus pattern must contain foreground")
    core = crop_bbox(consensus, box)
    h, w = core.shape
    out = np.full((h + 2 * distance, w + 2 * distance), int(halo_color), dtype=int)
    out[distance : distance + h, distance : distance + w] = core
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the repeated tile consensus, count one damaged copy's bad pixels, and wrap the consensus crop in a halo of that thickness.

    Input structure:
    - The background is black, with full orange divider rows and columns splitting the grid into equal rectangular tiles.
    - Every tile is a copy of the same multicolor pattern on black, except one damaged tile where 1-3 pattern pixels were replaced by a foreign color.
    - The foreign damage color appears only inside that one damaged tile.

    Rule:
    - Split the grid into tiles using the full divider lines, then take a per-coordinate majority vote across tiles to recover the consensus tile.
    - Find the unique tile that differs from the consensus and count its differing coordinates; this count is the halo thickness.
    - Take the consensus pattern's non-background bounding box crop and expand that rectangle outward by the counted thickness on all four sides.
    - Fill the new outer ring with the damaged tile's foreign color and keep the consensus crop unchanged in the center.

    Key insight:
    - The control number is not embedded directly; it is the number of pixels that disagree with the cross-tile consensus, so repair must happen before counting and expansion.
    """
    g = np.asarray(grid, dtype=int)
    tiles = _extract_tiles(g)
    consensus = _majority_tile(tiles)

    diff_counts = [int(np.count_nonzero(tile != consensus)) for tile in tiles]
    damaged_ids = [idx for idx, count in enumerate(diff_counts) if count > 0]
    assert len(damaged_ids) == 1, "expected exactly one damaged tile"

    damaged = tiles[damaged_ids[0]]
    diff_mask = damaged != consensus
    distance = int(diff_mask.sum())
    assert distance >= 1

    halo_colors = [int(v) for v in damaged[diff_mask].tolist()]
    halo_color = int(Counter(halo_colors).most_common(1)[0][0])
    return _consensus_halo_output(consensus, halo_color, distance)


def _neighbors4(cell: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr = int(r + dr)
        nc = int(c + dc)
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _random_polyomino(area: int, h: int, w: int) -> list[tuple[int, int]] | None:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    frontier = set(_neighbors4(start, h, w))
    while len(cells) < area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        if cell in cells:
            continue
        cells.add(cell)
        for nxt in _neighbors4(cell, h, w):
            if nxt not in cells:
                frontier.add(nxt)
    if len(cells) != area:
        return None
    return list(cells)


def _spanning_polyomino(area: int, h: int, w: int) -> list[tuple[int, int]] | None:
    for _ in range(POLYOMINO_RETRIES):
        cells = _random_polyomino(area, h, w)
        if cells is None:
            continue
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        if min(rs) == 0 and max(rs) == h - 1 and min(cs) == 0 and max(cs) == w - 1:
            return cells
    return None


def _paint_clustered_colors(shape_cells: list[tuple[int, int]], colors: list[int]) -> dict[tuple[int, int], int]:
    seeds = random.sample(shape_cells, len(colors))
    mapping: dict[tuple[int, int], int] = {}
    for cell in shape_cells:
        dist = [abs(cell[0] - seed[0]) + abs(cell[1] - seed[1]) for seed in seeds]
        idx = min(range(len(colors)), key=lambda i: (dist[i], i))
        mapping[cell] = int(colors[idx])
    return mapping


def _build_pattern(tile_h: int, tile_w: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    bbox_h_choices = [v for v in range(3, tile_h + 1)]
    bbox_w_choices = [v for v in range(3, tile_w + 1)]
    for _ in range(120):
        bbox_h = random.choice(bbox_h_choices)
        bbox_w = random.choice(bbox_w_choices)
        max_area = min(15, bbox_h * bbox_w)
        min_area = max(8, bbox_h + bbox_w - 1)
        if min_area > max_area:
            continue
        area = random.randint(min_area, max_area)
        cells = _spanning_polyomino(area, bbox_h, bbox_w)
        if cells is None:
            continue

        color_count = 2 if area < 10 else random.choice([2, 3])
        colors = random.sample(PATTERN_COLORS, color_count)
        local_colors = _paint_clustered_colors(cells, colors)

        pattern = np.full((tile_h, tile_w), BG, dtype=int)
        r_off = random.randint(0, tile_h - bbox_h)
        c_off = random.randint(0, tile_w - bbox_w)
        placed_cells: list[tuple[int, int]] = []
        for r, c in cells:
            rr = int(r + r_off)
            cc = int(c + c_off)
            pattern[rr, cc] = int(local_colors[(r, c)])
            placed_cells.append((rr, cc))

        used = set(int(pattern[r, c]) for r, c in placed_cells)
        if len(used) >= 2:
            return pattern, placed_cells
    raise RuntimeError("failed to build a valid pattern")


def _assemble_tiles(tiles: list[np.ndarray], row_count: int, col_count: int) -> np.ndarray:
    tile_h, tile_w = tiles[0].shape
    H = row_count * tile_h + (row_count - 1)
    W = col_count * tile_w + (col_count - 1)
    grid = np.full((H, W), BG, dtype=int)

    idx = 0
    for rr in range(row_count):
        for cc in range(col_count):
            r0 = rr * (tile_h + 1)
            c0 = cc * (tile_w + 1)
            grid[r0 : r0 + tile_h, c0 : c0 + tile_w] = tiles[idx]
            idx += 1

    for rr in range(1, row_count):
        grid[rr * tile_h + (rr - 1), :] = SEP
    for cc in range(1, col_count):
        grid[:, cc * tile_w + (cc - 1)] = SEP
    return grid


def _expected_output(pattern: np.ndarray, damage_count: int) -> np.ndarray:
    return _consensus_halo_output(pattern, DAMAGE, damage_count)


def _gen_attempt() -> np.ndarray | None:
    row_count, col_count = random.choice([(2, 3), (3, 2)])
    tile_h = random.randint(5, 7)
    tile_w = random.randint(5, 7)

    pattern, pattern_cells = _build_pattern(tile_h, tile_w)
    tile_count = row_count * col_count
    tiles = [pattern.copy() for _ in range(tile_count)]

    damage_count = random.randint(1, min(3, len(pattern_cells)))
    damaged_idx = random.randrange(tile_count)
    damaged_cells = random.sample(pattern_cells, damage_count)
    for r, c in damaged_cells:
        tiles[damaged_idx][r, c] = DAMAGE

    grid = _assemble_tiles(tiles, row_count, col_count)
    expected = _expected_output(pattern, damage_count)

    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
