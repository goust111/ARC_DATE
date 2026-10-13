from __future__ import annotations

import random
import numpy as np


BG = 0
MARKER = 7
PAIR_COLORS = (1, 2, 3, 4, 5, 6, 8, 9)

GRID_H_RANGE = (12, 28)
GRID_W_RANGE = (12, 28)
PAIR_COUNT_RANGE = (1, 4)
TILE_MIN = 5
BBOX_MIN = 3
PLACE_ATTEMPTS = 80
MAX_GENERATION_ATTEMPTS = 400


def _paint_segment(grid: np.ndarray, a: tuple[int, int], b: tuple[int, int], color: int) -> None:
    ar, ac = a
    br, bc = b
    if ar == br:
        c0, c1 = sorted((ac, bc))
        grid[ar, c0 : c1 + 1] = color
        return
    assert ac == bc, "segment endpoints must share a row or a column"
    r0, r1 = sorted((ar, br))
    grid[r0 : r1 + 1, ac] = color


def _path_cells(a: tuple[int, int], b: tuple[int, int], bend: tuple[int, int]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for p, q in ((a, bend), (bend, b)):
        pr, pc = p
        qr, qc = q
        if pr == qr:
            c0, c1 = sorted((pc, qc))
            for c in range(c0, c1 + 1):
                cells.add((pr, c))
        else:
            assert pc == qc, "segment endpoints must share a row or a column"
            r0, r1 = sorted((pr, qr))
            for r in range(r0, r1 + 1):
                cells.add((r, pc))
    return cells


def _random_partition(total: int, parts: int, min_size: int) -> list[int]:
    assert total >= parts * min_size
    slack = total - parts * min_size
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, slack + parts), parts - 1))
    aug = [0] + cuts + [slack + parts]
    return [aug[i + 1] - aug[i] - 1 + min_size for i in range(parts)]


def _tile_boxes(height: int, width: int, tile_rows: int, tile_cols: int) -> list[tuple[int, int, int, int]]:
    row_sizes = _random_partition(height, tile_rows, TILE_MIN)
    col_sizes = _random_partition(width, tile_cols, TILE_MIN)
    row_starts = [0]
    col_starts = [0]
    for size in row_sizes[:-1]:
        row_starts.append(row_starts[-1] + size)
    for size in col_sizes[:-1]:
        col_starts.append(col_starts[-1] + size)

    boxes = []
    for i, rs in enumerate(row_starts):
        for j, cs in enumerate(col_starts):
            rh = row_sizes[i]
            cw = col_sizes[j]
            boxes.append((rs, cs, rs + rh - 1, cs + cw - 1))
    return boxes


def main(grid: np.ndarray) -> np.ndarray:
    """Connect each same-color pixel pair with the L-shaped path chosen by its orange corner marker.

    Input structure:
    - The background is black.
    - Each non-black, non-orange color appears exactly twice as two single-pixel endpoints of one pair.
    - Orange single-pixel markers sit at one of the two unused corners of a pair's axis-aligned bounding rectangle.

    Rule:
    - For each same-color pair, form the rectangle whose opposite corners are the two pixels.
    - Find which of the other two rectangle corners contains the orange marker.
    - Draw the two straight cardinal segments from one endpoint to that corner and from the corner to the other endpoint, coloring the full L path with the pair color.
    - Remove every orange marker unless it is covered by one of the drawn paths.

    Key insight:
    - The paired pixels determine the two possible L routes, and the orange corner marker resolves which bend to use.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    out[out == MARKER] = BG

    marker_coords = [tuple(map(int, rc)) for rc in np.argwhere(g == MARKER)]
    marker_set = set(marker_coords)

    colors = [int(color) for color in np.unique(g) if color not in (BG, MARKER)]
    used_markers: set[tuple[int, int]] = set()
    for color in colors:
        coords = np.argwhere(g == color)
        assert len(coords) == 2, "each pair color must appear exactly twice"
        a = tuple(map(int, coords[0]))
        b = tuple(map(int, coords[1]))
        assert a[0] != b[0] and a[1] != b[1], "paired pixels must be diagonal corners"

        candidate_bends = [(a[0], b[1]), (b[0], a[1])]
        bends = [corner for corner in candidate_bends if corner in marker_set]
        assert len(bends) == 1, "expected exactly one orange bend marker per pair"
        bend = bends[0]
        used_markers.add(bend)

        _paint_segment(out, a, bend, color)
        _paint_segment(out, bend, b, color)

    assert used_markers == marker_set, "every orange marker must belong to exactly one pair"
    return out


def _sample_tiling(height: int, width: int, pair_count: int) -> tuple[int, int] | None:
    options = []
    for tile_rows in range(1, 4):
        for tile_cols in range(1, 4):
            if tile_rows * tile_cols < pair_count:
                continue
            if height < tile_rows * TILE_MIN or width < tile_cols * TILE_MIN:
                continue
            options.append((tile_rows, tile_cols))
    return random.choice(options) if options else None


def _sample_pair(tile: tuple[int, int, int, int], color: int) -> dict[str, object] | None:
    tr0, tc0, tr1, tc1 = tile
    tile_h = tr1 - tr0 + 1
    tile_w = tc1 - tc0 + 1
    max_h = tile_h - 2
    max_w = tile_w - 2
    if max_h < BBOX_MIN or max_w < BBOX_MIN:
        return None

    for _ in range(PLACE_ATTEMPTS):
        h = random.randint(BBOX_MIN, max_h)
        w = random.randint(BBOX_MIN, max_w)
        r0 = random.randint(tr0 + 1, tr1 - h)
        c0 = random.randint(tc0 + 1, tc1 - w)
        r1 = r0 + h - 1
        c1 = c0 + w - 1

        if random.random() < 0.5:
            a, b = (r0, c0), (r1, c1)
            bends = ((r0, c1), (r1, c0))
        else:
            a, b = (r0, c1), (r1, c0)
            bends = ((r0, c0), (r1, c1))
        bend = bends[0] if random.random() < 0.5 else bends[1]
        return {
            "color": int(color),
            "a": a,
            "b": b,
            "bend": bend,
            "path": _path_cells(a, b, bend),
        }
    return None


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    pair_count = random.randint(*PAIR_COUNT_RANGE)

    tiling = _sample_tiling(height, width, pair_count)
    if tiling is None:
        return None
    tile_rows, tile_cols = tiling
    tiles = _tile_boxes(height, width, tile_rows, tile_cols)
    chosen_tiles = random.sample(tiles, pair_count)
    colors = random.sample(PAIR_COLORS, pair_count)

    pairs: list[dict[str, object]] = []
    for tile, color in zip(chosen_tiles, colors):
        pair = _sample_pair(tile, color)
        if pair is None:
            return None
        pairs.append(pair)

    inp = np.full((height, width), BG, dtype=int)
    expected = np.full((height, width), BG, dtype=int)

    for pair in pairs:
        color = int(pair["color"])
        a = pair["a"]
        b = pair["b"]
        bend = pair["bend"]
        path = pair["path"]

        inp[a] = color
        inp[b] = color
        inp[bend] = MARKER
        for r, c in path:
            expected[r, c] = color

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
