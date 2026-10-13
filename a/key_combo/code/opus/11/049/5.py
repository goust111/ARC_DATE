from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
TILE_SIZE = 4
COLOR_POOL = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    box = _bbox(coords)
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy(), box


def _is_filled_rectangle(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    return len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _parse(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    comps = _components(g != bg)

    tile_candidates: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []
    legend_candidates: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []
    marker_cells: list[tuple[int, int, int]] = []

    for coords in comps:
        crop, box = _crop(g, coords)
        h, w = crop.shape
        area = len(coords)
        colors = set(map(int, crop.ravel()))
        colors.discard(bg)

        if area == 1:
            r, c = map(int, coords[0])
            marker_cells.append((r, c, int(g[r, c])))
            continue

        if h == TILE_SIZE and w == TILE_SIZE and area == TILE_SIZE * TILE_SIZE and len(colors) >= 3:
            tile_candidates.append((crop, box))
            continue

        if w == 2 and area == 2 * h and _is_filled_rectangle(coords) and len(colors) >= 3:
            legend_candidates.append((crop, box))

    assert len(tile_candidates) == 1, "expected exactly one 4x4 source tile"
    assert len(legend_candidates) == 1, "expected exactly one filled 2-column legend"

    tile, tile_box = tile_candidates[0]
    legend, legend_box = legend_candidates[0]
    legend_rows = legend.shape[0]

    assert len(marker_cells) == legend_rows, "marker count must match legend height"
    marker_rows = {r for r, _c, _color in marker_cells}
    assert len(marker_rows) == 1, "markers must form one horizontal row"
    marker_colors = [color for _r, _c, color in sorted(marker_cells, key=lambda x: x[1])]

    left_colors = {int(color) for color in legend[:, 0]}
    right_colors = {int(color) for color in legend[:, 1]}
    assert left_colors.isdisjoint(right_colors), "legend colors must map to one side only"
    assert all(color in left_colors or color in right_colors for color in marker_colors)

    return {
        "bg": bg,
        "tile": tile,
        "tile_box": tile_box,
        "legend": legend,
        "legend_box": legend_box,
        "left_colors": left_colors,
        "right_colors": right_colors,
        "marker_colors": marker_colors,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a horizontal tile strip by reading a two-column legend as original-vs-mirror instructions.

    Input structure:
    - The background is black, with one fully filled 4x4 multicolor tile on the left.
    - Elsewhere there is one fully filled 2-column legend block whose rows pair colors into two instruction classes.
    - Beneath that area sits one horizontal row of isolated marker pixels, one marker per output tile slot.

    Rule:
    - Read each legend row as a binary instruction: the left color means use the original tile, and the right color means use the left-right mirror of the tile.
    - Scan the marker pixels from left to right; each marker color is looked up in the legend to decide whether that output slot uses the original tile or the mirrored tile.
    - Concatenate the chosen 4x4 tiles from left to right to produce a 4 x (4N) output strip.

    Key insight:
    - The legend does not recolor anything; it partitions colors into two visible instruction sets, and the marker row queries that partition to choose tile orientation at each position.
    """
    parsed = _parse(grid)
    tile = parsed["tile"]
    mirrored = np.fliplr(tile)

    blocks = []
    for color in parsed["marker_colors"]:
        if color in parsed["left_colors"]:
            blocks.append(tile)
        else:
            assert color in parsed["right_colors"], "marker color must be encoded by the legend"
            blocks.append(mirrored)

    return np.concatenate(blocks, axis=1)


def _sample_tile() -> np.ndarray:
    for _ in range(100):
        palette = random.sample(COLOR_POOL, random.randint(3, 4))
        tile = np.array(random.choices(palette, k=TILE_SIZE * TILE_SIZE), dtype=int).reshape(TILE_SIZE, TILE_SIZE)

        forced = random.sample(range(TILE_SIZE * TILE_SIZE), len(palette))
        for idx, color in zip(forced, palette):
            tile[idx // TILE_SIZE, idx % TILE_SIZE] = color

        if np.array_equal(tile, np.fliplr(tile)):
            continue
        if len(set(map(int, tile.ravel()))) < 3:
            continue
        return tile
    raise RuntimeError("failed to sample a non-symmetric tile")


def _sample_legend(n: int) -> np.ndarray:
    if n <= 4:
        colors = random.sample(COLOR_POOL, 2 * n)
        left = colors[:n]
        right = colors[n:]
        random.shuffle(left)
        random.shuffle(right)
    else:
        repeated_side = random.choice((0, 1))
        colors = random.sample(COLOR_POOL, 9)
        if repeated_side == 0:
            left = colors[:4] + [random.choice(colors[:4])]
            right = colors[4:9]
        else:
            left = colors[:5]
            right = colors[5:9] + [random.choice(colors[5:9])]
        random.shuffle(left)
        random.shuffle(right)

    legend = np.column_stack([left, right]).astype(int)
    left_set = set(map(int, legend[:, 0]))
    right_set = set(map(int, legend[:, 1]))
    assert left_set.isdisjoint(right_set)
    return legend


def _sample_orientation_bits(n: int) -> list[int]:
    for _ in range(100):
        bits = [random.randint(0, 1) for _ in range(n)]
        if len(set(bits)) != 2:
            continue
        if not any(bits[i] == bits[i + 1] for i in range(n - 1)):
            continue
        if not any(bits[i] != bits[i + 1] for i in range(n - 1)):
            continue
        return bits
    raise RuntimeError("failed to sample a non-trivial orientation sequence")


def _build_input() -> np.ndarray:
    n = random.choices([3, 4, 5], weights=[3, 3, 4], k=1)[0]
    tile = _sample_tile()
    legend = _sample_legend(n)
    bits = _sample_orientation_bits(n)
    row_order = random.sample(range(n), n)
    marker_colors = [
        int(legend[row_order[i], bits[i]])
        for i in range(n)
    ]

    marker_span = 2 * n - 1
    width = random.randint(4 + 1 + marker_span, 4 + 1 + marker_span + 2)

    legend_r0 = random.randint(0, 2)
    marker_gap = random.randint(1, 2)
    marker_row = legend_r0 + n + marker_gap
    min_height = max(TILE_SIZE, marker_row + 1)
    height = random.randint(min_height, min(14, min_height + 4))
    tile_r0 = random.randint(0, height - TILE_SIZE)

    grid = np.full((height, width), BG, dtype=int)
    grid[tile_r0 : tile_r0 + TILE_SIZE, 0:TILE_SIZE] = tile

    legend_c0 = width - 2
    grid[legend_r0 : legend_r0 + n, legend_c0 : legend_c0 + 2] = legend

    marker_c0 = 5
    for i, color in enumerate(marker_colors):
        grid[marker_row, marker_c0 + 2 * i] = color

    return grid


def _validate_instance(grid: np.ndarray) -> None:
    parsed = _parse(grid)
    tile = parsed["tile"]
    mirrored = np.fliplr(tile)
    marker_colors = parsed["marker_colors"]
    bits = [0 if color in parsed["left_colors"] else 1 for color in marker_colors]
    n = len(bits)

    assert 3 <= n <= 5
    assert not np.array_equal(tile, mirrored)
    assert len(set(bits)) == 2
    assert any(bits[i] == bits[i + 1] for i in range(n - 1))
    assert any(bits[i] != bits[i + 1] for i in range(n - 1))

    out = main(grid)
    assert out.shape == (TILE_SIZE, TILE_SIZE * n)

    blocks = [out[:, i * TILE_SIZE : (i + 1) * TILE_SIZE] for i in range(n)]
    for block, bit in zip(blocks, bits):
        assert np.array_equal(block, tile if bit == 0 else mirrored)

    trivial = [
        np.concatenate([tile] * n, axis=1),
        np.concatenate([mirrored] * n, axis=1),
        np.concatenate([tile if i % 2 == 0 else mirrored for i in range(n)], axis=1),
        np.concatenate([mirrored if i % 2 == 0 else tile for i in range(n)], axis=1),
    ]
    assert not any(np.array_equal(out, candidate) for candidate in trivial)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_input()
        try:
            _validate_instance(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
