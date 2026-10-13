import random
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


BG = 0
BLOCK_COLORS = (1, 2, 3, 7)
CENTER_COLOR = 6
GRID_SIZE = 13
LAYERS = (1, 2, 3, 4)
MAX_ATTEMPTS = 400

Coord = Tuple[int, int]


def _ring_coords(center_r: int, center_c: int, dist: int) -> List[Coord]:
    coords: List[Coord] = []
    top = center_r - dist
    bottom = center_r + dist
    left = center_c - dist
    right = center_c + dist

    for c in range(left, right + 1):
        coords.append((top, c))
    for r in range(top + 1, bottom + 1):
        coords.append((r, right))
    for c in range(right - 1, left - 1, -1):
        coords.append((bottom, c))
    for r in range(bottom - 1, top, -1):
        coords.append((r, left))
    return coords


def _chebyshev(a: Coord, b: Coord) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _box_cells(top: int, left: int) -> List[Coord]:
    return [(top + dr, left + dc) for dr in range(3) for dc in range(3)]


def _box_layer(top: int, left: int, center: Coord) -> int:
    return min(_chebyshev(cell, center) for cell in _box_cells(top, left))


def _candidate_boxes(size: int, layer: int) -> List[Coord]:
    center = size // 2
    out: List[Coord] = []
    center_coord = (center, center)
    for top in range(size - 2):
        for left in range(size - 2):
            cells = _box_cells(top, left)
            if center_coord in cells:
                continue
            if _box_layer(top, left, center_coord) == layer:
                out.append((top, left))
    return out


def _make_output(block_info: Sequence[Tuple[int, int, int]], size: int) -> np.ndarray:
    center = size // 2
    out = np.full((size, size), BG, dtype=int)
    out[center, center] = CENTER_COLOR
    for layer, count, color in block_info:
        coords = _ring_coords(center, center, layer)
        for r, c in coords[:count]:
            out[r, c] = color
    return out


def _decode_blocks(grid: np.ndarray) -> List[Tuple[int, int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h == w and h % 2 == 1 and h >= 13, "expected an odd square grid large enough for four 3x3 blocks"
    center = h // 2
    assert int(g[center, center]) == CENTER_COLOR, "expected a unique magenta center pixel"
    colors = sorted(int(v) for v in np.unique(g) if v not in (BG, CENTER_COLOR))
    assert len(colors) == 4, "expected exactly four block colors"

    center_coord = (center, center)
    blocks: List[Tuple[int, int, int]] = []
    layers_seen = set()
    for color in colors:
        coords = [(int(r), int(c)) for r, c in np.argwhere(g == color)]
        assert 1 <= len(coords) <= 4, "each block must encode a count from one to four"

        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        assert max(rows) - min(rows) <= 2 and max(cols) - min(cols) <= 2, "each color must live inside one 3x3 region"

        layer = min(_chebyshev(coord, center_coord) for coord in coords)
        assert layer in LAYERS, "block layer must be one of 1..4"
        assert layer not in layers_seen, "expected exactly one block per ring layer"
        layers_seen.add(layer)
        blocks.append((layer, len(coords), color))

    assert layers_seen == set(LAYERS), "expected all four ring layers to be encoded"
    return sorted(blocks)


def main(grid: np.ndarray) -> np.ndarray:
    """Decode each 3x3 color-count block and partially fill the matching centered square ring.

    Input structure:
    - The grid is an odd square with black background and one magenta center pixel.
    - Four separated 3x3 local regions contain 1-4 pixels of a single color, with
      one block assigned to each ring layer 1..4 by the nearest Chebyshev
      distance of that block to the center.
    - The rest of the grid is black, so the colored pixels act only as scattered encoders.

    Rule:
    - For each 3x3 region, read its foreground color K and count its non-black pixels C.
    - Map the region to one centered square ring layer d according to the nearest
      pixel of that region to the center.
    - Produce a black output grid with the magenta center preserved, then traverse
      ring d in reading order from its top-left corner and fill only the first C
      cells of that ring with color K.

    Key insight:
    - Each block carries two visible signals at once: color chooses the paint, while
      pixel count chooses how far around the ring that paint extends.
    """
    block_info = _decode_blocks(np.asarray(grid, dtype=int))
    return _make_output(block_info, int(grid.shape[0]))


def _sample_counts(rng: random.Random) -> List[int]:
    if rng.random() < 0.55:
        counts = [1, 2, 3, 4]
        rng.shuffle(counts)
        return counts
    return [rng.randint(1, 4) for _ in LAYERS]


def _sample_boxes(rng: random.Random, size: int) -> Dict[int, Coord] | None:
    by_layer = {layer: _candidate_boxes(size, layer) for layer in LAYERS}
    order = list(LAYERS)
    rng.shuffle(order)
    chosen: Dict[int, Coord] = {}
    occupied: set[Coord] = set()

    def rec(i: int) -> bool:
        if i == len(order):
            return True
        layer = order[i]
        options = by_layer[layer][:]
        rng.shuffle(options)
        for top, left in options:
            cells = set(_box_cells(top, left))
            if cells & occupied:
                continue
            chosen[layer] = (top, left)
            occupied.update(cells)
            if rec(i + 1):
                return True
            for cell in cells:
                occupied.remove(cell)
            del chosen[layer]
        return False

    return chosen if rec(0) else None


def _paint_block(grid: np.ndarray, top_left: Coord, color: int, count: int, layer: int, center: Coord, rng: random.Random) -> None:
    top, left = top_left
    cells = _box_cells(top, left)
    frontier = [cell for cell in cells if _chebyshev(cell, center) == layer]
    assert frontier, "every box must expose at least one cell on its target layer"

    chosen = {rng.choice(frontier)}
    remaining = [cell for cell in cells if cell not in chosen]
    rng.shuffle(remaining)
    for cell in remaining[: count - 1]:
        chosen.add(cell)

    for r, c in chosen:
        grid[r, c] = color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    size = GRID_SIZE
    center = size // 2
    center_coord = (center, center)
    boxes = _sample_boxes(rng, size)
    if boxes is None:
        return None

    grid = np.full((size, size), BG, dtype=int)
    grid[center, center] = CENTER_COLOR

    colors = list(BLOCK_COLORS)
    rng.shuffle(colors)
    counts = _sample_counts(rng)
    block_info: List[Tuple[int, int, int]] = []

    for layer, color, count in zip(LAYERS, colors, counts):
        _paint_block(grid, boxes[layer], color, count, layer, center_coord, rng)
        block_info.append((layer, count, color))

    expected = _make_output(block_info, size)
    try:
        pred = main(grid)
    except Exception:
        return None
    if not np.array_equal(pred, expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
