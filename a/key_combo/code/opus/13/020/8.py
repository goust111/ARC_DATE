from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
TEAL = 8

MARKER_TO_DIR = {
    BLUE: (0, 1),
    RED: (0, -1),
    GREEN: (1, 0),
    PINK: (-1, 0),
}
RING_PALETTE = [BLUE, RED, GREEN, YELLOW, GREY, TEAL]
OUTER_RING_CHOICES = [YELLOW, GREY, TEAL]
GEN_MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert len(coords) > 0, "expected foreground structure"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= r < h and 0 <= c < w


def _layer_index(r: int, c: int, bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return min(r - r0, c - c0, r1 - r, c1 - c)


def _ring_count(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (min(r1 - r0 + 1, c1 - c0 + 1) + 1) // 2


def _paint_rings(grid: np.ndarray, bbox: tuple[int, int, int, int], ring_colors: list[int]) -> None:
    r0, c0, r1, c1 = bbox
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            grid[r, c] = ring_colors[_layer_index(r, c, bbox)]


def _extract_ring_colors(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> list[int]:
    ring_colors: list[int] = []
    for layer in range(_ring_count(bbox)):
        seen: set[int] = set()
        r0, c0, r1, c1 = bbox
        for r in range(r0 + layer, r1 - layer + 1):
            for c in range(c0 + layer, c1 - layer + 1):
                if _layer_index(r, c, bbox) != layer:
                    continue
                color = int(grid[r, c])
                if layer == 0 and color in MARKER_TO_DIR:
                    continue
                seen.add(color)
        assert len(seen) == 1, "each ring layer must have one underlying color"
        ring_colors.append(next(iter(seen)))
    return ring_colors


def _outer_markers(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = bbox
    markers: list[tuple[int, int, int]] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if _layer_index(r, c, bbox) != 0:
                continue
            color = int(grid[r, c])
            if color in MARKER_TO_DIR:
                markers.append((r, c, color))
    return markers


def _trace_cells(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    start: tuple[int, int],
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    dr, dc = direction
    r, c = start[0] + dr, start[1] + dc
    cells: list[tuple[int, int]] = []
    while _in_bounds(shape, r, c) and r0 <= r <= r1 and c0 <= c <= c1:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend direction-coded rays through concentric rings, then cyclically recolor only the rings hit by ray crossings.

    Input structure:
    - The background is black.
    - One centered filled rectangle is partitioned into 4-6 one-cell-thick concentric rectangular ring layers.
    - Each ring layer has a single underlying color, while 2-4 marker pixels on the outer ring use blue, red, green, or pink to encode inward ray directions.

    Rule:
    - Blue markers fire rightward rays, red markers leftward rays, green markers downward rays, and pink markers upward rays.
    - Each ray recolors every traversed in-structure cell after its source to grey, and any cell traversed by both one horizontal ray and one vertical ray becomes yellow instead.
    - Read the original outer-to-inner ring color order while ignoring the outer-ring marker overlays.
    - Any ring layer containing at least one yellow crossing cell has all of its non-ray-covered cells recolored to the next color in that nesting order; other ring layers stay unchanged.

    Key insight:
    - The output depends on coupling two independent structures: marker colors determine ray geometry, and the ring layer containing each ray intersection determines which nesting levels participate in the color cycle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    assert bg == BLACK, "expected black background"

    bbox = _bbox(g != bg)
    r0, c0, r1, c1 = bbox
    assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] != bg), "ring structure must fill its bounding box"

    ring_colors = _extract_ring_colors(g, bbox)
    assert 4 <= len(ring_colors) <= 6, "expected 4-6 ring layers"

    markers = _outer_markers(g, bbox)
    assert 2 <= len(markers) <= 4, "expected 2-4 outer-ring markers"

    hit_counts: Counter[tuple[int, int]] = Counter()
    for r, c, color in markers:
        for cell in _trace_cells(g.shape, bbox, (r, c), MARKER_TO_DIR[color]):
            hit_counts[cell] += 1

    out = g.copy()
    crossing_layers: set[int] = set()
    for (r, c), count in hit_counts.items():
        if count >= 2:
            crossing_layers.add(_layer_index(r, c, bbox))
        out[r, c] = YELLOW if count >= 2 else GREY

    successors = {
        layer: ring_colors[(layer + 1) % len(ring_colors)] for layer in range(len(ring_colors))
    }
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if (r, c) in hit_counts:
                continue
            layer = _layer_index(r, c, bbox)
            if layer in crossing_layers:
                out[r, c] = successors[layer]

    return out


def _choose_ring_colors(ring_count: int) -> list[int]:
    chosen = random.sample(RING_PALETTE, ring_count)
    outer_candidates = [color for color in chosen if color in OUTER_RING_CHOICES]
    assert outer_candidates
    outer = random.choice(outer_candidates)
    inner = [color for color in chosen if color != outer]
    random.shuffle(inner)
    return [outer] + inner


def _choose_marker_layout(size: int) -> tuple[list[int], list[int]]:
    mode = random.choice([(1, 1), (2, 1), (1, 2), (2, 2)])
    row_count, col_count = mode
    rows = sorted(random.sample(range(1, size - 1), row_count))
    cols = sorted(random.sample(range(1, size - 1), col_count))
    return rows, cols


def _place_markers(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    rows: list[int],
    cols: list[int],
) -> None:
    r0, c0, r1, c1 = bbox
    for rr in rows:
        if random.random() < 0.5:
            grid[r0 + rr, c0] = BLUE
        else:
            grid[r0 + rr, c1] = RED
    for cc in cols:
        if random.random() < 0.5:
            grid[r0, c0 + cc] = GREEN
        else:
            grid[r1, c0 + cc] = PINK


def _selected_layers(size: int, rows: list[int], cols: list[int]) -> set[int]:
    layers: set[int] = set()
    last = size - 1
    for rr in rows:
        for cc in cols:
            layers.add(min(rr, cc, last - rr, last - cc))
    return layers


def _gen_attempt() -> np.ndarray | None:
    ring_count = random.choices([4, 5, 6], weights=[4, 4, 2], k=1)[0]
    size = random.choice([2 * ring_count - 1, 2 * ring_count])

    height = random.randint(size + 2, min(19, size + 6))
    width = random.randint(size + 2, min(19, size + 6))
    r0 = random.randint(1, height - size - 1)
    c0 = random.randint(1, width - size - 1)
    bbox = (r0, c0, r0 + size - 1, c0 + size - 1)

    rows, cols = _choose_marker_layout(size)
    selected_layers = _selected_layers(size, rows, cols)
    if not selected_layers:
        return None
    if len(rows) * len(cols) >= 2 and random.random() < 0.45 and len(selected_layers) < 2:
        return None

    ring_colors = _choose_ring_colors(ring_count)
    grid = np.full((height, width), BLACK, dtype=int)
    _paint_rings(grid, bbox, ring_colors)
    _place_markers(grid, bbox, rows, cols)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None

    changed = int(np.count_nonzero(out != grid))
    if changed < len(rows) + len(cols) + 3:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-ring ARC input")
