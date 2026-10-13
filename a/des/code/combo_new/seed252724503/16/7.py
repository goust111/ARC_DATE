import random

import numpy as np


FRAME = 1
BG = 0


def _perimeter(bbox):
    r0, c0, r1, c1 = bbox
    cells = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c1))
    for c in range(c1, c0 - 1, -1):
        cells.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        cells.append((r, c0))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each ring marker around its own concentric frame by a layer-dependent number of border steps.

    Input structure:
    - The background is black.
    - Several one-pixel-thick blue rectangular frames are nested concentrically with a one-cell black gap between consecutive layers.
    - Each frame carries exactly one non-blue marker pixel on its perimeter.

    Rule:
    - Order the frames from outside to inside to obtain layer numbers 1, 2, 3, ...
    - On odd layers, move the marker clockwise by K steps; on even layers, move it counterclockwise by K steps, where K is the layer number.
    - Replace the old marker position with the blue frame color and keep all frame geometry unchanged.

    Key insight:
    - The layer index controls both the motion distance and the motion direction, so the same border walk rule must be applied separately on each concentric rectangle.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    coords = np.argwhere(g == FRAME)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    layers = []
    while r0 <= r1 and c0 <= c1:
        ring = _perimeter((int(r0), int(c0), int(r1), int(c1)))
        if all(g[r, c] != BG for r, c in ring):
            layers.append((int(r0), int(c0), int(r1), int(c1)))
        r0 += 2
        c0 += 2
        r1 -= 2
        c1 -= 2
    for idx, bbox in enumerate(layers, start=1):
        ring = _perimeter(bbox)
        marker_index = next(i for i, (r, c) in enumerate(ring) if g[r, c] not in (BG, FRAME))
        nr, nc = ring[(marker_index + (idx if idx % 2 else -idx)) % len(ring)]
        r, c = ring[marker_index]
        out[r, c] = FRAME
        out[nr, nc] = g[r, c]
    return out


def _draw_ring(grid, bbox):
    for r, c in _perimeter(bbox):
        grid[r, c] = FRAME


def generate_input() -> np.ndarray:
    for _ in range(300):
        layers = random.randint(2, 5)
        inner_h = random.randint(2, 5)
        inner_w = random.randint(2, 5)
        h = inner_h + 4 * layers - 2
        w = inner_w + 4 * layers - 2
        if h > 30 or w > 30:
            continue
        grid = np.full((h, w), BG, dtype=int)
        bbox = [0, 0, h - 1, w - 1]
        colors = random.sample([2, 3, 4, 5, 6, 7, 8, 9], layers)
        for color in colors:
            _draw_ring(grid, bbox)
            ring = _perimeter(tuple(bbox))
            r, c = random.choice(ring)
            grid[r, c] = color
            bbox = [bbox[0] + 2, bbox[1] + 2, bbox[2] - 2, bbox[3] - 2]
        return grid
    raise RuntimeError("failed to generate input")
