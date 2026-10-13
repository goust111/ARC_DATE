import random

import numpy as np


def _bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _ring_thickness(grid: np.ndarray, color: int):
    pts = np.argwhere(grid == color)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    t = 0
    while r0 + t <= r1 - t and np.all(grid[r0 + t, c0 : c1 + 1] == color):
        t += 1
    return int(r0), int(c0), int(r1), int(c1), t


def _draw_ring(out: np.ndarray, bbox, color: int, thickness: int):
    r0, c0, r1, c1 = bbox
    out[r0 : r0 + thickness, c0 : c1 + 1] = color
    out[r1 - thickness + 1 : r1 + 1, c0 : c1 + 1] = color
    out[r0 : r1 + 1, c0 : c0 + thickness] = color
    out[r0 : r1 + 1, c1 - thickness + 1 : c1 + 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-nest concentric colored frames so thickness decreases from the outside inward.

    Input structure:
    - The grid has a dominant background color.
    - Several differently colored rectangular hollow frames are already nested concentrically.
    - Each frame has a uniform integer thickness, but the input nesting order is unrelated to that thickness.

    Rule:
    - Measure each frame's border thickness from its own color band.
    - Sort the frames by thickness from largest to smallest.
    - Keep the overall outer bounding box fixed, redraw the thickest frame outside, then place each next frame directly inside the previous one.
    - Leave the final center area as background.

    Key insight:
    - Color does not determine order; the hidden sorting key is the geometric thickness of each ring.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg(g)
    infos = []
    for color in [int(v) for v in np.unique(g) if v != bg]:
        infos.append((color, *_ring_thickness(g, color)))
    outer = (
        min(info[1] for info in infos),
        min(info[2] for info in infos),
        max(info[3] for info in infos),
        max(info[4] for info in infos),
    )
    infos.sort(key=lambda x: -x[5])
    out = np.full_like(g, bg)
    bbox = list(outer)
    for color, _, _, _, _, thickness in infos:
        _draw_ring(out, bbox, color, thickness)
        bbox = [bbox[0] + thickness, bbox[1] + thickness, bbox[2] - thickness, bbox[3] - thickness]
    return out


def generate_input() -> np.ndarray:
    for _ in range(300):
        bg = random.choice([0, 1, 8])
        thicknesses = random.sample([1, 2, 3], random.randint(2, 3))
        colors = random.sample([c for c in range(10) if c != bg], len(thicknesses))
        total = sum(thicknesses)
        inner_h = random.randint(2, 6)
        inner_w = random.randint(2, 6)
        h = inner_h + 2 * total + random.randint(2, 6)
        w = inner_w + 2 * total + random.randint(2, 6)
        if h > 30 or w > 30:
            continue
        grid = np.full((h, w), bg, dtype=int)
        top = random.randint(1, h - (inner_h + 2 * total) - 1)
        left = random.randint(1, w - (inner_w + 2 * total) - 1)
        bbox = [top, left, top + inner_h + 2 * total - 1, left + inner_w + 2 * total - 1]
        order = colors[:]
        random.shuffle(order)
        for color, thickness in zip(order, thicknesses):
            _draw_ring(grid, bbox, color, thickness)
            bbox = [bbox[0] + thickness, bbox[1] + thickness, bbox[2] - thickness, bbox[3] - thickness]
        return grid
    raise RuntimeError("failed to generate input")
