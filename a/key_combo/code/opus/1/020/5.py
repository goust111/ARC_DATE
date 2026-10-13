import random

import numpy as np


def _draw_filled_rect(canvas, box, color):
    r0, r1, c0, c1 = box
    canvas[r0:r1 + 1, c0:c1 + 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Complete nested rectangles, then let innermost diagonal rays absorb hit layers.

    Input structure: Each foreground color gives sparse evidence for one
    axis-aligned rectangle, with rectangles strictly nested and the innermost
    one fully visible.
    Rule: Infer each color's bounding box and complete filled rectangles from
    largest to smallest. From the innermost rectangle's four corners, draw
    diagonal rays in the outward directions until each ray hits a different
    outer layer or exits. Every hit layer is recolored to the innermost color.
    Key insight: Completion builds the collision field that the later diagonal
    probes use to decide which outer layers are absorbed.
    """
    grid = np.asarray(grid, dtype=int)
    colors = [int(x) for x in np.unique(grid) if x != 0]
    layers = []
    for color in colors:
        coords = np.argwhere(grid == color)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        area = (int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1)
        layers.append({"color": color, "box": (int(r0), int(r1), int(c0), int(c1)), "area": area})
    layers.sort(key=lambda x: x["area"], reverse=True)
    canvas = np.zeros_like(grid)
    masks = {}
    for layer in layers:
        before = canvas.copy()
        _draw_filled_rect(canvas, layer["box"], layer["color"])
        masks[layer["color"]] = canvas == layer["color"]
        for old in masks:
            if old != layer["color"]:
                masks[old] &= canvas == old

    inner = min(layers, key=lambda x: x["area"])
    inner_color = inner["color"]
    r0, r1, c0, c1 = inner["box"]
    rays = [((r0, c0), (-1, -1)), ((r0, c1), (-1, 1)), ((r1, c0), (1, -1)), ((r1, c1), (1, 1))]
    hits = set()
    ray_cells = []
    for (sr, sc), (dr, dc) in rays:
        r, c = sr + dr, sc + dc
        current_ray = []
        while 0 <= r < canvas.shape[0] and 0 <= c < canvas.shape[1]:
            val = int(canvas[r, c])
            if val != 0 and val != inner_color:
                hits.add(val)
                break
            current_ray.append((r, c))
            r += dr
            c += dc
        ray_cells.extend(current_ray)
    out = canvas.copy()
    for color in hits:
        out[masks[color]] = inner_color
    for r, c in ray_cells:
        if out[r, c] == 0:
            out[r, c] = inner_color
    return out


def _sample_rect_border(grid, box, color):
    r0, r1, c0, c1 = box
    choices = []
    for c in range(c0, c1 + 1):
        choices.append((r0, c))
        choices.append((r1, c))
    for r in range(r0, r1 + 1):
        choices.append((r, c0))
        choices.append((r, c1))
    for r, c in random.sample(choices, min(len(choices), random.randint(8, 12))):
        grid[r, c] = color
    for p in [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]:
        grid[p] = color


def generate_input() -> np.ndarray:
    grid = np.zeros((15, 15), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7], random.choice([3, 4]))
    outer = (1, 13, 1, 13)
    middle = (3, 11, 2, 12)
    inner = random.choice([(6, 9, 5, 8), (5, 8, 6, 10), (6, 10, 4, 7)])
    boxes = [outer, middle]
    if len(colors) == 4:
        boxes.append((4, 10, 4, 10))
    boxes.append(inner)
    for box, color in zip(boxes[:-1], colors[:-1]):
        _sample_rect_border(grid, box, color)
    _draw_filled_rect(grid, boxes[-1], colors[-1])
    return grid

