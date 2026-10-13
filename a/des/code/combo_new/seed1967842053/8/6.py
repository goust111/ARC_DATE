import random
from collections import deque

import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            comps.append(cells)
    return comps


def _filled_area(cells, shape):
    """Return the component plus every enclosed hole cell."""
    h, w = shape
    occ = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        occ[r, c] = True

    outside = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not occ[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not occ[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not occ[ny, nx] and not outside[ny, nx]:
                outside[ny, nx] = True
                q.append((ny, nx))
    return occ | ~outside


def main(grid: np.ndarray) -> np.ndarray:
    """Shrink each nested terrain color inward by one cell to expose the layer below.

    Input structure:
    The grid is a top-down layered map. Black cells are height 0 background. Each
    non-black color appears as one visible 4-connected ring or solid region, and
    the colored regions are nested: an outer layer surrounds the next higher layer,
    which may surround another higher layer.

    Rule:
    The output has the same shape. For every visible color region, remove the
    cells on its 4-neighbor boundary. Removed cells reveal the containing lower
    layer color, or black for the outermost layer. Interior cells of that color
    remain unchanged. If a region has no interior after this one-cell erosion, all
    of its visible cells become the lower layer color.

    Key insight:
    The exposed color is not chosen from numeric color ids; it is determined by
    the spatial containment order of the nested regions, while the shrink itself
    is a separate 4-neighbor boundary-peeling operation.
    """
    arr = np.array(grid, copy=True)
    h, w = arr.shape
    layers = []

    for color in np.unique(arr):
        if int(color) == 0:
            continue
        comps = _components(arr == color)
        for cells in comps:
            cell_set = frozenset(cells)
            filled = _filled_area(cells, arr.shape)
            layers.append(
                {
                    "color": int(color),
                    "cells": cell_set,
                    "filled": filled,
                    "area": int(np.count_nonzero(filled)),
                }
            )

    if not layers:
        return arr

    n = len(layers)
    parent = [-1] * n
    for i, layer in enumerate(layers):
        sample = next(iter(layer["cells"]))
        candidates = []
        for j, other in enumerate(layers):
            if i != j and other["filled"][sample]:
                candidates.append(j)
        if candidates:
            parent[i] = min(candidates, key=lambda j: layers[j]["area"])

    by_cells = {layer["cells"]: i for i, layer in enumerate(layers)}
    out = arr.copy()
    for i, layer in enumerate(layers):
        lower = 0 if parent[i] == -1 else layers[parent[i]]["color"]
        for r, c in layer["cells"]:
            boundary = r == 0 or c == 0 or r == h - 1 or c == w - 1
            if not boundary:
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    if arr[r + dy, c + dx] != layer["color"]:
                        boundary = True
                        break
            if boundary:
                out[r, c] = lower

    return out


def _shape_mask(height, width, roughness):
    cy = (height - 1) / 2.0 + random.uniform(-0.2, 0.2)
    cx = (width - 1) / 2.0 + random.uniform(-0.2, 0.2)
    ay = max(1.0, height / 2.0 - random.uniform(0.1, 0.7))
    ax = max(1.0, width / 2.0 - random.uniform(0.1, 0.7))
    mask = np.zeros((height, width), dtype=bool)
    for r in range(height):
        for c in range(width):
            dy = abs((r - cy) / ay)
            dx = abs((c - cx) / ax)
            score = dy**2 + dx**2
            if roughness == 0:
                keep = score <= 1.0
            elif roughness == 1:
                keep = score + 0.18 * max(dx, dy) <= 1.02
            else:
                keep = score + 0.12 * abs(dx - dy) <= 1.0
            if keep:
                mask[r, c] = True
    if not mask.any():
        mask[height // 2, width // 2] = True
    return mask


def _erode(mask):
    h, w = mask.shape
    keep = np.zeros_like(mask)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            ok = True
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dy, c + dx
                if not (0 <= nr < h and 0 <= nc < w and mask[nr, nc]):
                    ok = False
                    break
            keep[r, c] = ok
    return keep


def _bbox(mask):
    ys, xs = np.where(mask)
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def _erode_steps(mask, steps):
    out = mask.copy()
    for _ in range(steps):
        out = _erode(out)
    return out


def generate_input() -> np.ndarray:
    rows = random.randint(12, 30)
    cols = random.randint(12, 30)
    grid = np.zeros((rows, cols), dtype=int)

    depth = random.choice((2, 3))
    colors = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], depth)
    outer_h = random.randint(max(7, rows // 2), rows - 2)
    outer_w = random.randint(max(7, cols // 2), cols - 2)
    top = random.randint(1, rows - outer_h - 1)
    left = random.randint(1, cols - outer_w - 1)

    layers = []
    outer = np.zeros((rows, cols), dtype=bool)
    outer_shape = _shape_mask(outer_h, outer_w, random.randrange(3))
    outer[top : top + outer_h, left : left + outer_w] = outer_shape
    layers.append(outer)

    parent = outer
    for level in range(1, depth):
        steps = random.choice((2, 3, 4))
        available = _erode_steps(parent, steps)
        while not available.any() and steps > 0:
            steps -= 1
            available = _erode_steps(parent, steps)
        if not available.any():
            available = parent.copy()
        y1, x1, y2, x2 = _bbox(available)
        avail_h, avail_w = y2 - y1, x2 - x1

        if level == depth - 1 and random.random() < 0.45:
            mh = random.randint(1, min(3, avail_h))
            mw = random.randint(1, min(3, avail_w))
        else:
            mh = random.randint(max(3, avail_h // 2), max(3, avail_h))
            mw = random.randint(max(3, avail_w // 2), max(3, avail_w))

        child = np.zeros((rows, cols), dtype=bool)
        for _ in range(80):
            shape = _shape_mask(mh, mw, random.randrange(3))
            max_y = y2 - mh
            max_x = x2 - mw
            if max_y < y1 or max_x < x1:
                continue
            cy = random.randint(y1, max_y)
            cx = random.randint(x1, max_x)
            candidate = np.zeros((rows, cols), dtype=bool)
            candidate[cy : cy + mh, cx : cx + mw] = shape
            if candidate.any() and np.all(candidate <= available):
                child = candidate
                break
            mh = max(1, min(mh, avail_h - 1))
            mw = max(1, min(mw, avail_w - 1))
        if not child.any():
            cy = (y1 + y2) // 2
            cx = (x1 + x2) // 2
            child[cy, cx] = True
        layers.append(child)
        parent = child

    for color, mask in zip(colors, layers):
        grid[mask] = color

    return grid


if __name__ == "__main__":
    sample = generate_input()
    print(sample)
    print(main(sample))
