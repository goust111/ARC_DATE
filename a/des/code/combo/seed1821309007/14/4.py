import random
from collections import deque

import numpy as np


BG = 7  # orange
GEN_ATTEMPTS = 400
GRID_H_RANGE = (10, 18)
GRID_W_RANGE = (10, 18)
OBJECT_COUNT_RANGE = (2, 5)
PALETTE = [1, 2, 3, 4, 5, 6, 8, 9]


def _components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [r0 for r0, _ in cells]
            cols = [c0 for _, c0 in cells]
            comps.append(
                {
                    "color": color,
                    "cells": cells,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = np.full_like(g, BG)
    for comp in _components(g, bg=BG):
        r0, c0, r1, c1 = comp["bbox"]
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert height != width, "every object bbox must be non-square"
        if width > height:
            dr, dc = 0, width
        else:
            dr, dc = height, 0
        for r, c in comp["cells"]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                out[nr, nc] = comp["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each foreground object along its bbox major axis.

    Horizontal bbox -> shift right by bbox width.
    Vertical bbox -> shift down by bbox height.
    Original pixels are erased and out-of-bounds pixels are clipped.
    """
    return _transform(grid)


def _random_connected_shape(height: int, width: int, rng: random.Random) -> np.ndarray:
    shape = np.zeros((height, width), dtype=bool)

    # Seed four anchors so the bbox is exactly the sampled non-square rectangle.
    anchors = {(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)}
    for r, c in anchors:
        shape[r, c] = True

    current = (0, 0)
    while current != (height - 1, width - 1):
        r, c = current
        moves = []
        if r < height - 1:
            moves.append((r + 1, c))
        if c < width - 1:
            moves.append((r, c + 1))
        current = rng.choice(moves)
        shape[current] = True

    current = (0, width - 1)
    while current != (height - 1, 0):
        r, c = current
        moves = []
        if r < height - 1:
            moves.append((r + 1, c))
        if c > 0:
            moves.append((r, c - 1))
        current = rng.choice(moves)
        shape[current] = True

    target_area = rng.randint(max(height + width - 1, (height * width) // 3), max(height + width, (2 * height * width) // 3))
    while int(shape.sum()) < target_area:
        frontier = []
        for r, c in np.argwhere(shape):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < height and 0 <= nc < width and not shape[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            break
        rr, cc = rng.choice(frontier)
        shape[rr, cc] = True

    assert shape[0].any() and shape[-1].any() and shape[:, 0].any() and shape[:, -1].any()
    return shape


def _dilate(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = True
    return out


def _can_place(occupied: np.ndarray, shape: np.ndarray, top: int, left: int) -> bool:
    h, w = shape.shape
    region = occupied[top : top + h, left : left + w]
    if np.any(region & shape):
        return False

    pad = np.zeros_like(occupied)
    pad[top : top + h, left : left + w] = shape
    return not np.any(_dilate(pad) & occupied)


def _sample_dims(axis: str, rng: random.Random) -> tuple[int, int]:
    if axis == "horizontal":
        height = rng.randint(2, 4)
        width = rng.randint(height + 1, min(7, height + 4))
    else:
        width = rng.randint(2, 4)
        height = rng.randint(width + 1, min(7, width + 4))
    return height, width


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    count = rng.randint(*OBJECT_COUNT_RANGE)
    axes = ["horizontal", "vertical"]
    while len(axes) < count:
        axes.append(rng.choice(["horizontal", "vertical"]))
    rng.shuffle(axes)

    colors = rng.sample(PALETTE, count)
    for axis, color in zip(axes, colors):
        height, width = _sample_dims(axis, rng)
        if height > h or width > w:
            return None
        shape = _random_connected_shape(height, width, rng)

        placements = []
        for top in range(h - height + 1):
            for left in range(w - width + 1):
                if _can_place(occupied, shape, top, left):
                    placements.append((top, left))
        if not placements:
            return None

        top, left = rng.choice(placements)
        region = grid[top : top + height, left : left + width]
        region[shape] = color
        occupied[top : top + height, left : left + width] |= shape

    comps = _components(grid, bg=BG)
    if len(comps) != count:
        return None
    if not any((bbox[3] - bbox[1]) > (bbox[2] - bbox[0]) for bbox in [comp["bbox"] for comp in comps]):
        return None
    if not any((bbox[2] - bbox[0]) > (bbox[3] - bbox[1]) for bbox in [comp["bbox"] for comp in comps]):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
