import random
from collections import deque

import numpy as np

BLACK = 0
GREEN = 3


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(cells)
    return parts


def _gap_side(grid: np.ndarray, bbox):
    r0, r1, c0, c1 = bbox
    if BLACK in grid[r0, c0:c1 + 1]:
        return "top"
    if BLACK in grid[r1, c0:c1 + 1]:
        return "bottom"
    if BLACK in grid[r0:r1 + 1, c0]:
        return "left"
    return "right"


def _rotate_to_top(crop: np.ndarray, side: str):
    if side == "top":
        return crop
    if side == "right":
        return np.rot90(crop, 1)
    if side == "bottom":
        return np.rot90(crop, 2)
    return np.rot90(crop, -1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each framed stamp according to its border gap and merge the corrected stamps.

    Input structure:
    - Black background with several separated green rectangular frames of identical outer size.
    - Each frame border has one single-pixel gap on exactly one side.
    - Inside each frame is a colored stamp that does not touch the border.

    Rule:
    - Treat a top-side gap as the canonical orientation.
    - Rotate each frame's interior so that its gap would move to the top.
    - Overlay all corrected interiors into one output canvas of the interior size.

    Key insight:
    - The missing border pixel is not damage; it is the orientation marker that tells how to undo each stamp's rotation before merging.
    """
    frames = _components(grid == GREEN)
    boxes = []
    for cells in frames:
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        boxes.append((min(rows), max(rows), min(cols), max(cols)))
    inner_h = boxes[0][1] - boxes[0][0] - 1
    inner_w = boxes[0][3] - boxes[0][2] - 1
    out = np.zeros((inner_h, inner_w), dtype=int)
    for bbox in boxes:
        r0, r1, c0, c1 = bbox
        side = _gap_side(grid, bbox)
        inner = grid[r0 + 1:r1, c0 + 1:c1].copy()
        corrected = _rotate_to_top(inner, side)
        mask = corrected != BLACK
        out[mask] = corrected[mask]
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    colors = [1, 2, 4, 5, 6, 7, 8, 9]
    for _ in range(400):
        inner_h = rng.randint(3, 5)
        inner_w = inner_h
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        n = rng.randint(2, 4)
        grid_h = n * (outer_h + 1) + 1
        grid_w = outer_w + 2
        if grid_h > 30 or grid_w > 30:
            continue
        grid = np.zeros((grid_h, grid_w), dtype=int)
        base = np.zeros((inner_h, inner_w), dtype=int)
        pixels = [(r, c) for r in range(inner_h) for c in range(inner_w)]
        rng.shuffle(pixels)
        chosen = pixels[: rng.randint(inner_h, inner_h * inner_w)]
        for r, c in chosen:
            base[r, c] = rng.choice(colors)
        assignments = [np.zeros_like(base) for _ in range(n)]
        for r, c in chosen:
            assignments[rng.randrange(n)][r, c] = base[r, c]
        for idx in range(n):
            if not np.any(assignments[idx]):
                r, c = chosen[idx % len(chosen)]
                assignments[idx][r, c] = base[r, c]
        y = 1
        for stamp in assignments:
            side = rng.choice(["top", "right", "bottom", "left"])
            if side == "top":
                inner = stamp
            elif side == "right":
                inner = np.rot90(stamp, -1)
            elif side == "bottom":
                inner = np.rot90(stamp, 2)
            else:
                inner = np.rot90(stamp, 1)
            grid[y:y + outer_h, 1:1 + outer_w] = BLACK
            grid[y, 1:1 + outer_w] = GREEN
            grid[y + outer_h - 1, 1:1 + outer_w] = GREEN
            grid[y:y + outer_h, 1] = GREEN
            grid[y:y + outer_h, outer_w] = GREEN
            if side == "top":
                grid[y, 1 + outer_w // 2] = BLACK
            elif side == "bottom":
                grid[y + outer_h - 1, 1 + outer_w // 2] = BLACK
            elif side == "left":
                grid[y + outer_h // 2, 1] = BLACK
            else:
                grid[y + outer_h // 2, outer_w] = BLACK
            grid[y + 1:y + outer_h - 1, 2:outer_w] = inner
            y += outer_h + 1
        return grid
    raise RuntimeError("failed to generate task 14/3")
