import random
from collections import deque

import numpy as np

BLACK = 0
YELLOW = 4


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate labeled objects by 180 degrees and mirror unlabeled symmetric objects across their unique axis.

    Input structure:
    - Black background with several disconnected monochrome 4-connected objects.
    - Some objects enclose one yellow label pixel inside their bounding box.
    - Unlabeled objects have exactly one symmetry axis, either horizontal or vertical.

    Rule:
    - For each object, clear its bounding box footprint from the output canvas.
    - If it contains a yellow label, rotate the object mask and the label position by 180 degrees within the bounding box.
    - Otherwise mirror the object mask across its unique symmetry axis within the bounding box.

    Key insight:
    - The yellow pixel is a mode switch: it changes the transformation from symmetry-preserving mirroring to a full 180-degree turn.
    """
    out = grid.copy()
    mask = (grid != BLACK) & (grid != YELLOW)
    for cells in _components(mask):
        color = int(grid[cells[0]])
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        crop = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        for r, c in cells:
            crop[r - r0, c - c0] = 1
        labels = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if grid[r, c] == YELLOW]
        out[r0:r1 + 1, c0:c1 + 1] = np.where((out[r0:r1 + 1, c0:c1 + 1] == color) | (out[r0:r1 + 1, c0:c1 + 1] == YELLOW), BLACK, out[r0:r1 + 1, c0:c1 + 1])
        if labels:
            new_crop = np.flipud(np.fliplr(crop))
            for rr, cc in zip(*np.where(new_crop == 1)):
                out[r0 + rr, c0 + cc] = color
            lr, lc = labels[0]
            out[r1 - (lr - r0), c1 - (lc - c0)] = YELLOW
        else:
            vflip = np.fliplr(crop)
            hflip = np.flipud(crop)
            new_crop = vflip if np.array_equal(crop, vflip) and not np.array_equal(crop, hflip) else hflip
            for rr, cc in zip(*np.where(new_crop == 1)):
                out[r0 + rr, c0 + cc] = color
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    colors = [1, 2, 3, 5, 6, 7, 8, 9]
    labeled_shapes = [
        np.array([[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]], dtype=int),
    ]
    unlabeled_shapes = [
        np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
        np.array([[1, 0, 0], [1, 1, 1], [1, 0, 0]], dtype=int),
    ]
    for _ in range(400):
        h = rng.randint(16, 24)
        w = rng.randint(16, 24)
        grid = np.zeros((h, w), dtype=int)
        occ = set()
        ok = True
        for color in rng.sample(colors, rng.randint(2, 4)):
            labeled = rng.choice([True, False])
            shape = rng.choice(labeled_shapes if labeled else unlabeled_shapes)
            sh, sw = shape.shape
            for _ in range(100):
                r0 = rng.randint(1, h - sh - 2)
                c0 = rng.randint(1, w - sw - 2)
                cells = {(r0 + r, c0 + c) for r in range(sh) for c in range(sw) if shape[r, c]}
                if any(cell in occ for cell in cells):
                    continue
                for r, c in cells:
                    grid[r, c] = color
                occ |= cells
                if labeled:
                    label_choices = [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]
                    lr, lc = rng.choice(label_choices)
                    grid[r0 + lr, c0 + lc] = YELLOW
                    occ.add((r0 + lr, c0 + lc))
                break
            else:
                ok = False
                break
        if ok:
            return grid
    raise RuntimeError("failed to generate task 13/8")
