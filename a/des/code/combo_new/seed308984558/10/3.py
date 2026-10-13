import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
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
            out.append(cells)
    return out


def _bbox(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _rotate(arr: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return arr
    if side == "right":
        return np.rot90(arr, -1)
    if side == "bottom":
        return np.rot90(arr, 2)
    return np.rot90(arr, 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each room motif by the doorway side and stamp a copy outside the opening.

    Input structure:
    - Grey hollow rectangles each have exactly one one-cell doorway gap on a side.
    - Inside each room is a colored motif on black.

    Rule:
    - Determine the doorway side from the missing border cell.
    - Rotate the room motif so it faces that side: top=0, right=CW, bottom=180, left=CCW.
    - Keep the room and original motif unchanged, and place the rotated copy just
      outside the doorway on the doorway normal.

    Key insight:
    - The gap encodes both where the copy goes and how it is rotated before projection.
    """
    out = grid.copy()
    for comp in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        side = "top"
        if np.any(grid[r0, c0 : c1 + 1] == BLACK):
            side = "top"
        elif np.any(grid[r1, c0 : c1 + 1] == BLACK):
            side = "bottom"
        elif np.any(grid[r0 : r1 + 1, c0] == BLACK):
            side = "left"
        else:
            side = "right"
        motif = grid[r0 + 1 : r1, c0 + 1 : c1].copy()
        motif[motif == GREY] = BLACK
        motif = _rotate(motif, side)
        if side == "top":
            rr, cc = r0 - motif.shape[0], c0 + 1
        elif side == "bottom":
            rr, cc = r1 + 1, c0 + 1
        elif side == "left":
            rr, cc = r0 + 1, c0 - motif.shape[1]
        else:
            rr, cc = r0 + 1, c1 + 1
        for i in range(motif.shape[0]):
            for j in range(motif.shape[1]):
                if motif[i, j] != BLACK and 0 <= rr + i < out.shape[0] and 0 <= cc + j < out.shape[1]:
                    out[rr + i, cc + j] = motif[i, j]
    return out


def generate_input() -> np.ndarray:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    grid = np.zeros((h, w), dtype=int)
    for color in random.sample(COLORS, random.randint(2, 3)):
        fh, fw = random.randint(5, 7), random.randint(5, 7)
        r0 = random.randint(2, h - fh - 2)
        c0 = random.randint(2, w - fw - 2)
        grid[r0, c0 : c0 + fw] = GREY
        grid[r0 + fh - 1, c0 : c0 + fw] = GREY
        grid[r0 : r0 + fh, c0] = GREY
        grid[r0 : r0 + fh, c0 + fw - 1] = GREY
        side = random.choice(["top", "bottom", "left", "right"])
        if side == "top":
            grid[r0, c0 + fw // 2] = BLACK
        elif side == "bottom":
            grid[r0 + fh - 1, c0 + fw // 2] = BLACK
        elif side == "left":
            grid[r0 + fh // 2, c0] = BLACK
        else:
            grid[r0 + fh // 2, c0 + fw - 1] = BLACK
        motif = np.zeros((fh - 2, fw - 2), dtype=int)
        motif[0, 0] = color
        motif[min(1, motif.shape[0] - 1), 0] = color
        motif[-1, min(1, motif.shape[1] - 1)] = color
        grid[r0 + 1 : r0 + fh - 1, c0 + 1 : c0 + fw - 1] = motif
    return grid
