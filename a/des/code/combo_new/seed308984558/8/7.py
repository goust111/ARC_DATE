import random
from collections import deque

import numpy as np


BLACK = 0
FRAME_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
PATTERN_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _components_of_color(grid: np.ndarray, color: int):
    mask = grid == color
    h, w = grid.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
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
            comps.append(cells)
    return comps


def _bbox(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each framed interior pattern according to the repaired border gaps.

    Input structure:
    - The grid contains disjoint colored rectangular frames on black.
    - Each frame has one or two black gaps on its border and a different-colored
      asymmetric pattern inside.
    - The interior pattern does not touch the border.

    Rule:
    - Detect whether the frame has a gap on the top/bottom side and/or on the left/
      right side.
    - Flip the whole interior canvas vertically for top/bottom gaps and horizontally
      for left/right gaps, applying both when both gap classes are present.
    - Restore the border to a complete rectangle in the frame color.
    - Replace the old interior with the transformed interior.

    Key insight:
    - The missing border cells are instructions: vertical-side gaps trigger left-right
      mirroring, horizontal-side gaps trigger up-down mirroring, and a pair of gaps
      composes into a 180-degree rotation.
    """
    out = grid.copy()
    for frame_color in [int(v) for v in np.unique(grid) if v != BLACK]:
        comps = _components_of_color(grid, frame_color)
        for comp in comps:
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            interior = grid[r0 + 1 : r1, c0 + 1 : c1].copy()
            border = grid[r0 : r1 + 1, c0 : c1 + 1]
            has_tb_gap = np.any(border[0] == BLACK) or np.any(border[-1] == BLACK)
            has_lr_gap = np.any(border[:, 0] == BLACK) or np.any(border[:, -1] == BLACK)
            if has_tb_gap:
                interior = np.flipud(interior)
            if has_lr_gap:
                interior = np.fliplr(interior)
            out[r0 : r1 + 1, c0 : c1 + 1] = BLACK
            out[r0, c0 : c1 + 1] = frame_color
            out[r1, c0 : c1 + 1] = frame_color
            out[r0 : r1 + 1, c0] = frame_color
            out[r0 : r1 + 1, c1] = frame_color
            out[r0 + 1 : r1, c0 + 1 : c1] = interior
    return out


def generate_input() -> np.ndarray:
    while True:
        h = random.randint(16, 26)
        w = random.randint(16, 26)
        grid = np.zeros((h, w), dtype=int)
        placed = []
        for frame_color in random.sample(FRAME_COLORS, random.randint(2, 4)):
            fh = random.randint(5, 7)
            fw = random.randint(5, 7)
            r0 = random.randint(1, h - fh - 1)
            c0 = random.randint(1, w - fw - 1)
            if any(not (r0 + fh - 1 < pr0 or pr1 < r0 or c0 + fw - 1 < pc0 or pc1 < c0) for pr0, pc0, pr1, pc1 in placed):
                continue
            placed.append((r0, c0, r0 + fh - 1, c0 + fw - 1))
            grid[r0, c0 : c0 + fw] = frame_color
            grid[r0 + fh - 1, c0 : c0 + fw] = frame_color
            grid[r0 : r0 + fh, c0] = frame_color
            grid[r0 : r0 + fh, c0 + fw - 1] = frame_color
            gap_sides = random.sample(["top", "bottom", "left", "right"], random.choice([1, 2]))
            if "top" in gap_sides:
                grid[r0, c0 + fw // 2] = BLACK
            if "bottom" in gap_sides:
                grid[r0 + fh - 1, c0 + fw // 2] = BLACK
            if "left" in gap_sides:
                grid[r0 + fh // 2, c0] = BLACK
            if "right" in gap_sides:
                grid[r0 + fh // 2, c0 + fw - 1] = BLACK
            inner_color = random.choice([c for c in PATTERN_COLORS if c != frame_color])
            inner = np.zeros((fh - 2, fw - 2), dtype=int)
            pattern = [(0, 0), (0, 1), (1, 1), (2, 1)]
            if inner.shape[0] >= 3 and inner.shape[1] >= 2:
                rr = random.randint(0, inner.shape[0] - 3)
                cc = random.randint(0, inner.shape[1] - 2)
                for dr, dc in pattern:
                    inner[rr + dr, cc + dc] = inner_color
            grid[r0 + 1 : r0 + fh - 1, c0 + 1 : c0 + fw - 1] = inner
        if np.count_nonzero(grid) > 0:
            return grid
