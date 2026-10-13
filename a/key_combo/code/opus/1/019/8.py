import random
from collections import deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components_color(grid, color):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != color or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in DIR4:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and grid[nx, ny] == color:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(comp)
    return comps


def _rect_frame(comp):
    cells = set(comp)
    min_r, max_r = min(r for r, _ in cells), max(r for r, _ in cells)
    min_c, max_c = min(c for _, c in cells), max(c for _, c in cells)
    if max_r - min_r < 2 or max_c - min_c < 2:
        return None
    border = {(r, c) for r in range(min_r, max_r + 1) for c in range(min_c, max_c + 1)
              if r in (min_r, max_r) or c in (min_c, max_c)}
    return (min_r, max_r, min_c, max_c) if cells == border else None


def main(grid: np.ndarray) -> np.ndarray:
    """Overlay three panels, then fill surviving hollow frames while protecting marker boxes.

    Input structure: Three equal-height panels contain hollow colored rectangles,
    magenta markers, and decorations on black backgrounds.
    Rule: Middle, then top, then bottom nonblack pixels are overlaid, making the
    bottom panel highest priority. Complete hollow rectangles in the composite
    fill their interior background with their frame color except inside the
    bounding box of surviving magenta markers; all magenta markers are erased.
    Key insight: Protection is computed only from markers that survive the layer
    overlay, so covered markers no longer affect the filled area.
    """
    grid = np.asarray(grid, dtype=int)
    total_h, w = grid.shape
    h = total_h // 3
    top = grid[:h]
    mid = grid[h:2 * h]
    bottom = grid[2 * h:3 * h]
    comp = np.zeros((h, w), dtype=int)
    for panel in (mid, top, bottom):
        comp[panel != 0] = panel[panel != 0]

    out = comp.copy()
    for color in [int(x) for x in np.unique(comp) if x not in (0, 6)]:
        for component in _components_color(comp, color):
            box = _rect_frame(component)
            if not box:
                continue
            min_r, max_r, min_c, max_c = box
            interior = [(r, c) for r in range(min_r + 1, max_r) for c in range(min_c + 1, max_c)]
            markers = [(r, c) for r, c in interior if comp[r, c] == 6]
            protected = set()
            if markers:
                pr0, pr1 = min(r for r, _ in markers), max(r for r, _ in markers)
                pc0, pc1 = min(c for _, c in markers), max(c for _, c in markers)
                protected = {(r, c) for r in range(pr0, pr1 + 1) for c in range(pc0, pc1 + 1)}
            for r, c in interior:
                if (r, c) not in protected and out[r, c] == 0:
                    out[r, c] = color
    out[out == 6] = 0
    return out


def _draw_frame(panel, top, left, height, width, color):
    bottom, right = top + height - 1, left + width - 1
    panel[top:bottom + 1, left] = color
    panel[top:bottom + 1, right] = color
    panel[top, left:right + 1] = color
    panel[bottom, left:right + 1] = color


def generate_input() -> np.ndarray:
    h = random.randint(6, 8)
    w = random.randint(10, 12)
    panels = [np.zeros((h, w), dtype=int) for _ in range(3)]
    # Top marker is deliberately covered by a bottom decoration.
    _draw_frame(panels[0], 1, 1, min(5, h - 1), 5, 1)
    panels[0][3, 3] = 6
    _draw_frame(panels[1], 2 if h > 6 else 1, w - 6, min(5, h - 2), 5, 2)
    panels[1][min(h - 2, 4), w - 4] = 6
    _draw_frame(panels[2], 1, w - 6, min(5, h - 1), 5, 3)
    r0, c0 = 2, w - 5
    panels[2][r0, c0] = 6
    panels[2][min(h - 2, r0 + 2), min(w - 2, c0 + 2)] = 6
    panels[2][3, 3] = random.choice([3, 4, 7, 8])
    return np.vstack(panels)
