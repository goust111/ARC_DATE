import random
from collections import deque

import numpy as np


RECT_COLORS = (1, 3, 4, 5, 7)


def _neighbors4(r, c, h, w):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(cells)
    return parts


def _panel_slices(grid):
    sep_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == 9)]
    bounds = [-1] + sep_cols + [grid.shape[1]]
    return [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1)]


def _translate_panel(panel):
    moved = np.zeros_like(panel)
    for cells in _components(panel != 0):
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        block = panel[r0 : r1 + 1, c0 : c1 + 1]
        base_colors = [panel[r, c] for r, c in cells if panel[r, c] != 8]
        color = max(set(base_colors), key=base_colors.count)
        cyan = np.argwhere(block == 8)[0]
        rr, cc = map(int, cyan)
        if (rr, cc) == (0, 0):
            dr, dc = -1, -1
        elif (rr, cc) == (0, block.shape[1] - 1):
            dr, dc = -1, 1
        elif (rr, cc) == (block.shape[0] - 1, 0):
            dr, dc = 1, -1
        else:
            dr, dc = 1, 1
        moved[r0 + dr : r1 + dr + 1, c0 + dc : c1 + dc + 1] = color
    return moved


def main(grid: np.ndarray) -> np.ndarray:
    """
    Summary:
    Translate every marked rectangle diagonally inside its panel, then compress all occupied rows into one output panel.

    Input Model:
    Maroon divider columns split the board into equal-width panels, and each colored rectangle contains one cyan corner pixel that encodes its one-step diagonal move.

    Algorithm:
    Process each panel independently, recover each full rectangle from its bounding box, move it according to the cyan corner, remove empty rows, then concatenate preserved rows by scanning panels from right to left while keeping top-to-bottom order inside each panel.

    Output:
    Return a single-panel-wide grid of the original height, bottom-packed with the retained rows and with divider columns and cyan markers removed.
    """
    grid = np.asarray(grid, dtype=int)
    translated = []
    for c0, c1 in _panel_slices(grid):
        translated.append(_translate_panel(grid[:, c0:c1]))
    kept_rows = []
    for panel in reversed(translated):
        for r in range(panel.shape[0]):
            if np.any(panel[r] != 0):
                kept_rows.append(panel[r].copy())
    out = np.zeros((grid.shape[0], translated[0].shape[1]), dtype=int)
    for idx, row in enumerate(reversed(kept_rows[-grid.shape[0] :])):
        out[-1 - idx] = row
    return out


def _generate_case(rng: random.Random):
    panels = rng.randint(2, 3)
    panel_w = rng.randint(4, 6)
    h = rng.randint(12, 18)
    w = panels * panel_w + (panels - 1)
    grid = np.zeros((h, w), dtype=int)
    for k in range(1, panels):
        grid[:, k * panel_w + (k - 1)] = 9
    translated_panels = []
    for p in range(panels):
        panel = np.zeros((h, panel_w), dtype=int)
        future = np.zeros_like(panel)
        rects = rng.randint(1, 3)
        colors = rng.sample(RECT_COLORS, rects)
        empties_before = set()
        for color in colors:
            for _ in range(500):
                rh = rng.randint(2, 3)
                rw = rng.randint(2, min(3, panel_w - 1))
                dr, dc = rng.choice(((-1, -1), (-1, 1), (1, -1), (1, 1)))
                r0 = rng.randint(1 if dr == -1 else 0, h - rh - (1 if dr == 1 else 0))
                c0 = rng.randint(1 if dc == -1 else 0, panel_w - rw - (1 if dc == 1 else 0))
                r1 = r0 + rh
                c1 = c0 + rw
                nr0, nr1 = r0 + dr, r1 + dr
                nc0, nc1 = c0 + dc, c1 + dc
                if np.any(panel[r0:r1, c0:c1] != 0):
                    continue
                if np.any(future[nr0:nr1, nc0:nc1] != 0):
                    continue
                block = np.full((rh, rw), color, dtype=int)
                corner = {
                    (-1, -1): (0, 0),
                    (-1, 1): (0, rw - 1),
                    (1, -1): (rh - 1, 0),
                    (1, 1): (rh - 1, rw - 1),
                }[(dr, dc)]
                block[corner] = 8
                panel[r0:r1, c0:c1] = block
                future[nr0:nr1, nc0:nc1] = color
                empties_before.update(range(r0, r1))
                break
        translated_panels.append(future)
        start = p * (panel_w + 1)
        grid[:, start : start + panel_w] = panel
    kept_rows = []
    for panel in reversed(translated_panels):
        for r in range(h):
            if np.any(panel[r] != 0):
                kept_rows.append(panel[r].copy())
    return grid, main(grid)


def generate_input() -> np.ndarray:
    return _generate_case(random.Random(random.randrange(1 << 30)))[0]
