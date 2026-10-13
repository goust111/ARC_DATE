import random
import numpy as np


BG = 0
GREY = 5
AZURE = 8


def _neighbors(p):
    r, c = p
    return [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]


def _ordered_path(grid):
    h, w = grid.shape
    cells = set(zip(*np.where(grid == AZURE)))
    ends = [p for p in cells if sum(n in cells for n in _neighbors(p)) == 1]
    edge = [p for p in ends if p[0] in (0, h - 2) or p[1] in (0, w - 1)]
    if not edge:
        return []
    cur = sorted(edge)[0]
    prev = None
    out = []
    seen = set()
    while cur and cur not in seen:
        seen.add(cur)
        out.append(cur)
        nxt = [n for n in _neighbors(cur) if n in cells and n != prev]
        prev, cur = cur, (nxt[0] if nxt else None)
    return out


def _in_gate(grid, r, c):
    h, w = grid.shape
    return ((0 < r < h - 1 and grid[r - 1, c] == GREY and grid[r + 1, c] == GREY) or
            (0 < c < w - 1 and grid[r, c - 1] == GREY and grid[r, c + 1] == GREY))


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor an azure path by the order in which it crosses grey gates.

    Input structure: A single azure orthogonal path runs from one grid edge to
    another through paired grey short-line gate frames. A bottom-row color strip
    has one color per gate.
    Rule: Trace the path from its edge endpoint. The bottom colors, read left to
    right, color consecutive path segments: start-to-first-gate, gate-to-gate,
    and final-gate-to-end. Grey gates and the color strip are preserved.
    Key insight: Gate crossing order, not gate position, indexes the palette for
    the path segments.
    """
    out = grid.copy()
    h, w = grid.shape
    palette = [int(grid[h - 1, c]) for c in range(w) if int(grid[h - 1, c]) not in (BG, GREY, AZURE)]
    idx = 0
    was_gate = False
    for r, c in _ordered_path(grid):
        if idx < len(palette):
            out[r, c] = palette[idx]
        gate = _in_gate(grid, r, c)
        if gate and not was_gate:
            idx = min(idx + 1, max(0, len(palette) - 1))
        was_gate = gate
    return out


def generate_input() -> np.ndarray:
    h, w = 20, 24
    grid = np.zeros((h, w), dtype=int)
    palette = random.sample([1, 2, 3, 4, 6, 7, 9], 4)
    grid[h - 1, :len(palette)] = palette
    pts = [(0, 3), (5, 3), (5, 10), (12, 10), (12, 18), (h - 2, 18)]
    path = []
    cr, cc = pts[0]
    path.append((cr, cc))
    for nr, nc in pts[1:]:
        while cr != nr:
            cr += 1 if nr > cr else -1
            path.append((cr, cc))
        while cc != nc:
            cc += 1 if nc > cc else -1
            path.append((cr, cc))
    for r, c in path:
        grid[r, c] = AZURE
    gate_cells = random.sample(path[3:-3], random.randint(2, 4))
    used = []
    for r, c in gate_cells:
        if any(abs(r - rr) + abs(c - cc) < 3 for rr, cc in used):
            continue
        if 1 <= r < h - 2:
            grid[r - 1, c] = grid[r + 1, c] = GREY
        used.append((r, c))
    return grid
