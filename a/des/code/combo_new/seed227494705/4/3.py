import random
import numpy as np


BG = 0
GREY = 5


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _ordered_path(cells, h, w):
    s = set(cells)
    ends = []
    for r, c in cells:
        deg = sum((r + dr, c + dc) in s for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
        if deg == 1:
            ends.append((r, c))
    edge_ends = [p for p in ends if p[0] in (0, h - 1) or p[1] in (0, w - 1)]
    if not edge_ends:
        edge_ends = [p for p in cells if p[0] in (0, h - 1) or p[1] in (0, w - 1)]
    if not edge_ends:
        return []
    start = sorted(edge_ends)[0]
    order = []
    prev = None
    cur = start
    visited = set()
    while cur is not None and cur not in visited:
        visited.add(cur)
        order.append(cur)
        nbrs = []
        r, c = cur
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            np_ = (r + dr, c + dc)
            if np_ in s and np_ != prev:
                nbrs.append(np_)
        prev, cur = cur, (nbrs[0] if nbrs else None)
    return order


def _gate_color(grid, r, c):
    h, w = grid.shape
    pairs = []
    if r > 0 and r < h - 1 and grid[r - 1, c] == grid[r + 1, c] and grid[r - 1, c] not in (BG, GREY):
        pairs.append(int(grid[r - 1, c]))
    if c > 0 and c < w - 1 and grid[r, c - 1] == grid[r, c + 1] and grid[r, c - 1] not in (BG, GREY):
        pairs.append(int(grid[r, c - 1]))
    return pairs[0] if pairs else None


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor grey polyline paths by the sequence of colored gates they cross.

    Input structure: Grey one-cell-wide non-branching orthogonal paths start on
    an outer edge. Colored two-pixel gates sit on opposite sides of some path
    cells, while all other cells are black background.
    Rule: Trace each grey path from its edge endpoint. The path remains grey
    until it crosses a colored gate; after crossing a gate, later path cells are
    recolored with that gate color until the next gate updates the active color.
    Gate pixels themselves are preserved.
    Key insight: The path color is a state carried forward from the most recent
    gate in traversal order.
    """
    out = grid.copy()
    h, w = grid.shape
    for comp in _components(grid == GREY):
        order = _ordered_path(comp, h, w)
        active = GREY
        for r, c in order:
            out[r, c] = active
            gc = _gate_color(grid, r, c)
            if gc is not None:
                active = gc
    return out


def generate_input() -> np.ndarray:
    h = random.choice([16, 18, 20, 22])
    w = random.choice([16, 18, 20, 22])
    grid = np.zeros((h, w), dtype=int)
    top_row = random.randint(2, h // 3)
    mid_row = random.randint(h // 2 - 1, h // 2 + 1)
    low_row = random.randint(2 * h // 3, h - 3)
    bends = sorted(random.sample(range(3, w - 3), 2))
    pts = [(top_row, 0), (top_row, bends[0]), (mid_row, bends[0]),
           (mid_row, bends[1]), (low_row, bends[1]), (low_row, w - 1)]
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
        grid[r, c] = GREY
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 4))
    possible = [p for p in path[3:-3] if 1 <= p[0] < h - 1 and 1 <= p[1] < w - 1]
    random.shuffle(possible)
    used = set()
    for col, (r, c) in zip(colors, possible):
        if any(abs(r - rr) + abs(c - cc) < 3 for rr, cc in used):
            continue
        if random.random() < 0.5 and grid[r - 1, c] == BG and grid[r + 1, c] == BG:
            grid[r - 1, c] = grid[r + 1, c] = col
            used.add((r, c))
        elif grid[r, c - 1] == BG and grid[r, c + 1] == BG:
            grid[r, c - 1] = grid[r, c + 1] = col
            used.add((r, c))
    return grid
