import random
from collections import deque

import numpy as np

BG = 0
GRAY = 5
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray):
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
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), max(rs), min(cs), max(cs)


def _norm_key(cells):
    r0, r1, c0, c1 = _bbox(cells)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    mask = np.zeros((h, w), dtype=np.int8)
    for r, c in cells:
        mask[r - r0, c - c0] = 1
    return tuple(mask.flatten())


def _enclosed_zero_components(template_mask: np.ndarray):
    h, w = template_mask.shape
    zero = template_mask == BG
    outside = np.zeros((h, w), dtype=bool)
    q = deque()

    for r in range(h):
        for c in (0, w - 1):
            if zero[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if zero[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and zero[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    comps = []
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not zero[r, c] or outside[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and zero[nr, nc] and not outside[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _template_component(g: np.ndarray):
    gray = g == GRAY
    h, w = g.shape
    comps = _components(gray)
    if not comps:
        return None

    chosen = None
    for comp in comps:
        r0, r1, c0, c1 = _bbox(comp)
        if r0 < h // 2:  # template in lower half
            continue
        if r1 - r0 < 2 or c1 - c0 < 4:
            continue
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if chosen is None or len(comp) > len(chosen[0]):
            chosen = (comp, (r0, r1, c0, c1), area)

    if chosen is None:
        # fallback to largest gray component
        comp = max(comps, key=len)
        return _bbox(comp)
    return chosen[1]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, w = g.shape

    frame = _template_component(g)
    if frame is None:
        return g.copy()

    r0, r1, c0, c1 = frame
    template = g[r0 : r1 + 1, c0 : c1 + 1]

    # Identify template holes.
    inside = template[1:-1, 1:-1]
    hole_cells = _enclosed_zero_components(inside)
    hole_map = {}
    for cells in hole_cells:
        if not cells:
            continue
        # Convert back to absolute
        abs_cells = {(r + r0 + 1, c + c0 + 1) for r, c in cells}
        key = _norm_key(cells)
        hole_map[key] = abs_cells

    # Identify movable objects above the template.
    movable_mask = (g != BG) & (g != GRAY)
    movable = []
    for comp in _components(movable_mask):
        if not comp:
            continue
        if any(r >= r0 for r, _ in comp):
            continue
        movable.append(comp)

    # Drop each object onto template top edge - 1 row.
    target_row = r0 - 1
    dropped = {}
    out = g.copy()

    # Erase objects first.
    for comp in movable:
        for r, c in comp:
            out[r, c] = BG

    for comp in movable:
        max_r = max(r for r, _ in comp)
        shift = max(0, target_row - max_r)
        color = int(g[comp[0]])
        translated = {(r + shift, c) for r, c in comp}
        key = _norm_key(comp)

        if key in hole_map:
            for hr, hc in hole_map[key]:
                out[hr, hc] = color
        # Keep object removed; objects are not kept at landing position.

    return out


def _place_shape(grid: np.ndarray, top: int, left: int, shape, color: int):
    h, w = grid.shape
    cells = []
    for dr, dc in shape:
        r = top + dr
        c = left + dc
        if 0 <= r < h and 0 <= c < w:
            grid[r, c] = color
            cells.append((r, c))
    return cells


def generate_input() -> np.ndarray:
    h, w = 14, 14
    g = np.zeros((h, w), dtype=int)

    # Template block near bottom.
    t_top = random.randint(6, 7)
    t_h = 6
    t_bottom = min(h - 2, t_top + t_h - 1)
    t_left = 1
    t_right = w - 2

    g[t_top : t_bottom + 1, t_left : t_right + 1] = GRAY

    hole_shapes = [
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 1), (1, 0), (1, 1), (2, 0)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ]

    # Make 2-3 template holes.
    hole_count = random.randint(2, 3)
    random.shuffle(hole_shapes)
    selected = hole_shapes[:hole_count]

    inner_h = t_bottom - t_top - 1
    inner_w = t_right - t_left - 1
    used = set()

    for shape in selected:
        sh_h = max(r for r, _ in shape) + 1
        sh_w = max(c for _, c in shape) + 1
        if inner_h <= sh_h or inner_w <= sh_w:
            continue

        placed = False
        for _ in range(220):
            min_top = t_top + 1
            max_top = t_bottom - sh_h
            min_left = t_left + 1
            max_left = t_right - sh_w
            if min_top > max_top or min_left > max_left:
                break
            top = random.randint(min_top, max_top)
            left = random.randint(min_left, max_left)
            cells = [(top + dr, left + dc) for dr, dc in shape]
            if any(c in used for c in cells):
                continue
            if any(g[r, c] != BG for r, c in cells):
                continue
            for cell in cells:
                g[cell[0], cell[1]] = BG
                used.add(cell)
            placed = True
            break

        if not placed:
            # deterministic fallback in template interior.
            min_top = t_top + 1
            max_top = t_bottom - sh_h
            min_left = t_left + 1
            max_left = t_right - sh_w
            if min_top <= max_top and min_left <= max_left:
                for top in range(min_top, max_top + 1):
                    for left in range(min_left, max_left + 1):
                        cells = [(top + dr, left + dc) for dr, dc in shape]
                        if any(c in used for c in cells):
                            continue
                        for cell in cells:
                            g[cell[0], cell[1]] = BG
                            used.add(cell)
                        placed = True
                        break
                    if placed:
                        break
                if placed:
                    break
            if not placed:
                top = t_top + 1
                left = t_left + 1
                for cell in ((top + dr, left + dc) for dr, dc in shape):
                    g[cell[0], cell[1]] = BG
                    used.add((int(cell[0]), int(cell[1])))

    # Place matching colored objects above template.
    colors = [1, 2, 3]
    random.shuffle(colors)
    random.shuffle(selected)

    used_cols = []

    for i, shape in enumerate(selected):
        sh_h = max(r for r, _ in shape) + 1
        color = colors[i % len(colors)]
        sh_w = max(c for _, c in shape) + 1
        placed = False

        top_max = max(1, t_top - sh_h - 2)

        for _ in range(300):
            top = random.randint(1, top_max)
            left = random.randint(0, w - sh_w - 1)

            cols = (left, left + sh_w - 1)
            if any(not (left > cr[1] or cols[1] < cr[0]) for cr in used_cols):
                # require disjoint column ranges between objects
                continue

            cells = [(top + dr, left + dc) for dr, dc in shape]
            if any(g[r, c] != BG for r, c in cells):
                continue

            for r, c in cells:
                g[r, c] = color
                used.add((r, c))
            used_cols.append(cols)
            placed = True
            break

        if not placed:
            # deterministic fallback in the upper-left region.
            top = 1 + i
            left = 1 + i * 3
            if left + sh_w < w:
                for r, c in ((top + dr, left + dc) for dr, dc in shape):
                    if 0 <= r < t_top and 0 <= c < w:
                        g[r, c] = color

    return g
