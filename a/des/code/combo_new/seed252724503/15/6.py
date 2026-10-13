import random
from collections import deque

import numpy as np


def _bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(grid: np.ndarray, color: int):
    mask = grid == color
    h, w = mask.shape
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
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each object's adjacent indicator pixel to choose target rectangles, then stamp that object's shape at every matching center.

    Input structure:
    - The grid has a dominant background color, several small monochrome stamp shapes, and several larger solid monochrome rectangles.
    - Each stamp has a single adjacent 1x1 indicator pixel directly above or below it, and the indicator color differs from the stamp color.
    - Rectangle colors may match one or more indicator colors.

    Rule:
    - Pair each indicator pixel with the directly adjacent stamp shape.
    - Remove the original stamp and its indicator from the output.
    - For every rectangle whose color matches that indicator color, overlay the stamp shape centered on the rectangle using the stamp's own color.
    - Leave unmatched parts of the target rectangles unchanged.

    Key insight:
    - The detached 1x1 pixel is a routing label: it picks destination rectangles by color, while the nearby object supplies the geometry to stamp.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg(g)
    out = g.copy()
    non_bg = [int(v) for v in np.unique(g) if v != bg]
    comps_by_color = {color: _components(g, color) for color in non_bg}

    pair_info = []
    used_cells = set()
    for color, comps in comps_by_color.items():
        for comp in comps:
            if len(comp) != 1:
                continue
            r, c = comp[0]
            for dr in (-1, 1):
                nr = r + dr
                if 0 <= nr < g.shape[0] and g[nr, c] != bg and g[nr, c] != color:
                    obj_color = int(g[nr, c])
                    obj = next(comp2 for comp2 in comps_by_color[obj_color] if (nr, c) in comp2)
                    pair_info.append((comp[0], color, obj, obj_color))
                    used_cells.update(obj)
                    used_cells.add(comp[0])
                    break

    targets = []
    for color, comps in comps_by_color.items():
        for comp in comps:
            if any(cell in used_cells for cell in comp):
                continue
            r0, c0, r1, c1 = _bbox(comp)
            if len(comp) == (r1 - r0 + 1) * (c1 - c0 + 1):
                targets.append((color, comp))

    for cell, _, obj, _ in pair_info:
        out[cell] = bg
        for r, c in obj:
            out[r, c] = bg

    for _, target_color, obj, obj_color in pair_info:
        r0, c0, r1, c1 = _bbox(obj)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in obj:
            mask[r - r0, c - c0] = True
        for color, target in targets:
            if color != target_color:
                continue
            tr0, tc0, tr1, tc1 = _bbox(target)
            center_r = (tr0 + tr1) // 2
            center_c = (tc0 + tc1) // 2
            start_r = center_r - mask.shape[0] // 2
            start_c = center_c - mask.shape[1] // 2
            for rr in range(mask.shape[0]):
                for cc in range(mask.shape[1]):
                    if mask[rr, cc] and 0 <= start_r + rr < out.shape[0] and 0 <= start_c + cc < out.shape[1]:
                        out[start_r + rr, start_c + cc] = obj_color
    return out


SHAPES = [
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
]


def generate_input() -> np.ndarray:
    for _ in range(300):
        h, w = random.randint(18, 26), random.randint(18, 26)
        bg = 0
        grid = np.full((h, w), bg, dtype=int)
        occ = np.zeros((h, w), dtype=bool)
        target_colors = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], random.randint(2, 4))
        for color in target_colors:
            for _ in range(random.randint(1, 2)):
                for _ in range(100):
                    hh, ww = random.randint(5, 7), random.randint(5, 7)
                    top = random.randint(1, h - hh - 1)
                    left = random.randint(1, w - ww - 1)
                    if occ[top : top + hh, left : left + ww].any():
                        continue
                    grid[top : top + hh, left : left + ww] = color
                    occ[top : top + hh, left : left + ww] = True
                    break
        stamp_colors = random.sample([c for c in range(10) if c not in (bg,) and c not in target_colors], random.randint(2, 3))
        ok = True
        for sc in stamp_colors:
            tgt = random.choice(target_colors)
            shape = random.choice(SHAPES)
            sh, sw = shape.shape
            placed = False
            for _ in range(100):
                top = random.randint(2, h - sh - 3)
                left = random.randint(1, w - sw - 1)
                indicator_above = random.choice([True, False])
                ir = top - 1 if indicator_above else top + sh
                ic = left + sw // 2
                if ir < 0 or ir >= h:
                    continue
                area = occ[max(0, top - 1) : min(h, top + sh + 1), max(0, left - 1) : min(w, left + sw + 1)]
                if area.any() or occ[ir, ic]:
                    continue
                for r in range(sh):
                    for c in range(sw):
                        if shape[r, c]:
                            grid[top + r, left + c] = sc
                            occ[top + r, left + c] = True
                grid[ir, ic] = tgt
                occ[ir, ic] = True
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return grid
    raise RuntimeError("failed to generate input")
