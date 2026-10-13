import numpy as np
from collections import deque

RNG = np.random.default_rng(303)


def _components(mask):
    h, w = mask.shape
    vis = np.zeros(mask.shape, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or vis[r, c]:
                continue
            q = [(r, c)]
            vis[r, c] = True
            comp = []
            while q:
                x, y = q.pop()
                comp.append((x, y))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = x + dr, y + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not vis[nr, nc]:
                        vis[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points):
    rs = [p[0] for p in points]
    cs = [p[1] for p in points]
    return min(rs), max(rs), min(cs), max(cs)


def _is_full_rect(points):
    rs = [p[0] for p in points]
    cs = [p[1] for p in points]
    top, bottom = min(rs), max(rs)
    left, right = min(cs), max(cs)
    if (bottom - top + 1) * (right - left + 1) != len(points):
        return None
    return top, bottom, left, right


def _match_translate(anchor_set, move_set, area, h, w):
    for dr in range(-h, h + 1):
        for dc in range(-w, w + 1):
            moved = set()
            ok = True
            for r, c in move_set:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    ok = False
                    break
                moved.add((nr, nc))
            if not ok:
                continue
            union = anchor_set | moved
            rect = _is_full_rect(union)
            if rect is None:
                continue
            top, bottom, left, right = rect
            if len(union) == area and (bottom - top + 1) * (right - left + 1) == area:
                return rect, moved
    return None, None


def main(grid: np.ndarray) -> np.ndarray:
    """Task 3: paired fragments become full rectangles by move+erase.

Input:
    3 colors, each two fragments of same color.
Output:
    Rectangles from pair-complement moves, sorted by area and overdrawn.

Reasoning:
    Detect move fragment touching top edge, compute unique translation that makes
    two fragments fill one solid rectangle, then draw rectangles in descending area.

Complexity:
    O(H*W*max_shift^2).
    """

    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    rects = []

    for color in (1, 2, 3):
        comps = _components(g == color)
        if len(comps) != 2:
            continue

        comp_sets = [set(tuple(p) for p in c) for c in comps]
        top_touch = []
        nontop = []
        for i, s in enumerate(comp_sets):
            if any(r == 0 for r, _ in s):
                top_touch.append((i, s))
            else:
                nontop.append((i, s))
        if len(top_touch) != 1 or len(nontop) != 1:
            continue

        move_set = set(top_touch[0][1])
        anchor_set = set(nontop[0][1])

        area = len(move_set) + len(anchor_set)
        rect, _ = _match_translate(anchor_set, move_set, area, h, w)
        if rect is None:
            # should not happen in generation
            continue

        top, bottom, left, right = rect
        # erase original move fragment
        for rr, cc in move_set:
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = 0

        rects.append((area, color, (top, bottom, left, right)))

    out[:] = 0
    # draw only composed rectangles; background stays black
    for _, color, (top, bottom, left, right) in sorted(rects, key=lambda x: -x[0]):
        out[top:bottom + 1, left:right + 1] = color

    return out


def generate_input() -> np.ndarray:
    h = 16
    w = 16
    g = np.zeros((h, w), dtype=int)

    specs = [
        (1, (3, 4)),
        (2, (4, 5)),
        (3, (4, 6)),
    ]

    occupied = set()
    for col, (rh, rw) in specs:
        area = rh * rw
        place_ok = False
        for _ in range(500):
            top = int(RNG.integers(2, h - rh - 1))
            left = int(RNG.integers(0, w - rw + 1))
            bottom = top + rh - 1
            right = left + rw - 1

            cells = {(rr, cc) for rr in range(top, bottom + 1) for cc in range(left, right + 1)}
            if occupied & cells:
                continue

            if any(r in range(top, bottom + 1) and c in range(left, right + 1) for r, c in occupied):
                pass

            # ensure move piece touches top border
            # move piece is top row; anchor is the rest
            move_cells = {(top, cc) for cc in range(left, right + 1)}
            anchor_cells = cells - move_cells
            if not anchor_cells or any((r == 0) for r, c in anchor_cells):
                continue

            occupied |= cells
            for rr, cc in move_cells:
                g[rr, cc] = col
            for rr, cc in anchor_cells:
                g[rr, cc] = col
            place_ok = True
            break

        if not place_ok:
            # safe deterministic fallback
            top = 2 + specs.index((col, (rh, rw))) * 4
            left = 1
            for rr in range(top, top + rh):
                for cc in range(left, left + rw):
                    if rr == top:
                        g[rr, cc] = col
                    else:
                        g[rr, cc] = col

    return g
