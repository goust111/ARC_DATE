import random
from collections import deque

import numpy as np


MARKER_COLORS = (1, 2, 4, 5, 6, 7, 8)


def _neighbors4(r, c, h, w):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _black_outside(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if grid[r, c] == 0 and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if grid[r, c] == 0 and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if grid[nr, nc] == 0 and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Summary:
    Flood-fill from the border to find which black cells are outside, then treat the remaining black components as enclosed regions.

    Input Model:
    Green rectangular rings may be closed or open; only closed rings have a rectangular black interior and four diagonal marker pixels placed one step outside the ring corners.

    Algorithm:
    For each enclosed black component, use its bounding box to locate the four marker colors, split the component into four equal quadrants, recolor those quadrants, and then erase the markers.

    Output:
    Return the transformed grid with closed interiors recolored, green boundaries preserved, markers removed, and open distractors untouched.
    """
    grid = np.asarray(grid, dtype=int)
    out = grid.copy()
    outside = _black_outside(grid)
    enclosed = (grid == 0) & (~outside)
    for cells in _components(enclosed):
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if len(cells) != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        marker_positions = [
            (r0 - 2, c0 - 2),
            (r0 - 2, c1 + 2),
            (r1 + 2, c0 - 2),
            (r1 + 2, c1 + 2),
        ]
        if any(not (0 <= mr < out.shape[0] and 0 <= mc < out.shape[1]) for mr, mc in marker_positions):
            continue
        colors = [grid[mr, mc] for mr, mc in marker_positions]
        if any(color not in MARKER_COLORS for color in colors):
            continue
        rm = r0 + (r1 - r0 + 1) // 2
        cm = c0 + (c1 - c0 + 1) // 2
        out[r0:rm, c0:cm] = colors[0]
        out[r0:rm, cm : c1 + 1] = colors[1]
        out[rm : r1 + 1, c0:cm] = colors[2]
        out[rm : r1 + 1, cm : c1 + 1] = colors[3]
        for mr, mc in marker_positions:
            out[mr, mc] = 0
    return out


def _draw_open_shape(grid, rng, occupied):
    h, w = grid.shape
    for _ in range(200):
        oh = rng.randint(4, 8)
        ow = rng.randint(4, 8)
        r0 = rng.randint(1, h - oh - 2)
        c0 = rng.randint(1, w - ow - 2)
        cells = {(r0, c) for c in range(c0, c0 + ow)}
        cells |= {(r0 + oh - 1, c) for c in range(c0, c0 + ow)}
        cells |= {(r, c0) for r in range(r0, r0 + oh)}
        cells |= {(r, c0 + ow - 1) for r in range(r0, r0 + oh)}
        gap_side = rng.choice(("top", "bottom", "left", "right"))
        if gap_side == "top":
            gap = (r0, rng.randint(c0 + 1, c0 + ow - 2))
        elif gap_side == "bottom":
            gap = (r0 + oh - 1, rng.randint(c0 + 1, c0 + ow - 2))
        elif gap_side == "left":
            gap = (rng.randint(r0 + 1, r0 + oh - 2), c0)
        else:
            gap = (rng.randint(r0 + 1, r0 + oh - 2), c0 + ow - 1)
        cells.remove(gap)
        halo = set()
        for r, c in cells:
            for nr in range(r - 1, r + 2):
                for nc in range(c - 1, c + 2):
                    halo.add((nr, nc))
        if any((rr, cc) in occupied for rr, cc in halo):
            continue
        for r, c in cells:
            grid[r, c] = 3
        occupied |= halo
        return True
    return False


def _generate_case(rng: random.Random):
    h = rng.randint(18, 27)
    w = rng.randint(18, 27)
    grid = np.zeros((h, w), dtype=int)
    expected = np.zeros((h, w), dtype=int)
    occupied = set()
    closed_count = rng.randint(2, 3)
    open_count = rng.randint(0, 2)
    for _ in range(closed_count):
        for _ in range(400):
            ih = rng.choice((4, 6, 8))
            iw = rng.choice((4, 6, 8))
            r0 = rng.randint(2, h - ih - 3)
            c0 = rng.randint(2, w - iw - 3)
            top = r0 - 1
            left = c0 - 1
            bottom = r0 + ih
            right = c0 + iw
            ring = set()
            for c in range(left, right + 1):
                ring.add((top, c))
                ring.add((bottom, c))
            for r in range(top, bottom + 1):
                ring.add((r, left))
                ring.add((r, right))
            markers = [
                (top - 1, left - 1),
                (top - 1, right + 1),
                (bottom + 1, left - 1),
                (bottom + 1, right + 1),
            ]
            footprint = set(ring) | set(markers)
            halo = set()
            for r, c in footprint:
                for nr in range(r - 1, r + 2):
                    for nc in range(c - 1, c + 2):
                        halo.add((nr, nc))
            if any(not (0 <= rr < h and 0 <= cc < w) for rr, cc in footprint):
                continue
            if any((rr, cc) in occupied for rr, cc in halo):
                continue
            colors = rng.sample(MARKER_COLORS, 4)
            for r, c in ring:
                grid[r, c] = 3
                expected[r, c] = 3
            for (mr, mc), color in zip(markers, colors):
                grid[mr, mc] = color
            rm = r0 + ih // 2
            cm = c0 + iw // 2
            expected[r0:rm, c0:cm] = colors[0]
            expected[r0:rm, cm : c0 + iw] = colors[1]
            expected[rm : r0 + ih, c0:cm] = colors[2]
            expected[rm : r0 + ih, cm : c0 + iw] = colors[3]
            occupied |= halo
            break
    for _ in range(open_count):
        _draw_open_shape(grid, rng, occupied)
        _draw_open_shape(expected, rng, set())  # ignored immediately below
    expected[(expected == 3) & (grid != 3)] = 0
    expected[(grid == 3) & (expected == 0)] = 3
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 3 and expected[r, c] == 0:
                expected[r, c] = 3
    return grid, main(grid)


def generate_input() -> np.ndarray:
    return _generate_case(random.Random(random.randrange(1 << 30)))[0]
