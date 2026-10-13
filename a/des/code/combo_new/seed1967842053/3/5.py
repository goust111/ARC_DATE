import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
AZURE = 8
COLORS = [1, 2, 3, 4, 6, 7, 9]
DIRECTIONS = ["up", "down", "left", "right"]


def _components(mask):
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps = []
    for sr, sc in zip(*np.where(mask)):
        if seen[sr, sc]:
            continue
        q = deque([(int(sr), int(sc))])
        seen[sr, sc] = True
        comp = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(comp):
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    return min(rows), max(rows), min(cols), max(cols)


def _is_rect_frame(comp):
    r0, r1, c0, c1 = _bbox(comp)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    expected = set()
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    return set(comp) == expected


def _longest_horizontal_run(points):
    by_row = {}
    for r, c in points:
        by_row.setdefault(r, []).append(c)
    best = None
    for r, cols in by_row.items():
        cols = sorted(cols)
        start = prev = cols[0]
        for c in cols[1:] + [None]:
            if c is not None and c == prev + 1:
                prev = c
                continue
            length = prev - start + 1
            if best is None or length > best[0]:
                best = (length, r, start, prev)
            start = prev = c
    return best


def _decode_arrow(comp):
    points = set(comp)
    run = _longest_horizontal_run(points)
    if run is None or run[0] < 2:
        return None
    length, arm_r, c0, c1 = run
    arm = {(arm_r, c) for c in range(c0, c1 + 1)}
    extras = points - arm
    if not extras:
        return None

    if all(r < arm_r for r, _ in extras):
        return "up", length
    if all(r > arm_r for r, _ in extras):
        return "down", length

    extra_cols = {c for _, c in extras}
    if len(extra_cols) == 1:
        tick_col = next(iter(extra_cols))
        if tick_col <= c0:
            return "left", length
        if tick_col >= c1:
            return "right", length
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """Roll each framed binary pattern according to its same-colored L arrow.

    Input structure:
    The grid has a grey background, several single-pixel rectangular containers
    with distinct colored borders, and black/azure binary pixels filling each
    container interior.  Near each container is one separated marker of the same
    color: a horizontal arm plus a short perpendicular tick.

    Rule:
    For every border color, find its rectangular container and same-colored
    arrow.  Decode the arrow arm length as the number of pixels in its
    horizontal run.  A tick above or below the arm means roll the container
    interior up or down; a tick spanning the left or right end means roll left
    or right.  The roll is cyclic by the decoded arm length, while borders and
    arrows stay unchanged.

    Key insight:
    Each arrow carries the target color, motion direction, and shift distance,
    so the visible code must be decoded before applying a cyclic translation to
    the matching container's interior pattern.
    """
    out = grid.copy()
    ignored = {BLACK, GREY, AZURE}
    for color in np.unique(grid):
        color = int(color)
        if color in ignored:
            continue
        comps = _components(grid == color)
        frames = [comp for comp in comps if _is_rect_frame(comp)]
        arrows = []
        for comp in comps:
            if comp in frames:
                continue
            decoded = _decode_arrow(comp)
            if decoded is not None:
                arrows.append(decoded)
        if not frames or not arrows:
            continue
        direction, length = arrows[0]
        for frame in frames:
            r0, r1, c0, c1 = _bbox(frame)
            interior = grid[r0 + 1:r1, c0 + 1:c1]
            if direction == "up":
                rolled = np.roll(interior, -length, axis=0)
            elif direction == "down":
                rolled = np.roll(interior, length, axis=0)
            elif direction == "left":
                rolled = np.roll(interior, -length, axis=1)
            else:
                rolled = np.roll(interior, length, axis=1)
            out[r0 + 1:r1, c0 + 1:c1] = rolled
    return out


def _arrow_size(direction, length):
    if direction in ("left", "right"):
        return 3, length
    return 2, length


def _arrow_cells(top, left, direction, length, rng):
    cells = []
    if direction == "up":
        arm_r = top + 1
        tick_c = left + (0 if int(rng.integers(0, 2)) == 0 else length - 1)
        cells.append((top, tick_c))
    elif direction == "down":
        arm_r = top
        tick_c = left + (0 if int(rng.integers(0, 2)) == 0 else length - 1)
        cells.append((top + 1, tick_c))
    elif direction == "left":
        arm_r = top + 1
        cells.extend([(top, left), (top + 2, left)])
    else:
        arm_r = top + 1
        end_c = left + length - 1
        cells.extend([(top, end_c), (top + 2, end_c)])
    cells.extend((arm_r, c) for c in range(left, left + length))
    return cells


def _paint_container(grid, top, left, ih, iw, color, rng):
    bottom = top + ih + 1
    right = left + iw + 1
    grid[top:bottom + 1, left] = color
    grid[top:bottom + 1, right] = color
    grid[top, left:right + 1] = color
    grid[bottom, left:right + 1] = color

    pattern = np.where(rng.integers(0, 2, size=(ih, iw)) == 0, BLACK, AZURE)
    if np.all(pattern == BLACK):
        pattern[int(rng.integers(0, ih)), int(rng.integers(0, iw))] = AZURE
    elif np.all(pattern == AZURE):
        pattern[int(rng.integers(0, ih)), int(rng.integers(0, iw))] = BLACK
    grid[top + 1:bottom, left + 1:right] = pattern


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(200):
        h = int(rng.integers(20, 31))
        w = int(rng.integers(20, 31))
        grid = np.full((h, w), GREY, dtype=int)
        reserved = np.zeros((h, w), dtype=bool)
        count = int(rng.integers(1, min(4, len(COLORS)) + 1))
        colors = random.sample(COLORS, count)
        offset = int(rng.integers(0, 4))
        directions = [DIRECTIONS[(offset + i) % 4] for i in range(count)]
        random.shuffle(directions)

        def can_place(r, c, hh, ww):
            if r < 0 or c < 0 or r + hh > h or c + ww > w:
                return False
            rr0 = max(0, r - 1)
            rr1 = min(h, r + hh + 1)
            cc0 = max(0, c - 1)
            cc1 = min(w, c + ww + 1)
            return not reserved[rr0:rr1, cc0:cc1].any()

        def reserve(r, c, hh, ww):
            rr0 = max(0, r - 1)
            rr1 = min(h, r + hh + 1)
            cc0 = max(0, c - 1)
            cc1 = min(w, c + ww + 1)
            reserved[rr0:rr1, cc0:cc1] = True

        success = True
        for color, direction in zip(colors, directions):
            placed = False
            for _attempt in range(600):
                ih = int(rng.integers(3, 8))
                iw = int(rng.integers(3, 8))
                oh, ow = ih + 2, iw + 2
                length = int(rng.integers(2, 8))
                ah, aw = _arrow_size(direction, length)
                cr = int(rng.integers(1, h - oh))
                cc = int(rng.integers(1, w - ow))
                if not can_place(cr, cc, oh, ow):
                    continue

                gap = int(rng.integers(1, 3))
                side = random.choice(("above", "below", "left", "right"))
                if side == "above":
                    ar = cr - gap - ah
                    lo = max(0, cc - aw + 1)
                    hi = min(w - aw, cc + ow - 1)
                    if lo > hi:
                        continue
                    ac = int(rng.integers(lo, hi + 1))
                elif side == "below":
                    ar = cr + oh + gap
                    lo = max(0, cc - aw + 1)
                    hi = min(w - aw, cc + ow - 1)
                    if lo > hi:
                        continue
                    ac = int(rng.integers(lo, hi + 1))
                elif side == "left":
                    ac = cc - gap - aw
                    lo = max(0, cr - ah + 1)
                    hi = min(h - ah, cr + oh - 1)
                    if lo > hi:
                        continue
                    ar = int(rng.integers(lo, hi + 1))
                else:
                    ac = cc + ow + gap
                    lo = max(0, cr - ah + 1)
                    hi = min(h - ah, cr + oh - 1)
                    if lo > hi:
                        continue
                    ar = int(rng.integers(lo, hi + 1))

                if not can_place(ar, ac, ah, aw):
                    continue
                _paint_container(grid, cr, cc, ih, iw, color, rng)
                for r, c in _arrow_cells(ar, ac, direction, length, rng):
                    grid[r, c] = color
                reserve(cr, cc, oh, ow)
                reserve(ar, ac, ah, aw)
                placed = True
                break
            if not placed:
                success = False
                break
        if success:
            return grid

    grid = np.full((18, 18), GREY, dtype=int)
    color = random.choice(COLORS)
    direction = random.choice(DIRECTIONS)
    _paint_container(grid, 6, 6, 5, 5, color, rng)
    for r, c in _arrow_cells(2, 6, direction, 4, rng):
        grid[r, c] = color
    return grid
