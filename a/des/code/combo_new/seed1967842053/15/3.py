import random
from collections import deque

import numpy as np


BACKGROUND = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _most_common_color(grid):
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _palette_patterns(grid, grey, background):
    h, w = grid.shape
    if h < 2 or w < 2:
        return {}

    rows = grid[h - 2 : h]
    patterns = {}
    c = 0
    while c < w:
        if rows[0, c] == grey and rows[1, c] == grey:
            c += 1
            continue

        start = c
        while c < w and not (rows[0, c] == grey and rows[1, c] == grey):
            c += 1

        s = start
        while s + 1 < c:
            pattern = rows[:, s : s + 2]
            marker = int(pattern[0, 0])
            if marker not in (grey, background):
                patterns[marker] = pattern.copy()
            s += 2

    return patterns


def _closed_rectangles(grid, grey):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    rectangles = []

    for sr in range(max(0, h - 2)):
        for sc in range(w):
            if seen[sr, sc] or grid[sr, sc] != grey:
                continue

            q = deque([(sr, sc)])
            seen[sr, sc] = True
            component = []
            while q:
                r, c = q.popleft()
                component.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < h - 2
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and grid[nr, nc] == grey
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rs = [r for r, _ in component]
            cs = [c for _, c in component]
            top, bottom = min(rs), max(rs)
            left, right = min(cs), max(cs)
            if bottom <= top + 1 or right <= left + 1:
                continue

            sub = grid[top : bottom + 1, left : right + 1]
            border = np.zeros(sub.shape, dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True

            if not np.all(sub[border] == grey):
                continue
            if any(not (r in (top, bottom) or c in (left, right)) for r, c in component):
                continue

            rectangles.append((top, left, bottom, right))

    return rectangles


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each grey rectangular enclosure by tiling the palette pattern keyed by its marker.

    Input structure:
    The grid has a most common background color, several one-pixel-wide grey rectangular
    closed borders above the bottom edge, and the bottom two rows form a palette strip.
    Each rectangle interior contains exactly one non-background marker pixel. The palette
    strip is split by grey vertical separators into 2x2 cells whose upper-left pixel is a
    marker color and whose full 2x2 contents are the pattern for that marker.

    Rule:
    For every grey closed rectangle, read the unique marker color inside it, find the
    matching 2x2 palette cell with the same upper-left marker color, and replace the whole
    rectangle interior with that 2x2 pattern tiled from the interior's upper-left corner.
    The grey rectangle borders and the bottom palette strip are left unchanged.

    Key insight:
    The interior marker is an index into the edge palette, and the selected miniature
    pattern must be repeated periodically rather than stretched or copied once.
    """
    background = _most_common_color(grid)
    grey = GREY if np.any(grid == GREY) else None
    if grey is None:
        return grid.copy()

    patterns = _palette_patterns(grid, grey, background)
    out = grid.copy()

    for top, left, bottom, right in _closed_rectangles(grid, grey):
        interior = grid[top + 1 : bottom, left + 1 : right]
        marker_mask = (interior != background) & (interior != grey)
        markers = np.unique(interior[marker_mask])
        if len(markers) != 1:
            continue

        marker = int(markers[0])
        if marker not in patterns:
            continue

        pattern = patterns[marker]
        ih, iw = interior.shape
        rr = np.arange(ih)[:, None] % 2
        cc = np.arange(iw)[None, :] % 2
        out[top + 1 : bottom, left + 1 : right] = pattern[rr, cc]

    return out


def _random_pattern(marker, rng, markers):
    accent_pool = [c for c in COLORS if c != marker and c not in markers]
    if not accent_pool:
        accent_pool = [c for c in COLORS if c != marker]
    accent = rng.choice(accent_pool)

    accent_masks = [
        ((0, 1),),
        ((1, 0),),
        ((1, 1),),
        ((0, 1), (1, 0)),
        ((0, 1), (1, 1)),
        ((1, 0), (1, 1)),
        ((0, 1), (1, 0), (1, 1)),
    ]
    pattern = np.full((2, 2), marker, dtype=int)
    for r, c in rng.choice(accent_masks):
        pattern[r, c] = accent
    return pattern


def _try_place_rectangle(grid, blocked, marker, rng):
    h, w = grid.shape
    max_ih = min(7, h - 6)
    max_iw = min(8, w - 4)
    if max_ih < 2 or max_iw < 2:
        return False

    for _ in range(250):
        ih = rng.randint(2, max_ih)
        iw = rng.randint(2, max_iw)
        oh, ow = ih + 2, iw + 2
        max_top = h - oh - 3
        max_left = w - ow - 1
        if max_top < 1 or max_left < 1:
            continue

        top = rng.randint(1, max_top)
        left = rng.randint(1, max_left)
        bottom = top + oh - 1
        right = left + ow - 1

        r0, r1 = max(0, top - 1), min(h - 2, bottom + 2)
        c0, c1 = max(0, left - 1), min(w, right + 2)
        if blocked[r0:r1, c0:c1].any():
            continue

        grid[top, left : right + 1] = GREY
        grid[bottom, left : right + 1] = GREY
        grid[top : bottom + 1, left] = GREY
        grid[top : bottom + 1, right] = GREY

        mr = rng.randint(top + 1, bottom - 1)
        mc = rng.randint(left + 1, right - 1)
        grid[mr, mc] = marker
        blocked[r0:r1, c0:c1] = True
        return True

    return False


def generate_input() -> np.ndarray:
    rng = random.Random()

    while True:
        count = rng.randint(2, 5)
        min_w = max(12, 3 * count + 1)
        h = rng.randint(14, 30)
        w = rng.randint(min_w, 30)

        grid = np.full((h, w), BACKGROUND, dtype=int)
        grid[-2:, :] = GREY

        markers = rng.sample(COLORS, count)
        patterns = {marker: _random_pattern(marker, rng, markers) for marker in markers}

        strip_width = 3 * count + 1
        offset = rng.randint(0, w - strip_width)
        palette_order = markers[:]
        rng.shuffle(palette_order)
        for i, marker in enumerate(palette_order):
            start = offset + 1 + 3 * i
            grid[-2:, start : start + 2] = patterns[marker]

        blocked = np.zeros((h, w), dtype=bool)
        placed = True
        region_order = markers[:]
        rng.shuffle(region_order)
        for marker in region_order:
            if not _try_place_rectangle(grid, blocked, marker, rng):
                placed = False
                break

        if not placed:
            continue
        if _most_common_color(grid) != BACKGROUND:
            continue

        return grid
