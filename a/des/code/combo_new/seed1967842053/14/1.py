"""ARC-style task generator and solver for seed1967842053/14/1."""

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
GRAY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _find_separator(grid):
    full_gray = np.all(grid == GRAY, axis=1)
    rows = np.flatnonzero(full_gray)
    if len(rows) == 0:
        return grid.shape[0]
    return int(rows[-1])


def _legend_patterns(grid, sep):
    h, w = grid.shape
    if sep >= h - 1:
        return {}

    legend = grid[sep + 1 :]
    dividers = [c for c in range(w) if np.all(legend[:, c] == GRAY)]
    if 0 not in dividers:
        dividers = [-1] + dividers
    if w - 1 not in dividers:
        dividers = dividers + [w]
    dividers = sorted(set(dividers))

    patterns = {}
    top = sep + 1
    for left_wall, right_wall in zip(dividers, dividers[1:]):
        cols = range(left_wall + 1, right_wall)
        if len(cols) < 2:
            continue
        key_col = left_wall + 1
        key = int(grid[top, key_col])
        if key in (BLACK, GRAY):
            continue

        buckets = {(0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []}
        for r in range(top, h):
            for c in cols:
                if r == top and c == key_col:
                    continue
                color = int(grid[r, c])
                if color == GRAY:
                    continue
                buckets[(r & 1, c & 1)].append(color)

        pattern = {}
        all_seen = [color for values in buckets.values() for color in values]
        fallback = Counter(all_seen).most_common(1)[0][0] if all_seen else BLACK
        for parity, values in buckets.items():
            pattern[parity] = Counter(values).most_common(1)[0][0] if values else fallback
        patterns[key] = pattern
    return patterns


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each closed region with the legend pattern selected by its seed color.

    Input structure:
    The grid has a black background, gray wall pixels forming a connected boundary
    network, and one non-black non-gray seed pixel inside each closed region.  The
    bottom legend is separated by a full gray horizontal line and is split by gray
    vertical dividers into equal cells; each legend cell has a seed-key color in its
    upper-left interior pixel and a repeated two-color 2x2 pattern sample elsewhere.

    Rule:
    Locate the bottom gray separator and read each legend cell to map a seed color
    to its 2x2 pattern by absolute row and column parity.  Above the separator,
    flood-fill every non-gray connected component.  If a component contains a seed,
    replace all cells in that component, including the seed, with the selected
    pattern color for each cell's global coordinate parity.  Gray boundaries and
    the legend area are copied unchanged.

    Key insight:
    The seed pixel is only a lookup key; the actual fill comes from the matching
    legend cell, and the pattern phase is shared globally rather than restarted for
    each enclosed region.
    """
    grid = np.asarray(grid)
    out = grid.copy()
    sep = _find_separator(grid)
    patterns = _legend_patterns(grid, sep)

    h, w = grid.shape
    limit_r = min(sep, h)
    seen = np.zeros((limit_r, w), dtype=bool)

    for sr in range(limit_r):
        for sc in range(w):
            if seen[sr, sc] or grid[sr, sc] == GRAY:
                continue

            q = deque([(sr, sc)])
            seen[sr, sc] = True
            cells = []
            seeds = []

            while q:
                r, c = q.popleft()
                cells.append((r, c))
                color = int(grid[r, c])
                if color not in (BLACK, GRAY):
                    seeds.append(color)
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < limit_r and 0 <= nc < w:
                        if not seen[nr, nc] and grid[nr, nc] != GRAY:
                            seen[nr, nc] = True
                            q.append((nr, nc))

            if not seeds:
                continue
            seed = seeds[0]
            pattern = patterns.get(seed)
            if pattern is None:
                continue
            for r, c in cells:
                out[r, c] = pattern[(r & 1, c & 1)]

    return out


def _split_total(total, parts, minimum):
    values = [minimum] * parts
    extra = total - minimum * parts
    for _ in range(extra):
        values[random.randrange(parts)] += 1
    random.shuffle(values)
    return values


def _pattern_from_colors(a, b):
    layouts = (
        ((a, b), (b, a)),
        ((a, b), (a, b)),
        ((a, a), (b, b)),
        ((a, b), (b, b)),
        ((a, a), (a, b)),
    )
    return random.choice(layouts)


def generate_input() -> np.ndarray:
    """Generate a random valid input grid for this task."""
    while True:
        room_rows, room_cols = random.choice(
            [(1, 2), (1, 3), (1, 4), (1, 5), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2)]
        )
        room_count = room_rows * room_cols
        slot_w = random.choice([4, 5])
        if room_count * slot_w + 1 > 30:
            continue
        width = room_count * slot_w + 1

        legend_h = random.choice([4, 5, 6])
        height_budget = 30 - legend_h
        min_room_h = 3 if room_rows <= 3 else 2
        min_main_h = room_rows * min_room_h + room_rows + 1
        if min_main_h > height_budget:
            continue
        main_h = random.randint(min_main_h, height_budget)
        total_room_h = main_h - room_rows - 1
        room_heights = _split_total(total_room_h, room_rows, min_room_h)

        min_room_w = 3
        total_room_w = width - room_cols - 1
        if total_room_w < room_cols * min_room_w:
            continue
        room_widths = _split_total(total_room_w, room_cols, min_room_w)
        break

    total_h = main_h + legend_h
    sep = main_h - 1
    grid = np.full((total_h, width), BLACK, dtype=int)

    horizontal = [0]
    pos = 0
    for room_h in room_heights:
        pos += room_h + 1
        horizontal.append(pos)

    vertical = [0]
    pos = 0
    for room_w in room_widths:
        pos += room_w + 1
        vertical.append(pos)

    for r in horizontal:
        grid[r, :] = GRAY
    for c in vertical:
        grid[: main_h, c] = GRAY
    grid[sep, :] = GRAY

    seed_colors = random.sample(COLORS, room_count)
    random.shuffle(seed_colors)
    cell_index = 0
    for rr in range(room_rows):
        r0 = horizontal[rr] + 1
        r1 = horizontal[rr + 1]
        for cc in range(room_cols):
            c0 = vertical[cc] + 1
            c1 = vertical[cc + 1]
            r = random.randrange(r0, r1)
            c = random.randrange(c0, c1)
            grid[r, c] = seed_colors[cell_index]
            cell_index += 1

    for r in range(sep + 1, total_h):
        for c in range(0, width, slot_w):
            grid[r, c] = GRAY
    grid[sep + 1 :, width - 1] = GRAY

    for i, key in enumerate(seed_colors):
        left_wall = i * slot_w
        cols = range(left_wall + 1, (i + 1) * slot_w)
        palette = [color for color in COLORS if color != key]
        a, b = random.sample(palette, 2)
        pattern = _pattern_from_colors(a, b)
        for r in range(sep + 1, total_h):
            for c in cols:
                grid[r, c] = pattern[r & 1][c & 1]
        grid[sep + 1, left_wall + 1] = key

    return grid
