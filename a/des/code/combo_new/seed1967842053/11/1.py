import random
from collections import deque

import numpy as np


BACKGROUND = 0
WALL = 5
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _region_labels(grid):
    rows, cols = grid.shape
    labels = np.full((rows, cols), -1, dtype=int)
    region_seed = []
    next_label = 0

    for start_r in range(rows):
        for start_c in range(cols):
            if grid[start_r, start_c] == WALL or labels[start_r, start_c] != -1:
                continue

            q = deque([(start_r, start_c)])
            labels[start_r, start_c] = next_label
            seed_colors = set()

            while q:
                r, c = q.popleft()
                color = int(grid[r, c])
                if color not in (BACKGROUND, WALL):
                    seed_colors.add(color)

                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < rows
                        and 0 <= nc < cols
                        and grid[nr, nc] != WALL
                        and labels[nr, nc] == -1
                    ):
                        labels[nr, nc] = next_label
                        q.append((nr, nc))

            region_seed.append(next(iter(seed_colors)) if len(seed_colors) == 1 else None)
            next_label += 1

    return labels, region_seed


def _side_seed_set(labels, region_seed, cells, dr, dc):
    rows, cols = labels.shape
    colors = set()
    for r, c in cells:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            label = labels[nr, nc]
            if label >= 0:
                color = region_seed[int(label)]
                if color is not None:
                    colors.add(color)
    return colors


def _is_junction(wall_mask, r, c):
    rows, cols = wall_mask.shape
    left = c > 0 and wall_mask[r, c - 1]
    right = c + 1 < cols and wall_mask[r, c + 1]
    up = r > 0 and wall_mask[r - 1, c]
    down = r + 1 < rows and wall_mask[r + 1, c]
    return (left or right) and (up or down)


def _has_horizontal_neighbor(wall_mask, r, c):
    cols = wall_mask.shape[1]
    return (c > 0 and wall_mask[r, c - 1]) or (c + 1 < cols and wall_mask[r, c + 1])


def _has_vertical_neighbor(wall_mask, r, c):
    rows = wall_mask.shape[0]
    return (r > 0 and wall_mask[r - 1, c]) or (r + 1 < rows and wall_mask[r + 1, c])


def _paint_segment(out, labels, region_seed, cells, side_a, side_b):
    seeds_a = _side_seed_set(labels, region_seed, cells, side_a[0], side_a[1])
    seeds_b = _side_seed_set(labels, region_seed, cells, side_b[0], side_b[1])

    if len(seeds_a) == 1 and len(seeds_b) == 0:
        color = next(iter(seeds_a))
    elif len(seeds_a) == 0 and len(seeds_b) == 1:
        color = next(iter(seeds_b))
    else:
        return

    for r, c in cells:
        out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Color neutral wall segments from the one seeded region that borders them.

    Input structure:
    A black background is divided by one-pixel-wide grey horizontal and vertical
    walls into closed regions. Some regions contain exactly one non-black,
    non-grey seed pixel, and seed colors are distinct; other regions are empty.

    Rule:
    Background pixels and seed pixels stay unchanged. Each maximal straight grey
    wall segment is checked on its two sides. If exactly one side touches seeded
    region(s), the whole segment becomes that seed color. If both sides have
    seeds, or neither side has a seed, the segment stays grey.

    Key insight:
    Wall color is decided by regional ownership across the wall, not by nearest
    distance: a color spreads only across an uncontested wall side.
    """
    out = np.array(grid, copy=True)
    wall_mask = grid == WALL
    labels, region_seed = _region_labels(grid)
    rows, cols = grid.shape

    junction = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        for c in range(cols):
            if wall_mask[r, c]:
                junction[r, c] = _is_junction(wall_mask, r, c)

    visited_h = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        c = 0
        while c < cols:
            if (
                not wall_mask[r, c]
                or junction[r, c]
                or visited_h[r, c]
                or not _has_horizontal_neighbor(wall_mask, r, c)
            ):
                c += 1
                continue

            cells = []
            cc = c
            while cc < cols and wall_mask[r, cc] and not junction[r, cc]:
                visited_h[r, cc] = True
                cells.append((r, cc))
                cc += 1

            _paint_segment(out, labels, region_seed, cells, (-1, 0), (1, 0))
            c = cc

    visited_v = np.zeros((rows, cols), dtype=bool)
    for c in range(cols):
        r = 0
        while r < rows:
            if (
                not wall_mask[r, c]
                or junction[r, c]
                or visited_v[r, c]
                or not _has_vertical_neighbor(wall_mask, r, c)
            ):
                r += 1
                continue

            cells = []
            rr = r
            while rr < rows and wall_mask[rr, c] and not junction[rr, c]:
                visited_v[rr, c] = True
                cells.append((rr, c))
                rr += 1

            _paint_segment(out, labels, region_seed, cells, (0, -1), (0, 1))
            r = rr

    return out


def _wall_positions(lengths):
    positions = [0]
    cur = 0
    for length in lengths:
        cur += length + 1
        positions.append(cur)
    return positions


def _branch_types(seed_mask):
    cell_rows, cell_cols = seed_mask.shape
    seen = set()
    for r in range(cell_rows):
        for c in range(cell_cols - 1):
            a = bool(seed_mask[r, c])
            b = bool(seed_mask[r, c + 1])
            if a and b:
                seen.add("both")
            elif a or b:
                seen.add("one")
            else:
                seen.add("none")
    for r in range(cell_rows - 1):
        for c in range(cell_cols):
            a = bool(seed_mask[r, c])
            b = bool(seed_mask[r + 1, c])
            if a and b:
                seen.add("both")
            elif a or b:
                seen.add("one")
            else:
                seen.add("none")
    return seen


def _random_seed_mask(cell_rows, cell_cols):
    for _ in range(200):
        mask = np.array(
            [[random.random() < 0.5 for _ in range(cell_cols)] for _ in range(cell_rows)],
            dtype=bool,
        )
        count = int(mask.sum())
        if 1 <= count <= len(SEED_COLORS) and _branch_types(mask) == {"both", "one", "none"}:
            return mask

    mask = np.zeros((cell_rows, cell_cols), dtype=bool)
    mask[0, :] = True
    if cell_cols > len(SEED_COLORS):
        mask[0, len(SEED_COLORS) :] = False
    return mask


def generate_input() -> np.ndarray:
    cell_rows = random.randint(2, 4)
    cell_cols = random.randint(2, 4)
    room_heights = [random.randint(3, 6) for _ in range(cell_rows)]
    room_widths = [random.randint(3, 6) for _ in range(cell_cols)]

    height = sum(room_heights) + cell_rows + 1
    width = sum(room_widths) + cell_cols + 1
    grid = np.full((height, width), BACKGROUND, dtype=int)

    wall_rows = _wall_positions(room_heights)
    wall_cols = _wall_positions(room_widths)
    grid[wall_rows, :] = WALL
    grid[:, wall_cols] = WALL

    seed_mask = _random_seed_mask(cell_rows, cell_cols)
    colors = SEED_COLORS[:]
    random.shuffle(colors)
    color_index = 0

    for r in range(cell_rows):
        r0 = wall_rows[r] + 1
        r1 = wall_rows[r + 1]
        for c in range(cell_cols):
            if not seed_mask[r, c]:
                continue
            c0 = wall_cols[c] + 1
            c1 = wall_cols[c + 1]
            seed_r = random.randint(r0, r1 - 1)
            seed_c = random.randint(c0, c1 - 1)
            grid[seed_r, seed_c] = colors[color_index]
            color_index += 1

    return grid
