import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GRAY = 5
TEMPLATE_COLORS = (1, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _extract_templates(grid):
    nonzero_rows = np.where(np.any(grid != BLACK, axis=1))[0]
    if len(nonzero_rows) == 0:
        return [], grid.shape[0]

    bottom = int(nonzero_rows[-1])
    top = max(0, bottom - 1)
    band = grid[top : bottom + 1]
    templates = []
    c = 0
    width = grid.shape[1]
    while c <= width - 2:
        block = band[:, c : c + 2]
        if block.shape == (2, 2) and np.all(block != BLACK):
            templates.append(block.copy())
            c += 2
            while c < width and np.all(band[:, c] == BLACK):
                c += 1
        else:
            c += 1
    return templates, top


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each enclosed room with the tiled bottom template selected by its red marker count.

    Input structure:
    The grid has a black background, gray 4-connected closed walls, and red marker
    pixels inside the enclosed black rooms.  The bottom two rows contain several
    left-to-right 2x2 color templates separated by black columns, outside the wall
    area.

    Rule:
    Count the red markers inside every gray-enclosed non-wall component.  If the
    count is N, choose the Nth 2x2 template from the bottom row.  Remove the
    template rows from the output, keep the gray walls unchanged, and fill every
    cell of that enclosed component, including red markers, by repeating the
    selected template from the component's upper-left bounding-box corner.

    Key insight:
    The red pixels are a visible count index into the ordered template library;
    after selecting the template, the task is periodic 2x2 tiling inside each
    closed region.
    """
    grid = np.asarray(grid)
    templates, crop_row = _extract_templates(grid)
    out = grid[:crop_row].copy()
    height, width = out.shape
    if height == 0 or width == 0 or not templates:
        return out

    passable = out != GRAY
    visited = np.zeros((height, width), dtype=bool)

    for sr in range(height):
        for sc in range(width):
            if visited[sr, sc] or not passable[sr, sc]:
                continue

            queue = deque([(sr, sc)])
            visited[sr, sc] = True
            component = []
            touches_edge = False
            red_count = 0

            while queue:
                r, c = queue.popleft()
                component.append((r, c))
                if r == 0 or c == 0 or r == height - 1 or c == width - 1:
                    touches_edge = True
                if out[r, c] == RED:
                    red_count += 1

                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and not visited[nr, nc]
                        and passable[nr, nc]
                    ):
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            if touches_edge or red_count < 1 or red_count > len(templates):
                continue

            template = templates[red_count - 1]
            min_r = min(r for r, _ in component)
            min_c = min(c for _, c in component)
            for r, c in component:
                out[r, c] = template[(r - min_r) % 2, (c - min_c) % 2]

    return out


def _random_template(seen):
    while True:
        block = np.array(random.choices(TEMPLATE_COLORS, k=4), dtype=int).reshape(2, 2)
        key = tuple(int(x) for x in block.ravel())
        if len(set(key)) >= 2 and key not in seen:
            seen.add(key)
            return block


def _connected_mask(min_area):
    for _ in range(100):
        height = random.randint(2, 5)
        width = random.randint(2, 5)
        target = random.randint(min_area, height * width)
        mask = np.zeros((height, width), dtype=bool)
        start = (random.randrange(height), random.randrange(width))
        mask[start] = True
        cells = [start]

        while int(mask.sum()) < target:
            frontier = []
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cell = random.choice(frontier)
            mask[cell] = True
            cells.append(cell)

        if int(mask.sum()) >= min_area:
            rows, cols = np.where(mask)
            return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]

    mask = np.ones((2, max(2, min_area)), dtype=bool)
    return mask


def _make_room(marker_count):
    interior = _connected_mask(marker_count)
    ih, iw = interior.shape
    room = np.zeros((ih + 2, iw + 2), dtype=int)
    interior_cells = []

    for r, c in zip(*np.where(interior)):
        lr, lc = int(r) + 1, int(c) + 1
        interior_cells.append((lr, lc))

    for lr, lc in interior_cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = lr + dr, lc + dc
                if 0 <= rr < room.shape[0] and 0 <= cc < room.shape[1]:
                    room[rr, cc] = GRAY

    for lr, lc in interior_cells:
        room[lr, lc] = BLACK

    reserved = room != BLACK
    for lr, lc in interior_cells:
        reserved[lr, lc] = True

    for lr, lc in random.sample(interior_cells, marker_count):
        room[lr, lc] = RED

    return room, reserved


def _try_build():
    width = random.randint(15, 30)
    top_height = random.randint(12, 28)
    grid = np.zeros((top_height + 2, width), dtype=int)

    object_count = random.randint(2, 4)
    marker_counts = random.sample([1, 2, 3, 4], object_count)
    random.shuffle(marker_counts)
    occupied = np.zeros((top_height, width), dtype=bool)

    for marker_count in marker_counts:
        room, reserved = _make_room(marker_count)
        rh, rw = room.shape
        if top_height - rh - 2 < 1 or width - rw - 2 < 1:
            return None

        placed = False
        for _ in range(300):
            top = random.randint(1, top_height - rh - 1)
            left = random.randint(1, width - rw - 1)
            r0, r1 = top - 1, top + rh + 1
            c0, c1 = left - 1, left + rw + 1
            if occupied[r0:r1, c0:c1].any():
                continue

            mask = room != BLACK
            grid[top : top + rh, left : left + rw][mask] = room[mask]
            occupied[r0:r1, c0:c1] = True
            placed = True
            break

        if not placed:
            return None

    seen = set()
    templates = [_random_template(seen) for _ in range(4)]
    gaps = [random.randint(1, 3) for _ in range(3)]
    total_width = 8 + sum(gaps)
    if total_width > width:
        gaps = [1, 1, 1]
        total_width = 11

    col = random.randint(0, width - total_width)
    for i, template in enumerate(templates):
        grid[top_height : top_height + 2, col : col + 2] = template
        col += 2
        if i < 3:
            col += gaps[i]

    return grid


def generate_input() -> np.ndarray:
    for _ in range(100):
        grid = _try_build()
        if grid is not None:
            return grid

    grid = np.zeros((16, 15), dtype=int)
    counts = [1, 2, 3, 4]
    positions = [(1, 1), (1, 8), (8, 1), (8, 8)]
    for count, (top, left) in zip(counts, positions):
        room, _ = _make_room(count)
        rh, rw = room.shape
        mask = room != BLACK
        grid[top : top + rh, left : left + rw][mask] = room[mask]

    seen = set()
    col = 1
    for template in [_random_template(seen) for _ in range(4)]:
        grid[14:16, col : col + 2] = template
        col += 3
    return grid
