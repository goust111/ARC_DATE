from collections import deque
import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """Draw a one-cell inner border in each walled room using that room's seed color.

    Input structure:
    The grid has a most-common background color, a connected one-pixel-wide wall
    structure in one other color, and several rectangular enclosed rooms separated
    by those walls. Each room contains exactly one isolated non-background,
    non-wall seed pixel, and seed colors are visually distinct.

    Rule:
    Treat wall pixels as barriers and find each non-wall connected room region.
    For any region containing exactly one seed color C, recolor every background
    pixel in that region that is four-neighbor adjacent to a wall pixel to C.
    Keep the wall pixels and the seed pixel itself unchanged.

    Key insight:
    The seed does not spread through the whole room; it identifies the room and
    supplies the color for exactly the one-cell layer touching the inside walls.
    """
    arr = np.asarray(grid)
    out = arr.copy()
    if arr.ndim != 2 or arr.size == 0:
        return out

    values, counts = np.unique(arr, return_counts=True)
    bg = values[int(np.argmax(counts))]
    non_bg = values[values != bg]
    if len(non_bg) == 0:
        return out

    non_bg_counts = counts[values != bg]
    wall = non_bg[int(np.argmax(non_bg_counts))]

    rows, cols = arr.shape
    visited = np.zeros((rows, cols), dtype=bool)
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for start_r in range(rows):
        for start_c in range(cols):
            if visited[start_r, start_c] or arr[start_r, start_c] == wall:
                continue

            q = deque([(start_r, start_c)])
            visited[start_r, start_c] = True
            cells = []
            seed_colors = []

            while q:
                r, c = q.popleft()
                cells.append((r, c))
                color = arr[r, c]
                if color != bg and color != wall:
                    seed_colors.append(color)

                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < rows
                        and 0 <= nc < cols
                        and not visited[nr, nc]
                        and arr[nr, nc] != wall
                    ):
                        visited[nr, nc] = True
                        q.append((nr, nc))

            distinct_seeds = []
            for color in seed_colors:
                if not any(color == seen for seen in distinct_seeds):
                    distinct_seeds.append(color)
            if len(distinct_seeds) != 1:
                continue

            seed_color = distinct_seeds[0]
            for r, c in cells:
                if arr[r, c] != bg:
                    continue
                touches_wall = False
                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and arr[nr, nc] == wall:
                        touches_wall = True
                        break
                if touches_wall:
                    out[r, c] = seed_color

    return out


def _wall_positions(sizes):
    positions = [0]
    pos = 0
    for size in sizes:
        pos += size + 1
        positions.append(pos)
    return positions


def generate_input() -> np.ndarray:
    layouts = [
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 1),
        (2, 2),
        (2, 3),
        (2, 4),
        (3, 1),
        (3, 2),
    ]

    for _ in range(1000):
        room_rows, room_cols = random.choice(layouts)
        heights = [random.randint(4, 8) for _ in range(room_rows)]
        widths = [random.randint(4, 8) for _ in range(room_cols)]
        rows = sum(heights) + room_rows + 1
        cols = sum(widths) + room_cols + 1
        if rows > 30 or cols > 30:
            continue

        room_area = sum(heights) * sum(widths)
        wall_area = rows * cols - room_area
        room_count = room_rows * room_cols
        if room_area - room_count > wall_area:
            break
    else:
        room_rows, room_cols = 2, 2
        heights = [5, 6]
        widths = [5, 6]
        rows = sum(heights) + room_rows + 1
        cols = sum(widths) + room_cols + 1
        room_count = room_rows * room_cols

    colors = list(range(10))
    bg = random.choice(colors)
    wall_choices = [color for color in colors if color != bg]
    wall = random.choice(wall_choices)
    seed_choices = [color for color in colors if color not in (bg, wall)]
    seed_colors = random.sample(seed_choices, room_count)

    grid = np.full((rows, cols), bg, dtype=int)
    wall_rows = _wall_positions(heights)
    wall_cols = _wall_positions(widths)

    for r in wall_rows:
        grid[r, :] = wall
    for c in wall_cols:
        grid[:, c] = wall

    color_index = 0
    for rr in range(room_rows):
        top = wall_rows[rr] + 1
        bottom = wall_rows[rr + 1] - 1
        for cc in range(room_cols):
            left = wall_cols[cc] + 1
            right = wall_cols[cc + 1] - 1

            if random.random() < 0.5:
                candidates = []
                for r in range(top, bottom + 1):
                    candidates.append((r, left))
                    candidates.append((r, right))
                for c in range(left + 1, right):
                    candidates.append((top, c))
                    candidates.append((bottom, c))
                seed_r, seed_c = random.choice(candidates)
            else:
                seed_r = random.randint(top, bottom)
                seed_c = random.randint(left, right)

            grid[seed_r, seed_c] = seed_colors[color_index]
            color_index += 1

    return grid
