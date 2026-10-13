import random
from collections import deque

import numpy as np


BACKGROUND = 0
WALL = 5
SEED = 7
GRID_LIMIT = 30
ROOM_LAYOUTS = ((2, 2), (2, 3), (2, 4), (3, 2))
ROOM_SIZE_RANGE = (4, 6)
MARGIN_CAP = 3
GEN_RETRIES = 300
LEGEND_COLORS = tuple(color for color in range(10) if color not in {BACKGROUND, WALL})


def _largest_component(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            area = 0
            while q:
                rr, cc = q.popleft()
                area += 1
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            best = max(best, area)
    return best


def _open_components(mask: np.ndarray) -> list[tuple[list[tuple[int, int]], bool]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            touches_edge = False
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_edge = True
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append((coords, touches_edge))
    return components


def _legend_slice(row: np.ndarray) -> slice:
    best = None
    c = 0
    while c < len(row):
        if int(row[c]) == BACKGROUND:
            c += 1
            continue
        start = c
        while c < len(row) and int(row[c]) != BACKGROUND:
            c += 1
        if best is None or c - start > best[1] - best[0]:
            best = (start, c)
    assert best is not None, "top row must contain a non-background legend strip"
    legend = [int(x) for x in row[best[0] : best[1]]]
    assert len(set(legend)) == len(legend), "legend colors must be distinct"
    return slice(best[0], best[1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed region with the legend color indexed by the number of seeds inside it.

    Input structure:
    - The background is black, and the top row contains a contiguous left-to-right legend strip of distinct colors.
    - Below the legend, one foreground color forms a connected wall structure over the background.
    - The wall structure encloses several 4-connected background regions, each containing some seed pixels of a second foreground color.

    Rule:
    - Read the legend strip so its 1st cell corresponds to seed-count 1, its 2nd cell to seed-count 2, and so on.
    - Ignore wall cells and split the remaining cells into 4-connected regions; only regions that do not touch the outer border are enclosed rooms.
    - For each enclosed room, count its seed pixels k and fill the entire room with the k-th legend color, covering both seeds and background but leaving the walls and legend unchanged.

    Key insight:
    - The seeds are not direct color codes; they are counting markers that must be converted into a legend lookup before coloring the whole closed region.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    if g.ndim != 2 or g.shape[0] < 2 or g.shape[1] == 0:
        return out

    legend = [int(x) for x in g[0, _legend_slice(g[0])]]
    body = g[1:]

    body_colors = [int(color) for color in np.unique(body) if int(color) != BACKGROUND]
    assert len(body_colors) >= 2, "body must contain both walls and seeds"

    wall_color = max(
        body_colors,
        key=lambda color: (_largest_component(body == color), int(np.count_nonzero(body == color))),
    )
    seed_options = [color for color in body_colors if color != wall_color]
    seed_color = min(
        seed_options,
        key=lambda color: (_largest_component(body == color), int(np.count_nonzero(body == color))),
    )

    for coords, touches_edge in _open_components(body != wall_color):
        if touches_edge:
            continue
        seed_count = sum(1 for r, c in coords if int(body[r, c]) == seed_color)
        if 1 <= seed_count <= len(legend):
            fill_color = legend[seed_count - 1]
            for r, c in coords:
                out[r + 1, c] = fill_color

    return out


def _wall_lines(start: int, sizes: list[int]) -> list[int]:
    lines = [start]
    cursor = start
    for size in sizes:
        cursor += size + 1
        lines.append(cursor)
    return lines


def _draw_structure(
    grid: np.ndarray, top: int, left: int, room_heights: list[int], room_widths: list[int]
) -> list[tuple[int, int, int, int]]:
    wall_rows = _wall_lines(top, room_heights)
    wall_cols = _wall_lines(left, room_widths)

    for r in wall_rows:
        grid[r, wall_cols[0] : wall_cols[-1] + 1] = WALL
    for c in wall_cols:
        grid[wall_rows[0] : wall_rows[-1] + 1, c] = WALL

    rooms = []
    for i in range(len(room_heights)):
        for j in range(len(room_widths)):
            rooms.append((wall_rows[i] + 1, wall_rows[i + 1] - 1, wall_cols[j] + 1, wall_cols[j + 1] - 1))
    return rooms


def _checkerboard_cells(box: tuple[int, int, int, int], parity: int) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = box
    return [
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if (r + c) % 2 == parity
    ]


def _place_seeds(grid: np.ndarray, box: tuple[int, int, int, int], count: int) -> bool:
    candidates = []
    for parity in (0, 1):
        cells = _checkerboard_cells(box, parity)
        if len(cells) >= count:
            candidates.append(cells)
    if not candidates:
        return False
    chosen = random.choice(candidates)
    for r, c in random.sample(chosen, count):
        grid[r, c] = SEED
    return True


def _expected_output(
    grid: np.ndarray,
    rooms: list[tuple[int, int, int, int]],
    room_counts: list[int],
    legend_colors: list[int],
) -> np.ndarray:
    out = grid.copy()
    for (r0, r1, c0, c1), count in zip(rooms, room_counts):
        out[r0 : r1 + 1, c0 : c1 + 1] = legend_colors[count - 1]
    return out


def _gen_attempt() -> np.ndarray | None:
    room_rows, room_cols = random.choice(ROOM_LAYOUTS)
    n_regions = room_rows * room_cols

    room_heights = [random.randint(*ROOM_SIZE_RANGE) for _ in range(room_rows)]
    room_widths = [random.randint(*ROOM_SIZE_RANGE) for _ in range(room_cols)]
    structure_h = sum(room_heights) + room_rows + 1
    structure_w = sum(room_widths) + room_cols + 1

    slack_h = GRID_LIMIT - (1 + structure_h)
    slack_w = GRID_LIMIT - structure_w
    if slack_h < 0 or slack_w < 0:
        return None

    top_margin = random.randint(0, min(MARGIN_CAP, slack_h))
    bottom_margin = random.randint(0, min(MARGIN_CAP, slack_h - top_margin))
    left_margin = random.randint(0, min(MARGIN_CAP, slack_w))
    right_margin = random.randint(0, min(MARGIN_CAP, slack_w - left_margin))

    height = 1 + top_margin + structure_h + bottom_margin
    width = structure_w + left_margin + right_margin
    if width < n_regions:
        return None

    grid = np.full((height, width), BACKGROUND, dtype=int)
    legend_colors = random.sample(list(LEGEND_COLORS), n_regions)
    legend_start = random.randint(0, width - n_regions)
    grid[0, legend_start : legend_start + n_regions] = legend_colors

    rooms = _draw_structure(grid, 1 + top_margin, left_margin, room_heights, room_widths)
    room_counts = list(range(1, n_regions + 1))
    random.shuffle(room_counts)

    for box, count in zip(rooms, room_counts):
        if not _place_seeds(grid, box, count):
            return None

    expected = _expected_output(grid, rooms, room_counts, legend_colors)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid inputs after {GEN_RETRIES} attempts")
