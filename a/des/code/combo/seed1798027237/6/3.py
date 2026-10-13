import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
VISIBLE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

# Generator configuration.
ROOM_ROWS_RANGE = (2, 3)
ROOM_COLS_RANGE = (2, 3)
ROOM_SIZE_RANGE = (3, 5)
LEGEND_LEN_RANGE = (2, 4)
TOP_MARGIN_RANGE = (1, 4)
LEFT_MARGIN_RANGE = (3, 6)
OTHER_MARGIN_RANGE = (1, 4)
GEN_RETRIES = 300


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _extract_legend(grid: np.ndarray) -> tuple[list[int], set[tuple[int, int]]]:
    colors: list[int] = []
    cells: set[tuple[int, int]] = set()
    seen: set[int] = set()

    for r in range(grid.shape[0]):
        color = int(grid[r, 0])
        if color in (BLACK, GREY) or color in seen:
            break
        colors.append(color)
        cells.add((r, 0))
        seen.add(color)

    assert colors, "expected a non-empty vertical legend in the top-left corner"
    return colors, cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey wall by the seed count of the enclosed region it borders.

    The top-left vertical legend is a count->color lookup table. For each enclosed
    non-grey region, count its non-black seed pixels, map that count through the
    legend, and recolor every adjacent grey wall. Shared walls use the larger seed
    count; regions with zero seeds leave their walls grey.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    legend, legend_cells = _extract_legend(g)
    passable = g != GREY
    h, w = g.shape

    labels = np.full((h, w), -1, dtype=int)
    closed_flags: list[bool] = []
    seed_counts: list[int] = []

    for r in range(h):
        for c in range(w):
            if not passable[r, c] or labels[r, c] != -1:
                continue

            label = len(closed_flags)
            queue = deque([(r, c)])
            labels[r, c] = label
            coords: list[tuple[int, int]] = []
            touches_edge = False

            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                if cr in (0, h - 1) or cc in (0, w - 1):
                    touches_edge = True
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = label
                        queue.append((nr, nc))

            closed = not touches_edge
            closed_flags.append(closed)
            if closed:
                count = sum(
                    1
                    for cr, cc in coords
                    if int(g[cr, cc]) != BLACK and (cr, cc) not in legend_cells
                )
            else:
                count = 0
            seed_counts.append(count)

    out = g.copy()
    for r, c in np.argwhere(g == GREY):
        best_count = 0
        for nr, nc in _neighbors4(int(r), int(c)):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            label = int(labels[nr, nc])
            if label < 0 or not closed_flags[label]:
                continue
            best_count = max(best_count, seed_counts[label])
        if best_count > 0:
            assert best_count <= len(legend), "seed count exceeds legend length"
            out[r, c] = legend[best_count - 1]
    return out


def _room_index(row: int, col: int, n_cols: int) -> int:
    return row * n_cols + col


def _border_room_indices(n_rows: int, n_cols: int) -> list[int]:
    indices = []
    for rr in range(n_rows):
        for cc in range(n_cols):
            if rr in (0, n_rows - 1) or cc in (0, n_cols - 1):
                indices.append(_room_index(rr, cc, n_cols))
    return indices


def _border_adjacent_pairs(n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    pairs = []
    for rr in range(n_rows):
        for cc in range(n_cols):
            idx = _room_index(rr, cc, n_cols)
            if cc + 1 < n_cols:
                right = _room_index(rr, cc + 1, n_cols)
                if rr in (0, n_rows - 1):
                    pairs.append((idx, right))
            if rr + 1 < n_rows:
                down = _room_index(rr + 1, cc, n_cols)
                if cc in (0, n_cols - 1):
                    pairs.append((idx, down))
    return pairs


def _choose_counts(n_rows: int, n_cols: int, legend_len: int) -> list[int]:
    n_rooms = n_rows * n_cols
    counts = [random.randint(0, legend_len) for _ in range(n_rooms)]

    high_idx, low_idx = random.choice(_border_adjacent_pairs(n_rows, n_cols))
    counts[high_idx] = legend_len
    counts[low_idx] = random.randint(1, legend_len - 1)

    zero_candidates = [idx for idx in _border_room_indices(n_rows, n_cols) if idx not in (high_idx, low_idx)]
    zero_idx = random.choice(zero_candidates)
    counts[zero_idx] = 0
    return counts


def _draw_rooms(
    grid: np.ndarray,
    top: int,
    left: int,
    room_heights: list[int],
    room_widths: list[int],
) -> list[tuple[int, int, int, int]]:
    struct_h = sum(room_heights) + len(room_heights) + 1
    struct_w = sum(room_widths) + len(room_widths) + 1
    bottom = top + struct_h - 1
    right = left + struct_w - 1

    wall_r = top
    grid[wall_r, left : right + 1] = GREY
    for room_h in room_heights:
        wall_r += room_h + 1
        grid[wall_r, left : right + 1] = GREY

    wall_c = left
    grid[top : bottom + 1, wall_c] = GREY
    for room_w in room_widths:
        wall_c += room_w + 1
        grid[top : bottom + 1, wall_c] = GREY

    boxes = []
    cur_r = top
    for room_h in room_heights:
        cur_c = left
        for room_w in room_widths:
            boxes.append((cur_r + 1, cur_r + room_h, cur_c + 1, cur_c + room_w))
            cur_c += room_w + 1
        cur_r += room_h + 1
    return boxes


def _sample_seed_cells(box: tuple[int, int, int, int], count: int) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = box
    cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    if count == 0:
        return []

    for _ in range(40):
        random.shuffle(cells)
        chosen: list[tuple[int, int]] = []
        for cell in cells:
            if all(abs(cell[0] - rr) + abs(cell[1] - cc) > 1 for rr, cc in chosen):
                chosen.append(cell)
                if len(chosen) == count:
                    return chosen
    return random.sample(cells, count)


def _gen_attempt() -> np.ndarray | None:
    n_rows = random.randint(*ROOM_ROWS_RANGE)
    n_cols = random.randint(*ROOM_COLS_RANGE)
    legend_len = random.randint(*LEGEND_LEN_RANGE)

    room_heights = [random.randint(*ROOM_SIZE_RANGE) for _ in range(n_rows)]
    room_widths = [random.randint(*ROOM_SIZE_RANGE) for _ in range(n_cols)]

    top = random.randint(*TOP_MARGIN_RANGE)
    left = random.randint(*LEFT_MARGIN_RANGE)
    bottom_margin = random.randint(*OTHER_MARGIN_RANGE)
    right_margin = random.randint(*OTHER_MARGIN_RANGE)

    struct_h = sum(room_heights) + n_rows + 1
    struct_w = sum(room_widths) + n_cols + 1
    height = max(legend_len + random.randint(0, 3), top + struct_h + bottom_margin)
    width = left + struct_w + right_margin

    legend_colors = random.sample(VISIBLE_COLORS, legend_len)
    seed_palette = [color for color in VISIBLE_COLORS if color not in legend_colors]

    grid = np.full((height, width), BLACK, dtype=int)
    for r, color in enumerate(legend_colors):
        grid[r, 0] = color

    room_boxes = _draw_rooms(grid, top, left, room_heights, room_widths)
    counts = _choose_counts(n_rows, n_cols, legend_len)

    seed_cells: list[tuple[int, int]] = []
    for box, count in zip(room_boxes, counts):
        for r, c in _sample_seed_cells(box, count):
            grid[r, c] = random.choice(seed_palette)
            seed_cells.append((r, c))

    if len(seed_cells) >= 2 and len({int(grid[r, c]) for r, c in seed_cells}) == 1:
        r, c = random.choice(seed_cells)
        alternatives = [color for color in seed_palette if color != int(grid[r, c])]
        if alternatives:
            grid[r, c] = random.choice(alternatives)

    try:
        out = main(grid)
    except Exception:
        return None

    wall_mask = grid == GREY
    recolored_walls = {int(v) for v in np.unique(out[wall_mask]) if int(v) != GREY}
    if len(recolored_walls) < 2:
        return None
    if not np.any(wall_mask & (out == GREY)):
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
