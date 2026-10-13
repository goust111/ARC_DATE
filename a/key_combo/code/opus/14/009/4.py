from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

PIECE_COLORS = (BLUE, RED, YELLOW)
GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    green = np.argwhere(grid == GREEN)
    if green.size == 0:
        raise ValueError("expected a green grid frame")
    return _bbox_coords(green)


def _gravity_vector(grid: np.ndarray) -> tuple[int, int]:
    markers = np.argwhere(grid == GREY)
    if markers.shape[0] != 1:
        raise ValueError("expected one grey gravity marker")
    r, c = map(int, markers[0])
    h, w = grid.shape
    if r == 0:
        return -1, 0
    if r == h - 1:
        return 1, 0
    if c == 0:
        return 0, -1
    if c == w - 1:
        return 0, 1
    raise ValueError("gravity marker must lie on a grid edge")


def _passable_mask(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[bool]]:
    r0, c0, r1, c1 = _frame_bbox(grid)
    inside = np.zeros_like(grid, dtype=bool)
    inside[r0 + 1 : r1, c0 + 1 : c1] = True
    passable = inside & (grid != GREEN) & (grid != GREY)
    comps = _components(passable)
    comp_index = np.full(grid.shape, -1, dtype=int)
    is_rect: list[bool] = []
    for idx, coords in enumerate(comps):
        comp_index[coords[:, 0], coords[:, 1]] = idx
        cr0, cc0, cr1, cc1 = _bbox_coords(coords)
        area = int(coords.shape[0])
        bbox_area = (cr1 - cr0 + 1) * (cc1 - cc0 + 1)
        is_rect.append(area == bbox_area)
    return passable, comp_index, comps, is_rect


def _slide_destination(start: tuple[int, int], passable: np.ndarray, dr: int, dc: int) -> tuple[int, int]:
    h, w = passable.shape
    r, c = start
    while True:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w and passable[nr, nc]:
            r, c = nr, nc
        else:
            return r, c


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark sealed lattice cells orange and slide the channel pixels along the grey-indicated gravity direction.

    Input structure:
    - A black background contains one green 1-pixel-thick rectangular cell grid with a closed outer border and some broken inner wall segments.
    - A single grey border pixel indicates gravity: top/up, bottom/down, left/left, right/right.
    - Some cell interiors contain one colored pixel chosen from blue, red, or yellow; a broken background region may span several cells through wall gaps.

    Rule:
    - Treat the green frame and grey marker as blockers, and treat black plus colored interior pixels as the passable space inside the outer green border.
    - For each 4-connected passable region inside the frame, compare its area to its bounding-box area: exact rectangles are sealed single cells, while non-rectangular regions are broken channels.
    - Fill every sealed region with orange, keep every broken channel black, keep colored pixels fixed in sealed cells, and slide colored pixels in channels step by step along gravity until the next step would hit a blocker or leave the grid.

    Key insight:
    - The flood-fill classification does not decide the final color of the moving pixels; it decides whether each colored pixel is locked inside a sealed cell or allowed to travel through a broken channel.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    passable, comp_index, _comps, is_rect = _passable_mask(g)
    dr, dc = _gravity_vector(g)

    out = g.copy()
    out[passable] = BLACK

    piece_positions = [
        (int(r), int(c), int(g[r, c]))
        for r, c in np.argwhere((g != BLACK) & (g != GREEN) & (g != GREY))
    ]

    for r, c, color in piece_positions:
        idx = int(comp_index[r, c])
        if idx < 0:
            out[r, c] = color
            continue
        if is_rect[idx]:
            out[r, c] = color
        else:
            rr, cc = _slide_destination((r, c), passable, dr, dc)
            out[rr, cc] = color

    for idx, rect in enumerate(is_rect):
        if not rect:
            continue
        coords = np.argwhere(comp_index == idx)
        out[coords[:, 0], coords[:, 1]] = ORANGE
    for r, c, color in piece_positions:
        idx = int(comp_index[r, c])
        if idx >= 0 and is_rect[idx]:
            out[r, c] = color

    return out


def _line_positions(offset: int, n_cells: int, cell_size: int) -> list[int]:
    return [offset + k * (cell_size + 1) for k in range(n_cells + 1)]


def _cell_centers(offset: int, n_cells: int, cell_size: int) -> list[int]:
    return [offset + 1 + r * (cell_size + 1) + (cell_size - 1) // 2 for r in range(n_cells)]


def _draw_lattice(
    grid: np.ndarray,
    top: int,
    left: int,
    n_rows: int,
    n_cols: int,
    cell_size: int,
) -> tuple[list[int], list[int]]:
    row_lines = _line_positions(top, n_rows, cell_size)
    col_lines = _line_positions(left, n_cols, cell_size)
    for rr in row_lines:
        grid[rr, col_lines[0] : col_lines[-1] + 1] = GREEN
    for cc in col_lines:
        grid[row_lines[0] : row_lines[-1] + 1, cc] = GREEN
    return row_lines, col_lines


def _punch_gap(
    grid: np.ndarray,
    row_lines: list[int],
    col_lines: list[int],
    cell_size: int,
    orientation: str,
    r: int,
    c: int,
    rng: np.random.Generator,
) -> None:
    gap_len = int(rng.integers(1, min(2, cell_size) + 1))
    span = cell_size - gap_len + 1
    offset = int(rng.integers(0, span))
    if orientation == "h":
        wall_r = row_lines[r + 1]
        c0 = col_lines[c] + 1 + offset
        grid[wall_r, c0 : c0 + gap_len] = BLACK
    else:
        wall_c = col_lines[c + 1]
        r0 = row_lines[r] + 1 + offset
        grid[r0 : r0 + gap_len, wall_c] = BLACK


def _cell_of_coord(r: int, c: int, top: int, left: int, cell_size: int) -> tuple[int, int]:
    return (
        (r - (top + 1)) // (cell_size + 1),
        (c - (left + 1)) // (cell_size + 1),
    )


def _interior_coords_for_cell(
    row_lines: list[int],
    col_lines: list[int],
    cell_r: int,
    cell_c: int,
) -> list[tuple[int, int]]:
    coords = []
    for r in range(row_lines[cell_r] + 1, row_lines[cell_r + 1]):
        for c in range(col_lines[cell_c] + 1, col_lines[cell_c + 1]):
            coords.append((r, c))
    return coords


def _edge_segments(n_rows: int, n_cols: int) -> list[tuple[str, int, int]]:
    edges: list[tuple[str, int, int]] = []
    for r in range(n_rows - 1):
        for c in range(n_cols):
            edges.append(("h", r, c))
    for r in range(n_rows):
        for c in range(n_cols - 1):
            edges.append(("v", r, c))
    return edges


def _gravity_marker_position(
    h: int,
    w: int,
    direction: str,
    rng: np.random.Generator,
) -> tuple[int, int]:
    if direction == "up":
        return 0, int(rng.integers(1, w - 1))
    if direction == "down":
        return h - 1, int(rng.integers(1, w - 1))
    if direction == "left":
        return int(rng.integers(1, h - 1)), 0
    return int(rng.integers(1, h - 1)), w - 1


def _blank_classification(
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[bool], tuple[int, int, int, int]]:
    passable, comp_index, comps, is_rect = _passable_mask(grid)
    return passable, comp_index, comps, is_rect, _frame_bbox(grid)


def _channel_start_options(
    passable: np.ndarray,
    comp_index: np.ndarray,
    comp_id: int,
    row_lines: list[int],
    col_lines: list[int],
    top: int,
    left: int,
    cell_size: int,
    dr: int,
    dc: int,
) -> list[tuple[tuple[int, int], tuple[int, int], int]]:
    options: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    used_cells: set[tuple[int, int]] = set()
    h, w = passable.shape
    for cell_r in range(len(row_lines) - 1):
        for cell_c in range(len(col_lines) - 1):
            best: tuple[tuple[int, int], tuple[int, int], int] | None = None
            for r, c in _interior_coords_for_cell(row_lines, col_lines, cell_r, cell_c):
                if not (0 <= r < h and 0 <= c < w):
                    continue
                if int(comp_index[r, c]) != comp_id:
                    continue
                dest = _slide_destination((r, c), passable, dr, dc)
                dist = abs(dest[0] - r) + abs(dest[1] - c)
                if dist == 0:
                    continue
                cand = ((r, c), dest, dist)
                if best is None or dist > best[2]:
                    best = cand
            if best is not None and (cell_r, cell_c) not in used_cells:
                used_cells.add((cell_r, cell_c))
                options.append(best)
    return options


def _sealed_cells(
    comp_index: np.ndarray,
    is_rect: list[bool],
    row_lines: list[int],
    col_lines: list[int],
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for cell_r in range(len(row_lines) - 1):
        for cell_c in range(len(col_lines) - 1):
            r = row_lines[cell_r] + 1
            c = col_lines[cell_c] + 1
            idx = int(comp_index[r, c])
            if idx >= 0 and is_rect[idx]:
                cells.append((cell_r, cell_c))
    return cells


def _best_cell_coord(
    row_lines: list[int],
    col_lines: list[int],
    cell_r: int,
    cell_c: int,
) -> tuple[int, int]:
    rs = list(range(row_lines[cell_r] + 1, row_lines[cell_r + 1]))
    cs = list(range(col_lines[cell_c] + 1, col_lines[cell_c + 1]))
    return rs[(len(rs) - 1) // 2], cs[(len(cs) - 1) // 2]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_rows = int(rng.integers(3, 6))
    n_cols = int(rng.integers(3, 6))
    cell_size = int(rng.integers(3, 5))

    frame_h = n_rows * cell_size + (n_rows + 1)
    frame_w = n_cols * cell_size + (n_cols + 1)
    extra_h = 30 - frame_h
    extra_w = 30 - frame_w
    if extra_h < 2 or extra_w < 2:
        return None

    top = int(rng.integers(1, min(3, extra_h - 1) + 1))
    left = int(rng.integers(1, min(3, extra_w - 1) + 1))
    bottom = int(rng.integers(1, extra_h - top + 1))
    right = int(rng.integers(1, extra_w - left + 1))
    h = frame_h + top + bottom
    w = frame_w + left + right

    grid = np.full((h, w), BLACK, dtype=int)
    row_lines, col_lines = _draw_lattice(grid, top, left, n_rows, n_cols, cell_size)

    edges = _edge_segments(n_rows, n_cols)
    rng.shuffle(edges)
    breach_low = max(1, len(edges) // 5)
    breach_high = max(breach_low, len(edges) // 3)
    breach_count = int(rng.integers(breach_low, breach_high + 1))
    for orientation, r, c in edges[:breach_count]:
        _punch_gap(grid, row_lines, col_lines, cell_size, orientation, r, c, rng)

    direction = ("up", "down", "left", "right")[int(rng.integers(0, 4))]
    marker_r, marker_c = _gravity_marker_position(h, w, direction, rng)
    grid[marker_r, marker_c] = GREY
    dr, dc = _gravity_vector(grid)

    passable, comp_index, comps, is_rect, _bbox = _blank_classification(grid)
    rect_ids = [idx for idx, rect in enumerate(is_rect) if rect]
    channel_ids = [idx for idx, rect in enumerate(is_rect) if not rect]
    if not rect_ids or not channel_ids:
        return None

    movable_options: list[tuple[int, tuple[int, int], tuple[int, int], int]] = []
    for comp_id in channel_ids:
        options = _channel_start_options(
            passable, comp_index, comp_id, row_lines, col_lines, top, left, cell_size, dr, dc
        )
        if not options:
            continue
        best = max(options, key=lambda item: item[2])
        movable_options.append((comp_id, best[0], best[1], best[2]))
    if not movable_options:
        return None

    rng.shuffle(movable_options)
    movable_count = min(len(movable_options), int(rng.integers(1, min(3, len(movable_options)) + 1)))
    chosen_movable = movable_options[:movable_count]

    used_destinations = {dest for _comp_id, _start, dest, _dist in chosen_movable}
    if len(used_destinations) != len(chosen_movable):
        return None

    sealed_cells = _sealed_cells(comp_index, is_rect, row_lines, col_lines)
    if len(sealed_cells) < 1:
        return None
    rng.shuffle(sealed_cells)
    locked_count = min(len(sealed_cells), int(rng.integers(1, min(4, len(sealed_cells)) + 1)))
    chosen_locked = sealed_cells[:locked_count]

    used_cells = {
        _cell_of_coord(start[0], start[1], top, left, cell_size) for _comp_id, start, _dest, _dist in chosen_movable
    }
    for cell in chosen_locked:
        if cell in used_cells:
            return None

    total_pieces = len(chosen_movable) + len(chosen_locked)
    if total_pieces < 2:
        return None

    color_pool = [PIECE_COLORS[int(rng.integers(0, len(PIECE_COLORS)))] for _ in range(total_pieces)]
    for (_comp_id, start, _dest, _dist), color in zip(chosen_movable, color_pool[: len(chosen_movable)]):
        grid[start] = color
    for cell, color in zip(chosen_locked, color_pool[len(chosen_movable) :]):
        grid[_best_cell_coord(row_lines, col_lines, *cell)] = color

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    inside = np.zeros_like(grid, dtype=bool)
    fr0, fc0, fr1, fc1 = _frame_bbox(grid)
    inside[fr0 + 1 : fr1, fc0 + 1 : fc1] = True
    if not np.any(out[inside] == ORANGE):
        return None
    if not np.any(passable & (out == BLACK)):
        return None

    moved = 0
    for _comp_id, start, dest, _dist in chosen_movable:
        if start != dest and out[dest] in PIECE_COLORS and out[start] == BLACK:
            moved += 1
    if moved != len(chosen_movable):
        return None

    distinct_rect = len(rect_ids)
    if distinct_rect == 1 and not chosen_locked:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
