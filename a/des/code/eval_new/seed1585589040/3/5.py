import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GREY = 5
PATH_COLORS = (1, 4, 6, 7, 8, 9)
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow a forked colored path, choose turns from nearby square sizes, and encode the visited color sequence into the end frame.

    Input structure:
    - The background is black.
    - A one-pixel-wide orthogonal path starts at a single red pixel and ends at a single green pixel.
    - At each fork, the incoming path can continue in exactly two turn directions, and a nearby solid 2x2 or 3x3 square sits by the fork.
    - A single grey hollow rectangle sits next to the green endpoint and provides an empty encoding area.

    Rule:
    - Remove the nearby solid squares from the traversable path mask, then trace the path from red to green.
    - Whenever the trace reaches a fork, read the size of the square touching a diagonal neighbor of that fork: size 2 means take the left turn, size 3 means take the right turn, relative to the current travel direction.
    - Record the colors of the traversed path segments in order, excluding the red start and green end.
    - Keep the whole input unchanged except for the grey frame interior, which is filled row-major with the recorded color sequence; unused frame cells stay black.

    Key insight:
    - The square size, not its color, chooses the branch direction, and the chosen path's ordered color segments become a compact pixel-array code inside the endpoint frame.
    """
    g = np.asarray(grid, dtype=int)
    start = _unique_pos(g, RED)
    end = _unique_pos(g, GREEN)
    frame = _find_frame(g)
    squares, indicator_mask = _detect_indicator_squares(g)

    traversable = (g != BLACK) & (g != GREY) & ~indicator_mask
    path_mask = _component_from(start, traversable)
    assert bool(path_mask[end]), "green endpoint must lie on the traced path"

    sequence = _trace_sequence(g, path_mask, start, end, squares)

    out = g.copy()
    inner = out[frame[0] + 1 : frame[2], frame[1] + 1 : frame[3]]
    inner[:] = BLACK
    for idx, color in enumerate(sequence):
        if idx >= int(inner.size):
            break
        rr = idx // inner.shape[1]
        cc = idx % inner.shape[1]
        inner[rr, cc] = int(color)
    return out


def _unique_pos(grid: np.ndarray, color: int) -> tuple[int, int]:
    pts = np.argwhere(grid == int(color))
    assert len(pts) == 1, f"expected exactly one pixel of color {color}"
    return int(pts[0, 0]), int(pts[0, 1])


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(grid == GREY)
    assert len(pts) > 0, "grey frame missing"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    assert r1 - r0 >= 3 and c1 - c0 >= 3, "frame must be hollow"
    assert bool((grid[r0, c0 : c1 + 1] == GREY).all())
    assert bool((grid[r1, c0 : c1 + 1] == GREY).all())
    assert bool((grid[r0 : r1 + 1, c0] == GREY).all())
    assert bool((grid[r0 : r1 + 1, c1] == GREY).all())
    assert bool((grid[r0 + 1 : r1, c0 + 1 : c1] == BLACK).all())
    return r0, c0, r1, c1


def _detect_indicator_squares(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    h, w = grid.shape
    used = np.zeros_like(grid, dtype=bool)
    squares: list[dict] = []
    for size in (3, 2):
        for r in range(h - size + 1):
            for c in range(w - size + 1):
                patch = grid[r : r + size, c : c + size]
                color = int(patch[0, 0])
                if color in (BLACK, GREY):
                    continue
                if not bool((patch == color).all()):
                    continue
                if bool(used[r : r + size, c : c + size].any()):
                    continue
                cells = {(rr, cc) for rr in range(r, r + size) for cc in range(c, c + size)}
                squares.append(
                    {
                        "size": int(size),
                        "color": int(color),
                        "bbox": (int(r), int(c), int(r + size - 1), int(c + size - 1)),
                        "cells": cells,
                    }
                )
                used[r : r + size, c : c + size] = True
    return squares, used


def _component_from(start: tuple[int, int], mask: np.ndarray) -> np.ndarray:
    seen = np.zeros_like(mask, dtype=bool)
    if not bool(mask[start]):
        return seen
    q = deque([start])
    seen[start] = True
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, mask.shape[0], mask.shape[1]):
            if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _trace_sequence(
    grid: np.ndarray,
    path_mask: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    squares: list[dict],
) -> list[int]:
    h, w = grid.shape
    current = start
    prev: tuple[int, int] | None = None
    sequence: list[int] = []
    last_color: int | None = None

    while True:
        if current == end:
            break
        if current != start:
            color = int(grid[current])
            if color not in (RED, GREEN) and color != last_color:
                sequence.append(color)
                last_color = color

        nxts = []
        for nr, nc in _neighbors4(current[0], current[1], h, w):
            if not bool(path_mask[nr, nc]):
                continue
            if prev is not None and (nr, nc) == prev:
                continue
            nxts.append((nr, nc))

        if prev is None:
            assert len(nxts) == 1, "start must have one outgoing path cell"
            nxt = nxts[0]
        elif len(nxts) == 1:
            nxt = nxts[0]
        elif len(nxts) == 2:
            dr = int(current[0] - prev[0])
            dc = int(current[1] - prev[1])
            left = (int(current[0] - dc), int(current[1] + dr))
            right = (int(current[0] + dc), int(current[1] - dr))
            assert set(nxts) == {left, right}, "fork must offer left and right turns only"
            indicator = _indicator_size_at(current, squares)
            nxt = left if indicator == 2 else right
        else:
            raise AssertionError("path trace became ambiguous")

        prev, current = current, nxt

    return sequence


def _indicator_size_at(pos: tuple[int, int], squares: list[dict]) -> int:
    r, c = pos
    matches = []
    for sq in squares:
        r0, c0, r1, c1 = map(int, sq["bbox"])
        is_touching_corner = (
            (r1 == r - 1 and c1 == c - 1)
            or (r1 == r - 1 and c0 == c + 1)
            or (r0 == r + 1 and c1 == c - 1)
            or (r0 == r + 1 and c0 == c + 1)
        )
        if is_touching_corner:
            matches.append(sq)
    assert len(matches) == 1, "each fork must have exactly one nearby square"
    return int(matches[0]["size"])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _turn_left(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return -dc, dr


def _turn_right(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return dc, -dr


def _add(a: tuple[int, int], b: tuple[int, int], k: int = 1) -> tuple[int, int]:
    return int(a[0] + b[0] * k), int(a[1] + b[1] * k)


def _build_square_cells(
    fork: tuple[int, int],
    diag: tuple[int, int],
    size: int,
) -> set[tuple[int, int]]:
    sr, sc = diag
    row_start = fork[0] - size if sr < 0 else fork[0] + 1
    row_end = fork[0] if sr < 0 else fork[0] + size + 1
    col_start = fork[1] - size if sc < 0 else fork[1] + 1
    col_end = fork[1] if sc < 0 else fork[1] + size + 1
    return {(r, c) for r in range(row_start, row_end) for c in range(col_start, col_end)}


def _rect_border_cells(top: int, left: int, height: int, width: int) -> set[tuple[int, int]]:
    bottom = top + height - 1
    right = left + width - 1
    cells = set()
    for c in range(left, right + 1):
        cells.add((top, c))
        cells.add((bottom, c))
    for r in range(top, bottom + 1):
        cells.add((r, left))
        cells.add((r, right))
    return cells


def _rect_cells(top: int, left: int, height: int, width: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(top, top + height) for c in range(left, left + width)}


def _path_neighbors(cell: tuple[int, int], path_cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {nb for nb in (_add(cell, d) for d in DIRS) if nb in path_cells}


def _ray_cells(origin: tuple[int, int], step: tuple[int, int], path_cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    cells = set()
    current = _add(origin, step)
    while current in path_cells:
        cells.add(current)
        current = _add(current, step)
    return cells


def _extend_segment(
    cells: dict[tuple[int, int], int],
    start: tuple[int, int],
    direction: tuple[int, int],
    length: int,
    color: int,
) -> tuple[int, int] | None:
    current = start
    for _ in range(length):
        nxt = _add(current, direction)
        if nxt in cells:
            return None
        neighbors = _path_neighbors(nxt, set(cells))
        if neighbors - {current}:
            return None
        cells[nxt] = int(color)
        current = nxt
    return current


def _sequence_to_expected(grid: np.ndarray, frame: tuple[int, int, int, int], sequence: list[int]) -> np.ndarray:
    out = grid.copy()
    inner = out[frame[0] + 1 : frame[2], frame[1] + 1 : frame[3]]
    inner[:] = BLACK
    for idx, color in enumerate(sequence):
        rr = idx // inner.shape[1]
        cc = idx % inner.shape[1]
        inner[rr, cc] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    fork_count = random.randint(1, 3)
    start = (0, 0)
    cells: dict[tuple[int, int], int] = {start: RED}
    incoming = random.choice(DIRS)
    main_colors = [random.choice(PATH_COLORS)]
    turns = [random.choice(("L", "R")) for _ in range(fork_count)]
    main_lengths = [random.randint(2, 5) for _ in range(fork_count + 1)]
    branch_lengths = [random.randint(2, 4) for _ in range(fork_count)]

    fork_infos: list[dict] = []
    current = start

    for idx in range(fork_count):
        current = _extend_segment(cells, current, incoming, main_lengths[idx], main_colors[idx])
        if current is None:
            return None

        left_color, right_color = random.sample([c for c in PATH_COLORS if c != main_colors[idx]], 2)
        chosen_turn = turns[idx]
        chosen_dir = _turn_left(incoming) if chosen_turn == "L" else _turn_right(incoming)
        other_dir = _turn_right(incoming) if chosen_turn == "L" else _turn_left(incoming)
        chosen_color = left_color if chosen_turn == "L" else right_color
        other_color = right_color if chosen_turn == "L" else left_color

        branch_end = _extend_segment(cells, current, other_dir, branch_lengths[idx], other_color)
        if branch_end is None:
            return None

        fork_infos.append(
            {
                "pos": current,
                "incoming": incoming,
                "turn": chosen_turn,
                "left_color": left_color,
                "right_color": right_color,
            }
        )
        main_colors.append(chosen_color)
        incoming = chosen_dir

    current = _extend_segment(cells, current, incoming, main_lengths[-1], main_colors[-1])
    if current is None:
        return None
    end = _add(current, incoming)
    if end in cells:
        return None
    if _path_neighbors(end, set(cells)) - {current}:
        return None
    cells[end] = GREEN

    path_cells = set(cells)
    square_cells: dict[tuple[int, int], int] = {}
    occupied = set(path_cells)

    for info in fork_infos:
        size = 2 if info["turn"] == "L" else 3
        matched_side = random.choice(("left", "right"))
        square_color = int(info["left_color"] if matched_side == "left" else info["right_color"])
        forward = info["incoming"]
        backward = (-forward[0], -forward[1])
        left_dir = _turn_left(forward)
        right_dir = _turn_right(forward)
        if matched_side == "left":
            diags = [
                (forward[0] + right_dir[0], forward[1] + right_dir[1]),
                (backward[0] + right_dir[0], backward[1] + right_dir[1]),
            ]
        else:
            diags = [
                (forward[0] + left_dir[0], forward[1] + left_dir[1]),
                (backward[0] + left_dir[0], backward[1] + left_dir[1]),
            ]
        random.shuffle(diags)

        placed = False
        for diag in diags:
            coords = _build_square_cells(info["pos"], diag, size)
            if coords & occupied:
                continue
            allowed_path_touch = _ray_cells(info["pos"], (diag[0], 0), path_cells) | _ray_cells(
                info["pos"], (0, diag[1]), path_cells
            )
            ok = True
            for cell in coords:
                neighbors = _path_neighbors(cell, path_cells)
                if neighbors - allowed_path_touch:
                    ok = False
                    break
            if not ok:
                continue
            for cell in coords:
                square_cells[cell] = square_color
            occupied |= coords
            placed = True
            break
        if not placed:
            return None

    inner_shapes = [(h, w) for h in range(2, 5) for w in range(2, 5) if h * w >= len(main_colors)]
    inner_h, inner_w = random.choice(inner_shapes)
    outer_h, outer_w = inner_h + 2, inner_w + 2
    if incoming[0] == 0:
        anchor = random.randint(1, outer_h - 2)
        top = end[0] - anchor
        left = end[1] + 1 if incoming[1] > 0 else end[1] - outer_w
    else:
        anchor = random.randint(1, outer_w - 2)
        top = end[0] + 1 if incoming[0] > 0 else end[0] - outer_h
        left = end[1] - anchor

    frame_bbox = (top, left, top + outer_h - 1, left + outer_w - 1)
    frame_area = _rect_cells(top, left, outer_h, outer_w)
    if frame_area & occupied:
        return None
    frame_border = _rect_border_cells(top, left, outer_h, outer_w)
    adjacent_to_end = {cell for cell in frame_border if abs(cell[0] - end[0]) + abs(cell[1] - end[1]) == 1}
    if len(adjacent_to_end) != 1:
        return None
    for cell in frame_border:
        if _path_neighbors(cell, path_cells) - {end}:
            return None

    all_objects = set(path_cells) | set(square_cells) | frame_area
    rows = [p[0] for p in all_objects]
    cols = [p[1] for p in all_objects]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    bbox_h = max_r - min_r + 1
    bbox_w = max_c - min_c + 1
    if bbox_h > 30 or bbox_w > 30:
        return None

    H = random.randint(bbox_h, 30)
    W = random.randint(bbox_w, 30)
    row_shift = random.randint(0, H - bbox_h) - min_r
    col_shift = random.randint(0, W - bbox_w) - min_c

    def shift(cell: tuple[int, int]) -> tuple[int, int]:
        return int(cell[0] + row_shift), int(cell[1] + col_shift)

    grid = np.zeros((H, W), dtype=int)
    for cell, color in cells.items():
        rr, cc = shift(cell)
        grid[rr, cc] = int(color)
    for cell, color in square_cells.items():
        rr, cc = shift(cell)
        grid[rr, cc] = int(color)

    shifted_frame = (
        int(frame_bbox[0] + row_shift),
        int(frame_bbox[1] + col_shift),
        int(frame_bbox[2] + row_shift),
        int(frame_bbox[3] + col_shift),
    )
    for rr, cc in _rect_border_cells(shifted_frame[0], shifted_frame[1], outer_h, outer_w):
        grid[rr, cc] = GREY

    expected = _sequence_to_expected(grid, shifted_frame, main_colors)
    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input after 500 attempts")
