from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

FRAME_COLOR = GREY
START_COLOR = GREEN
MARKER_COLORS = (BLUE, RED, GREEN)
SPRITE_COLORS = (4, 6, 7, 8, 9)
DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))  # right, down, left, up
GEN_RETRIES = 800


def _bbox_of_color(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == color)
    assert coords.size > 0, f"expected color {color} to be present"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_marker_axes(grid: np.ndarray, frame: tuple[int, int, int, int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    r0, c0, r1, c1 = frame
    row_marks = set()
    col_marks = set()
    for c in range(c0 + 1, c1):
        if int(grid[r0, c]) != FRAME_COLOR:
            col_marks.add(c)
        if int(grid[r1, c]) != FRAME_COLOR:
            col_marks.add(c)
    for r in range(r0 + 1, r1):
        if int(grid[r, c0]) != FRAME_COLOR:
            row_marks.add(r)
        if int(grid[r, c1]) != FRAME_COLOR:
            row_marks.add(r)
    assert row_marks and col_marks, "expected colored frame markers on both axes"
    return tuple(sorted(row_marks)), tuple(sorted(col_marks))


def _find_start(grid: np.ndarray, frame: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, _c1 = frame
    rows = [r for r in range(r0 + 1, r1) if int(grid[r, c0 + 1]) == START_COLOR]
    assert len(rows) == 1, "expected a unique green start pixel adjacent to the left inner frame edge"
    return int(rows[0]), int(c0 + 1)


def _simulate_path(
    grid: np.ndarray,
    frame: tuple[int, int, int, int],
    start: tuple[int, int],
    row_marks: tuple[int, ...],
    col_marks: tuple[int, ...],
) -> tuple[list[tuple[int, int]], list[tuple[int, int, str]]]:
    r0, c0, r1, c1 = frame
    row_mark_set = set(map(int, row_marks))
    col_mark_set = set(map(int, col_marks))
    temp = np.array(grid, dtype=int, copy=True)

    path = [start]
    turns: list[tuple[int, int, str]] = []
    r, c = map(int, start)
    dir_idx = 0
    max_steps = (r1 - r0 - 1) * (c1 - c0 - 1) + 5

    for _ in range(max_steps):
        row_hit = r in row_mark_set
        col_hit = c in col_mark_set
        if row_hit:
            dir_idx = (dir_idx + 1) % 4
            turns.append((r, c, "both" if col_hit else "row"))
        elif col_hit:
            dir_idx = (dir_idx - 1) % 4
            turns.append((r, c, "col"))

        dr, dc = DIRS[dir_idx]
        nr, nc = r + dr, c + dc
        if not (r0 < nr < r1 and c0 < nc < c1):
            break
        if int(temp[nr, nc]) != BG:
            break

        temp[nr, nc] = START_COLOR
        path.append((int(nr), int(nc)))
        r, c = int(nr), int(nc)
    return path, turns


def _dist_to_path(point: tuple[int, int], path: list[tuple[int, int]]) -> int:
    r, c = point
    return min(abs(r - pr) + abs(c - pc) for pr, pc in path)


def _selected_intersections(
    row_marks: tuple[int, ...],
    col_marks: tuple[int, ...],
    path: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], int]]:
    distances: dict[tuple[int, int], int] = {}
    selected = []
    for r in row_marks:
        for c in col_marks:
            point = (int(r), int(c))
            dist = _dist_to_path(point, path)
            distances[point] = dist
            if dist <= 1:
                selected.append(point)
    selected.sort()
    return selected, distances


def _stamp_sprite(
    out: np.ndarray,
    sprite: np.ndarray,
    center: tuple[int, int],
    frame: tuple[int, int, int, int],
) -> None:
    r0, c0, r1, c1 = frame
    cr, cc = center
    for sr in range(3):
        for sc in range(3):
            color = int(sprite[sr, sc])
            if color == BG:
                continue
            tr = int(cr + sr - 1)
            tc = int(cc + sc - 1)
            if r0 < tr < r1 and c0 < tc < c1:
                out[tr, tc] = color


def _stamped_cells(
    sprite: np.ndarray,
    center: tuple[int, int],
    frame: tuple[int, int, int, int],
) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = frame
    cr, cc = center
    cells: set[tuple[int, int]] = set()
    for sr in range(3):
        for sc in range(3):
            if int(sprite[sr, sc]) == BG:
                continue
            tr = int(cr + sr - 1)
            tc = int(cc + sc - 1)
            if r0 < tr < r1 and c0 < tc < c1:
                cells.add((tr, tc))
    return cells


def _build_output(
    grid: np.ndarray,
    frame: tuple[int, int, int, int],
    sprite: np.ndarray,
    path: list[tuple[int, int]],
    selected: list[tuple[int, int]],
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for r, c in path:
        out[r, c] = BG
    out[:3, :3] = BG
    for point in selected:
        _stamp_sprite(out, sprite, point, frame)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a turning ray inside a marked frame and stamp the sprite only at crossed marker intersections.

    Input structure:
    - Black background with a grey hollow rectangular frame on the right side.
    - Colored border markers replace some non-corner frame pixels on the top/bottom and left/right edges.
    - A 3x3 sprite palette sits in the upper-left corner, and a green start pixel lies one cell inside the frame's left wall.

    Rule:
    - Read the frame-marker rows from the left/right edges and the frame-marker columns from the top/bottom edges.
    - Starting from the green pixel and facing right, advance one cell at a time through frame-interior background cells; on a marked row turn clockwise, on a marked column turn counterclockwise, and if both apply use the clockwise turn.
    - Stop when the next step would hit the frame wall, leave the frame interior, or enter any already-colored cell.
    - Form every row/column marker intersection inside the frame and place the 3x3 sprite, clipped to the frame interior, only at intersections whose Manhattan distance to the ray path is at most 1.
    - Erase the green path, the start pixel, and the original 3x3 palette area.

    Key insight:
    - The same frame markers simultaneously define the ray's turning lines and the candidate stamp coordinates, so moving a marker changes both the path and which intersections can be selected.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= 4 and g.shape[1] >= 4, "grid is too small"

    frame = _bbox_of_color(g, FRAME_COLOR)
    sprite = g[:3, :3].copy()
    row_marks, col_marks = _extract_marker_axes(g, frame)
    start = _find_start(g, frame)
    path, _turns = _simulate_path(g, frame, start, row_marks, col_marks)
    selected, _distances = _selected_intersections(row_marks, col_marks, path)
    return _build_output(g, frame, sprite, path, selected)


def _choose_spaced_positions(values: list[int], count: int, min_gap: int) -> tuple[int, ...] | None:
    valid = [combo for combo in combinations(values, count) if all(b - a >= min_gap for a, b in zip(combo, combo[1:]))]
    if not valid:
        return None
    return tuple(map(int, random.choice(valid)))


def _random_sprite() -> np.ndarray:
    sprite = np.zeros((3, 3), dtype=int)
    cells = [(r, c) for r in range(3) for c in range(3)]
    count = random.randint(3, 5)
    picked = [(1, 1)] + random.sample([cell for cell in cells if cell != (1, 1)], count - 1)

    palette_size = 2 if count == 3 else random.randint(2, min(4, count))
    palette = random.sample(SPRITE_COLORS, palette_size)
    colors = [random.choice(palette) for _ in range(count)]
    if len(set(colors)) == 1:
        colors[-1] = random.choice([c for c in palette if c != colors[-1]])

    for (r, c), color in zip(picked, colors):
        sprite[r, c] = int(color)
    return sprite


def _assign_marker_colors(total: int) -> list[int]:
    colors = [random.choice(MARKER_COLORS) for _ in range(total)]
    if len(set(colors)) == 1:
        colors[-1] = random.choice([c for c in MARKER_COLORS if c != colors[-1]])
    return colors


def _draw_frame_and_markers(
    grid: np.ndarray,
    frame: tuple[int, int, int, int],
    row_marks: tuple[int, ...],
    col_marks: tuple[int, ...],
) -> None:
    r0, c0, r1, c1 = frame
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR

    row_colors = _assign_marker_colors(len(row_marks))
    for r, color in zip(row_marks, row_colors):
        side_col = c0 if random.random() < 0.5 else c1
        grid[r, side_col] = int(color)

    col_colors = _assign_marker_colors(len(col_marks))
    for c, color in zip(col_marks, col_colors):
        side_row = r0 if random.random() < 0.5 else r1
        grid[side_row, c] = int(color)


def _make_instance(require_priority: bool, require_adjacent: bool) -> tuple[np.ndarray, dict] | None:
    frame_h = random.randint(10, 14)
    frame_w = random.randint(11, 15)
    top_margin = random.randint(2, 5)
    bottom_margin = random.randint(1, 4)
    left_margin = random.randint(5, 10)
    right_margin = random.randint(1, 3)
    height = top_margin + frame_h + bottom_margin
    width = left_margin + frame_w + right_margin
    if height > 30 or width > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    frame = (top_margin, left_margin, top_margin + frame_h - 1, left_margin + frame_w - 1)
    r0, c0, r1, c1 = frame
    interior_rows = list(range(r0 + 1, r1))
    interior_cols = list(range(c0 + 2, c1))
    row_count = random.randint(2, 3)
    col_count = random.randint(2, 3)
    row_marks = _choose_spaced_positions(interior_rows, row_count, min_gap=2)
    col_marks = _choose_spaced_positions(interior_cols, col_count, min_gap=2)
    if row_marks is None or col_marks is None:
        return None

    start_candidates = [r for r in interior_rows if r not in row_marks]
    if not start_candidates:
        return None
    start_row = random.choice(start_candidates)
    start = (int(start_row), int(c0 + 1))

    sprite = _random_sprite()
    grid[:3, :3] = sprite
    _draw_frame_and_markers(grid, frame, row_marks, col_marks)
    grid[start] = START_COLOR

    path, turns = _simulate_path(grid, frame, start, row_marks, col_marks)
    selected, distances = _selected_intersections(row_marks, col_marks, path)

    row_turns = sum(kind in {"row", "both"} for _r, _c, kind in turns)
    col_turns = sum(kind == "col" for _r, _c, kind in turns)
    exact = [point for point, dist in distances.items() if dist == 0]
    adjacent = [point for point, dist in distances.items() if dist == 1]
    priority_hits = [(r, c) for r, c, kind in turns if kind == "both"]

    if len(turns) < 2 or row_turns < 1 or col_turns < 1:
        return None
    if len(path) < 5:
        return None
    if len(selected) < 2 or len(selected) >= len(distances):
        return None
    if require_adjacent:
        if not adjacent:
            return None
    elif not exact:
        return None
    if require_priority and not priority_hits:
        return None
    painted: set[tuple[int, int]] = set()
    for point in selected:
        stamp_cells = _stamped_cells(sprite, point, frame)
        if painted & stamp_cells:
            return None
        painted.update(stamp_cells)

    output = _build_output(grid, frame, sprite, path, selected)
    if np.any(output[:3, :3] != BG):
        return None
    if np.any(output[r0 + 1 : r1, c0 + 1 : c1] == START_COLOR):
        return None

    meta = {
        "frame": frame,
        "sprite": sprite,
        "path": path,
        "turns": turns,
        "selected": selected,
        "distances": distances,
        "output": output,
        "priority_hits": priority_hits,
    }
    return grid, meta


def _semantic_validate(inp: np.ndarray, meta: dict) -> bool:
    out = main(inp)
    if not np.array_equal(out, meta["output"]):
        return False

    frame = meta["frame"]
    r0, c0, r1, c1 = frame
    for point, dist in meta["distances"].items():
        cr, cc = point
        center_color = int(out[cr, cc])
        if dist <= 1 and center_color == BG and int(meta["sprite"][1, 1]) != BG:
            return False
        if dist > 1 and center_color != BG and int(meta["sprite"][1, 1]) != BG:
            return False

    if np.any(out[:3, :3] != BG):
        return False
    if np.any(out[r0 + 1 : r1, c0 + 1 : c1] == START_COLOR):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        require_priority = random.random() < 0.3
        require_adjacent = random.random() < 0.55
        built = _make_instance(require_priority=require_priority, require_adjacent=require_adjacent)
        if built is None:
            continue
        inp, meta = built
        if _semantic_validate(inp, meta):
            return inp
    raise RuntimeError("failed to generate a valid non-degenerate ARC instance")
