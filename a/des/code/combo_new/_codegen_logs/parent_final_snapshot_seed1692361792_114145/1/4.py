from __future__ import annotations

import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8

DIR_COLOR_TO_VEC = {
    AZURE: (-1, 0),
    ORANGE: (1, 0),
    YELLOW: (0, -1),
    MAGENTA: (0, 1),
}
DIR_TO_COLOR = {vec: color for color, vec in DIR_COLOR_TO_VEC.items()}
DIRECTIONS = tuple(DIR_TO_COLOR)

ROW_COLOR_POOL = (1, 2, 3, 9)

ARROW_TEMPLATES = {
    (-1, 0): frozenset({(0, 1), (1, 0), (1, 2), (2, 1), (3, 1)}),
    (1, 0): frozenset({(0, 1), (1, 1), (2, 0), (2, 2), (3, 1)}),
    (0, -1): frozenset({(1, 0), (0, 1), (2, 1), (1, 2), (1, 3)}),
    (0, 1): frozenset({(1, 0), (1, 1), (0, 2), (2, 2), (1, 3)}),
}
BODY_TEMPLATES = {
    (-1, 0): ((1, 1), (2, 1)),
    (1, 0): ((0, 1), (1, 1)),
    (0, -1): ((1, 1), (1, 2)),
    (0, 1): ((1, 0), (1, 1)),
}


def _neighbors(r: int, c: int, connectivity: int) -> tuple[tuple[int, int], ...]:
    if connectivity == 4:
        return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
    return (
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    )


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _find_target_region(
    grid: np.ndarray,
) -> tuple[int, int, list[int], list[int]]:
    candidates: list[tuple[int, int, int, list[int], list[int]]] = []
    for pts in _components(grid == GREY, connectivity=4):
        r0, c0, r1, c1 = _bbox(pts)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        sh, sw = sub.shape
        if sh < 9 or sw < 9:
            continue
        row_lines = [i for i in range(sh) if np.all(sub[i, :] == GREY)]
        col_lines = [j for j in range(sw) if np.all(sub[:, j] == GREY)]
        if len(row_lines) < 3 or len(col_lines) < 3:
            continue
        if row_lines[0] != 0 or row_lines[-1] != sh - 1:
            continue
        if col_lines[0] != 0 or col_lines[-1] != sw - 1:
            continue
        if any(b - a != 4 for a, b in zip(row_lines, row_lines[1:], strict=False)):
            continue
        if any(b - a != 4 for a, b in zip(col_lines, col_lines[1:], strict=False)):
            continue
        row_set = set(row_lines)
        col_set = set(col_lines)
        ok = True
        for rr, cc in np.argwhere(sub == GREY):
            if int(rr) not in row_set and int(cc) not in col_set:
                ok = False
                break
        if not ok:
            continue
        candidates.append((len(pts), r0, c0, [r0 + rr for rr in row_lines], [c0 + cc for cc in col_lines]))

    assert candidates, "expected one grey grid region"
    max_area = max(item[0] for item in candidates)
    best = [item for item in candidates if item[0] == max_area]
    assert len(best) == 1, "grey target region must be unique"
    _, r0, c0, row_lines, col_lines = best[0]
    return r0, c0, row_lines, col_lines


def _read_row_legend(
    grid: np.ndarray, r0: int, c0: int, row_lines: list[int]
) -> dict[int, int]:
    assert c0 >= 1, "row legend requires one column to the left of the frame"
    mapping: dict[int, int] = {}
    for i in range(len(row_lines) - 1):
        rr = row_lines[i] + 2
        color = int(grid[rr, c0 - 1])
        assert color not in (BLACK, GREY), "legend pixel must be a non-background non-grey color"
        assert color not in mapping, "row legend colors must be unique"
        mapping[color] = i
    return mapping


def _read_column_hints(
    grid: np.ndarray, r0: int, col_lines: list[int]
) -> dict[tuple[int, int], int]:
    assert r0 >= 1, "direction hints require one row above the frame"
    mapping: dict[tuple[int, int], int] = {}
    for j in range(len(col_lines) - 1):
        cc = col_lines[j] + 2
        color = int(grid[r0 - 1, cc])
        assert color in DIR_COLOR_TO_VEC, "unexpected direction-hint color"
        direction = DIR_COLOR_TO_VEC[color]
        assert direction not in mapping, "each direction should map to a unique column"
        mapping[direction] = j
    return mapping


def _detect_arrows(grid: np.ndarray) -> list[tuple[int, tuple[int, int]]]:
    arrows: list[tuple[int, tuple[int, int]]] = []
    for color in np.unique(grid):
        color = int(color)
        if color in (BLACK, GREY):
            continue
        for pts in _components(grid == color, connectivity=8):
            if len(pts) != 5:
                continue
            r0, c0, _r1, _c1 = _bbox(pts)
            shape = frozenset((r - r0, c - c0) for r, c in pts)
            if shape in ARROW_TEMPLATES.values():
                for direction, template in ARROW_TEMPLATES.items():
                    if shape == template:
                        arrows.append((color, direction))
                        break
    return arrows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the grey target table with arrow shafts routed by direction hints and row-color legend pixels.

    Input structure:
    - The background is black.
    - One grey rectangular target area is subdivided by grey internal grid lines into equal 3x3 cells.
    - A one-pixel color legend sits immediately to the left of the target area, one color per row.
    - One-pixel direction hints sit immediately above the target columns, and 5-pixel monochrome arrow objects are scattered elsewhere.

    Rule:
    - Detect the target table, read the left legend as a color-to-row mapping, and read the top hints as a direction-to-column mapping.
    - For each 5-pixel arrow, infer its facing direction from its geometry and use its color to choose the destination row.
    - Clear the interiors of all target cells to black, then draw only the arrow's 2-pixel shaft into the matched 3x3 cell while preserving the shaft orientation.
    - If multiple arrows choose the same cell, their identical shaft pixels overlap harmlessly.

    Key insight:
    - The arrow encodes two different pieces of information at once: its geometry selects a column via the top direction code, while its color selects a row via the side legend.
    """
    g = np.asarray(grid, dtype=int)
    r0, c0, row_lines, col_lines = _find_target_region(g)
    row_by_color = _read_row_legend(g, r0, c0, row_lines)
    col_by_dir = _read_column_hints(g, r0, col_lines)

    out = g.copy()
    for i in range(len(row_lines) - 1):
        for j in range(len(col_lines) - 1):
            out[row_lines[i] + 1 : row_lines[i + 1], col_lines[j] + 1 : col_lines[j + 1]] = BLACK

    for color, direction in _detect_arrows(g):
        if color not in row_by_color or direction not in col_by_dir:
            continue
        cell_r = row_by_color[color]
        cell_c = col_by_dir[direction]
        base_r = row_lines[cell_r] + 1
        base_c = col_lines[cell_c] + 1
        for dr, dc in BODY_TEMPLATES[direction]:
            out[base_r + dr, base_c + dc] = color

    return out


def _draw_target(
    grid: np.ndarray,
    top: int,
    left: int,
    row_colors: list[int],
    dir_order: list[tuple[int, int]],
) -> tuple[int, int]:
    rows = len(row_colors)
    cols = len(dir_order)
    frame_h = 4 * rows + 1
    frame_w = 4 * cols + 1
    bottom = top + frame_h - 1
    right = left + frame_w - 1

    for i in range(rows + 1):
        rr = top + 4 * i
        grid[rr, left : right + 1] = GREY
    for j in range(cols + 1):
        cc = left + 4 * j
        grid[top : bottom + 1, cc] = GREY

    for i, color in enumerate(row_colors):
        grid[top + 2 + 4 * i, left - 1] = color
    for j, direction in enumerate(dir_order):
        grid[top - 1, left + 2 + 4 * j] = DIR_TO_COLOR[direction]

    return bottom, right


def _clear_target_cells(grid: np.ndarray, top: int, left: int, rows: int, cols: int) -> None:
    for i in range(rows):
        for j in range(cols):
            grid[top + 1 + 4 * i : top + 4 + 4 * i, left + 1 + 4 * j : left + 4 + 4 * j] = BLACK


def _paint_expected(
    inp: np.ndarray,
    top: int,
    left: int,
    row_colors: list[int],
    dir_order: list[tuple[int, int]],
    assignments: list[tuple[int, tuple[int, int]]],
) -> np.ndarray:
    expected = inp.copy()
    _clear_target_cells(expected, top, left, len(row_colors), len(dir_order))
    dir_to_col = {direction: j for j, direction in enumerate(dir_order)}
    for row_index, direction in assignments:
        base_r = top + 1 + 4 * row_index
        base_c = left + 1 + 4 * dir_to_col[direction]
        color = row_colors[row_index]
        for dr, dc in BODY_TEMPLATES[direction]:
            expected[base_r + dr, base_c + dc] = color
    return expected


def _sample_assignments(row_count: int) -> list[tuple[int, tuple[int, int]]]:
    assignments: list[tuple[int, tuple[int, int]]] = []
    for direction in random.sample(list(DIRECTIONS), len(DIRECTIONS)):
        assignments.append((random.randrange(row_count), direction))
    missing_rows = [row for row in range(row_count) if all(row != used_row for used_row, _ in assignments)]
    for row in missing_rows:
        assignments.append((row, random.choice(DIRECTIONS)))
    for _ in range(random.randint(0, row_count + 2)):
        assignments.append((random.randrange(row_count), random.choice(DIRECTIONS)))
    if random.random() < 0.5:
        dup_count = random.randint(1, min(2, len(assignments)))
        for index in random.sample(range(len(assignments)), dup_count):
            assignments.append(assignments[index])
    random.shuffle(assignments)
    return assignments


def _place_arrow(
    grid: np.ndarray,
    blocked: np.ndarray,
    color: int,
    direction: tuple[int, int],
) -> bool:
    template = ARROW_TEMPLATES[direction]
    ph = 1 + max(r for r, _ in template)
    pw = 1 + max(c for _, c in template)
    h, w = grid.shape
    positions = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
    random.shuffle(positions)

    for top, left in positions:
        br0 = max(0, top - 1)
        bc0 = max(0, left - 1)
        br1 = min(h - 1, top + ph)
        bc1 = min(w - 1, left + pw)
        if np.any(blocked[br0 : br1 + 1, bc0 : bc1 + 1]):
            continue
        for dr, dc in template:
            grid[top + dr, left + dc] = color
        blocked[br0 : br1 + 1, bc0 : bc1 + 1] = True
        return True
    return False


def _generate_once() -> np.ndarray | None:
    row_count = random.randint(2, len(ROW_COLOR_POOL))
    col_count = 4
    row_colors = random.sample(list(ROW_COLOR_POOL), row_count)
    dir_order = random.sample(list(DIRECTIONS), col_count)

    frame_h = 4 * row_count + 1
    frame_w = 4 * col_count + 1
    grid_h = random.randint(frame_h + 6, 30)
    grid_w = random.randint(frame_w + 6, 30)
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)

    max_top = grid_h - frame_h - 1
    max_left = grid_w - frame_w - 1
    if max_top < 1 or max_left < 2:
        return None

    mode = random.choice(("left", "top"))
    if mode == "left":
        top = random.randint(1, max_top)
        left = random.randint(2, min(4, max_left))
    else:
        top = random.randint(1, min(3, max_top))
        left = random.randint(2, max_left)

    bottom, right = _draw_target(grid, top, left, row_colors, dir_order)
    assignments = _sample_assignments(row_count)

    blocked = np.zeros_like(grid, dtype=bool)
    blocked[max(0, top - 2) : min(grid_h, bottom + 3), max(0, left - 2) : min(grid_w, right + 3)] = True
    for row_index, direction in assignments:
        if not _place_arrow(grid, blocked, row_colors[row_index], direction):
            return None

    expected = _paint_expected(grid, top, left, row_colors, dir_order, assignments)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        sample = _generate_once()
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate a valid sample")
