import random
from collections import Counter

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_separator_row(grid: np.ndarray, bg: int) -> tuple[int, int]:
    for r in range(grid.shape[0]):
        row = grid[r]
        if row[0] != bg and np.all(row == row[0]):
            return r, int(row[0])
    raise AssertionError("separator row not found")


def _extract_legend_order(grid: np.ndarray, bg: int, sep_color: int, sep_row: int) -> list[int]:
    best_row = None
    best_cols: list[int] = []
    for r in range(sep_row):
        cols = [c for c, v in enumerate(grid[r]) if int(v) not in (bg, sep_color)]
        if len(cols) > len(best_cols):
            best_row = r
            best_cols = cols
    assert best_row is not None and best_cols, "legend row not found"
    return [int(grid[best_row, c]) for c in best_cols]


def _l_candidate(
    pts: set[tuple[int, int]], occupied: np.ndarray, bend_row: int, bend_col: int
) -> set[tuple[int, int]] | None:
    if any(r != bend_row and c != bend_col for r, c in pts):
        return None

    row_cols = sorted(c for r, c in pts if r == bend_row)
    col_rows = sorted(r for r, c in pts if c == bend_col)
    if not row_cols or not col_rows:
        return None

    if max(row_cols) <= bend_col:
        row_end = min(row_cols)
    elif min(row_cols) >= bend_col:
        row_end = max(row_cols)
    else:
        return None

    if max(col_rows) <= bend_row:
        col_end = min(col_rows)
    elif min(col_rows) >= bend_row:
        col_end = max(col_rows)
    else:
        return None

    if row_end == bend_col or col_end == bend_row:
        return None

    cells: set[tuple[int, int]] = set()
    for c in range(min(bend_col, row_end), max(bend_col, row_end) + 1):
        if not occupied[bend_row, c]:
            return None
        cells.add((bend_row, c))
    for r in range(min(bend_row, col_end), max(bend_row, col_end) + 1):
        if not occupied[r, bend_col]:
            return None
        cells.add((r, bend_col))
    return cells


def _reconstruct_path(
    pts: set[tuple[int, int]], occupied: np.ndarray
) -> set[tuple[int, int]]:
    assert pts, "path color must remain visible somewhere"

    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    row_counts = Counter(rows)
    col_counts = Counter(cols)

    if len(row_counts) == 1:
        row = rows[0]
        c0, c1 = min(cols), max(cols)
        cells = {(row, c) for c in range(c0, c1 + 1)}
        if all(occupied[row, c] for c in range(c0, c1 + 1)):
            return cells

    if len(col_counts) == 1:
        col = cols[0]
        r0, r1 = min(rows), max(rows)
        cells = {(r, col) for r in range(r0, r1 + 1)}
        if all(occupied[r, col] for r in range(r0, r1 + 1)):
            return cells

    strong_rows = [r for r, count in row_counts.items() if count >= 2]
    strong_cols = [c for c, count in col_counts.items() if count >= 2]
    if len(strong_rows) == 1 and len(strong_cols) == 1:
        cells = _l_candidate(pts, occupied, strong_rows[0], strong_cols[0])
        if cells is not None:
            return cells

    candidates: list[tuple[tuple[int, int, int], set[tuple[int, int]]]] = []
    for bend_row in row_counts:
        for bend_col in col_counts:
            cells = _l_candidate(pts, occupied, bend_row, bend_col)
            if cells is None:
                continue
            support = row_counts[bend_row] + col_counts[bend_col]
            extra = len(cells) - len(pts)
            score = (extra, -support, len(cells))
            candidates.append((score, cells))
    assert candidates, "failed to reconstruct path"
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _solve_canvas(canvas: np.ndarray, legend_order: list[int], bg: int) -> np.ndarray:
    occupied = canvas != bg
    masks: dict[int, set[tuple[int, int]]] = {}
    for color in legend_order:
        pts = {(int(r), int(c)) for r, c in np.argwhere(canvas == color)}
        masks[color] = _reconstruct_path(pts, occupied)

    out = np.full(canvas.shape, bg, dtype=int)
    for color in reversed(legend_order):
        for r, c in masks[color]:
            out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-render crossed single-pixel paths using the priority order encoded by the legend row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected 2D grid"

    bg = _most_common_color(g)
    sep_row, sep_color = _find_separator_row(g, bg)
    legend_order = _extract_legend_order(g, bg, sep_color, sep_row)
    canvas = g[sep_row + 1 :, :]
    return _solve_canvas(canvas, legend_order, bg)


# --------------- generator ---------------

CANVAS_H_RANGE = (8, 14)
CANVAS_W_RANGE = (9, 15)
PATH_COUNT_CHOICES = (2, 3, 4)
PATH_TYPES = ("H", "V", "L")
STRAIGHT_LEN_MIN = 5
L_ARM_LEN_MIN = 3
TOP_BLANK_MAX = 2
MID_BLANK_MAX = 1
MAX_GENERATION_ATTEMPTS = 400
SEPARATOR_COLOR = 5


def _make_h_path(color: int, row: int, c0: int, c1: int) -> dict:
    cells = {(row, c) for c in range(c0, c1 + 1)}
    return {
        "type": "H",
        "color": color,
        "row": row,
        "col": None,
        "cells": cells,
        "endpoints": {(row, c0), (row, c1)},
        "bend": None,
    }


def _make_v_path(color: int, col: int, r0: int, r1: int) -> dict:
    cells = {(r, col) for r in range(r0, r1 + 1)}
    return {
        "type": "V",
        "color": color,
        "row": None,
        "col": col,
        "cells": cells,
        "endpoints": {(r0, col), (r1, col)},
        "bend": None,
    }


def _make_l_path(color: int, bend_row: int, bend_col: int, row_end: int, col_end: int) -> dict:
    cells = {(bend_row, c) for c in range(min(bend_col, row_end), max(bend_col, row_end) + 1)}
    cells.update((r, bend_col) for r in range(min(bend_row, col_end), max(bend_row, col_end) + 1))
    return {
        "type": "L",
        "color": color,
        "row": bend_row,
        "col": bend_col,
        "cells": cells,
        "endpoints": {(bend_row, row_end), (col_end, bend_col)},
        "bend": (bend_row, bend_col),
    }


def _pairwise_compatible(path_a: dict, path_b: dict) -> bool:
    shared = path_a["cells"] & path_b["cells"]
    if len(shared) > 1:
        return False
    if not shared:
        return True
    pt = next(iter(shared))
    blocked_a = set(path_a["endpoints"])
    blocked_b = set(path_b["endpoints"])
    if path_a["bend"] is not None:
        blocked_a.add(path_a["bend"])
    if path_b["bend"] is not None:
        blocked_b.add(path_b["bend"])
    return pt not in blocked_a and pt not in blocked_b


def _sample_path(
    rng: random.Random,
    color: int,
    kind: str,
    height: int,
    width: int,
    used_rows: set[int],
    used_cols: set[int],
    existing: list[dict],
) -> dict | None:
    for _ in range(200):
        if kind == "H":
            row_choices = [r for r in range(1, height - 1) if r not in used_rows]
            if not row_choices:
                return None
            row = rng.choice(row_choices)
            max_len = width - 2
            if max_len < STRAIGHT_LEN_MIN:
                return None
            length = rng.randint(STRAIGHT_LEN_MIN, max_len)
            c0 = rng.randint(1, width - length - 1)
            path = _make_h_path(color, row, c0, c0 + length - 1)
        elif kind == "V":
            col_choices = [c for c in range(1, width - 1) if c not in used_cols]
            if not col_choices:
                return None
            col = rng.choice(col_choices)
            max_len = height - 2
            if max_len < STRAIGHT_LEN_MIN:
                return None
            length = rng.randint(STRAIGHT_LEN_MIN, max_len)
            r0 = rng.randint(1, height - length - 1)
            path = _make_v_path(color, col, r0, r0 + length - 1)
        else:
            row_choices = [r for r in range(1, height - 1) if r not in used_rows]
            col_choices = [c for c in range(1, width - 1) if c not in used_cols]
            if not row_choices or not col_choices:
                return None
            bend_row = rng.choice(row_choices)
            bend_col = rng.choice(col_choices)
            hdirs = [-1, 1]
            vdirs = [-1, 1]
            rng.shuffle(hdirs)
            rng.shuffle(vdirs)
            found = None
            for hdir in hdirs:
                if hdir < 0:
                    max_h = bend_col - 1
                else:
                    max_h = width - 2 - bend_col
                if max_h < L_ARM_LEN_MIN:
                    continue
                hlen = rng.randint(L_ARM_LEN_MIN, max_h)
                row_end = bend_col + hdir * hlen
                for vdir in vdirs:
                    if vdir < 0:
                        max_v = bend_row - 1
                    else:
                        max_v = height - 2 - bend_row
                    if max_v < L_ARM_LEN_MIN:
                        continue
                    vlen = rng.randint(L_ARM_LEN_MIN, max_v)
                    col_end = bend_row + vdir * vlen
                    found = (row_end, col_end)
                    break
                if found is not None:
                    break
            if found is None:
                continue
            row_end, col_end = found
            path = _make_l_path(color, bend_row, bend_col, row_end, col_end)

        if all(_pairwise_compatible(path, other) for other in existing):
            return path
    return None


def _render_canvas(paths: list[dict], draw_order: list[int], shape: tuple[int, int], bg: int) -> np.ndarray:
    canvas = np.full(shape, bg, dtype=int)
    for idx in draw_order:
        path = paths[idx]
        for r, c in path["cells"]:
            canvas[r, c] = path["color"]
    return canvas


def _assemble_grid(
    rng: random.Random,
    canvas: np.ndarray,
    legend_order: list[int],
    bg: int,
) -> np.ndarray:
    top_blank = rng.randint(0, TOP_BLANK_MAX)
    mid_blank = rng.randint(0, MID_BLANK_MAX)
    legend_row = top_blank
    sep_row = legend_row + 1 + mid_blank
    total_h = canvas.shape[0] + sep_row + 1
    total_w = canvas.shape[1]
    grid = np.full((total_h, total_w), bg, dtype=int)

    legend_width = 2 * len(legend_order) - 1
    start_col = rng.randint(1, total_w - legend_width - 1)
    for i, color in enumerate(legend_order):
        grid[legend_row, start_col + 2 * i] = color

    grid[sep_row, :] = SEPARATOR_COLOR
    grid[sep_row + 1 :, :] = canvas
    return grid


def _visibility_ok(path: dict, visible_pts: set[tuple[int, int]]) -> bool:
    if path["type"] == "H" or path["type"] == "V":
        return len(visible_pts) >= 3
    row = path["row"]
    col = path["col"]
    row_vis = [pt for pt in visible_pts if pt[0] == row]
    col_vis = [pt for pt in visible_pts if pt[1] == col]
    row_counts = Counter(r for r, _ in visible_pts)
    col_counts = Counter(c for _, c in visible_pts)
    return (
        len(row_vis) >= 2
        and len(col_vis) >= 2
        and sum(count >= 2 for count in row_counts.values()) == 1
        and sum(count >= 2 for count in col_counts.values()) == 1
    )


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    path_count = rng.choice(PATH_COUNT_CHOICES)
    min_width = max(CANVAS_W_RANGE[0], 2 * path_count + 1)
    canvas_h = rng.randint(*CANVAS_H_RANGE)
    canvas_w = rng.randint(min_width, CANVAS_W_RANGE[1])

    types = [rng.choice(PATH_TYPES) for _ in range(path_count)]
    if not any(t in ("H", "L") for t in types):
        return None
    if not any(t in ("V", "L") for t in types):
        return None

    palette = [c for c in range(10) if c != SEPARATOR_COLOR]
    bg = rng.choice(palette)
    colors = [c for c in palette if c != bg]
    rng.shuffle(colors)
    path_colors = colors[:path_count]

    paths: list[dict] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    for color, kind in zip(path_colors, types):
        path = _sample_path(rng, color, kind, canvas_h, canvas_w, used_rows, used_cols, paths)
        if path is None:
            return None
        paths.append(path)
        if path["row"] is not None:
            used_rows.add(path["row"])
        if path["col"] is not None:
            used_cols.add(path["col"])

    intersections = [path_a["cells"] & path_b["cells"] for i, path_a in enumerate(paths) for path_b in paths[i + 1 :]]
    overlap_count = sum(1 for shared in intersections if shared)
    if overlap_count == 0:
        return None

    legend_order = path_colors[:]
    rng.shuffle(legend_order)
    desired_priority = {color: idx for idx, color in enumerate(legend_order)}
    desired_order = sorted(range(path_count), key=lambda idx: desired_priority[paths[idx]["color"]], reverse=True)
    desired_canvas = _render_canvas(paths, desired_order, (canvas_h, canvas_w), bg)

    input_canvas = None
    for _ in range(20):
        draw_order = list(range(path_count))
        rng.shuffle(draw_order)
        candidate = _render_canvas(paths, draw_order, (canvas_h, canvas_w), bg)
        if not np.array_equal(candidate, desired_canvas):
            input_canvas = candidate
            break
    if input_canvas is None:
        return None

    for path in paths:
        color = path["color"]
        visible_pts = {(int(r), int(c)) for r, c in np.argwhere(input_canvas == color)}
        if not _visibility_ok(path, visible_pts):
            return None

    grid = _assemble_grid(rng, input_canvas, legend_order, bg)
    if _most_common_color(grid) != bg:
        return None
    if not np.array_equal(main(grid), desired_canvas):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
