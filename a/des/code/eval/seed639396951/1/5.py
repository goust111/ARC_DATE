import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
END = 7
START = 9
SEGMENT_PALETTE = (1, 2, 3, 4, 6, 8)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
HORIZONTAL_DIRS = ((0, -1), (0, 1))
VERTICAL_DIRS = ((-1, 0), (1, 0))

GEN_ATTEMPTS = 400
GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
SEGMENT_COUNT_RANGE = (3, 6)
SEGMENT_LENGTH_CHOICES = (2, 3, 3, 4, 4, 5)
STRAIGHT_DISTANCE_RANGE = (2, 5)
L_LEG_RANGE = (1, 4)
TOP_MARGIN_RANGE = (4, 8)
LEFT_MARGIN_RANGE = (1, 5)
BOTTOM_MARGIN_RANGE = (1, 4)
RIGHT_MARGIN_RANGE = (1, 5)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _reverse(direction: tuple[int, int] | None) -> tuple[int, int] | None:
    if direction is None:
        return None
    return (-direction[0], -direction[1])


def _line_inclusive(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        step = 1 if b[1] > a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    if a[1] == b[1]:
        step = 1 if b[0] > a[0] else -1
        return [(r, a[1]) for r in range(a[0], b[0] + step, step)]
    raise ValueError("line must be horizontal or vertical")


def _direct_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    return _line_inclusive(a, b)[1:-1]


def _hv_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    turn = (a[0], b[1])
    leg1 = _line_inclusive(a, turn)
    leg2 = _line_inclusive(turn, b)[1:]
    return (leg1 + leg2)[1:-1]


def _vh_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    turn = (b[0], a[1])
    leg1 = _line_inclusive(a, turn)
    leg2 = _line_inclusive(turn, b)[1:]
    return (leg1 + leg2)[1:-1]


def _find_legend(grid: np.ndarray) -> list[int]:
    grey_comps = _components(grid == GREY)
    candidates: list[tuple[tuple[int, int, int], list[int]]] = []

    for comp in grey_comps:
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 4:
            continue
        if comp.shape[0] != 2 * h + 2 * w - 4:
            continue
        ok = True
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                is_border = r in (r0, r1) or c in (c0, c1)
                if is_border != bool(grid[r, c] == GREY):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue

        interior = [
            (c, r, int(grid[r, c]))
            for r in range(r0 + 1, r1)
            for c in range(c0 + 1, c1)
            if int(grid[r, c]) not in (BG, GREY)
        ]
        if len(interior) < 2:
            continue
        colors = [color for _c, _r, color in sorted(interior)]
        candidates.append(((r0 + c0, r0, c0), colors))

    assert candidates, "legend frame not found"
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _segment_endpoints(coords: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    if int(rows.min()) == int(rows.max()):
        ordered = sorted(((int(r), int(c)) for r, c in coords), key=lambda p: p[1])
        return ordered[0], ordered[-1]
    if int(cols.min()) == int(cols.max()):
        ordered = sorted(((int(r), int(c)) for r, c in coords), key=lambda p: p[0])
        return ordered[0], ordered[-1]
    raise AssertionError("segment must be straight")


def _ordered_segments(grid: np.ndarray, colors: list[int]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    ordered: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for color in colors:
        comps = _components(grid == color)
        segs = [comp for comp in comps if comp.shape[0] >= 2]
        assert len(segs) == 1, "each legend color must have exactly one main-grid segment"
        ordered.append(_segment_endpoints(segs[0]))
    return ordered


def _singleton(grid: np.ndarray, color: int) -> tuple[int, int]:
    pts = np.argwhere(grid == color)
    assert pts.shape[0] == 1, f"expected exactly one cell of color {color}"
    return int(pts[0, 0]), int(pts[0, 1])


def _passable(out: np.ndarray, drawn: set[tuple[int, int]], cell: tuple[int, int]) -> bool:
    r, c = cell
    return int(out[r, c]) == BG or cell in drawn


def _choose_path(
    out: np.ndarray,
    drawn: set[tuple[int, int]],
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[tuple[int, int]] | None:
    if start == end:
        return []

    if start[0] == end[0] or start[1] == end[1]:
        path = _direct_path(start, end)
        return path if all(_passable(out, drawn, cell) for cell in path) else None

    for path in (_hv_path(start, end), _vh_path(start, end)):
        if all(_passable(out, drawn, cell) for cell in path):
            return path
    return None


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _solve_mask(
    grid: np.ndarray,
    segments: list[tuple[tuple[int, int], tuple[int, int]]],
    start: tuple[int, int],
    end: tuple[int, int],
    mask_bits: tuple[int, ...],
) -> tuple[int, np.ndarray] | None:
    out = np.array(grid, dtype=int, copy=True)
    drawn: set[tuple[int, int]] = set()
    current = start
    total_cost = 0

    for idx, bit in enumerate(mask_bits):
        entry = segments[idx][bit]
        exit_ = segments[idx][1 - bit]
        total_cost += _manhattan(current, entry)
        path = _choose_path(out, drawn, current, entry)
        if path is None:
            return None
        for cell in path:
            out[cell] = GREY
        drawn.update(path)
        current = exit_

    total_cost += _manhattan(current, end)
    path = _choose_path(out, drawn, current, end)
    if path is None:
        return None
    for cell in path:
        out[cell] = GREY
    drawn.update(path)
    return total_cost, out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    legend_colors = _find_legend(g)
    segments = _ordered_segments(g, legend_colors)
    start = _singleton(g, START)
    end = _singleton(g, END)

    best_key: tuple[int, tuple[int, ...]] | None = None
    best_out: np.ndarray | None = None
    n = len(segments)

    for mask in range(1 << n):
        bits = tuple((mask >> idx) & 1 for idx in range(n))
        solved = _solve_mask(g, segments, start, end, bits)
        if solved is None:
            continue
        total_cost, out = solved
        key = (total_cost, bits)
        if best_key is None or key < best_key:
            best_key = key
            best_out = out

    assert best_out is not None, "no valid endpoint assignment found"
    return best_out


def _all_local_cells(layout: dict) -> list[tuple[int, int]]:
    cells = [layout["start"], layout["end"]]
    for connector in layout["connectors"]:
        cells.extend(connector)
    for segment in layout["segments"]:
        cells.extend(segment["cells"])
    return cells


def _can_use(points: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    return all(point not in occupied for point in points)


def _sample_connector(
    current: tuple[int, int],
    prev_dir: tuple[int, int] | None,
    occupied: set[tuple[int, int]],
    rng: random.Random,
) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]] | None:
    for _ in range(40):
        use_l = rng.random() < 0.5
        reverse_prev = _reverse(prev_dir)

        if not use_l:
            dirs = [direction for direction in DIRS4 if direction != reverse_prev]
            direction = rng.choice(dirs)
            dist = rng.randint(*STRAIGHT_DISTANCE_RANGE)
            target = (current[0] + direction[0] * dist, current[1] + direction[1] * dist)
            path = _direct_path(current, target)
            if _can_use(path + [target], occupied):
                return path, target, direction
            continue

        hdirs = [direction for direction in HORIZONTAL_DIRS if direction != reverse_prev]
        if not hdirs:
            continue
        hdir = rng.choice(hdirs)
        vdir = rng.choice(VERTICAL_DIRS)
        leg_h = rng.randint(*L_LEG_RANGE)
        leg_v = rng.randint(*L_LEG_RANGE)
        target = (current[0] + vdir[0] * leg_v, current[1] + hdir[1] * leg_h)
        path = _hv_path(current, target)
        if _can_use(path + [target], occupied):
            return path, target, vdir
    return None


def _sample_segment(
    entry: tuple[int, int],
    arrival_dir: tuple[int, int],
    occupied: set[tuple[int, int]],
    rng: random.Random,
) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    for _ in range(40):
        seg_dirs = [direction for direction in DIRS4 if direction != _reverse(arrival_dir)]
        seg_dir = rng.choice(seg_dirs)
        seg_len = rng.choice(SEGMENT_LENGTH_CHOICES)
        cells = [
            (entry[0] + seg_dir[0] * step, entry[1] + seg_dir[1] * step)
            for step in range(seg_len)
        ]
        if _can_use(cells, occupied):
            return cells, seg_dir
    return None


def _build_local_layout(colors: list[int], rng: random.Random) -> dict | None:
    start = (0, 0)
    occupied: set[tuple[int, int]] = {start}
    current = start
    prev_dir: tuple[int, int] | None = None
    connectors: list[list[tuple[int, int]]] = []
    segments: list[dict] = []

    for color in colors:
        connector = _sample_connector(current, prev_dir, occupied, rng)
        if connector is None:
            return None
        path_cells, entry, arrival_dir = connector
        segment = _sample_segment(entry, arrival_dir, occupied, rng)
        if segment is None:
            return None
        seg_cells, seg_dir = segment

        occupied.update(path_cells)
        occupied.update(seg_cells)
        connectors.append(path_cells)
        segments.append({"color": color, "cells": seg_cells})
        current = seg_cells[-1]
        prev_dir = seg_dir

    connector = _sample_connector(current, prev_dir, occupied, rng)
    if connector is None:
        return None
    path_cells, end, _arrival_dir = connector
    occupied.update(path_cells)
    occupied.add(end)
    connectors.append(path_cells)

    return {
        "start": start,
        "end": end,
        "connectors": connectors,
        "segments": segments,
    }


def _render_layout(layout: dict, colors: list[int], rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    legend_w = len(colors) + 2
    legend_h = 3

    cells = np.array(_all_local_cells(layout), dtype=int)
    r0, c0, r1, c1 = _bbox(cells)
    bbox_h = r1 - r0 + 1
    bbox_w = c1 - c0 + 1

    top_margin = rng.randint(*TOP_MARGIN_RANGE)
    left_margin = rng.randint(*LEFT_MARGIN_RANGE)
    bottom_margin = rng.randint(*BOTTOM_MARGIN_RANGE)
    right_margin = rng.randint(*RIGHT_MARGIN_RANGE)

    h = top_margin + bbox_h + bottom_margin
    w = max(legend_w + 1, left_margin + bbox_w + right_margin)
    if h < GRID_H_RANGE[0]:
        h = GRID_H_RANGE[0]
    if w < GRID_W_RANGE[0]:
        w = GRID_W_RANGE[0]
    if h > GRID_H_RANGE[1] or w > GRID_W_RANGE[1]:
        return None

    row_shift = top_margin - r0
    col_shift = left_margin - c0

    out = np.full((h, w), BG, dtype=int)
    out[0, :legend_w] = GREY
    out[legend_h - 1, :legend_w] = GREY
    out[:legend_h, 0] = GREY
    out[:legend_h, legend_w - 1] = GREY
    for idx, color in enumerate(colors):
        out[1, idx + 1] = color

    connector_cells: list[tuple[int, int]] = []
    sr, sc = layout["start"]
    out[sr + row_shift, sc + col_shift] = START
    er, ec = layout["end"]
    out[er + row_shift, ec + col_shift] = END

    for connector in layout["connectors"]:
        for r, c in connector:
            gr = r + row_shift
            gc = c + col_shift
            out[gr, gc] = GREY
            connector_cells.append((gr, gc))

    for segment in layout["segments"]:
        color = int(segment["color"])
        for r, c in segment["cells"]:
            out[r + row_shift, c + col_shift] = color

    inp = np.array(out, copy=True)
    for r, c in connector_cells:
        inp[r, c] = BG
    return inp, out


def _input_structure_ok(grid: np.ndarray, colors: list[int]) -> bool:
    if int((grid == START).sum()) != 1 or int((grid == END).sum()) != 1:
        return False
    grey_comps = _components(grid == GREY)
    if len(grey_comps) != 1:
        return False
    for color in colors:
        comps = _components(grid == color)
        if len(comps) != 2:
            return False
        sizes = sorted(int(comp.shape[0]) for comp in comps)
        if sizes[0] != 1 or sizes[1] < 2:
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.randint(*SEGMENT_COUNT_RANGE)
    colors = rng.sample(SEGMENT_PALETTE, count)
    layout = _build_local_layout(colors, rng)
    if layout is None:
        return None

    rendered = _render_layout(layout, colors, rng)
    if rendered is None:
        return None
    inp, out = rendered

    if not _input_structure_ok(inp, colors):
        return None
    if np.array_equal(inp, out):
        return None
    try:
        pred = main(inp)
    except Exception:
        return None
    if not np.array_equal(pred, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
