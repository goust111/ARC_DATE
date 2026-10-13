from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
PATH = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # clockwise: U, R, D, L
DIR_INDEX = {d: i for i, d in enumerate(DIRS)}

GEN_RETRIES = 400
PATH_BUILD_RETRIES = 120
PLACEMENT_RETRIES = 120
SEGMENT_COUNT_RANGE = (2, 6)
SEGMENT_STEP_RANGE = (2, 5)
MAX_PATH_SPAN = 20


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS:
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc), h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _legend_cells(corner: str, h: int, w: int) -> set[tuple[int, int]]:
    if corner == "tl":
        rows = (0, 1)
        cols = (0, 1)
    elif corner == "tr":
        rows = (0, 1)
        cols = (w - 2, w - 1)
    elif corner == "bl":
        rows = (h - 2, h - 1)
        cols = (0, 1)
    else:
        rows = (h - 2, h - 1)
        cols = (w - 2, w - 1)
    return {(int(r), int(c)) for r in rows for c in cols}


def _reserved_corner_zone(corner: str, h: int, w: int) -> set[tuple[int, int]]:
    if corner == "tl":
        rows = range(0, min(3, h))
        cols = range(0, min(3, w))
    elif corner == "tr":
        rows = range(0, min(3, h))
        cols = range(max(0, w - 3), w)
    elif corner == "bl":
        rows = range(max(0, h - 3), h)
        cols = range(0, min(3, w))
    else:
        rows = range(max(0, h - 3), h)
        cols = range(max(0, w - 3), w)
    return {(int(r), int(c)) for r in rows for c in cols}


def _detect_legend(grid: np.ndarray, bg: int) -> tuple[str, int, int, set[tuple[int, int]]]:
    h, w = grid.shape
    assert h >= 4 and w >= 4
    candidates: list[tuple[str, int, int, set[tuple[int, int]]]] = []
    for corner in ("tl", "tr", "bl", "br"):
        cells = _legend_cells(corner, h, w)
        rows = sorted({r for r, _ in cells})
        cols = sorted({c for _, c in cells})
        block = grid[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
        if block.shape != (2, 2):
            continue
        a = int(block[0, 0])
        b = int(block[0, 1])
        c = int(block[1, 0])
        d = int(block[1, 1])
        if bg in (a, b, c, d):
            continue
        if a != d or b != c or a == b:
            continue
        candidates.append((corner, a, b, cells))
    assert len(candidates) == 1, f"expected one corner legend, got {len(candidates)}"
    return candidates[0]


def _extract_path(
    grid: np.ndarray, bg: int, legend: set[tuple[int, int]]
) -> tuple[set[tuple[int, int]], tuple[int, int], int]:
    mask = grid != int(bg)
    for r, c in legend:
        mask[int(r), int(c)] = False
    comps = [comp for comp in _components(mask) if len(comp) >= 2]
    assert comps, "expected a non-legend path component"
    path = max(comps, key=len)
    path_set = {(int(r), int(c)) for r, c in path}

    color_counts = Counter(int(grid[r, c]) for r, c in path)
    path_color = int(color_counts.most_common(1)[0][0])
    start_cells = [(int(r), int(c)) for r, c in path if int(grid[r, c]) != int(path_color)]
    assert len(start_cells) == 1, f"expected one colored start endpoint, got {len(start_cells)}"
    start = start_cells[0]

    endpoints: list[tuple[int, int]] = []
    for r, c in path:
        deg = sum((nr, nc) in path_set for nr, nc in _neighbors4(int(r), int(c), *grid.shape))
        if deg == 1:
            endpoints.append((int(r), int(c)))
        else:
            assert deg == 2, f"path must be a simple polyline, got degree {deg}"
    assert len(endpoints) == 2, f"expected exactly two endpoints, got {len(endpoints)}"
    assert start in endpoints, "colored start must be an endpoint"

    return path_set, start, path_color


def _ordered_path(path_set: set[tuple[int, int]], start: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    order = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxts = [p for p in _neighbors4(cur[0], cur[1], h, w) if p in path_set and p != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "path traversal became ambiguous"
        prev, cur = cur, nxts[0]
        order.append(cur)
    assert len(order) == len(path_set), "path traversal did not cover the full path"
    return order


def _turn_kind(in_dir: tuple[int, int], out_dir: tuple[int, int]) -> str:
    delta = (DIR_INDEX[out_dir] - DIR_INDEX[in_dir]) % 4
    assert delta in (1, 3), f"expected a 90-degree turn, got delta={delta}"
    return "right" if delta == 1 else "left"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a single grey polyline by propagating a signal that switches color at each turn according to a corner legend.

    Input structure:
    - The background is black.
    - One thin orthogonal path forms a single 4-connected polyline; most of it is grey and one endpoint is recolored to mark the start.
    - One corner contains a 2x2 legend whose two diagonal colors encode left-turn and right-turn updates.

    Rule:
    - Read the corner legend: the top-left/bottom-right diagonal color means "switch to this color after a left turn", and the other diagonal color means the right-turn color.
    - Traverse the path from the unique non-grey endpoint, recoloring the current straight segment with the current signal color.
    - When a bend is reached, determine whether the path turns left or right relative to travel direction, switch the signal to the legend color for that turn, then recolor the next segment with the new signal; recolor the terminal endpoint with the final signal and place one same-color arrow pixel on the boundary in the forward direction.

    Key insight:
    - The 2x2 legend is not split by rows or columns: its diagonals encode left-versus-right turn behavior, so the path color depends on local turn direction rather than on position alone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = _bg_color(g)

    _corner, left_color, right_color, legend = _detect_legend(g, bg)
    path_set, start, _path_color = _extract_path(g, bg, legend)
    order = _ordered_path(path_set, start, (h, w))
    assert len(order) >= 2, "path must contain at least two cells"

    out = g.copy()
    signal = int(g[start[0], start[1]])
    dirs = [
        (int(order[i + 1][0] - order[i][0]), int(order[i + 1][1] - order[i][1]))
        for i in range(len(order) - 1)
    ]

    segment_start = 0
    for i in range(1, len(order) - 1):
        if dirs[i - 1] == dirs[i]:
            continue
        for j in range(segment_start, i + 1):
            rr, cc = order[j]
            out[int(rr), int(cc)] = int(signal)
        turn = _turn_kind(dirs[i - 1], dirs[i])
        signal = int(left_color if turn == "left" else right_color)
        segment_start = i + 1

    for j in range(segment_start, len(order)):
        rr, cc = order[j]
        out[int(rr), int(cc)] = int(signal)

    end_r, end_c = order[-1]
    dr, dc = dirs[-1]
    if dr < 0:
        arrow = (0, int(end_c))
    elif dr > 0:
        arrow = (h - 1, int(end_c))
    elif dc < 0:
        arrow = (int(end_r), 0)
    else:
        arrow = (int(end_r), w - 1)
    if arrow == (end_r, end_c):
        arrow = (int(end_r + dr), int(end_c + dc))
    if 0 <= arrow[0] < h and 0 <= arrow[1] < w:
        assert arrow not in path_set and arrow not in legend, "arrow cell must be separate from the path and legend"
        out[int(arrow[0]), int(arrow[1])] = int(signal)

    return out


def _build_relative_path() -> list[tuple[int, int]] | None:
    num_segments = random.randint(*SEGMENT_COUNT_RANGE)
    turns = [random.choice((-1, 1)) for _ in range(num_segments - 1)]
    start_dir = random.randrange(4)
    occupied = {(0, 0)}
    order = [(0, 0)]

    def rec(seg_idx: int, current: tuple[int, int], dir_idx: int) -> bool:
        steps_choices = list(range(SEGMENT_STEP_RANGE[0], SEGMENT_STEP_RANGE[1] + 1))
        random.shuffle(steps_choices)
        for steps in steps_choices:
            added: list[tuple[int, int]] = []
            prev = current
            dr, dc = DIRS[dir_idx]
            ok = True
            for _ in range(steps):
                nxt = (int(prev[0] + dr), int(prev[1] + dc))
                if nxt in occupied or abs(nxt[0]) > MAX_PATH_SPAN or abs(nxt[1]) > MAX_PATH_SPAN:
                    ok = False
                    break
                for nbr_dr, nbr_dc in DIRS:
                    nbr = (int(nxt[0] + nbr_dr), int(nxt[1] + nbr_dc))
                    if nbr in occupied and nbr != prev:
                        ok = False
                        break
                if not ok:
                    break
                occupied.add(nxt)
                order.append(nxt)
                added.append(nxt)
                prev = nxt
            if ok:
                rows = [r for r, _ in occupied]
                cols = [c for _, c in occupied]
                if max(rows) - min(rows) + 1 <= MAX_PATH_SPAN and max(cols) - min(cols) + 1 <= MAX_PATH_SPAN:
                    if seg_idx == num_segments - 1:
                        return True
                    next_dir = (dir_idx + turns[seg_idx]) % 4
                    if rec(seg_idx + 1, prev, next_dir):
                        return True
            for cell in reversed(added):
                order.pop()
                occupied.remove(cell)
        return False

    if rec(0, (0, 0), start_dir):
        return order
    return None


def _boundary_arrow(end: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    r, c = end
    dr, dc = direction
    if dr < 0:
        return (0, int(c))
    if dr > 0:
        return (h - 1, int(c))
    if dc < 0:
        return (int(r), 0)
    return (int(r), w - 1)


def _place_path(
    relative_order: list[tuple[int, int]], corner: str
) -> tuple[int, int, list[tuple[int, int]], tuple[int, int]] | None:
    rows = [r for r, _ in relative_order]
    cols = [c for _, c in relative_order]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    path_h = max_r - min_r + 1
    path_w = max_c - min_c + 1
    final_dir = (
        int(relative_order[-1][0] - relative_order[-2][0]),
        int(relative_order[-1][1] - relative_order[-2][1]),
    )

    min_h = max(10, path_h + 2)
    min_w = max(10, path_w + 2)
    max_h = min(30, min_h + 8)
    max_w = min(30, min_w + 8)
    if min_h > max_h or min_w > max_w:
        return None

    for _ in range(PLACEMENT_RETRIES):
        h = random.randint(min_h, max_h)
        w = random.randint(min_w, max_w)
        top = random.randint(1, h - path_h - 1)
        left = random.randint(1, w - path_w - 1)
        order = [(int(r - min_r + top), int(c - min_c + left)) for r, c in relative_order]
        path_set = set(order)
        reserved = _reserved_corner_zone(corner, h, w)
        if path_set & reserved:
            continue
        arrow = _boundary_arrow(order[-1], final_dir, (h, w))
        if arrow in path_set or arrow in reserved:
            continue
        return h, w, order, arrow
    return None


def _legend_block(corner: str, h: int, w: int, left_color: int, right_color: int) -> list[tuple[int, int, int]]:
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - 2
    elif corner == "bl":
        r0, c0 = h - 2, 0
    else:
        r0, c0 = h - 2, w - 2
    return [
        (int(r0), int(c0), int(left_color)),
        (int(r0), int(c0 + 1), int(right_color)),
        (int(r0 + 1), int(c0), int(right_color)),
        (int(r0 + 1), int(c0 + 1), int(left_color)),
    ]


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        relative_order = None
        for _ in range(PATH_BUILD_RETRIES):
            relative_order = _build_relative_path()
            if relative_order is not None:
                break
        if relative_order is None:
            continue

        corner = random.choice(("tl", "tr", "bl", "br"))
        placed = _place_path(relative_order, corner)
        if placed is None:
            continue
        h, w, order, arrow = placed

        start_color, left_color, right_color = random.sample(PALETTE, 3)
        grid = np.full((h, w), BG, dtype=int)
        for r, c in order:
            grid[int(r), int(c)] = PATH
        sr, sc = order[0]
        grid[int(sr), int(sc)] = int(start_color)
        for r, c, color in _legend_block(corner, h, w, left_color, right_color):
            grid[int(r), int(c)] = int(color)

        if arrow in _legend_cells(corner, h, w):
            continue

        try:
            out = main(grid)
        except AssertionError:
            continue
        if out.shape != grid.shape or np.array_equal(out, grid):
            continue
        return grid

    raise RuntimeError(f"Failed to generate a valid input after {GEN_RETRIES} attempts")
