import random
from collections import Counter, deque

import numpy as np


DIR_FROM_EDGE = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}

GEN_RETRIES = 400
GROUP_RETRIES = 240
SEGMENT_RETRIES = 200
GRID_H_RANGE = (14, 28)
GRID_W_RANGE = (14, 28)
GROUP_COUNT_BY_SIZE = {
    0: (1, 2),
    18: (2, 3),
    24: (3, 4),
}
SEGMENT_COUNT_RANGE = (2, 5)
SEGMENT_LENGTH_MIN = 2
SEGMENT_LENGTH_CAP = 8
INPUT_MARGIN = 1


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in grid.ravel())
    return int(max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0])


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _is_straight_segment(comp: list[tuple[int, int]]) -> bool:
    rows = {r for r, _ in comp}
    cols = {c for _, c in comp}
    if len(rows) == 1:
        ordered = sorted(c for _, c in comp)
        return ordered == list(range(ordered[0], ordered[0] + len(ordered)))
    if len(cols) == 1:
        ordered = sorted(r for r, _ in comp)
        return ordered == list(range(ordered[0], ordered[0] + len(ordered)))
    return False


def _marker_edge(r: int, c: int, h: int, w: int) -> str:
    if r == 0:
        return "top"
    if r == h - 1:
        return "bottom"
    if c == 0:
        return "left"
    if c == w - 1:
        return "right"
    raise AssertionError("marker must be on the grid edge")


def _choose_axis_sign(coord: int, span: int, seg_len: int) -> tuple[int, str]:
    steps = int(seg_len) - 1
    pos_ok = coord + steps < span
    neg_ok = coord - steps >= 0
    assert pos_ok or neg_ok, "segment cannot fit in either direction"
    if pos_ok and not neg_ok:
        return 1, "forced"
    if neg_ok and not pos_ok:
        return -1, "forced"
    center = (span - 1) / 2.0
    if coord < center:
        return 1, "center"
    if coord > center:
        return -1, "center"
    return -1, "center"


def _append_segment(
    path: list[tuple[int, int]],
    used: set[tuple[int, int]],
    dr: int,
    dc: int,
    seg_len: int,
) -> bool:
    prev = path[-1]
    for step in range(1, int(seg_len)):
        nxt = (prev[0] + dr * step, prev[1] + dc * step)
        if nxt in used:
            return False
        for adj in _neighbors4(*nxt):
            if adj != path[-1] and adj in used:
                return False
        path.append(nxt)
        used.add(nxt)
    return True


def _build_polyline(
    shape: tuple[int, int],
    start: tuple[int, int],
    lengths: list[int],
) -> tuple[list[tuple[int, int]] | None, list[str]]:
    h, w = shape
    edge = _marker_edge(start[0], start[1], h, w)
    first_dr, first_dc = DIR_FROM_EDGE[edge]
    path = [start]
    used = {start}
    decisions: list[str] = []

    if not _append_segment(path, used, first_dr, first_dc, int(lengths[0])):
        return None, decisions

    prev_axis = "v" if first_dr != 0 else "h"
    for seg_len in lengths[1:]:
        cur_r, cur_c = path[-1]
        if prev_axis == "v":
            sign, why = _choose_axis_sign(cur_c, w, int(seg_len))
            if not _append_segment(path, used, 0, sign, int(seg_len)):
                return None, decisions
            prev_axis = "h"
        else:
            sign, why = _choose_axis_sign(cur_r, h, int(seg_len))
            if not _append_segment(path, used, sign, 0, int(seg_len)):
                return None, decisions
            prev_axis = "v"
        decisions.append(why)

    return path, decisions


def _group_specs(grid: np.ndarray) -> tuple[int, dict[int, dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    specs: dict[int, dict[str, object]] = {}

    for color in sorted(int(v) for v in np.unique(g) if int(v) != int(bg)):
        comps = _components_of_color(g, color)
        marker_candidates = [
            comp[0]
            for comp in comps
            if len(comp) == 1 and (comp[0][0] in (0, h - 1) or comp[0][1] in (0, w - 1))
        ]
        assert len(marker_candidates) == 1, f"expected exactly one edge marker for color {color}"
        marker = marker_candidates[0]

        lengths: list[int] = []
        for comp in comps:
            if len(comp) == 1 and comp[0] == marker:
                continue
            assert _is_straight_segment(comp), f"non-marker component for color {color} must be straight"
            lengths.append(int(len(comp)))
        assert lengths, f"color {color} must have at least one line segment"
        specs[color] = {"marker": marker, "lengths": sorted(lengths)}

    return bg, specs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconnect each color's scattered line fragments into one edge-started polyline ordered by fragment length.

    Input structure:
    - The background is the dominant color.
    - Each non-background color has one isolated single-pixel marker on the grid edge and several separate 4-connected straight line segments elsewhere.
    - Segments are width-1 horizontal or vertical monochrome components; multiple segments can share a color.
    - The marker is not 4-adjacent to any same-color segment and its edge position determines the first drawing direction.

    Rule:
    - For each color, collect all same-color line segments and sort them by pixel length from shortest to longest.
    - Start at that color's edge marker and draw the shortest segment first in the inward direction implied by the touched edge.
    - Draw each later segment from the previous endpoint, alternating orientation between vertical and horizontal.
    - When a later segment could extend in either sign along its axis, choose the sign that stays inside the grid; if both signs fit, choose the one pointing toward the grid center.
    - Remove the original scattered segments and markers, and keep only the reconstructed concatenated polyline for each color.

    Key insight:
    - The spatial placement of the input fragments is irrelevant: the solution groups fragments by color, orders them by length, then uses the edge marker only as the start point and initial direction for rebuilding the path.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg, specs = _group_specs(g)
    out = np.full((h, w), bg, dtype=int)

    for color in sorted(specs):
        marker = tuple(specs[color]["marker"])
        lengths = list(specs[color]["lengths"])
        path, _decisions = _build_polyline((h, w), marker, lengths)
        assert path is not None, f"could not reconstruct path for color {color}"
        for r, c in path:
            if int(out[r, c]) not in (int(bg), int(color)):
                raise AssertionError("reconstructed polylines must not overlap")
            out[r, c] = int(color)

    return out


def _dilate_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    expanded.add((rr, cc))
    return expanded


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]] | set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _desired_group_count(h: int, w: int) -> tuple[int, int]:
    smaller = min(h, w)
    bounds = GROUP_COUNT_BY_SIZE[0]
    for threshold, value in GROUP_COUNT_BY_SIZE.items():
        if smaller >= threshold:
            bounds = value
    return bounds


def _edge_positions(edge: str, h: int, w: int) -> list[tuple[int, int]]:
    if edge == "top":
        return [(0, c) for c in range(1, w - 1)]
    if edge == "bottom":
        return [(h - 1, c) for c in range(1, w - 1)]
    if edge == "left":
        return [(r, 0) for r in range(1, h - 1)]
    return [(r, w - 1) for r in range(1, h - 1)]


def _sample_lengths(h: int, w: int) -> list[int]:
    max_len = min(SEGMENT_LENGTH_CAP, max(SEGMENT_LENGTH_MIN + 2, min(h, w) - 3))
    count_hi = min(SEGMENT_COUNT_RANGE[1], max_len - SEGMENT_LENGTH_MIN + 1)
    count = random.randint(SEGMENT_COUNT_RANGE[0], count_hi)
    return sorted(random.sample(range(SEGMENT_LENGTH_MIN, max_len + 1), count))


def _sample_group(
    shape: tuple[int, int],
    reserved_out: set[tuple[int, int]],
    branch_mode: str,
) -> dict[str, object] | None:
    h, w = shape
    edges = ["top", "bottom", "left", "right"]
    random.shuffle(edges)
    for _ in range(GROUP_RETRIES):
        edge = random.choice(edges)
        marker = random.choice(_edge_positions(edge, h, w))
        lengths = _sample_lengths(h, w)
        path, decisions = _build_polyline(shape, marker, lengths)
        if path is None:
            continue
        if branch_mode == "forced" and "forced" not in decisions:
            continue
        if branch_mode == "center" and "center" not in decisions:
            continue
        path_set = set(path)
        halo = _dilate_cells(path_set, shape)
        if halo & reserved_out:
            continue
        return {
            "marker": marker,
            "lengths": lengths,
            "path": path,
            "path_set": path_set,
            "halo": halo,
        }
    return None


def _can_place_object(
    cells: list[tuple[int, int]] | set[tuple[int, int]],
    shape: tuple[int, int],
    reserved: set[tuple[int, int]],
) -> bool:
    h, w = shape
    cell_set = set(cells)
    for r, c in cell_set:
        if not (0 <= r < h and 0 <= c < w):
            return False
    return not (_dilate_cells(cell_set, shape) & reserved)


def _random_segment_cells(shape: tuple[int, int], length: int) -> list[tuple[int, int]] | None:
    h, w = shape
    orientation = random.choice(["h", "v"])
    if orientation == "h":
        if w < length + 2:
            return None
        r = random.randint(INPUT_MARGIN, h - 1 - INPUT_MARGIN)
        c0 = random.randint(INPUT_MARGIN, w - INPUT_MARGIN - length)
        return [(r, c0 + k) for k in range(length)]
    if h < length + 2:
        return None
    r0 = random.randint(INPUT_MARGIN, h - INPUT_MARGIN - length)
    c = random.randint(INPUT_MARGIN, w - 1 - INPUT_MARGIN)
    return [(r0 + k, c) for k in range(length)]


def _place_input_objects(
    shape: tuple[int, int],
    groups: list[dict[str, object]],
    colors: list[int],
    bg: int,
) -> np.ndarray | None:
    h, w = shape
    inp = np.full((h, w), bg, dtype=int)
    reserved: set[tuple[int, int]] = set()

    order = list(range(len(groups)))
    random.shuffle(order)
    for idx in order:
        group = groups[idx]
        color = colors[idx]
        marker = [tuple(group["marker"])]
        if not _can_place_object(marker, shape, reserved):
            return None
        _paint_cells(inp, marker, color)
        reserved |= _dilate_cells(set(marker), shape)

    segment_jobs: list[tuple[int, int]] = []
    for idx, group in enumerate(groups):
        for seg_len in sorted(group["lengths"], reverse=True):
            segment_jobs.append((idx, int(seg_len)))
    random.shuffle(segment_jobs)

    for idx, seg_len in segment_jobs:
        color = colors[idx]
        placed = False
        for _ in range(SEGMENT_RETRIES):
            cells = _random_segment_cells(shape, seg_len)
            if cells is None or not _can_place_object(cells, shape, reserved):
                continue
            _paint_cells(inp, cells, color)
            reserved |= _dilate_cells(set(cells), shape)
            placed = True
            break
        if not placed:
            return None

    return inp


def _build_expected_output(
    shape: tuple[int, int],
    groups: list[dict[str, object]],
    colors: list[int],
    bg: int,
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for group, color in zip(groups, colors):
        _paint_cells(out, group["path"], color)
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    bg = random.randint(0, 9)
    lo, hi = _desired_group_count(h, w)
    group_count = random.randint(lo, hi)
    colors = random.sample([c for c in range(10) if c != bg], group_count)

    groups: list[dict[str, object]] = []
    reserved_out: set[tuple[int, int]] = set()

    for idx in range(group_count):
        branch_mode = "forced" if idx % 2 == 0 else "center"
        group = _sample_group((h, w), reserved_out, branch_mode)
        if group is None:
            return None
        groups.append(group)
        reserved_out |= set(group["halo"])

    expected = _build_expected_output((h, w), groups, colors, bg)
    inp = _place_input_objects((h, w), groups, colors, bg)
    if inp is None:
        return None

    if Counter(int(v) for v in inp.ravel())[bg] <= max(
        count for color, count in Counter(int(v) for v in inp.ravel()).items() if color != bg
    ):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
