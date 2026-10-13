from __future__ import annotations

import random

import numpy as np


BG = 8
TURN = 9
SEGMENT_COLORS = (1, 2, 3, 4, 5, 6, 7)
MIN_SEGMENTS = 3
MAX_SEGMENTS = 6
MIN_LENGTH = 3
MAX_LENGTH = 6
MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _is_straight_line(coords: list[tuple[int, int]]) -> bool:
    rows = sorted({r for r, _ in coords})
    cols = sorted({c for _, c in coords})
    if len(rows) == 1:
        return cols == list(range(cols[0], cols[-1] + 1))
    if len(cols) == 1:
        return rows == list(range(rows[0], rows[-1] + 1))
    return False


def _parse_segment(grid: np.ndarray, color: int) -> dict:
    comps = _components(grid == color)
    line_comps = [comp for comp in comps if MIN_LENGTH <= len(comp) <= MAX_LENGTH and _is_straight_line(comp)]
    assert len(line_comps) == 1, f"color {color} must have exactly one straight line component"
    assert all(len(comp) == 1 or comp is line_comps[0] for comp in comps), f"color {color} has invalid extra components"

    comp = line_comps[0]
    rows = sorted({r for r, _ in comp})
    cols = sorted({c for _, c in comp})
    if len(rows) == 1:
        axis = "h"
        row = rows[0]
        min_end = (row, cols[0])
        max_end = (row, cols[-1])
        neg_cell = (row, cols[0] - 1)
        pos_cell = (row, cols[-1] + 1)
    else:
        axis = "v"
        col = cols[0]
        min_end = (rows[0], col)
        max_end = (rows[-1], col)
        neg_cell = (rows[0] - 1, col)
        pos_cell = (rows[-1] + 1, col)

    h, w = grid.shape

    def in_bounds(cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < h and 0 <= cell[1] < w

    candidates = []
    if in_bounds(neg_cell) and int(grid[neg_cell]) not in (BG, TURN):
        candidates.append((-1, neg_cell))
    if in_bounds(pos_cell) and int(grid[pos_cell]) not in (BG, TURN):
        candidates.append((1, pos_cell))
    assert len(candidates) == 1, f"segment {color} must have exactly one endpoint connector"
    sign, connector = candidates[0]
    next_color = int(grid[connector])
    assert next_color != color

    if sign == 1:
        entry = min_end
        exit_ = max_end
    else:
        entry = max_end
        exit_ = min_end

    return {
        "color": int(color),
        "axis": axis,
        "length": int(len(comp)),
        "sign": int(sign),
        "entry": entry,
        "exit": exit_,
        "next_color": next_color,
    }


def _assemble_output(segments: list[dict]) -> np.ndarray:
    painted: list[tuple[int, int, int]] = []
    occupied: set[tuple[int, int]] = set()
    joints: list[tuple[int, int]] = []
    current = (0, 0)
    current_axis = segments[0]["axis"]

    for i, seg in enumerate(segments):
        if i > 0:
            current_axis = "v" if current_axis == "h" else "h"
            joints.append(current)
        sign = int(seg["sign"])
        if current_axis == "h":
            delta = (0, sign)
        else:
            delta = (sign, 0)

        cells = [
            (current[0] + delta[0] * step, current[1] + delta[1] * step)
            for step in range(int(seg["length"]))
        ]
        for j, cell in enumerate(cells):
            if j == 0 and i > 0:
                assert cell == current
                continue
            assert cell not in occupied, "assembled snake must not self-intersect"
        for cell in cells:
            occupied.add(cell)
            painted.append((cell[0], cell[1], int(seg["color"])))
        current = cells[-1]

    rows = [r for r, _, _ in painted]
    cols = [c for _, c, _ in painted]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for r, c, color in painted:
        out[r - r0, c - c0] = int(color)
    for r, c in joints:
        out[r - r0, c - c0] = TURN
    assert max(out.shape) <= 30
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Chain scattered colored line segments into one compact snake path and recolor each elbow maroon.

    Input structure:
    - bg = azure.
    - A unique maroon start pixel sits in the top-left corner area.
    - Every other foreground color owns exactly one straight horizontal or vertical segment of length 3-6.
    - Each segment has one singleton connector pixel immediately outside one endpoint, and that connector's color names the next segment.

    Rule:
    - Find the segment with minimum Manhattan distance to the maroon start pixel and use it as the first path segment.
    - Traverse every segment from its endpoint without a connector to its endpoint with the connector, then follow the connector color to the next segment.
    - Reassemble the chain into a tight-cropped path: keep the first segment's axis, force each later segment to alternate between horizontal and vertical, and align each next segment's start endpoint to the previous segment's end endpoint.
    - Preserve each segment's start-to-end direction when it is moved, and recolor every shared turning cell between consecutive segments to maroon.

    Key insight:
    - The endpoint connector encodes both chain order and segment direction, while the rebuilt path must alternate axes at every step.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    starts = list(map(tuple, np.argwhere(g == TURN)))
    assert len(starts) == 1, "expected exactly one maroon start pixel"
    start = starts[0]

    colors = sorted(set(map(int, np.unique(g))) - {BG, TURN})
    assert MIN_SEGMENTS <= len(colors) <= MAX_SEGMENTS
    segments = {color: _parse_segment(g, color) for color in colors}

    def dist_to_segment(seg: dict) -> int:
        if seg["axis"] == "h":
            row = seg["entry"][0]
            c0 = min(seg["entry"][1], seg["exit"][1])
            c1 = max(seg["entry"][1], seg["exit"][1])
            return min(abs(start[0] - row) + abs(start[1] - c) for c in range(c0, c1 + 1))
        col = seg["entry"][1]
        r0 = min(seg["entry"][0], seg["exit"][0])
        r1 = max(seg["entry"][0], seg["exit"][0])
        return min(abs(start[0] - r) + abs(start[1] - col) for r in range(r0, r1 + 1))

    ranked = sorted((dist_to_segment(seg), color) for color, seg in segments.items())
    assert len(ranked) >= 1
    assert len(ranked) == 1 or ranked[0][0] < ranked[1][0], "nearest segment must be unique"
    first_color = ranked[0][1]

    order = [first_color]
    used = {first_color}
    while len(order) < len(segments):
        nxt = int(segments[order[-1]]["next_color"])
        assert nxt in segments, f"connector points to unknown segment color {nxt}"
        assert nxt not in used, "connector cycle closed before covering every segment"
        order.append(nxt)
        used.add(nxt)

    return _assemble_output([segments[color] for color in order])


def _normalize_shape(line_cells: list[tuple[int, int]], connector: tuple[int, int]) -> dict:
    all_cells = line_cells + [connector]
    min_r = min(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)
    norm_line = [(r - min_r, c - min_c) for r, c in line_cells]
    norm_connector = (connector[0] - min_r, connector[1] - min_c)
    norm_all = norm_line + [norm_connector]
    return {
        "line": norm_line,
        "connector": norm_connector,
        "all": norm_all,
        "height": max(r for r, _ in norm_all) + 1,
        "width": max(c for _, c in norm_all) + 1,
    }


def _input_shape(axis: str, length: int, sign: int) -> dict:
    if axis == "h":
        line = [(0, c) for c in range(length)]
        connector = (0, length) if sign == 1 else (0, -1)
    else:
        line = [(r, 0) for r in range(length)]
        connector = (length, 0) if sign == 1 else (-1, 0)
    return _normalize_shape(line, connector)


def _translate(shape: dict, top: int, left: int) -> dict:
    return {
        "line": [(top + r, left + c) for r, c in shape["line"]],
        "connector": (top + shape["connector"][0], left + shape["connector"][1]),
        "all": [(top + r, left + c) for r, c in shape["all"]],
    }


def _can_place(shape: dict, top: int, left: int, occupied: set[tuple[int, int]], h: int, w: int) -> bool:
    placed = _translate(shape, top, left)
    for r, c in placed["all"]:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in occupied:
            return False
    for r, c in placed["all"]:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in occupied:
                    return False
    return True


def _candidate_positions(
    shape: dict,
    occupied: set[tuple[int, int]],
    h: int,
    w: int,
    *,
    top_min: int,
    top_max: int,
    left_min: int,
    left_max: int,
    min_line_dist: int,
) -> list[tuple[int, int, int]]:
    candidates: list[tuple[int, int, int]] = []
    top_max = min(top_max, h - shape["height"])
    left_max = min(left_max, w - shape["width"])
    for top in range(top_min, top_max + 1):
        for left in range(left_min, left_max + 1):
            if not _can_place(shape, top, left, occupied, h, w):
                continue
            placed = _translate(shape, top, left)
            line_dist = min(r + c for r, c in placed["line"])
            if line_dist < min_line_dist:
                continue
            candidates.append((top, left, line_dist))
    return candidates


def _render_input(h: int, w: int, segments: list[dict], placements: dict[int, dict]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    grid[0, 0] = TURN
    for seg in segments:
        placed = placements[int(seg["color"])]
        for r, c in placed["line"]:
            grid[r, c] = int(seg["color"])
        cr, cc = placed["connector"]
        grid[cr, cc] = int(seg["next_color"])
    return grid


def _sample_segments() -> list[dict] | None:
    count = random.randint(MIN_SEGMENTS, MAX_SEGMENTS)
    colors = random.sample(list(SEGMENT_COLORS), count)
    segments = []
    for i, color in enumerate(colors):
        segments.append(
            {
                "color": int(color),
                "axis": random.choice(("h", "v")),
                "length": random.randint(MIN_LENGTH, MAX_LENGTH),
                "sign": random.choice((-1, 1)),
                "next_color": int(colors[(i + 1) % count]),
            }
        )
    try:
        _assemble_output(segments)
    except AssertionError:
        return None
    return segments


def _place_segments(segments: list[dict], h: int, w: int) -> np.ndarray | None:
    occupied = {(0, 0)}
    placements: dict[int, dict] = {}

    first_shape = _input_shape(segments[0]["axis"], int(segments[0]["length"]), int(segments[0]["sign"]))
    first_candidates = _candidate_positions(
        first_shape,
        occupied,
        h,
        w,
        top_min=1,
        top_max=5,
        left_min=1,
        left_max=7,
        min_line_dist=3,
    )
    if not first_candidates:
        return None
    first_top, first_left, first_dist = random.choice(first_candidates)
    first_placed = _translate(first_shape, first_top, first_left)
    placements[int(segments[0]["color"])] = first_placed
    occupied.update(first_placed["all"])

    rest = segments[1:]
    rest = sorted(
        rest,
        key=lambda seg: (-max(_input_shape(seg["axis"], int(seg["length"]), int(seg["sign"]))["height"],
                              _input_shape(seg["axis"], int(seg["length"]), int(seg["sign"]))["width"]),
                         int(seg["color"])),
    )
    for seg in rest:
        shape = _input_shape(seg["axis"], int(seg["length"]), int(seg["sign"]))
        candidates = _candidate_positions(
            shape,
            occupied,
            h,
            w,
            top_min=0,
            top_max=h - 1,
            left_min=0,
            left_max=w - 1,
            min_line_dist=first_dist + 4,
        )
        if not candidates:
            return None
        top, left, _dist = random.choice(candidates)
        placed = _translate(shape, top, left)
        placements[int(seg["color"])] = placed
        occupied.update(placed["all"])

    ordered = [placements[int(seg["color"])] for seg in segments]
    if len(ordered) != len(segments):
        return None
    return _render_input(h, w, segments, placements)


def _gen_attempt() -> np.ndarray | None:
    segments = _sample_segments()
    if segments is None:
        return None

    expected = _assemble_output(segments)
    h = random.randint(max(12, expected.shape[0] + 6), 30)
    w = random.randint(max(12, expected.shape[1] + 8), 30)
    inp = _place_segments(segments, h, w)
    if inp is None:
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid chained-segment puzzle input")
