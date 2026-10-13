from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
MARKER_COLORS = (2, 3, 4, 7)

GEN_ATTEMPTS = 400
GEN_SIZE_RANGE = (18, 25)
GEN_LENGTH_RANGE = (3, 6)

_DIR_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
_PERP_VECTORS = {
    "up": (0, 1),
    "down": (0, 1),
    "left": (1, 0),
    "right": (1, 0),
}
_GEN_SCHEDULE = [
    ("vertical", "up", 2),
    ("horizontal", "down", 3),
    ("vertical", "left", 4),
    ("horizontal", "right", 7),
    ("horizontal", "up", 3),
    ("vertical", "down", 4),
    ("horizontal", "left", 7),
    ("vertical", "right", 2),
    ("vertical", "up", 7),
    ("horizontal", "down", 2),
    ("vertical", "left", 3),
    ("horizontal", "right", 4),
    ("horizontal", "up", 4),
    ("vertical", "down", 7),
    ("horizontal", "left", 2),
    ("vertical", "right", 3),
]
_GEN_COUNTER = 0


def _is_consecutive(values: np.ndarray) -> bool:
    vals = np.sort(np.asarray(values, dtype=int))
    return bool(np.all(np.diff(vals) == 1))


def _endpoint_favor_values(start: int, end: int, favor_first: bool) -> list[int]:
    if favor_first:
        hi = (start + end - 1) // 2
        return list(range(start, hi + 1))
    lo = (start + end + 2) // 2
    return list(range(lo, end + 1))


def _line_info_from_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    blue = np.argwhere(g == BLUE)
    assert 3 <= len(blue) <= 6, "expected one blue line segment of length 3-6"

    rows = np.unique(blue[:, 0])
    cols = np.unique(blue[:, 1])
    if len(rows) == 1:
        orientation = "horizontal"
        row = int(rows[0])
        line_cols = np.sort(blue[:, 1])
        assert _is_consecutive(line_cols), "horizontal line must be consecutive"
        c0 = int(line_cols[0])
        c1 = int(line_cols[-1])
        r0 = r1 = row
        endpoints = ((row, c0), (row, c1))
    else:
        assert len(cols) == 1, "line must be horizontal or vertical"
        orientation = "vertical"
        col = int(cols[0])
        line_rows = np.sort(blue[:, 0])
        assert _is_consecutive(line_rows), "vertical line must be consecutive"
        r0 = int(line_rows[0])
        r1 = int(line_rows[-1])
        c0 = c1 = col
        endpoints = ((r0, col), (r1, col))

    expected_cells = (r1 - r0 + 1) * (c1 - c0 + 1)
    assert len(blue) == expected_cells, "blue pixels must form a solid 1-cell-thick segment"

    markers = np.argwhere((g != BLACK) & (g != BLUE))
    assert len(markers) == 1, "expected exactly one marker pixel"
    mr, mc = map(int, markers[0])
    marker_color = int(g[mr, mc])
    assert marker_color in MARKER_COLORS, "marker color must be one of red/green/yellow/orange"

    direction_hits = []
    if mr < r0 - 1 and c0 <= mc <= c1:
        direction_hits.append("up")
    if mr > r1 + 1 and c0 <= mc <= c1:
        direction_hits.append("down")
    if mc < c0 - 1 and r0 <= mr <= r1:
        direction_hits.append("left")
    if mc > c1 + 1 and r0 <= mr <= r1:
        direction_hits.append("right")
    assert len(direction_hits) == 1, "marker must lie in exactly one legal side band"
    direction = direction_hits[0]

    d0 = abs(mr - endpoints[0][0]) + abs(mc - endpoints[0][1])
    d1 = abs(mr - endpoints[1][0]) + abs(mc - endpoints[1][1])
    assert d0 != d1, "marker must identify a unique nearer endpoint"
    apex = endpoints[0] if d0 < d1 else endpoints[1]

    return {
        "shape": tuple(map(int, g.shape)),
        "orientation": orientation,
        "length": int(len(blue)),
        "bbox": (r0, c0, r1, c1),
        "endpoints": endpoints,
        "marker": (mr, mc),
        "marker_color": marker_color,
        "direction": direction,
        "apex": apex,
    }


def _render_output(
    shape: tuple[int, int],
    apex: tuple[int, int],
    length: int,
    direction: str,
    marker_color: int,
    marker: tuple[int, int],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    dr, dc = _DIR_VECTORS[direction]
    pr, pc = _PERP_VECTORS[direction]
    ar, ac = map(int, apex)

    for k in range(int(length)):
        cr = ar + dr * k
        cc = ac + dc * k
        for offset in range(-k, k + 1):
            rr = cr + pr * offset
            rc = cc + pc * offset
            assert 0 <= rr < shape[0] and 0 <= rc < shape[1], "pyramid must fit in the canvas"
            out[rr, rc] = BLUE if (abs(offset) % 2 == 0) else int(marker_color)

    mr, mc = marker
    out[mr, mc] = int(marker_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn a blue line segment into a direction-coded alternating-color pyramid while preserving the marker pixel.

    Input structure:
    - The grid has black background, exactly one blue 1-cell-thick horizontal or vertical line segment, and exactly one non-black non-blue marker pixel.
    - The marker color is one of red, green, yellow, or orange, and its side band relative to the line's bounding box is uniquely one of up, down, left, or right.
    - The marker is far enough from the line that the full pyramid implied by the line length and chosen direction fits inside the grid.

    Rule:
    - Read the blue segment as the source object: its length is the pyramid height.
    - Read the marker's side band as the growth direction and its color as the alternating secondary color.
    - Choose the line endpoint that is strictly nearer to the marker as the apex, then draw rows of lengths 1, 3, ..., 2L-1 along the chosen direction, with each row perpendicular to that direction.
    - Within each drawn row, cells at even distance from the row center are blue and cells at odd distance are the marker color; finally keep the marker pixel itself in its original color.

    Key insight:
    - The single marker pixel is a two-parameter code: its position selects the pyramid orientation and its color selects the second color of the alternating fill.
    """
    info = _line_info_from_grid(np.asarray(grid, dtype=int))
    return _render_output(
        info["shape"],
        info["apex"],
        info["length"],
        info["direction"],
        info["marker_color"],
        info["marker"],
    )


def _sample_line_layout(height: int, width: int, orientation: str, length: int) -> tuple[int, int]:
    if orientation == "vertical":
        top_min, top_max = length, height - 2 * length
        col_min, col_max = length, width - length - 1
        assert top_min <= top_max and col_min <= col_max
        return random.randint(top_min, top_max), random.randint(col_min, col_max)

    row_min, row_max = length, height - length - 1
    left_min, left_max = length, width - 2 * length
    assert row_min <= row_max and left_min <= left_max
    return random.randint(row_min, row_max), random.randint(left_min, left_max)


def _build_line_info(
    height: int,
    width: int,
    orientation: str,
    length: int,
    anchor_a: int,
    anchor_b: int,
) -> dict:
    if orientation == "vertical":
        r0 = int(anchor_a)
        c = int(anchor_b)
        r1 = r0 + length - 1
        endpoints = ((r0, c), (r1, c))
        bbox = (r0, c, r1, c)
    else:
        r = int(anchor_a)
        c0 = int(anchor_b)
        c1 = c0 + length - 1
        endpoints = ((r, c0), (r, c1))
        bbox = (r, c0, r, c1)

    return {
        "shape": (int(height), int(width)),
        "orientation": orientation,
        "length": int(length),
        "bbox": bbox,
        "endpoints": endpoints,
    }


def _marker_for_side(info: dict, side: str, marker_color: int, favor_first: bool) -> tuple[int, int]:
    height, width = info["shape"]
    r0, c0, r1, c1 = info["bbox"]
    orientation = info["orientation"]
    length = info["length"]

    if side == "up":
        mr = random.randint(0, r0 - length)
        if orientation == "horizontal":
            choices = _endpoint_favor_values(c0, c1, favor_first)
            mc = random.choice(choices)
        else:
            mc = c0
        return mr, mc

    if side == "down":
        mr = random.randint(r1 + length, height - 1)
        if orientation == "horizontal":
            choices = _endpoint_favor_values(c0, c1, favor_first)
            mc = random.choice(choices)
        else:
            mc = c0
        return mr, mc

    if side == "left":
        mc = random.randint(0, c0 - length)
        if orientation == "vertical":
            choices = _endpoint_favor_values(r0, r1, favor_first)
            mr = random.choice(choices)
        else:
            mr = r0
        return mr, mc

    mc = random.randint(c1 + length, width - 1)
    if orientation == "vertical":
        choices = _endpoint_favor_values(r0, r1, favor_first)
        mr = random.choice(choices)
    else:
        mr = r0
    return mr, mc


def _build_input(
    info: dict,
    side: str,
    marker_color: int,
    favor_first: bool,
) -> np.ndarray:
    height, width = info["shape"]
    r0, c0, r1, c1 = info["bbox"]
    grid = np.zeros((height, width), dtype=int)
    if info["orientation"] == "vertical":
        grid[r0 : r1 + 1, c0] = BLUE
    else:
        grid[r0, c0 : c1 + 1] = BLUE

    mr, mc = _marker_for_side(info, side, marker_color, favor_first)
    grid[mr, mc] = int(marker_color)
    return grid


def _compare_without_markers(
    a: np.ndarray,
    b: np.ndarray,
    marker_a: tuple[int, int],
    marker_b: tuple[int, int],
) -> bool:
    aa = a.copy()
    bb = b.copy()
    for rr, cc in {tuple(marker_a), tuple(marker_b)}:
        aa[rr, cc] = BLACK
        bb[rr, cc] = BLACK
    return bool(np.array_equal(aa, bb))


def _validate_semantics(grid: np.ndarray) -> bool:
    try:
        info = _line_info_from_grid(grid)
        out = main(grid)
    except AssertionError:
        return False

    expected = _render_output(
        info["shape"],
        info["apex"],
        info["length"],
        info["direction"],
        info["marker_color"],
        info["marker"],
    )
    if not np.array_equal(out, expected):
        return False

    other_colors = [color for color in MARKER_COLORS if color != info["marker_color"]]
    recolored = grid.copy()
    recolored[info["marker"]] = random.choice(other_colors)
    recolored_out = main(recolored)
    if _compare_without_markers(out, recolored_out, info["marker"], info["marker"]):
        return False

    alt_side_outputs = 0
    alt_info_base = {
        "shape": info["shape"],
        "orientation": info["orientation"],
        "length": info["length"],
        "bbox": info["bbox"],
        "endpoints": info["endpoints"],
    }
    for side in ("up", "down", "left", "right"):
        if side == info["direction"]:
            continue
        for favor_first in (True, False):
            alt_grid = _build_input(alt_info_base, side, info["marker_color"], favor_first)
            try:
                alt_out = main(alt_grid)
                alt_marker = tuple(map(int, np.argwhere((alt_grid != BLACK) & (alt_grid != BLUE))[0]))
            except AssertionError:
                continue
            if not _compare_without_markers(out, alt_out, info["marker"], alt_marker):
                alt_side_outputs += 1
                break
    return alt_side_outputs == 3


def _generate_attempt(orientation: str, side: str, marker_color: int) -> np.ndarray | None:
    height = random.randint(*GEN_SIZE_RANGE)
    width = random.randint(*GEN_SIZE_RANGE)
    length = random.randint(*GEN_LENGTH_RANGE)
    if min(height, width) < 3 * length:
        return None

    try:
        anchor_a, anchor_b = _sample_line_layout(height, width, orientation, length)
    except AssertionError:
        return None

    info = _build_line_info(height, width, orientation, length, anchor_a, anchor_b)
    perpendicular = (orientation == "vertical" and side in ("left", "right")) or (
        orientation == "horizontal" and side in ("up", "down")
    )
    favor_first = random.choice((True, False)) if perpendicular else True
    grid = _build_input(info, side, marker_color, favor_first)
    return grid if _validate_semantics(grid) else None


def generate_input() -> np.ndarray:
    global _GEN_COUNTER

    for offset in range(GEN_ATTEMPTS):
        orientation, side, marker_color = _GEN_SCHEDULE[(_GEN_COUNTER + offset) % len(_GEN_SCHEDULE)]
        grid = _generate_attempt(orientation, side, marker_color)
        if grid is not None:
            _GEN_COUNTER = (_GEN_COUNTER + offset + 1) % len(_GEN_SCHEDULE)
            return grid
    raise RuntimeError("generate_input: failed to generate a valid puzzle instance")
