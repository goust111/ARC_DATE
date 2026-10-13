from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
GRAY = 5
SEGMENT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_MIN = 18
GRID_MAX = 30
FRAME_OUTER_MIN = 7
FRAME_OUTER_MAX = 15
FRAME_MARGIN_MIN = 3
MAX_SEGMENTS = 5
MAX_SEGMENT_LEN = 7
GEN_RETRIES = 400
PLACE_RETRIES = 200


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(coords)
    rows = [r for r, _c in pts]
    cols = [c for _r, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _components_by_color(grid: np.ndarray, ignore: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in ignore or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    frame = np.argwhere(grid == GRAY)
    assert frame.size > 0, "expected a gray frame"
    r0, c0 = frame.min(axis=0)
    r1, c1 = frame.max(axis=0)
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose an interior"

    expected = np.zeros_like(grid, dtype=bool)
    expected[r0 : r1 + 1, c0] = True
    expected[r0 : r1 + 1, c1] = True
    expected[r0, c0 : c1 + 1] = True
    expected[r1, c0 : c1 + 1] = True
    assert np.array_equal(grid == GRAY, expected), "gray pixels must form exactly one rectangular border"
    return int(r0), int(c0), int(r1), int(c1)


def _segment_shape(coords: list[tuple[int, int]]) -> tuple[str, list[tuple[int, int]], int]:
    rows = sorted({r for r, _c in coords})
    cols = sorted({c for _r, c in coords})
    if len(coords) == 1:
        return "point", [coords[0]], 1
    if len(rows) == 1:
        row = rows[0]
        assert cols == list(range(cols[0], cols[-1] + 1)), "horizontal segment must be contiguous"
        return "horizontal", [(row, cols[0]), (row, cols[-1])], len(cols)
    if len(cols) == 1:
        col = cols[0]
        assert rows == list(range(rows[0], rows[-1] + 1)), "vertical segment must be contiguous"
        return "vertical", [(rows[0], col), (rows[-1], col)], len(rows)
    raise AssertionError("each colored object must be a straight line segment")


def _touch_info(
    endpoints: list[tuple[int, int]],
    frame_mask: np.ndarray,
) -> tuple[tuple[int, int], str]:
    h, w = frame_mask.shape
    hits: list[tuple[tuple[int, int], str]] = []
    for r, c in endpoints:
        for side, (dr, dc) in (
            ("bottom", (-1, 0)),
            ("top", (1, 0)),
            ("right", (0, -1)),
            ("left", (0, 1)),
        ):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and frame_mask[nr, nc]:
                hits.append(((r, c), side))
    assert len(hits) == 1, "exactly one segment endpoint must touch the frame"
    return hits[0]


def _fold_spec(
    frame_bbox: tuple[int, int, int, int],
    side: str,
    hinge_row_or_col: int,
) -> tuple[tuple[int, int], tuple[int, int], int]:
    fr0, fc0, fr1, fc1 = frame_bbox
    ir0, ic0, ir1, ic1 = fr0 + 1, fc0 + 1, fr1 - 1, fc1 - 1
    assert ir0 <= ir1 and ic0 <= ic1

    if side == "left":
        row = hinge_row_or_col
        assert ir0 <= row <= ir1
        start = (row, ic0)
        up = row - ir0 + 1
        down = ir1 - row + 1
        if down >= up:
            return start, (1, 0), down
        return start, (-1, 0), up
    if side == "right":
        row = hinge_row_or_col
        assert ir0 <= row <= ir1
        start = (row, ic1)
        up = row - ir0 + 1
        down = ir1 - row + 1
        if down >= up:
            return start, (1, 0), down
        return start, (-1, 0), up
    if side == "top":
        col = hinge_row_or_col
        assert ic0 <= col <= ic1
        start = (ir0, col)
        left = col - ic0 + 1
        right = ic1 - col + 1
        if right >= left:
            return start, (0, 1), right
        return start, (0, -1), left
    if side == "bottom":
        col = hinge_row_or_col
        assert ic0 <= col <= ic1
        start = (ir1, col)
        left = col - ic0 + 1
        right = ic1 - col + 1
        if right >= left:
            return start, (0, 1), right
        return start, (0, -1), left
    raise ValueError(side)


def _folded_cells(
    frame_bbox: tuple[int, int, int, int],
    side: str,
    hinge_row_or_col: int,
    length: int,
) -> list[tuple[int, int]]:
    start, step, limit = _fold_spec(frame_bbox, side, hinge_row_or_col)
    dr, dc = step
    return [(start[0] + dr * k, start[1] + dc * k) for k in range(min(length, limit))]


def _input_cells(
    frame_bbox: tuple[int, int, int, int],
    side: str,
    hinge_row_or_col: int,
    length: int,
) -> list[tuple[int, int]]:
    fr0, fc0, fr1, fc1 = frame_bbox
    if side == "left":
        row = hinge_row_or_col
        return [(row, fc0 - d) for d in range(length, 0, -1)]
    if side == "right":
        row = hinge_row_or_col
        return [(row, fc1 + d) for d in range(1, length + 1)]
    if side == "top":
        col = hinge_row_or_col
        return [(fr0 - d, col) for d in range(length, 0, -1)]
    if side == "bottom":
        col = hinge_row_or_col
        return [(fr1 + d, col) for d in range(1, length + 1)]
    raise ValueError(side)


def _outside_capacity(shape: tuple[int, int], frame_bbox: tuple[int, int, int, int], side: str) -> int:
    h, w = shape
    fr0, fc0, fr1, fc1 = frame_bbox
    if side == "left":
        return fc0
    if side == "right":
        return w - fc1 - 1
    if side == "top":
        return fr0
    if side == "bottom":
        return h - fr1 - 1
    raise ValueError(side)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold each outside line segment into the hollow gray frame by turning it 90 degrees at its contacted endpoint.

    Input structure:
    - The background is black.
    - A unique one-pixel-thick gray rectangular frame encloses a black interior.
    - Outside the frame, each non-gray foreground object is one straight horizontal or vertical segment, and exactly one of its endpoints is orthogonally adjacent to one frame edge.

    Rule:
    - Erase every outside colored segment.
    - Use the touching endpoint to determine the hinge row or column on the contacted frame edge.
    - Horizontal segments touching the left or right edge become vertical strokes drawn just inside that edge; vertical segments touching the top or bottom edge become horizontal strokes drawn just inside that edge.
    - Preserve the segment color and requested length, truncating at the far inner wall if needed; if both perpendicular fold directions are possible, use the interior ray with more available cells.

    Key insight:
    - The segment is not translated as a rigid bar: the solver must first detect which endpoint touches which side of the frame, then reinterpret the outside run as a perpendicular run anchored to the corresponding inner wall.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    fr0, fc0, fr1, fc1 = _frame_bbox(g)
    ir0, ic0, ir1, ic1 = fr0 + 1, fc0 + 1, fr1 - 1, fc1 - 1
    assert np.all(g[ir0 : ir1 + 1, ic0 : ic1 + 1] == BLACK), "frame interior must start empty"

    frame_mask = g == GRAY
    out = g.copy()
    out[(g != BLACK) & (g != GRAY)] = BLACK

    for comp in _components_by_color(g, ignore={BLACK, GRAY}):
        coords = comp["coords"]
        color = comp["color"]
        for r, c in coords:
            assert not (fr0 <= r <= fr1 and fc0 <= c <= fc1), "segments must stay outside the frame"

        shape, endpoints, length = _segment_shape(coords)
        touch_cell, side = _touch_info(endpoints, frame_mask)
        if side in {"left", "right"}:
            assert shape in {"horizontal", "point"}, "segments touching left/right must be horizontal"
            row = touch_cell[0]
            assert fr0 < row < fr1, "touching point must be on a side, not a corner"
            cells = _folded_cells((fr0, fc0, fr1, fc1), side, row, length)
        else:
            assert side in {"top", "bottom"}
            assert shape in {"vertical", "point"}, "segments touching top/bottom must be vertical"
            col = touch_cell[1]
            assert fc0 < col < fc1, "touching point must be on a side, not a corner"
            cells = _folded_cells((fr0, fc0, fr1, fc1), side, col, length)

        for r, c in cells:
            assert ir0 <= r <= ir1 and ic0 <= c <= ic1
            current = int(out[r, c])
            assert current in {BLACK, color}, "folded segments may not overlap in different colors"
            out[r, c] = color

    return out


def _sample_frame(shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    frame_h = random.randint(FRAME_OUTER_MIN, min(FRAME_OUTER_MAX, h - 2 * FRAME_MARGIN_MIN))
    frame_w = random.randint(FRAME_OUTER_MIN, min(FRAME_OUTER_MAX, w - 2 * FRAME_MARGIN_MIN))

    center_r = (h - frame_h) // 2
    center_c = (w - frame_w) // 2
    r_lo = max(FRAME_MARGIN_MIN, center_r - 2)
    r_hi = min(h - frame_h - FRAME_MARGIN_MIN, center_r + 2)
    c_lo = max(FRAME_MARGIN_MIN, center_c - 2)
    c_hi = min(w - frame_w - FRAME_MARGIN_MIN, center_c + 2)
    r0 = random.randint(r_lo, r_hi)
    c0 = random.randint(c_lo, c_hi)
    return r0, c0, r0 + frame_h - 1, c0 + frame_w - 1


def _valid_hinges(frame_bbox: tuple[int, int, int, int], side: str) -> list[int]:
    fr0, fc0, fr1, fc1 = frame_bbox
    ir0, ic0, ir1, ic1 = fr0 + 1, fc0 + 1, fr1 - 1, fc1 - 1
    vals: list[int] = []
    if side in {"left", "right"}:
        for row in range(ir0, ir1 + 1):
            up = row - ir0 + 1
            down = ir1 - row + 1
            if up != down and max(up, down) >= 2:
                vals.append(row)
    else:
        for col in range(ic0, ic1 + 1):
            left = col - ic0 + 1
            right = ic1 - col + 1
            if left != right and max(left, right) >= 2:
                vals.append(col)
    return vals


def _input_conflict(
    shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    occupied: set[tuple[int, int]],
    cells: list[tuple[int, int]],
) -> bool:
    h, w = shape
    fr0, fc0, fr1, fc1 = frame_bbox
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return True
        if fr0 <= r <= fr1 and fc0 <= c <= fc1:
            return True
        if (r, c) in occupied:
            return True
    for r, c in cells:
        for nr in range(max(0, r - 1), min(h, r + 2)):
            for nc in range(max(0, c - 1), min(w, c + 2)):
                if (nr, nc) in cell_set:
                    continue
                if (nr, nc) in occupied:
                    return True
    return False


def _draw_frame(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = frame_bbox
    grid[r0 : r1 + 1, c0] = GRAY
    grid[r0 : r1 + 1, c1] = GRAY
    grid[r0, c0 : c1 + 1] = GRAY
    grid[r1, c0 : c1 + 1] = GRAY


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    frame_bbox = _sample_frame((h, w))

    segment_count = random.randint(2, MAX_SEGMENTS)
    colors = random.sample(SEGMENT_COLORS, segment_count)
    side_plan = [random.choice(("left", "right")), random.choice(("top", "bottom"))]
    while len(side_plan) < segment_count:
        side_plan.append(random.choice(("left", "right", "top", "bottom")))
    random.shuffle(side_plan)

    require_truncation = random.choice((True, False))
    placed_truncation = False
    occupied_input: set[tuple[int, int]] = set()
    occupied_output: set[tuple[int, int]] = set()
    segments: list[dict] = []

    for idx, side in enumerate(side_plan):
        hinges = _valid_hinges(frame_bbox, side)
        if not hinges:
            return None

        must_truncate = require_truncation and not placed_truncation and idx == segment_count - 1
        chosen: dict | None = None
        for _ in range(PLACE_RETRIES):
            hinge = random.choice(hinges)
            output_limit = len(_folded_cells(frame_bbox, side, hinge, MAX_SEGMENT_LEN + 10))
            outside_limit = min(MAX_SEGMENT_LEN, _outside_capacity((h, w), frame_bbox, side))
            if outside_limit < 2 or output_limit < 2:
                continue

            short_lengths = list(range(2, min(output_limit, outside_limit) + 1))
            long_lengths = list(range(max(output_limit + 1, 2), outside_limit + 1))

            wants_trunc = must_truncate or (not placed_truncation and require_truncation and random.choice((True, False)))
            branch_options: list[tuple[bool, list[int]]] = []
            if wants_trunc and long_lengths:
                branch_options.append((True, long_lengths))
            if short_lengths:
                branch_options.append((False, short_lengths))
            if not wants_trunc and long_lengths:
                branch_options.append((True, long_lengths))
            if not branch_options:
                continue

            truncate, lengths = random.choice(branch_options)
            length = random.choice(lengths)
            input_cells = _input_cells(frame_bbox, side, hinge, length)
            output_cells = _folded_cells(frame_bbox, side, hinge, length)
            if _input_conflict((h, w), frame_bbox, occupied_input, input_cells):
                continue
            if any(cell in occupied_output for cell in output_cells):
                continue
            chosen = {
                "side": side,
                "hinge": hinge,
                "length": length,
                "input_cells": input_cells,
                "output_cells": output_cells,
                "truncate": truncate,
            }
            break

        if chosen is None:
            return None
        occupied_input.update(chosen["input_cells"])
        occupied_output.update(chosen["output_cells"])
        placed_truncation = placed_truncation or chosen["truncate"]
        chosen["color"] = colors[idx]
        segments.append(chosen)

    if require_truncation and not placed_truncation:
        return None

    inp = np.full((h, w), BLACK, dtype=int)
    _draw_frame(inp, frame_bbox)
    for segment in segments:
        color = segment["color"]
        for r, c in segment["input_cells"]:
            inp[r, c] = color

    expected = inp.copy()
    expected[(expected != BLACK) & (expected != GRAY)] = BLACK
    for segment in segments:
        color = segment["color"]
        for r, c in segment["output_cells"]:
            expected[r, c] = color

    actual = main(inp)
    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid folded-segment frame input")
