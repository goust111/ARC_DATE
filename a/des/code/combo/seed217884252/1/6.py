from __future__ import annotations

import random

import numpy as np


BG = 7
PALETTE = tuple(color for color in range(10) if color != BG)

HEIGHT_RANGE = (10, 18)
WIDTH_RANGE = (10, 18)
EXTRA_HORIZONTAL_RANGE = (0, 2)
EXTRA_VERTICAL_RANGE = (0, 2)
ANCHORED_EXTRA_PROB = 0.65
PLACEMENT_RETRIES = 100
MAX_GENERATION_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _make_horizontal(color: int, row: int, c0: int, c1: int) -> dict[str, int | str]:
    left = min(int(c0), int(c1))
    right = max(int(c0), int(c1))
    assert left < right
    return {
        "color": int(color),
        "orientation": "h",
        "fixed": int(row),
        "start": left,
        "end": right,
        "length": right - left + 1,
    }


def _make_vertical(color: int, col: int, r0: int, r1: int) -> dict[str, int | str]:
    top = min(int(r0), int(r1))
    bottom = max(int(r0), int(r1))
    assert top < bottom
    return {
        "color": int(color),
        "orientation": "v",
        "fixed": int(col),
        "start": top,
        "end": bottom,
        "length": bottom - top + 1,
    }


def _segment_cells(segment: dict[str, int | str]) -> set[tuple[int, int]]:
    if segment["orientation"] == "h":
        row = int(segment["fixed"])
        return {(row, c) for c in range(int(segment["start"]), int(segment["end"]) + 1)}
    col = int(segment["fixed"])
    return {(r, col) for r in range(int(segment["start"]), int(segment["end"]) + 1)}


def _segment_endpoints(segment: dict[str, int | str]) -> set[tuple[int, int]]:
    if segment["orientation"] == "h":
        row = int(segment["fixed"])
        return {
            (row, int(segment["start"])),
            (row, int(segment["end"])),
        }
    col = int(segment["fixed"])
    return {
        (int(segment["start"]), col),
        (int(segment["end"]), col),
    }


def _paint_segment(grid: np.ndarray, segment: dict[str, int | str]) -> None:
    color = int(segment["color"])
    if segment["orientation"] == "h":
        row = int(segment["fixed"])
        grid[row, int(segment["start"]) : int(segment["end"]) + 1] = color
    else:
        col = int(segment["fixed"])
        grid[int(segment["start"]) : int(segment["end"]) + 1, col] = color


def _segments_from_grid(grid: np.ndarray, bg: int) -> list[dict[str, int | str]]:
    segments: list[dict[str, int | str]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        coords = np.argwhere(grid == color)
        assert coords.shape == (2, 2), f"expected exactly two endpoints for color {color}"
        (r0, c0), (r1, c1) = coords
        r0 = int(r0)
        c0 = int(c0)
        r1 = int(r1)
        c1 = int(c1)
        assert (r0 == r1) ^ (c0 == c1), f"color {color} must share exactly one row or column"
        if r0 == r1:
            segments.append(_make_horizontal(color, r0, c0, c1))
        else:
            segments.append(_make_vertical(color, c0, r0, r1))
    return segments


def _assert_unambiguous_ties(segments: list[dict[str, int | str]]) -> None:
    for orientation in ("h", "v"):
        same_orientation = [segment for segment in segments if segment["orientation"] == orientation]
        for i, first in enumerate(same_orientation):
            for second in same_orientation[i + 1 :]:
                if int(first["fixed"]) != int(second["fixed"]):
                    continue
                if int(first["length"]) != int(second["length"]):
                    continue
                if _intervals_overlap(
                    int(first["start"]),
                    int(first["end"]),
                    int(second["start"]),
                    int(second["end"]),
                ):
                    raise AssertionError("same-direction overlapping ties are visually ambiguous")


def _render_segments(shape: tuple[int, int], bg: int, segments: list[dict[str, int | str]]) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    horizontals = sorted(
        [segment for segment in segments if segment["orientation"] == "h"],
        key=lambda segment: (-int(segment["length"]), int(segment["fixed"]), int(segment["start"])),
    )
    verticals = sorted(
        [segment for segment in segments if segment["orientation"] == "v"],
        key=lambda segment: (-int(segment["length"]), int(segment["fixed"]), int(segment["start"])),
    )
    for segment in horizontals + verticals:
        _paint_segment(out, segment)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color endpoint pair, painting horizontal segments before vertical segments.
    Longer same-direction segments are drawn first, so shorter ones cover them where they overlap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    segments = _segments_from_grid(g, bg)
    assert segments, "expected at least one segment"
    _assert_unambiguous_ties(segments)
    return _render_segments(g.shape, bg, segments)


def _build_core_segments(h: int, w: int, colors: list[int]) -> tuple[list[dict[str, int | str]], int, int] | None:
    for _ in range(PLACEMENT_RETRIES):
        row = random.randint(2, h - 3)
        col = random.randint(2, w - 3)

        max_left = col
        max_right = w - 1 - col
        max_up = row
        max_down = h - 1 - row
        if min(max_left, max_right, max_up, max_down) < 2:
            continue

        long_left = random.randint(2, max_left)
        long_right = random.randint(2, max_right)
        short_left = random.randint(1, long_left - 1)
        short_right = random.randint(1, long_right - 1)

        long_up = random.randint(2, max_up)
        long_down = random.randint(2, max_down)
        short_up = random.randint(1, long_up - 1)
        short_down = random.randint(1, long_down - 1)

        segments = [
            _make_horizontal(colors[0], row, col - long_left, col + long_right),
            _make_horizontal(colors[1], row, col - short_left, col + short_right),
            _make_vertical(colors[2], col, row - long_up, row + long_down),
            _make_vertical(colors[3], col, row - short_up, row + short_down),
        ]
        return segments, row, col
    return None


def _place_extra_horizontal(
    h: int,
    w: int,
    color: int,
    blocked_rows: set[int],
    used_endpoints: set[tuple[int, int]],
    used_cells: set[tuple[int, int]],
    anchor_col: int | None,
) -> dict[str, int | str] | None:
    candidate_rows = [row for row in range(h) if row not in blocked_rows]
    if not candidate_rows:
        return None

    for _ in range(PLACEMENT_RETRIES):
        row = random.choice(candidate_rows)
        force_anchor = anchor_col is not None and random.random() < ANCHORED_EXTRA_PROB
        if force_anchor:
            left = random.randint(0, anchor_col - 1)
            right = random.randint(anchor_col + 1, w - 1)
        else:
            left = random.randint(0, w - 2)
            right = random.randint(left + 1, w - 1)

        segment = _make_horizontal(color, row, left, right)
        endpoints = _segment_endpoints(segment)
        cells = _segment_cells(segment)
        if not cells.isdisjoint(used_endpoints):
            continue
        if not endpoints.isdisjoint(used_cells):
            continue
        return segment
    return None


def _place_extra_vertical(
    h: int,
    w: int,
    color: int,
    blocked_cols: set[int],
    used_endpoints: set[tuple[int, int]],
    used_cells: set[tuple[int, int]],
    anchor_row: int | None,
) -> dict[str, int | str] | None:
    candidate_cols = [col for col in range(w) if col not in blocked_cols]
    if not candidate_cols:
        return None

    for _ in range(PLACEMENT_RETRIES):
        col = random.choice(candidate_cols)
        force_anchor = anchor_row is not None and random.random() < ANCHORED_EXTRA_PROB
        if force_anchor:
            top = random.randint(0, anchor_row - 1)
            bottom = random.randint(anchor_row + 1, h - 1)
        else:
            top = random.randint(0, h - 2)
            bottom = random.randint(top + 1, h - 1)

        segment = _make_vertical(color, col, top, bottom)
        endpoints = _segment_endpoints(segment)
        cells = _segment_cells(segment)
        if not cells.isdisjoint(used_endpoints):
            continue
        if not endpoints.isdisjoint(used_cells):
            continue
        return segment
    return None


def _render_input(shape: tuple[int, int], bg: int, segments: list[dict[str, int | str]]) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    for segment in segments:
        for row, col in _segment_endpoints(segment):
            grid[row, col] = int(segment["color"])
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*HEIGHT_RANGE)
    w = random.randint(*WIDTH_RANGE)

    extra_h = random.randint(*EXTRA_HORIZONTAL_RANGE)
    extra_v = random.randint(*EXTRA_VERTICAL_RANGE)
    total_segments = 4 + extra_h + extra_v
    if total_segments > len(PALETTE):
        return None

    colors = random.sample(PALETTE, total_segments)
    core = _build_core_segments(h, w, colors[:4])
    if core is None:
        return None
    segments, core_row, core_col = core

    used_rows = {core_row}
    used_cols = {core_col}
    used_endpoints: set[tuple[int, int]] = set()
    used_cells: set[tuple[int, int]] = set()
    for segment in segments:
        used_endpoints |= _segment_endpoints(segment)
        used_cells |= _segment_cells(segment)

    color_index = 4
    for _ in range(extra_h):
        segment = _place_extra_horizontal(
            h=h,
            w=w,
            color=colors[color_index],
            blocked_rows=used_rows,
            used_endpoints=used_endpoints,
            used_cells=used_cells,
            anchor_col=core_col,
        )
        if segment is None:
            return None
        segments.append(segment)
        used_rows.add(int(segment["fixed"]))
        used_endpoints |= _segment_endpoints(segment)
        used_cells |= _segment_cells(segment)
        color_index += 1

    for _ in range(extra_v):
        segment = _place_extra_vertical(
            h=h,
            w=w,
            color=colors[color_index],
            blocked_cols=used_cols,
            used_endpoints=used_endpoints,
            used_cells=used_cells,
            anchor_row=core_row,
        )
        if segment is None:
            return None
        segments.append(segment)
        used_cols.add(int(segment["fixed"]))
        used_endpoints |= _segment_endpoints(segment)
        used_cells |= _segment_cells(segment)
        color_index += 1

    inp = _render_input((h, w), BG, segments)
    try:
        out = main(inp)
    except AssertionError:
        return None

    assert out.shape == inp.shape
    if np.array_equal(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid line-connection puzzle")
