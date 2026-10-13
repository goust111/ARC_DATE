from __future__ import annotations

import random
from collections import Counter

import numpy as np


FRAME = 5
BG = 0
BLACK = 0
ARROW_PALETTE = [1, 2, 3, 4]
SEGMENT_PALETTE = [6, 7, 8, 9]
GEN_ATTEMPTS = 400
INNER_ATTEMPTS = 300


def _solid_frame_color(grid: np.ndarray) -> int:
    h, w = grid.shape
    assert h >= 5 and w >= 5, "grid must contain a frame and interior"
    border = np.concatenate([grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]])
    colors = set(map(int, border))
    assert len(colors) == 1, "expected a one-color rectangular frame"
    return int(next(iter(colors)))


def _dominant_color(arr: np.ndarray) -> int:
    counts = Counter(map(int, arr.ravel()))
    return int(counts.most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _connected_single_component(coords: np.ndarray) -> bool:
    pts = {tuple(map(int, p)) for p in coords}
    if not pts:
        return False
    stack = [next(iter(pts))]
    seen = set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for nxt in _neighbors4(*cur):
            if nxt in pts and nxt not in seen:
                stack.append(nxt)
    return seen == pts


def _consecutive(values: list[int]) -> bool:
    return values == list(range(values[0], values[0] + len(values)))


def _analyze_arrow(coords: np.ndarray, shape: tuple[int, int]) -> dict | None:
    h, w = shape
    pts = [tuple(map(int, p)) for p in coords]
    if len(pts) != 6:
        return None

    row_counts = Counter(r for r, _ in pts)
    col_counts = Counter(c for _, c in pts)
    row_hist = sorted(row_counts.values())
    col_hist = sorted(col_counts.values())

    # Horizontal 4-bar + vertical 3-crossbar.
    if row_hist == [1, 1, 4] and col_hist == [1, 1, 1, 3]:
        bar_row = next(r for r, n in row_counts.items() if n == 4)
        cross_col = next(c for c, n in col_counts.items() if n == 3)
        bar_cols = sorted(c for r, c in pts if r == bar_row)
        cross_rows = sorted(r for r, c in pts if c == cross_col)
        if not _consecutive(bar_cols):
            return None
        if cross_rows != [bar_row - 1, bar_row, bar_row + 1]:
            return None
        if not (0 < bar_row < h - 1):
            return None

        min_col, max_col = bar_cols[0], bar_cols[-1]
        left_dist = cross_col - min_col
        right_dist = max_col - cross_col
        if {left_dist, right_dist} != {1, 2}:
            return None

        if left_dist < right_dist:
            direction = (0, -1)
            tip = (bar_row, min_col)
            target = (bar_row, 0)
            opposite = (bar_row, w - 1)
        else:
            direction = (0, 1)
            tip = (bar_row, max_col)
            target = (bar_row, w - 1)
            opposite = (bar_row, 0)

        return {
            "kind": "arrow",
            "orientation": "H",
            "cells": pts,
            "bar_min": (bar_row, min_col),
            "bar_max": (bar_row, max_col),
            "joint": (bar_row, cross_col),
            "tip": tip,
            "direction": direction,
            "target": target,
            "opposite": opposite,
            "center_key": (2 * bar_row, min_col + max_col),
        }

    # Vertical 4-bar + horizontal 3-crossbar.
    if row_hist == [1, 1, 1, 3] and col_hist == [1, 1, 4]:
        cross_row = next(r for r, n in row_counts.items() if n == 3)
        bar_col = next(c for c, n in col_counts.items() if n == 4)
        bar_rows = sorted(r for r, c in pts if c == bar_col)
        cross_cols = sorted(c for r, c in pts if r == cross_row)
        if not _consecutive(bar_rows):
            return None
        if cross_cols != [bar_col - 1, bar_col, bar_col + 1]:
            return None
        if not (0 < bar_col < w - 1):
            return None

        min_row, max_row = bar_rows[0], bar_rows[-1]
        up_dist = cross_row - min_row
        down_dist = max_row - cross_row
        if {up_dist, down_dist} != {1, 2}:
            return None

        if up_dist < down_dist:
            direction = (-1, 0)
            tip = (min_row, bar_col)
            target = (0, bar_col)
            opposite = (h - 1, bar_col)
        else:
            direction = (1, 0)
            tip = (max_row, bar_col)
            target = (h - 1, bar_col)
            opposite = (0, bar_col)

        return {
            "kind": "arrow",
            "orientation": "V",
            "cells": pts,
            "bar_min": (min_row, bar_col),
            "bar_max": (max_row, bar_col),
            "joint": (cross_row, bar_col),
            "tip": tip,
            "direction": direction,
            "target": target,
            "opposite": opposite,
            "center_key": (min_row + max_row, 2 * bar_col),
        }

    return None


def _analyze_segment(coords: np.ndarray, shape: tuple[int, int]) -> dict | None:
    h, w = shape
    pts = sorted(tuple(map(int, p)) for p in coords)
    if not (2 <= len(pts) <= 4):
        return None

    rows = sorted({r for r, _ in pts})
    cols = sorted({c for _, c in pts})

    if len(rows) == 1 and len(cols) == len(pts) and _consecutive(cols):
        row = rows[0]
        if row not in {1, h - 2}:
            return None
        return {
            "kind": "segment",
            "orientation": "H",
            "cells": pts,
            "top_row": row,
        }

    if len(cols) == 1 and len(rows) == len(pts) and _consecutive(rows):
        col = cols[0]
        if col not in {1, w - 2}:
            return None
        return {
            "kind": "segment",
            "orientation": "V",
            "cells": pts,
            "top_row": rows[0],
        }

    return None


def _ray_interior(arrow: dict) -> list[tuple[int, int]]:
    r, c = arrow["tip"]
    dr, dc = arrow["direction"]
    cells = []
    rr, cc = r + dr, c + dc
    while (rr, cc) != arrow["target"]:
        cells.append((rr, cc))
        rr += dr
        cc += dc
    return cells


def _parse_grid(grid: np.ndarray) -> tuple[int, int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h <= 30 and w <= 30, "ARC grids must fit within 30x30"

    frame = _solid_frame_color(g)
    bg = _dominant_color(g[1:-1, 1:-1])
    assert bg != frame, "interior background must differ from frame"

    colors = sorted(set(map(int, np.unique(g))) - {bg, frame})
    arrows: list[dict] = []
    segments: list[dict] = []

    for color in colors:
        coords = np.argwhere(g == color)
        assert _connected_single_component(coords), "each non-frame color must form one 4-connected object"
        arrow = _analyze_arrow(coords, g.shape)
        if arrow is not None:
            arrow["color"] = int(color)
            arrows.append(arrow)
            continue
        segment = _analyze_segment(coords, g.shape)
        assert segment is not None, "each non-frame object must be either a valid arrow or a valid side segment"
        segment["color"] = int(color)
        segments.append(segment)

    assert 2 <= len(arrows) <= 4, "expected two to four arrows"
    assert len(arrows) == len(segments), "arrow and segment counts must match"
    assert len({seg["top_row"] for seg in segments}) == len(segments), "segment top rows must be distinct"

    all_endpoints = [arrow["target"] for arrow in arrows] + [arrow["opposite"] for arrow in arrows]
    assert len(set(all_endpoints)) == len(all_endpoints), "all frame endpoints must be unique"
    for arrow in arrows:
        assert g[arrow["target"]] == frame and g[arrow["opposite"]] == frame

    occupied = set()
    for obj in arrows + segments:
        occupied.update(obj["cells"])
    for arrow in arrows:
        for cell in _ray_interior(arrow):
            assert cell not in occupied, "arrow rays should cross only background before the frame"

    return frame, bg, arrows, segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor arrow-selected frame endpoints using the color order encoded by side stripes.

    Input structure:
    - The grid has a one-pixel solid rectangular frame around a black interior background.
    - Inside the frame are 2-4 monochrome six-pixel asymmetric T arrows; the 4-cell bar gives the ray axis,
      and the 3-cell crossbar sits one cell closer to the pointed end.
    - The same interior also contains 2-4 monochrome one-cell-thick short line segments placed flush against
      one inner side of the frame, with distinct topmost rows.

    Rule:
    - For each arrow, use the crossbar offset to determine which bar end is the tip, then extend a ray from
      that tip to the frame and record the hit frame pixel as the arrow's target endpoint.
    - Sort the short line segments by their topmost occupied row from top to bottom to obtain an ordered color list.
    - Sort the arrows by the geometric midpoint of their 4-cell main bar in row-major order.
    - Recolor the k-th arrow's target endpoint with the k-th sorted segment color, and recolor that endpoint's
      opposite frame endpoint on the same row or column to black.
    - Leave the interior objects, background, and all other frame pixels unchanged.

    Key insight:
    - The arrows determine where to paint, while the side segments determine which colors to use; neither stream
      of information is sufficient alone, and they only meet when the ordered colors are assigned to the ordered
      arrow endpoints on the frame.
    """
    g = np.asarray(grid, dtype=int)
    _frame, _bg, arrows, segments = _parse_grid(g)

    ordered_colors = [seg["color"] for seg in sorted(segments, key=lambda seg: seg["top_row"])]
    ordered_arrows = sorted(arrows, key=lambda arrow: arrow["center_key"])

    out = g.copy()
    for arrow, color in zip(ordered_arrows, ordered_colors):
        out[arrow["target"]] = int(color)
        out[arrow["opposite"]] = BLACK
    return out


def _arrow_cells(orientation: str, direction: str, axis: int, start: int) -> dict:
    if orientation == "H":
        row = axis
        cols = list(range(start, start + 4))
        cross_col = start + (1 if direction == "left" else 2)
        cells = {(row, c) for c in cols}
        cells.add((row - 1, cross_col))
        cells.add((row + 1, cross_col))
        tip = (row, start) if direction == "left" else (row, start + 3)
        step = (0, -1) if direction == "left" else (0, 1)
    else:
        col = axis
        rows = list(range(start, start + 4))
        cross_row = start + (1 if direction == "up" else 2)
        cells = {(r, col) for r in rows}
        cells.add((cross_row, col - 1))
        cells.add((cross_row, col + 1))
        tip = (start, col) if direction == "up" else (start + 3, col)
        step = (-1, 0) if direction == "up" else (1, 0)
    return {"cells": cells, "tip": tip, "step": step}


def _ray_from_tip(tip: tuple[int, int], step: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c = tip
    dr, dc = step
    cells = []
    rr, cc = r + dr, c + dc
    while 0 < rr < h - 1 and 0 < cc < w - 1:
        cells.append((rr, cc))
        rr += dr
        cc += dc
    return cells


def _expanded(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 1 <= rr < h - 1 and 1 <= cc < w - 1:
                    out.add((rr, cc))
    return out


def _sample_arrow_spec(
    rng: random.Random,
    h: int,
    w: int,
    protected: set[tuple[int, int]],
    ray_forbidden: set[tuple[int, int]],
    endpoint_used: set[tuple[int, int]],
    row_axes: set[int],
    col_axes: set[int],
) -> dict | None:
    for _ in range(INNER_ATTEMPTS):
        orientation = rng.choice(["H", "V"])
        if orientation == "H":
            axis = rng.randint(2, h - 3)
            if axis in row_axes:
                continue
            direction = rng.choice(["left", "right"])
            start = rng.randint(1, w - 5)
            spec = _arrow_cells(orientation, direction, axis, start)
            target = (axis, 0) if direction == "left" else (axis, w - 1)
            opposite = (axis, w - 1) if direction == "left" else (axis, 0)
        else:
            axis = rng.randint(2, w - 3)
            if axis in col_axes:
                continue
            direction = rng.choice(["up", "down"])
            start = rng.randint(1, h - 5)
            spec = _arrow_cells(orientation, direction, axis, start)
            target = (0, axis) if direction == "up" else (h - 1, axis)
            opposite = (h - 1, axis) if direction == "up" else (0, axis)

        cells = spec["cells"]
        if cells & protected:
            continue
        if cells & ray_forbidden:
            continue
        if target in endpoint_used or opposite in endpoint_used:
            continue

        ray = _ray_from_tip(spec["tip"], spec["step"], h, w)
        if any(cell in protected for cell in ray):
            continue

        arrow = {
            "orientation": orientation,
            "direction": direction,
            "axis": axis,
            "start": start,
            "cells": cells,
            "target": target,
            "opposite": opposite,
            "tip": spec["tip"],
            "step": spec["step"],
            "ray": ray,
        }
        return arrow
    return None


def _sample_segment_spec(
    rng: random.Random,
    h: int,
    w: int,
    protected: set[tuple[int, int]],
    ray_forbidden: set[tuple[int, int]],
    top_rows_used: set[int],
) -> dict | None:
    for _ in range(INNER_ATTEMPTS):
        orientation = rng.choice(["H", "V"])
        length = rng.randint(2, 4)
        if orientation == "H":
            row = rng.choice([1, h - 2])
            if row in top_rows_used:
                continue
            start_col = rng.randint(1, w - 1 - length)
            cells = {(row, c) for c in range(start_col, start_col + length)}
            top_row = row
        else:
            col = rng.choice([1, w - 2])
            start_row = rng.randint(1, h - 1 - length)
            if start_row in top_rows_used:
                continue
            cells = {(r, col) for r in range(start_row, start_row + length)}
            top_row = start_row

        if cells & protected:
            continue
        if cells & ray_forbidden:
            continue
        return {"orientation": orientation, "cells": cells, "top_row": top_row}
    return None


def _build_grid(
    h: int,
    w: int,
    arrows: list[dict],
    segments: list[dict],
    arrow_colors: list[int],
    segment_colors: list[int],
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = FRAME
    for arrow, color in zip(arrows, arrow_colors):
        for r, c in arrow["cells"]:
            grid[r, c] = int(color)
    for seg, color in zip(segments, segment_colors):
        for r, c in seg["cells"]:
            grid[r, c] = int(color)
    return grid


def _validate_generated(grid: np.ndarray) -> None:
    frame, bg, arrows, segments = _parse_grid(grid)
    assert frame == FRAME, "generated frame color must stay fixed"
    assert bg == BG, "generated background color must stay fixed"

    ordered_arrows = sorted(arrows, key=lambda arrow: arrow["center_key"])
    ordered_segments = sorted(segments, key=lambda seg: seg["top_row"])
    out = main(grid)

    diff = np.argwhere(out != grid)
    assert diff.shape[0] == 2 * len(arrows), "each arrow should recolor exactly two frame endpoints"
    for r, c in diff:
        assert r in {0, grid.shape[0] - 1} or c in {0, grid.shape[1] - 1}, "only frame cells may change"

    for arrow, segment in zip(ordered_arrows, ordered_segments):
        assert out[arrow["target"]] == segment["color"], "target endpoint must receive ordered segment color"
        assert out[arrow["opposite"]] == BLACK, "opposite endpoint must turn black"

    occupied = set()
    for obj in arrows + segments:
        occupied.update(obj["cells"])
    for arrow in arrows:
        assert set(_ray_interior(arrow)).isdisjoint(occupied), "generated rays must be visually unobstructed"


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        h = rng.randint(16, 22)
        w = rng.randint(16, 22)
        count = rng.randint(2, 4)
        arrow_colors = rng.sample(ARROW_PALETTE, count)
        segment_colors = rng.sample(SEGMENT_PALETTE, count)

        protected: set[tuple[int, int]] = set()
        ray_forbidden: set[tuple[int, int]] = set()
        endpoint_used: set[tuple[int, int]] = set()
        row_axes: set[int] = set()
        col_axes: set[int] = set()
        arrows: list[dict] = []

        ok = True
        for _idx in range(count):
            arrow = _sample_arrow_spec(rng, h, w, protected, ray_forbidden, endpoint_used, row_axes, col_axes)
            if arrow is None:
                ok = False
                break
            arrows.append(arrow)
            protected |= _expanded(arrow["cells"], h, w)
            ray_forbidden |= set(arrow["ray"])
            endpoint_used.add(arrow["target"])
            endpoint_used.add(arrow["opposite"])
            if arrow["orientation"] == "H":
                row_axes.add(arrow["axis"])
            else:
                col_axes.add(arrow["axis"])
        if not ok:
            continue

        top_rows_used: set[int] = set()
        segments: list[dict] = []
        for _idx in range(count):
            segment = _sample_segment_spec(rng, h, w, protected, ray_forbidden, top_rows_used)
            if segment is None:
                ok = False
                break
            segments.append(segment)
            protected |= _expanded(segment["cells"], h, w)
            top_rows_used.add(segment["top_row"])
        if not ok:
            continue

        grid = _build_grid(h, w, arrows, segments, arrow_colors, segment_colors)
        try:
            _validate_generated(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid input for puzzle 4")
