from __future__ import annotations

import random
from typing import Callable

import numpy as np


PAIR_COLORS = (1, 2, 3, 4, 6, 9)
BAR_COLOR = 5
CONTROL_COLOR = 7
BG_COLOR = 0

GRID_MIN = 16
GRID_MAX = 24
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(np.asarray(grid, dtype=int))))
    if exclude:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _segment_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    if r0 == r1:
        a, b = sorted((c0, c1))
        return [(int(r0), c) for c in range(a, b + 1)]
    a, b = sorted((r0, r1))
    return [(r, int(c0)) for r in range(a, b + 1)]


def _draw_segments(shape: tuple[int, int], bg: int, segments: list[dict], color_map: dict[int, int] | None = None) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for orientation in ("horizontal", "vertical"):
        for seg in segments:
            if seg["orientation"] != orientation:
                continue
            color = int(color_map.get(seg["color"], seg["color"])) if color_map else int(seg["color"])
            for r, c in seg["cells"]:
                out[r, c] = color
    return out


def _marker_info(grid: np.ndarray) -> tuple[int, tuple[int, int], tuple[int, int], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    bar = np.argwhere(g == BAR_COLOR)
    assert bar.shape == (3, 2), "expected one 3-pixel marker bar"
    rows = sorted(set(map(int, bar[:, 0])))
    cols = sorted(set(map(int, bar[:, 1])))
    if len(rows) == 1 and len(cols) == 3 and cols == list(range(cols[0], cols[0] + 3)):
        center = (rows[0], cols[1])
        axis = "horizontal"
    elif len(cols) == 1 and len(rows) == 3 and rows == list(range(rows[0], rows[0] + 3)):
        center = (rows[1], cols[0])
        axis = "vertical"
    else:
        raise AssertionError("marker bar must be a straight contiguous length-3 segment")

    control = np.argwhere(g == CONTROL_COLOR)
    assert control.shape == (1, 2), "expected one control pixel"
    control_rc = tuple(map(int, control[0]))
    dr = center[0] - control_rc[0]
    dc = center[1] - control_rc[1]
    assert abs(dr) + abs(dc) == 1, "control must be 4-neighbor-adjacent to the marker center"
    if axis == "horizontal":
        assert dc == 0, "horizontal bars must have vertical control offsets"
    else:
        assert dr == 0, "vertical bars must have horizontal control offsets"

    ray: list[tuple[int, int]] = []
    r, c = center[0] + dr, center[1] + dc
    h, w = g.shape
    while 0 <= r < h and 0 <= c < w:
        ray.append((int(r), int(c)))
        r += dr
        c += dc
    return bg, center, (int(dr), int(dc)), ray


def _extract_segments(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    colors = _colors_present(g, exclude={bg, BAR_COLOR, CONTROL_COLOR})
    assert 4 <= len(colors) <= 6, "expected 4-6 same-color endpoint pairs"

    segments: list[dict] = []
    for color in colors:
        coords = np.argwhere(g == int(color))
        assert coords.shape == (2, 2), "each segment color must appear exactly twice"
        (r0, c0), (r1, c1) = coords.tolist()
        assert (r0 == r1) ^ (c0 == c1), "each pair must share one row or one column"
        if r0 == r1:
            c0, c1 = sorted((int(c0), int(c1)))
            cells = _segment_cells(int(r0), c0, int(r1), c1)
            orientation = "horizontal"
        else:
            r0, r1 = sorted((int(r0), int(r1)))
            cells = _segment_cells(r0, int(c0), r1, int(c1))
            orientation = "vertical"
        segments.append(
            {
                "color": int(color),
                "orientation": orientation,
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "cells": cells,
                "cell_set": set(cells),
            }
        )
    return segments


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg, center, step, ray = _marker_info(g)
    segments = _extract_segments(g, bg)
    composite = _draw_segments(g.shape, bg, segments)

    hits: list[tuple[int, dict]] = []
    for seg in segments:
        distances = [idx + 1 for idx, cell in enumerate(ray) if cell in seg["cell_set"]]
        if distances:
            hits.append((min(distances), seg))

    hits.sort(key=lambda item: item[0])
    assert 2 <= len(hits) < len(segments), "ray must hit at least two but not all segments"
    first_distances = [dist for dist, _seg in hits]
    assert len(set(first_distances)) == len(first_distances), "ray-contact order must be unambiguous"

    ordered_colors = [int(seg["color"]) for _dist, seg in hits]
    color_map = {
        ordered_colors[i]: ordered_colors[(i + 1) % len(ordered_colors)]
        for i in range(len(ordered_colors))
    }
    return {
        "bg": int(bg),
        "center": center,
        "step": step,
        "ray": ray,
        "ray_set": set(ray),
        "segments": segments,
        "hits": hits,
        "hit_colors": ordered_colors,
        "color_map": color_map,
        "composite": composite,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color endpoint pair, derive a cyclic recoloring from the marker ray's hit order, and redraw the segments.

    Input structure:
    - A black background contains 4-6 foreground colors, each appearing in exactly two endpoint pixels.
    - Every same-color pair is axis-aligned: the two endpoints share one row or one column.
    - A single 3-pixel gray marker bar and one adjacent orange control pixel specify a ray direction.

    Rule:
    - Connect every same-color pair with its full horizontal or vertical segment, drawing all horizontal segments first and all vertical segments second so vertical colors win at crossings.
    - Cast one ray from the bar center in the direction away from the orange control pixel and list the segments it intersects by first-contact distance along that ray.
    - Turn that ordered hit list into a cyclic color substitution, redraw every segment with its substituted color when present, and erase the gray bar plus orange control back to black.

    Key insight:
    - The ray does not paint anything itself; it uses the hidden segment geometry, including covered horizontal crossings, to define a global cyclic recoloring of the whole segment set.
    """
    info = _analyze(np.asarray(grid, dtype=int))
    return _draw_segments(np.asarray(grid).shape, info["bg"], info["segments"], info["color_map"])


def _sample_positions(candidates: list[int], count: int, min_gap: int) -> list[int] | None:
    if count == 0:
        return []
    if len(candidates) < count:
        return None
    for _ in range(200):
        picks = sorted(random.sample(candidates, count))
        if all(b - a >= min_gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _covering_span(center_col: int, width: int, min_len: int = 4) -> tuple[int, int] | None:
    options = []
    for c0 in range(0, center_col):
        for c1 in range(center_col + 1, width):
            if c1 - c0 + 1 >= min_len:
                options.append((c0, c1))
    return random.choice(options) if options else None


def _noncovering_span(center_col: int, width: int, min_len: int = 4) -> tuple[int, int] | None:
    options = []
    for c0 in range(0, width):
        for c1 in range(c0 + min_len - 1, width):
            if c0 <= center_col <= c1:
                continue
            options.append((c0, c1))
    return random.choice(options) if options else None


def _vertical_span(row_lo: int, row_hi: int, min_len: int = 4) -> tuple[int, int] | None:
    if row_hi - row_lo + 1 < min_len:
        return None
    options = []
    for r0 in range(row_lo, row_hi + 1):
        for r1 in range(r0 + min_len - 1, row_hi + 1):
            options.append((r0, r1))
    return random.choice(options) if options else None


def _transforms() -> list[Callable[[np.ndarray], np.ndarray]]:
    return [
        lambda g: g.copy(),
        lambda g: np.rot90(g, 1).copy(),
        lambda g: np.rot90(g, 2).copy(),
        lambda g: np.rot90(g, 3).copy(),
        lambda g: np.fliplr(g).copy(),
        lambda g: np.flipud(g).copy(),
        lambda g: np.transpose(g).copy(),
        lambda g: np.fliplr(np.transpose(g)).copy(),
    ]


def _visible_shortcut(grid: np.ndarray) -> np.ndarray:
    info = _analyze(grid)
    seen: list[int] = []
    for r, c in info["ray"]:
        color = int(info["composite"][r, c])
        if color != info["bg"] and color not in seen:
            seen.append(color)
    if len(seen) < 2:
        return _draw_segments(np.asarray(grid).shape, info["bg"], info["segments"])
    color_map = {seen[i]: seen[(i + 1) % len(seen)] for i in range(len(seen))}
    return _draw_segments(np.asarray(grid).shape, info["bg"], info["segments"], color_map)


def _has_occluded_hit(info: dict) -> bool:
    for _dist, seg in info["hits"]:
        ray_cells = [cell for cell in seg["cells"] if cell in info["ray_set"]]
        if ray_cells and all(int(info["composite"][r, c]) != int(seg["color"]) for r, c in ray_cells):
            return True
    return False


def _has_behind_axis_nonhit(info: dict) -> bool:
    center = info["center"]
    step = info["step"]
    for seg in info["segments"]:
        if seg["color"] in info["hit_colors"]:
            continue
        for r, c in seg["cells"]:
            if step[0] != 0 and c == center[1]:
                if (r - center[0]) * step[0] < 0:
                    return True
            if step[1] != 0 and r == center[0]:
                if (c - center[1]) * step[1] < 0:
                    return True
    return False


def _build_input(shape: tuple[int, int], center: tuple[int, int], segments: list[dict]) -> np.ndarray | None:
    h, w = shape
    cr, cc = center
    grid = np.zeros((h, w), dtype=int)
    occupied = {
        (cr, cc - 1),
        (cr, cc),
        (cr, cc + 1),
        (cr - 1, cc),
    }
    for seg in segments:
        a = (seg["r0"], seg["c0"])
        b = (seg["r1"], seg["c1"])
        if a in occupied or b in occupied or a == b:
            return None
        occupied.add(a)
        occupied.add(b)
        grid[a] = seg["color"]
        grid[b] = seg["color"]
    grid[cr, cc - 1 : cc + 2] = BAR_COLOR
    grid[cr - 1, cc] = CONTROL_COLOR
    return grid


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    center = (random.randint(3, 5), random.randint(4, width - 5))
    cr, cc = center

    pair_count = random.randint(4, 6)
    hit_count = random.randint(2, pair_count - 1)
    hit_horizontal_count = hit_count - 1
    extra_nonhits = pair_count - hit_count - 1

    below_rows = list(range(cr + 2, height - 1))
    above_rows = list(range(0, cr - 1))
    if hit_horizontal_count > len(below_rows) or not above_rows:
        return None

    hit_rows = _sample_positions(below_rows, hit_horizontal_count, min_gap=2 if len(below_rows) >= 2 * hit_horizontal_count else 1)
    if hit_rows is None:
        return None
    back_row = random.choice(above_rows)

    segments: list[dict] = []

    vh_span = _vertical_span(cr + 1, height - 2, min_len=4)
    if vh_span is None:
        return None
    vr0, vr1 = vh_span
    occluded_rows = [row for row in hit_rows if vr0 < row <= vr1]
    if not occluded_rows:
        vr1 = max(hit_rows)
        if vr1 - vr0 + 1 < 4:
            vr1 = min(height - 2, vr0 + 3)
        occluded_rows = [row for row in hit_rows if vr0 < row <= vr1]
    if not occluded_rows:
        return None
    segments.append({"orientation": "vertical", "r0": vr0, "c0": cc, "r1": vr1, "c1": cc})

    for row in hit_rows:
        span = _covering_span(cc, width, min_len=5)
        if span is None:
            return None
        c0, c1 = span
        segments.append({"orientation": "horizontal", "r0": row, "c0": c0, "r1": row, "c1": c1})

    span = _covering_span(cc, width, min_len=5)
    if span is None:
        return None
    c0, c1 = span
    segments.append({"orientation": "horizontal", "r0": back_row, "c0": c0, "r1": back_row, "c1": c1})

    used_rows = {back_row, *hit_rows}
    used_cols = {cc}
    used_samecol_above = False

    extra_types = ["horizontal_miss", "vertical_other", "vertical_samecol_above"]
    for _ in range(extra_nonhits):
        placed = False
        for _inner in range(80):
            choice = random.choice(extra_types)
            if choice == "horizontal_miss":
                row_candidates = [r for r in range(cr + 1, height - 1) if r not in used_rows]
                if not row_candidates:
                    continue
                row = random.choice(row_candidates)
                span = _noncovering_span(cc, width, min_len=4)
                if span is None:
                    continue
                c0, c1 = span
                segments.append({"orientation": "horizontal", "r0": row, "c0": c0, "r1": row, "c1": c1})
                used_rows.add(row)
                placed = True
                break
            if choice == "vertical_samecol_above":
                if used_samecol_above:
                    continue
                span = _vertical_span(0, cr - 2, min_len=3)
                if span is None:
                    continue
                r0, r1 = span
                segments.append({"orientation": "vertical", "r0": r0, "c0": cc, "r1": r1, "c1": cc})
                used_samecol_above = True
                placed = True
                break
            col_candidates = [c for c in range(0, width) if c not in used_cols and c not in {cc - 1, cc + 1}]
            if not col_candidates:
                continue
            col = random.choice(col_candidates)
            span = _vertical_span(0, height - 1, min_len=4)
            if span is None:
                continue
            r0, r1 = span
            segments.append({"orientation": "vertical", "r0": r0, "c0": col, "r1": r1, "c1": col})
            used_cols.add(col)
            placed = True
            break
        if not placed:
            return None

    random.shuffle(segments)
    colors = random.sample(PAIR_COLORS, len(segments))
    for seg, color in zip(segments, colors):
        seg["color"] = int(color)
        seg["cells"] = _segment_cells(seg["r0"], seg["c0"], seg["r1"], seg["c1"])
        seg["cell_set"] = set(seg["cells"])

    grid = _build_input((height, width), center, segments)
    if grid is None:
        return None

    transformed = random.choice(_transforms())(grid)
    try:
        info = _analyze(transformed)
    except AssertionError:
        return None
    if len(info["segments"]) != pair_count:
        return None
    if not _has_occluded_hit(info):
        return None
    if not _has_behind_axis_nonhit(info):
        return None
    solved = main(transformed)
    if np.array_equal(solved, _visible_shortcut(transformed)):
        return None
    if np.array_equal(solved, transformed):
        return None
    return transformed


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input")
