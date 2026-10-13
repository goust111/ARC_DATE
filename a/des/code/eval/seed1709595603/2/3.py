from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
MARKER_COLORS = (1, 2, 3, 6, 7, 8, 9)
GEN_ATTEMPTS = 300

Cell = tuple[int, int]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> list[Cell]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _trace_gap_cells(grid: np.ndarray, marker: Cell) -> list[Cell]:
    r0, c0 = marker
    starts = [
        (rr, cc)
        for rr, cc in _neighbors4(r0, c0)
        if _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == GREY
    ]
    assert len(starts) == 1, "each marker must touch exactly one pipe endpoint"

    prev = marker
    curr = starts[0]
    seen_states: set[tuple[Cell, Cell]] = set()
    claims: list[Cell] = []

    while True:
        state = (prev, curr)
        assert state not in seen_states, "pipe tracing should not loop"
        seen_states.add(state)

        next_steps = [
            (rr, cc)
            for rr, cc in _neighbors4(*curr)
            if _in_bounds(grid.shape, rr, cc)
            and (rr, cc) != prev
            and int(grid[rr, cc]) == GREY
        ]
        assert len(next_steps) <= 1, "pipes must not branch"
        if next_steps:
            prev, curr = curr, next_steps[0]
            continue

        dr = curr[0] - prev[0]
        dc = curr[1] - prev[1]
        rr = curr[0] + dr
        cc = curr[1] + dc
        jumped: list[Cell] = []
        landing: Cell | None = None
        while _in_bounds(grid.shape, rr, cc):
            value = int(grid[rr, cc])
            if value == BLACK:
                jumped.append((rr, cc))
                rr += dr
                cc += dc
                continue
            if value == GREY and jumped:
                landing = (rr, cc)
            break

        if landing is None:
            break
        claims.extend(jumped)
        prev, curr = curr, landing

    return claims


def _recover_gap_claims(grid: np.ndarray) -> dict[Cell, set[int]]:
    g = np.asarray(grid, dtype=int)
    markers = np.argwhere((g != BLACK) & (g != GREY))
    claims: dict[Cell, set[int]] = defaultdict(set)

    for r0, c0 in markers:
        r = int(r0)
        c = int(c0)
        color = int(g[r, c])
        for cell in _trace_gap_cells(g, (r, c)):
            claims[cell].add(color)

    return {cell: set(colors) for cell, colors in claims.items()}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace each labelled pipe from its marked endpoint, jump across straight
    background gaps, and repaint those gaps with the marker color. Cells claimed
    by multiple colors become yellow.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for cell, colors in _recover_gap_claims(g).items():
        out[cell] = YELLOW if len(colors) > 1 else next(iter(colors))

    return out


def _segment_cells(a: Cell, b: Cell) -> list[Cell]:
    ar, ac = a
    br, bc = b
    assert ar == br or ac == bc, "segments must be axis-aligned"
    dr = 0 if ar == br else (1 if br > ar else -1)
    dc = 0 if ac == bc else (1 if bc > ac else -1)
    steps = abs(br - ar) + abs(bc - ac)
    cells = [(ar, ac)]
    for _ in range(steps):
        ar += dr
        ac += dc
        cells.append((ar, ac))
    return cells


def _polyline(vertices: list[Cell]) -> tuple[list[Cell], list[list[Cell]]]:
    assert len(vertices) >= 2
    cells: list[Cell] = []
    segments: list[list[Cell]] = []
    for idx in range(len(vertices) - 1):
        seg = _segment_cells(vertices[idx], vertices[idx + 1])
        segments.append(seg)
        if idx == 0:
            cells.extend(seg)
        else:
            cells.extend(seg[1:])
    assert len(cells) == len(set(cells)), "generated path must be simple"
    return cells, segments


def _sample_gap(segment: list[Cell], rng: random.Random, anchor: Cell | None = None) -> set[Cell]:
    safe_start = 2
    safe_stop = len(segment) - 3
    safe_count = safe_stop - safe_start + 1
    assert safe_count >= 1, "gap must stay away from corners and endpoints"
    max_len = min(3, safe_count)

    if anchor is None:
        gap_len = rng.randint(1, max_len)
        start = rng.randint(safe_start, safe_stop - gap_len + 1)
    else:
        anchor_idx = segment.index(anchor)
        assert safe_start <= anchor_idx <= safe_stop, "anchor must stay inside the safe segment interior"
        gap_len = rng.randint(1, max_len)
        lo = max(safe_start, anchor_idx - gap_len + 1)
        hi = min(anchor_idx, safe_stop - gap_len + 1)
        start = rng.randint(lo, hi)

    end = start + gap_len
    return set(segment[start:end])


def _marker_for_side(side: str, endpoint: Cell, shape: tuple[int, int]) -> Cell:
    h, w = shape
    r, c = endpoint
    if side == "top":
        assert r == 1
        return (0, c)
    if side == "bottom":
        assert r == h - 2
        return (h - 1, c)
    if side == "left":
        assert c == 1
        return (r, 0)
    assert side == "right"
    assert c == w - 2
    return (r, w - 1)


def _make_vertical_vertices(h: int, slab_left: int, slab_right: int, rng: random.Random) -> list[Cell]:
    cols = list(range(slab_left, slab_right + 1))
    variant = rng.choice(["straight", "dogleg", "s"])

    if variant == "straight":
        c0 = rng.choice(cols)
        return [(1, c0), (h - 2, c0)]

    if variant == "dogleg":
        c0, c1 = rng.sample(cols, 2)
        y1 = rng.randint(2, h - 3)
        return [(1, c0), (y1, c0), (y1, c1), (h - 2, c1)]

    c0, c1, c2 = rng.sample(cols, 3)
    y1 = rng.randint(2, h - 6)
    y2 = rng.randint(y1 + 2, h - 3)
    return [(1, c0), (y1, c0), (y1, c1), (y2, c1), (y2, c2), (h - 2, c2)]


def _make_horizontal_vertices(w: int, slab_top: int, slab_bottom: int, rng: random.Random) -> list[Cell]:
    rows = list(range(slab_top, slab_bottom + 1))
    variant = rng.choice(["straight", "dogleg", "s"])

    if variant == "straight":
        r0 = rng.choice(rows)
        return [(r0, 1), (r0, w - 2)]

    if variant == "dogleg":
        r0, r1 = rng.sample(rows, 2)
        x1 = rng.randint(2, w - 3)
        return [(r0, 1), (r0, x1), (r1, x1), (r1, w - 2)]

    r0, r1, r2 = rng.sample(rows, 3)
    x1 = rng.randint(2, w - 6)
    x2 = rng.randint(x1 + 2, w - 3)
    return [(r0, 1), (r0, x1), (r1, x1), (r1, x2), (r2, x2), (r2, w - 2)]


def _choose_gaps(segments: list[list[Cell]], rng: random.Random) -> set[Cell] | None:
    candidates = [seg for seg in segments if len(seg) >= 5]
    if not candidates:
        return None
    gap_count = rng.randint(1, min(3, len(candidates)))
    gaps: set[Cell] = set()
    for seg in rng.sample(candidates, gap_count):
        gaps.update(_sample_gap(seg, rng))
    return gaps


def _expected_output_color(colors: set[int]) -> int:
    return YELLOW if len(colors) > 1 else next(iter(colors))


def _validate_generated(grid: np.ndarray, expected_claims: dict[Cell, set[int]]) -> bool:
    recovered = _recover_gap_claims(grid)
    if recovered != expected_claims:
        return False

    out = main(grid)
    for cell, colors in expected_claims.items():
        if int(out[cell]) != _expected_output_color(colors):
            return False
    return True


def _render_grid(
    shape: tuple[int, int], paths: list[dict[str, object]]
) -> tuple[np.ndarray, dict[Cell, set[int]]]:
    grid = np.full(shape, BLACK, dtype=int)
    expected_claims: dict[Cell, set[int]] = defaultdict(set)

    for path in paths:
        color = int(path["color"])
        cells = set(path["cells"])
        gaps = set(path["gaps"])
        marker = path["marker"]

        assert marker not in cells
        for cell in cells - gaps:
            grid[cell] = GREY
        for cell in gaps:
            expected_claims[cell].add(color)
        mr, mc = marker
        assert int(grid[mr, mc]) == BLACK
        grid[mr, mc] = color

    return grid, {cell: set(colors) for cell, colors in expected_claims.items()}


def _build_cross_paths(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(11, 18)
    w = rng.randint(11, 18)
    shape = (h, w)
    colors = rng.sample(MARKER_COLORS, 2)

    cross_r = rng.randint(3, h - 4)
    cross_c = rng.randint(3, w - 4)

    vertical_vertices = [(1, cross_c), (h - 2, cross_c)]
    horizontal_vertices = [(cross_r, 1), (cross_r, w - 2)]

    v_cells, v_segments = _polyline(vertical_vertices)
    h_cells, h_segments = _polyline(horizontal_vertices)

    v_gaps = _sample_gap(v_segments[0], rng, anchor=(cross_r, cross_c))
    h_gaps = _sample_gap(h_segments[0], rng, anchor=(cross_r, cross_c))

    v_side = rng.choice(["top", "bottom"])
    h_side = rng.choice(["left", "right"])
    paths = [
        {
            "color": colors[0],
            "cells": v_cells,
            "gaps": v_gaps,
            "marker": _marker_for_side(v_side, v_cells[0] if v_side == "top" else v_cells[-1], shape),
        },
        {
            "color": colors[1],
            "cells": h_cells,
            "gaps": h_gaps,
            "marker": _marker_for_side(h_side, h_cells[0] if h_side == "left" else h_cells[-1], shape),
        },
    ]

    grid, expected_claims = _render_grid(shape, paths)
    if not any(len(colors_here) > 1 for colors_here in expected_claims.values()):
        return None
    if not _validate_generated(grid, expected_claims):
        return None
    return grid


def _build_non_cross_paths(rng: random.Random) -> np.ndarray | None:
    orientation = rng.choice(["vertical", "horizontal"])
    path_count = rng.randint(2, 4)
    colors = rng.sample(MARKER_COLORS, path_count)

    slab_sizes = [rng.randint(4, 5) for _ in range(path_count)]
    slab_gaps = [rng.randint(1, 2) for _ in range(path_count - 1)]
    leading_margin = rng.randint(0, 2)
    trailing_margin = rng.randint(0, 2)

    if orientation == "vertical":
        h = rng.randint(12, 20)
        w = 2 + leading_margin + trailing_margin + sum(slab_sizes) + sum(slab_gaps)
        if w > 30:
            return None

        paths: list[dict[str, object]] = []
        cursor = 1 + leading_margin
        for idx in range(path_count):
            slab_left = cursor
            slab_right = slab_left + slab_sizes[idx] - 1
            vertices = _make_vertical_vertices(h, slab_left, slab_right, rng)
            cells, segments = _polyline(vertices)
            gaps = _choose_gaps(segments, rng)
            if gaps is None:
                return None
            side = rng.choice(["top", "bottom"])
            marker = _marker_for_side(side, cells[0] if side == "top" else cells[-1], (h, w))
            paths.append(
                {
                    "color": colors[idx],
                    "cells": cells,
                    "gaps": gaps,
                    "marker": marker,
                }
            )
            cursor = slab_right + 1
            if idx < path_count - 1:
                cursor += slab_gaps[idx]
        shape = (h, w)
    else:
        w = rng.randint(12, 20)
        h = 2 + leading_margin + trailing_margin + sum(slab_sizes) + sum(slab_gaps)
        if h > 30:
            return None

        paths = []
        cursor = 1 + leading_margin
        for idx in range(path_count):
            slab_top = cursor
            slab_bottom = slab_top + slab_sizes[idx] - 1
            vertices = _make_horizontal_vertices(w, slab_top, slab_bottom, rng)
            cells, segments = _polyline(vertices)
            gaps = _choose_gaps(segments, rng)
            if gaps is None:
                return None
            side = rng.choice(["left", "right"])
            marker = _marker_for_side(side, cells[0] if side == "left" else cells[-1], (h, w))
            paths.append(
                {
                    "color": colors[idx],
                    "cells": cells,
                    "gaps": gaps,
                    "marker": marker,
                }
            )
            cursor = slab_bottom + 1
            if idx < path_count - 1:
                cursor += slab_gaps[idx]
        shape = (h, w)

    grid, expected_claims = _render_grid(shape, paths)
    if not expected_claims:
        return None
    if not _validate_generated(grid, expected_claims):
        return None
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    if rng.random() < 0.5:
        return _build_cross_paths(rng)
    return _build_non_cross_paths(rng)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after 300 attempts")
