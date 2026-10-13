from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import colors_present, get_bg, in_bounds


BG = 0
RAY_COLOR = 5
PAIR_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_H_RANGE = (14, 20)
GRID_W_RANGE = (14, 20)
PAIR_COUNT_RANGE = (3, 4)
MARKER_COUNT_RANGE = (2, 3)
DIAG_LENGTH_RANGE = (3, 7)
GEN_RETRIES = 500


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    dr = r1 - r0
    dc = c1 - c0
    assert dr != 0 and abs(dr) == abs(dc)
    step_r = _sign(dr)
    step_c = _sign(dc)
    return [(r0 + k * step_r, c0 + k * step_c) for k in range(abs(dr) + 1)]


def _boundary_direction(r: int, c: int, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    on_top = r == 0
    on_bottom = r == h - 1
    on_left = c == 0
    on_right = c == w - 1
    assert on_top or on_bottom or on_left or on_right
    assert sum((on_top, on_bottom, on_left, on_right)) == 1
    if on_top:
        return (1, 0)
    if on_bottom:
        return (-1, 0)
    if on_left:
        return (0, 1)
    return (0, -1)


def _ray_positions(marker: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, tuple[int, int]]]:
    r, c = marker
    dr, dc = _boundary_direction(r, c, shape)
    out = []
    dist = 0
    rr, cc = r, c
    while in_bounds(shape, rr, cc):
        out.append((dist, (rr, cc)))
        dist += 1
        rr += dr
        cc += dc
    return out


def _build_specs(grid: np.ndarray) -> tuple[list[dict], list[tuple[int, int]], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == BG

    colors = [color for color in colors_present(g, exclude={bg}) if int(color) != RAY_COLOR]
    segments = []
    endpoints: set[tuple[int, int]] = set()
    for color in colors:
        coords = np.argwhere(g == int(color))
        assert coords.shape == (2, 2), "each non-grey foreground color must appear exactly twice"
        a = tuple(map(int, coords[0]))
        b = tuple(map(int, coords[1]))
        cells = _segment_cells(a, b)
        segments.append({"color": int(color), "a": a, "b": b, "cells": cells})
        endpoints.add(a)
        endpoints.add(b)

    markers = [tuple(map(int, rc)) for rc in np.argwhere(g == RAY_COLOR)]
    for r, c in markers:
        _boundary_direction(r, c, g.shape)
        assert (r, c) not in endpoints
    return segments, markers, endpoints


def _render_from_specs(
    shape: tuple[int, int],
    segments: list[dict],
    markers: list[tuple[int, int]],
    endpoints: set[tuple[int, int]],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for seg in segments:
        for r, c in seg["cells"]:
            out[r, c] = int(seg["color"])

    for marker in markers:
        for dist, (r, c) in _ray_positions(marker, shape):
            if dist % 2 == 1:
                continue
            if (r, c) in endpoints:
                continue
            out[r, c] = RAY_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw diagonal same-color segments, then let dotted boundary rays engrave gaps where they land.

    Input structure:
    - Black background contains 3-4 non-grey colors, each appearing exactly twice as endpoint pairs.
    - Each same-color pair lies on a strict 45-degree diagonal and defines one diagonal segment.
    - There are 2-3 grey single pixels on the boundary, each acting as a ray marker.

    Rule:
    - For every non-grey color, fill every cell on the diagonal segment between its two endpoints.
    - For every grey boundary marker, cast a ray straight inward perpendicular to that side.
    - Paint grey on ray distances 0, 2, 4, ...; skip distances 1, 3, 5, ...
    - If a painted grey ray cell lands on a diagonal-segment cell, replace that segment cell with black by keeping the grey pixel there.
    - Original colored endpoints are protected: when an even-distance ray position coincides with an endpoint, leave the endpoint color unchanged and do not paint grey there.

    Key insight:
    - The output depends on combining two independent geometries: diagonal segments supply potential targets, and the dotted ray phase selects which crossings become erased gaps and which crossings survive.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    segments, markers, endpoints = _build_specs(g)
    return _render_from_specs(g.shape, segments, markers, endpoints)


def _random_segment_candidates(h: int, w: int) -> list[dict]:
    candidates = []
    max_length = min(DIAG_LENGTH_RANGE[1], h - 3, w - 3)
    min_length = min(DIAG_LENGTH_RANGE[0], max_length)
    for length in range(min_length, max_length + 1):
        for step_c in (-1, 1):
            for r0 in range(1, h - 1 - length):
                for c0 in range(1, w - 1):
                    c1 = c0 + step_c * length
                    if not (1 <= c1 <= w - 2):
                        continue
                    a = (r0, c0)
                    b = (r0 + length, c1)
                    candidates.append({"a": a, "b": b, "cells": _segment_cells(a, b)})
    random.shuffle(candidates)
    return candidates


def _choose_segments(h: int, w: int, n_pairs: int) -> list[dict] | None:
    candidates = _random_segment_candidates(h, w)
    segments: list[dict] = []
    occupied: set[tuple[int, int]] = set()

    for cand in candidates:
        cells = set(cand["cells"])
        if cells & occupied:
            continue
        segments.append(cand)
        occupied |= cells
        if len(segments) == n_pairs:
            return segments
    return None


def _marker_analysis(
    marker: tuple[int, int],
    shape: tuple[int, int],
    segment_cells: set[tuple[int, int]],
    endpoints: set[tuple[int, int]],
) -> dict:
    even_nonendpoint = set()
    even_endpoint = set()
    odd_hits = set()
    drawn_cells = set()
    direction = _boundary_direction(marker[0], marker[1], shape)
    side = {
        (1, 0): "top",
        (-1, 0): "bottom",
        (0, 1): "left",
        (0, -1): "right",
    }[direction]

    for dist, cell in _ray_positions(marker, shape):
        if dist % 2 == 0:
            if cell in endpoints:
                if cell in segment_cells:
                    even_endpoint.add(cell)
            else:
                drawn_cells.add(cell)
                if cell in segment_cells:
                    even_nonendpoint.add(cell)
        elif cell in segment_cells:
            odd_hits.add(cell)

    return {
        "marker": marker,
        "direction": direction,
        "side": side,
        "drawn_cells": drawn_cells,
        "even_nonendpoint": even_nonendpoint,
        "even_endpoint": even_endpoint,
        "odd_hits": odd_hits,
    }


def _all_marker_candidates(
    h: int,
    w: int,
    segment_cells: set[tuple[int, int]],
    endpoints: set[tuple[int, int]],
) -> list[dict]:
    markers = []
    boundary = (
        [(0, c) for c in range(1, w - 1)]
        + [(h - 1, c) for c in range(1, w - 1)]
        + [(r, 0) for r in range(1, h - 1)]
        + [(r, w - 1) for r in range(1, h - 1)]
    )
    for marker in boundary:
        if marker in endpoints:
            continue
        info = _marker_analysis(marker, (h, w), segment_cells, endpoints)
        if info["even_nonendpoint"] or info["even_endpoint"] or info["odd_hits"]:
            markers.append(info)
    random.shuffle(markers)
    return markers


def _marker_subset_ok(marker_infos: list[dict], require_protected: bool) -> bool:
    sides = {info["side"] for info in marker_infos}
    if len(sides) < 2:
        return False

    used_rows: set[int] = set()
    used_cols: set[int] = set()
    drawn_union: set[tuple[int, int]] = set()
    erased_union: set[tuple[int, int]] = set()
    odd_union: set[tuple[int, int]] = set()
    protected_union: set[tuple[int, int]] = set()

    for info in marker_infos:
        r, c = info["marker"]
        dr, dc = info["direction"]
        if dr != 0:
            if c in used_cols:
                return False
            used_cols.add(c)
        else:
            if r in used_rows:
                return False
            used_rows.add(r)

        if drawn_union & info["drawn_cells"]:
            return False
        drawn_union |= info["drawn_cells"]
        erased_union |= info["even_nonendpoint"]
        odd_union |= info["odd_hits"]
        protected_union |= info["even_endpoint"]

    if not erased_union:
        return False
    if not odd_union:
        return False
    if require_protected and not protected_union:
        return False
    if len(drawn_union) <= len(marker_infos):
        return False
    return True


def _choose_markers(
    h: int,
    w: int,
    segment_cells: set[tuple[int, int]],
    endpoints: set[tuple[int, int]],
    n_markers: int,
    require_protected: bool,
) -> list[dict] | None:
    candidates = _all_marker_candidates(h, w, segment_cells, endpoints)
    if len(candidates) < n_markers:
        return None

    for _ in range(300):
        subset = random.sample(candidates, n_markers)
        if _marker_subset_ok(subset, require_protected):
            return subset
    return None


def _render_input(
    h: int,
    w: int,
    segments: list[dict],
    marker_infos: list[dict],
    colors: list[int],
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for color, seg in zip(colors, segments):
        a = seg["a"]
        b = seg["b"]
        grid[a] = int(color)
        grid[b] = int(color)
    for info in marker_infos:
        r, c = info["marker"]
        grid[r, c] = RAY_COLOR
    return grid


def _semantic_stats(
    shape: tuple[int, int],
    segments: list[dict],
    marker_infos: list[dict],
    endpoints: set[tuple[int, int]],
) -> dict:
    segment_cells = {cell for seg in segments for cell in seg["cells"]}
    interior_cells = segment_cells - endpoints
    erased = set().union(*(info["even_nonendpoint"] for info in marker_infos))
    odd_hits = set().union(*(info["odd_hits"] for info in marker_infos))
    protected = set().union(*(info["even_endpoint"] for info in marker_infos))
    drawn = set().union(*(info["drawn_cells"] for info in marker_infos))

    out = _render_from_specs(shape, segments, [info["marker"] for info in marker_infos], endpoints)
    line_only = np.full(shape, BG, dtype=int)
    for seg in segments:
        for r, c in seg["cells"]:
            line_only[r, c] = int(seg["color"])

    return {
        "segment_cells": segment_cells,
        "interior_cells": interior_cells,
        "erased": erased,
        "odd_hits": odd_hits,
        "protected": protected,
        "drawn": drawn,
        "out": out,
        "line_only": line_only,
    }


def _attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n_pairs = random.randint(*PAIR_COUNT_RANGE)
    n_markers = random.randint(*MARKER_COUNT_RANGE)
    require_protected = random.random() < 0.45

    base_segments = _choose_segments(h, w, n_pairs)
    if base_segments is None:
        return None

    colors = random.sample(PAIR_COLORS, n_pairs)
    segments = []
    endpoints: set[tuple[int, int]] = set()
    segment_cells: set[tuple[int, int]] = set()
    for color, seg in zip(colors, base_segments):
        spec = {"color": int(color), "a": seg["a"], "b": seg["b"], "cells": seg["cells"]}
        segments.append(spec)
        endpoints.add(seg["a"])
        endpoints.add(seg["b"])
        segment_cells.update(seg["cells"])

    marker_infos = _choose_markers(h, w, segment_cells, endpoints, n_markers, require_protected)
    if marker_infos is None:
        return None

    grid = _render_input(h, w, segments, marker_infos, colors)
    expected = _render_from_specs((h, w), segments, [info["marker"] for info in marker_infos], endpoints)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    stats = _semantic_stats((h, w), segments, marker_infos, endpoints)
    if not stats["erased"]:
        return None
    if not stats["odd_hits"]:
        return None
    if require_protected and not stats["protected"]:
        return None
    if not (stats["interior_cells"] - stats["erased"]):
        return None
    if not (stats["drawn"] - stats["segment_cells"]):
        return None
    if np.array_equal(stats["line_only"], stats["out"]):
        return None

    grey_count = int(np.count_nonzero(expected == RAY_COLOR))
    if grey_count <= n_markers:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-engraving ARC input")
