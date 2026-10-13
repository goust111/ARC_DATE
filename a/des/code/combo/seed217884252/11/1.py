from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
FAMILIES = ("h", "v", "d1", "d2")
ORIENTED_STEPS = (
    ("h", (0, 1)),
    ("h", (0, -1)),
    ("v", (1, 0)),
    ("v", (-1, 0)),
    ("d1", (1, 1)),
    ("d1", (-1, -1)),
    ("d2", (1, -1)),
    ("d2", (-1, 1)),
)


def _line_point(family: str, const: int, t: int) -> tuple[int, int]:
    if family == "h":
        return const, t
    if family == "v":
        return t, const
    if family == "d1":
        return t, t + const
    assert family == "d2"
    return t, const - t


def _segment_cells_from_visible(coords: np.ndarray) -> tuple[str, int, int, int, set[tuple[int, int]]]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    if np.all(rows == rows[0]):
        family = "h"
        const = int(rows[0])
        t_vals = cols
    elif np.all(cols == cols[0]):
        family = "v"
        const = int(cols[0])
        t_vals = rows
    elif np.all(cols - rows == cols[0] - rows[0]):
        family = "d1"
        const = int(cols[0] - rows[0])
        t_vals = rows
    elif np.all(cols + rows == cols[0] + rows[0]):
        family = "d2"
        const = int(cols[0] + rows[0])
        t_vals = rows
    else:
        raise AssertionError("visible pixels for one color must lie on a supported straight line")

    t0 = int(np.min(t_vals))
    t1 = int(np.max(t_vals))
    cells = {_line_point(family, const, t) for t in range(t0, t1 + 1)}
    return family, const, t0, t1, cells


def _interior_on_segment(segment: dict, cell: tuple[int, int]) -> bool:
    r, c = cell
    family = segment["family"]
    if family == "h":
        return int(segment["t0"]) < c < int(segment["t1"])
    if family == "v":
        return int(segment["t0"]) < r < int(segment["t1"])
    return int(segment["t0"]) < r < int(segment["t1"])


def _unique_topological_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    n = len(colors)
    graph = [set() for _ in range(n)]
    indeg = [0] * n
    for u, v in edges:
        if v not in graph[u]:
            graph[u].add(v)
            indeg[v] += 1

    order: list[int] = []
    remaining = set(range(n))
    while remaining:
        zeros = sorted(i for i in remaining if indeg[i] == 0)
        assert len(zeros) == 1, "occlusion constraints must determine a unique total order"
        cur = zeros[0]
        remaining.remove(cur)
        order.append(colors[cur])
        for nxt in graph[cur]:
            indeg[nxt] -= 1
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split crossing colored line segments into independent layers and output the
    bottom-to-top layer order as a vertical color strip.

    Input structure:
    - Background is black.
    - Each non-black color belongs to exactly one straight line segment.
    - Segments are horizontal, vertical, or 45-degree diagonal.
    - At a crossing only the top segment's color is visible.

    Rule:
    - Reconstruct each segment's full span from its visible same-color collinear pixels.
    - Each crossing says the visible segment is above every hidden segment there.
    - Combine those local constraints into a unique global bottom-to-top order.
    - Output an Nx1 column, filled from top to bottom with the segment colors in
      bottom-to-top stacking order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    colors = sorted(int(c) for c in np.unique(g) if int(c) != BG)
    assert colors, "expected at least one segment"

    segments = []
    color_to_index = {}
    for idx, color in enumerate(colors):
        coords = np.argwhere(g == color)
        assert coords.shape[0] >= 2, "each segment must keep at least two visible pixels"
        family, const, t0, t1, cells = _segment_cells_from_visible(coords)
        segments.append(
            {
                "color": color,
                "family": family,
                "const": const,
                "t0": t0,
                "t1": t1,
                "cells": cells,
            }
        )
        color_to_index[color] = idx

    cell_to_members: dict[tuple[int, int], list[int]] = {}
    for idx, segment in enumerate(segments):
        for cell in segment["cells"]:
            cell_to_members.setdefault(cell, []).append(idx)

    edges: set[tuple[int, int]] = set()
    for cell, members in cell_to_members.items():
        if len(members) < 2:
            continue
        r, c = cell
        visible_color = int(g[r, c])
        assert visible_color != BG, "a crossing cell must show the top segment color"
        assert visible_color in color_to_index, "visible crossing color must belong to a segment"
        top_idx = color_to_index[visible_color]
        assert top_idx in members, "visible crossing color must lie on that reconstructed segment"
        for other_idx in members:
            if other_idx != top_idx:
                edges.add((other_idx, top_idx))

    order = _unique_topological_order(colors, edges)
    return np.asarray(order, dtype=int).reshape((-1, 1))


# --------------- generator ---------------

SEGMENT_COUNT_RANGE = (3, 6)
STEP_LEN_RANGE = (4, 7)
END_EXTEND_RANGE = (2, 5)
MARGIN_RANGE = (2, 5)
MAX_GRID_SIDE = 26
MAX_ATTEMPTS = 600
COLOR_POOL = tuple(range(1, 10))


def _segment_cells_from_endpoints(start: tuple[int, int], end: tuple[int, int]) -> tuple[str, int, int, int, set[tuple[int, int]]]:
    r0, c0 = start
    r1, c1 = end
    dr = r1 - r0
    dc = c1 - c0
    if dr == 0 and dc != 0:
        family = "h"
        const = r0
        t0, t1 = sorted((c0, c1))
    elif dc == 0 and dr != 0:
        family = "v"
        const = c0
        t0, t1 = sorted((r0, r1))
    elif abs(dr) == abs(dc) and dr != 0 and dc != 0 and (dr > 0) == (dc > 0):
        family = "d1"
        const = c0 - r0
        t0, t1 = sorted((r0, r1))
    elif abs(dr) == abs(dc) and dr != 0 and dc != 0 and (dr > 0) != (dc > 0):
        family = "d2"
        const = c0 + r0
        t0, t1 = sorted((r0, r1))
    else:
        raise AssertionError("segment endpoints must define a supported straight line")

    cells = {_line_point(family, const, t) for t in range(t0, t1 + 1)}
    return family, const, t0, t1, cells


def _shifted_cells(cells: Iterable[tuple[int, int]], shift_r: int, shift_c: int) -> set[tuple[int, int]]:
    return {(r + shift_r, c + shift_c) for r, c in cells}


def _build_hidden_scene(rng: random.Random, n_segments: int) -> list[dict] | None:
    colors = rng.sample(COLOR_POOL, n_segments)
    intersections = [(0, 0)]
    middle_steps: list[tuple[str, tuple[int, int]]] = []

    prev_family = None
    for _ in range(n_segments - 2):
        options = [(family, step) for family, step in ORIENTED_STEPS if family != prev_family]
        family, step = rng.choice(options)
        middle_steps.append((family, step))
        prev_family = family
        step_len = rng.randint(*STEP_LEN_RANGE)
        last_r, last_c = intersections[-1]
        intersections.append((last_r + step_len * step[0], last_c + step_len * step[1]))

    next_family = middle_steps[0][0] if middle_steps else None
    start_options = [(family, step) for family, step in ORIENTED_STEPS if family != next_family]
    start_family, start_step = rng.choice(start_options)

    prev_family = middle_steps[-1][0] if middle_steps else start_family
    end_options = [(family, step) for family, step in ORIENTED_STEPS if family != prev_family]
    end_family, end_step = rng.choice(end_options)

    hidden_segments: list[dict] = []
    before = rng.randint(*END_EXTEND_RANGE)
    after = rng.randint(*END_EXTEND_RANGE)
    center_r, center_c = intersections[0]
    start = (center_r - before * start_step[0], center_c - before * start_step[1])
    end = (center_r + after * start_step[0], center_c + after * start_step[1])
    family, const, t0, t1, cells = _segment_cells_from_endpoints(start, end)
    hidden_segments.append(
        {
            "color": colors[0],
            "family": family,
            "const": const,
            "t0": t0,
            "t1": t1,
            "start": start,
            "end": end,
            "cells": cells,
        }
    )

    for i, (family_name, step) in enumerate(middle_steps, start=1):
        a_r, a_c = intersections[i - 1]
        b_r, b_c = intersections[i]
        before = rng.randint(*END_EXTEND_RANGE)
        after = rng.randint(*END_EXTEND_RANGE)
        start = (a_r - before * step[0], a_c - before * step[1])
        end = (b_r + after * step[0], b_c + after * step[1])
        family, const, t0, t1, cells = _segment_cells_from_endpoints(start, end)
        assert family == family_name
        hidden_segments.append(
            {
                "color": colors[i],
                "family": family,
                "const": const,
                "t0": t0,
                "t1": t1,
                "start": start,
                "end": end,
                "cells": cells,
            }
        )

    before = rng.randint(*END_EXTEND_RANGE)
    after = rng.randint(*END_EXTEND_RANGE)
    center_r, center_c = intersections[-1]
    start = (center_r - before * end_step[0], center_c - before * end_step[1])
    end = (center_r + after * end_step[0], center_c + after * end_step[1])
    family, const, t0, t1, cells = _segment_cells_from_endpoints(start, end)
    assert family == end_family
    hidden_segments.append(
        {
            "color": colors[-1],
            "family": family,
            "const": const,
            "t0": t0,
            "t1": t1,
            "start": start,
            "end": end,
            "cells": cells,
        }
    )
    return hidden_segments


def _scene_is_clear(segments: list[dict]) -> bool:
    cell_counts: dict[tuple[int, int], int] = {}
    for seg in segments:
        for cell in seg["cells"]:
            cell_counts[cell] = cell_counts.get(cell, 0) + 1
            if cell_counts[cell] > 2:
                return False

    for i, seg_a in enumerate(segments):
        for seg_b in segments[i + 1 :]:
            inter = seg_a["cells"] & seg_b["cells"]
            if not inter:
                continue
            if len(inter) > 1:
                return False
            if seg_a["family"] == seg_b["family"]:
                return False
            cell = next(iter(inter))
            if not _interior_on_segment(seg_a, cell):
                return False
            if not _interior_on_segment(seg_b, cell):
                return False
    return True


def _embed_scene(rng: random.Random, segments: list[dict]) -> tuple[np.ndarray, list[dict]] | None:
    all_cells = [cell for seg in segments for cell in seg["cells"]]
    rows = [r for r, _c in all_cells]
    cols = [c for _r, c in all_cells]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    top = rng.randint(*MARGIN_RANGE)
    bottom = rng.randint(*MARGIN_RANGE)
    left = rng.randint(*MARGIN_RANGE)
    right = rng.randint(*MARGIN_RANGE)

    height = max_r - min_r + 1 + top + bottom
    width = max_c - min_c + 1 + left + right
    if height > MAX_GRID_SIDE or width > MAX_GRID_SIDE:
        return None

    shift_r = top - min_r
    shift_c = left - min_c
    shifted_segments: list[dict] = []
    for seg in segments:
        shifted_segments.append(
            {
                **seg,
                "cells": _shifted_cells(seg["cells"], shift_r, shift_c),
                "t0": seg["t0"] + shift_c if seg["family"] == "h" else seg["t0"] + shift_r,
                "t1": seg["t1"] + shift_c if seg["family"] == "h" else seg["t1"] + shift_r,
                "const": (
                    seg["const"] + shift_r
                    if seg["family"] == "h"
                    else seg["const"] + shift_c
                    if seg["family"] == "v"
                    else seg["const"] + shift_c - shift_r
                    if seg["family"] == "d1"
                    else seg["const"] + shift_r + shift_c
                ),
                "start": (seg["start"][0] + shift_r, seg["start"][1] + shift_c),
                "end": (seg["end"][0] + shift_r, seg["end"][1] + shift_c),
            }
        )

    grid = np.full((height, width), BG, dtype=int)
    for seg in shifted_segments:
        rr, cc = np.array(sorted(seg["cells"]), dtype=int).T
        grid[rr, cc] = seg["color"]
    return grid, shifted_segments


def _render_occluded_input(grid_shape: tuple[int, int], ordered_segments: list[dict]) -> np.ndarray:
    grid = np.full(grid_shape, BG, dtype=int)
    for seg in ordered_segments:
        rr, cc = np.array(sorted(seg["cells"]), dtype=int).T
        grid[rr, cc] = seg["color"]
    return grid


def _visible_counts_ok(grid: np.ndarray) -> bool:
    for color in [int(c) for c in np.unique(grid) if int(c) != BG]:
        if int(np.sum(grid == color)) < 2:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    n_segments = rng.randint(*SEGMENT_COUNT_RANGE)
    hidden_segments = _build_hidden_scene(rng, n_segments)
    if hidden_segments is None or not _scene_is_clear(hidden_segments):
        return None

    embedded = _embed_scene(rng, hidden_segments)
    if embedded is None:
        return None
    blank_grid, shifted_segments = embedded
    inp = _render_occluded_input(blank_grid.shape, shifted_segments)

    if not _visible_counts_ok(inp):
        return None

    expected = np.asarray([seg["color"] for seg in shifted_segments], dtype=int).reshape((-1, 1))
    try:
        out = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(out, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid layered-line input")
