from __future__ import annotations

import numpy as np

from arc_utils import find_components, get_bg
from itertools import combinations
import random

BBox = tuple[int, int, int, int]
Projection = tuple[int, int, BBox]


def _full_edge_bar(g: np.ndarray, bg: int) -> tuple[str, int]:
    h, w = g.shape
    edge_values = [
        ("top", g[0, :]),
        ("bottom", g[h - 1, :]),
        ("left", g[:, 0]),
        ("right", g[:, w - 1]),
    ]

    candidates: list[tuple[str, int]] = []
    for name, values in edge_values:
        color = int(values[0])
        if color != bg and np.all(values == color):
            candidates.append((name, color))

    assert len(candidates) == 1, "expected exactly one solid non-background edge bar"
    return candidates[0]


def _is_bar_component(comp: dict, edge: str, wall_color: int, h: int, w: int) -> bool:
    if int(comp["color"]) != wall_color:
        return False

    r0, c0, r1, c1 = comp["bbox"]
    return (
        (edge == "top" and r0 == r1 == 0 and c0 == 0 and c1 == w - 1)
        or (edge == "bottom" and r0 == r1 == h - 1 and c0 == 0 and c1 == w - 1)
        or (edge == "left" and c0 == c1 == 0 and r0 == 0 and r1 == h - 1)
        or (edge == "right" and c0 == c1 == w - 1 and r0 == 0 and r1 == h - 1)
    )


def _segment_projections(g: np.ndarray, bg: int, edge: str, wall_color: int) -> list[Projection]:
    h, w = g.shape
    projections: list[Projection] = []
    bar_count = 0

    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    for comp in components:
        if _is_bar_component(comp, edge, wall_color, h, w):
            bar_count += 1
            continue

        r0, c0, r1, c1 = comp["bbox"]
        color = int(comp["color"])
        area = int(comp["area"])

        if edge in {"top", "bottom"}:
            assert r0 == r1, "all non-wall objects must be horizontal segments"
            assert area == c1 - c0 + 1, "segment must be a solid one-cell-thick bar"
            if edge == "top":
                fill_bbox = (1, c0, r0, c1)
                distance = r0 - 1
            else:
                fill_bbox = (r0, c0, h - 2, c1)
                distance = h - 2 - r0
        else:
            assert c0 == c1, "all non-wall objects must be vertical segments"
            assert area == r1 - r0 + 1, "segment must be a solid one-cell-thick bar"
            if edge == "left":
                fill_bbox = (r0, 1, r1, c0)
                distance = c0 - 1
            else:
                fill_bbox = (r0, c0, r1, w - 2)
                distance = w - 2 - c0

        assert distance >= 0, "segments must sit between the background and the edge bar"
        projections.append((int(distance), color, fill_bbox))

    assert bar_count == 1, "expected exactly one wall component"
    assert projections, "expected at least one colored segment"
    return projections


def _assert_no_equal_distance_color_conflicts(projections: list[Projection]) -> None:
    for distance in {projection[0] for projection in projections}:
        seen: dict[tuple[int, int], int] = {}
        for candidate_distance, color, (r0, c0, r1, c1) in projections:
            if candidate_distance != distance:
                continue
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    previous = seen.get((r, c))
                    assert previous is None or previous == color, "equal-distance projections must not conflict"
                    seen[(r, c)] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend every colored segment straight toward the single solid edge bar, with nearer segments covering farther projections.

    Input structure
    - The background is the dominant color.
    - Exactly one full grid edge is a solid non-background bar.
    - All other foreground objects are 4-connected, monochrome, one-cell-thick segments parallel to that edge.

    Rule
    - Each segment paints the rectangular strip from itself to the cell before the edge bar, perpendicular to the segment.
    - Strips farther from the edge bar are painted first; closer strips overwrite them wherever their rectangular regions overlap.
    - The original edge bar remains unchanged.

    Key insight
    - The edge bar acts as the direction cue and occluding wall: every segment casts a same-color projection toward it, and distance to the wall determines visible layering.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    edge, wall_color = _full_edge_bar(g, bg)
    projections = _segment_projections(g, bg, edge, wall_color)
    _assert_no_equal_distance_color_conflicts(projections)

    out = g.copy()
    for _distance, color, (r0, c0, r1, c1) in sorted(projections, key=lambda item: item[0], reverse=True):
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out

# --------------- generator (o5x) ---------------

# Generator configuration.
GRID_SIZE_RANGE = (15, 19)
EDGES = ("top", "bottom", "left", "right")
HORIZONTAL_EDGES = {"top", "bottom"}
BACKGROUND_COLOR = 0
WALL_COLOR = 5
SEGMENT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
SEGMENT_COUNT_RANGE = (3, 4)

# Geometry and retry limits.
DISTANCE_RANGE = (2, 14)
MIN_DISTANCE_GAP = 2
PERP_DISTANCE_MARGIN = 3
AXIS_LENGTH_MARGIN = 2
SEGMENT_LENGTH_RANGE = (2, 9)
RETRY_LIMIT = 300


def _gen_distance_set(count: int, max_distance: int) -> list[int] | None:
    min_distance, _ = DISTANCE_RANGE
    valid = [
        combo
        for combo in combinations(range(min_distance, max_distance + 1), count)
        if all(b - a >= MIN_DISTANCE_GAP for a, b in zip(combo, combo[1:]))
    ]
    if not valid:
        return None
    return list(random.choice(valid))


def _gen_ordered_lengths(count: int, max_length: int) -> list[int]:
    min_length, _ = SEGMENT_LENGTH_RANGE
    lengths = [random.randint(min_length, max_length) for _ in range(count)]
    lengths.sort()
    return lengths


def _gen_spans(lengths: list[int], axis_len: int) -> list[tuple[int, int]]:
    overlap_pair = set(random.sample(range(len(lengths)), 2))
    shared_axis_cell = random.randrange(axis_len)
    spans: list[tuple[int, int]] = []

    for i, length in enumerate(lengths):
        if i in overlap_pair:
            lo = max(0, shared_axis_cell - length + 1)
            hi = min(shared_axis_cell, axis_len - length)
            start = random.randint(lo, hi)
        else:
            start = random.randint(0, axis_len - length)
        spans.append((start, start + length - 1))

    return spans


def _gen_segments(edge: str, h: int, w: int) -> list[tuple[int, int, tuple[int, int]]] | None:
    count = random.randint(*SEGMENT_COUNT_RANGE)
    if edge in HORIZONTAL_EDGES:
        perp_dim, axis_len = h, w
    else:
        perp_dim, axis_len = w, h
    max_distance = min(DISTANCE_RANGE[1], perp_dim - PERP_DISTANCE_MARGIN)
    max_length = min(axis_len - AXIS_LENGTH_MARGIN, SEGMENT_LENGTH_RANGE[1])

    distances = _gen_distance_set(count, max_distance)
    if distances is None:
        return None

    lengths = _gen_ordered_lengths(count, max_length)
    spans = _gen_spans(lengths, axis_len)
    colors = random.sample(SEGMENT_COLORS, count)
    return list(zip(distances, colors, spans))


def _gen_projection_bbox(edge: str, h: int, w: int, distance: int, span: tuple[int, int]) -> BBox:
    s0, s1 = span
    if edge == "top":
        row = distance + 1
        return 1, s0, row, s1
    if edge == "bottom":
        row = h - 2 - distance
        return row, s0, h - 2, s1
    if edge == "left":
        col = distance + 1
        return s0, 1, s1, col
    col = w - 2 - distance
    return s0, col, s1, w - 2


def _gen_fill_bbox(grid: np.ndarray, bbox: BBox, color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _gen_paint_wall(grid: np.ndarray, edge: str) -> None:
    if edge == "top":
        grid[0, :] = WALL_COLOR
    elif edge == "bottom":
        grid[-1, :] = WALL_COLOR
    elif edge == "left":
        grid[:, 0] = WALL_COLOR
    else:
        grid[:, -1] = WALL_COLOR


def _gen_draw_segment(grid: np.ndarray, edge: str, distance: int, span: tuple[int, int], color: int) -> None:
    s0, s1 = span
    h, w = grid.shape
    if edge == "top":
        grid[distance + 1, s0 : s1 + 1] = color
    elif edge == "bottom":
        grid[h - 2 - distance, s0 : s1 + 1] = color
    elif edge == "left":
        grid[s0 : s1 + 1, distance + 1] = color
    else:
        grid[s0 : s1 + 1, w - 2 - distance] = color


def _gen_render_output(h: int, w: int, edge: str, segments: list[tuple[int, int, tuple[int, int]]]) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    _gen_paint_wall(out, edge)
    for distance, color, span in sorted(segments, key=lambda item: item[0], reverse=True):
        bbox = _gen_projection_bbox(edge, h, w, distance, span)
        _gen_fill_bbox(out, bbox, color)
    return out


def _gen_derive_input(out: np.ndarray, edge: str, segments: list[tuple[int, int, tuple[int, int]]]) -> np.ndarray:
    h, w = out.shape
    inp = out.copy()
    for distance, _color, span in segments:
        bbox = _gen_projection_bbox(edge, h, w, distance, span)
        _gen_fill_bbox(inp, bbox, BACKGROUND_COLOR)
    for distance, color, span in segments:
        _gen_draw_segment(inp, edge, distance, span, color)
    _gen_paint_wall(inp, edge)
    return inp


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)
    edge = random.choice(EDGES)
    segments = _gen_segments(edge, h, w)
    if segments is None:
        return None

    out = _gen_render_output(h, w, edge, segments)
    inp = _gen_derive_input(out, edge, segments)
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {RETRY_LIMIT} attempts")
