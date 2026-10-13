from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BG = 0
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]


@dataclass(frozen=True)
class Segment:
    color: int
    orient: str  # "h" or "v"
    fixed: int
    start: int
    end: int


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool).tolist()
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(values):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    return runs


def _legend_bounds(grid: np.ndarray) -> tuple[int, int]:
    col_runs = _runs(np.any(grid != BG, axis=0))
    assert len(col_runs) >= 2, "expected a right-side legend separated from the line network"
    return col_runs[-1]


def _extract_legend_colors(legend: np.ndarray) -> list[int]:
    colors = [int(color) for color in np.unique(legend) if int(color) != BG]
    assert colors, "legend must contain at least one non-background color"

    ordered: list[tuple[int, int, int]] = []
    for color in colors:
        coords = np.argwhere(legend == int(color))
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        assert coords.shape[0] == (r1 - r0 + 1) * (c1 - c0 + 1), "legend entries must be solid rectangles"
        ordered.append((r0, c0, int(color)))
    ordered.sort()
    return [color for _r0, _c0, color in ordered]


def _extract_segments(scene: np.ndarray) -> dict[int, Segment]:
    colors = [int(color) for color in np.unique(scene) if int(color) != BG]
    assert colors, "scene must contain line segments"

    segments: dict[int, Segment] = {}
    for color in colors:
        coords = np.argwhere(scene == int(color))
        rows = np.unique(coords[:, 0])
        cols = np.unique(coords[:, 1])

        if len(rows) == 1 and len(cols) >= 2:
            segments[color] = Segment(color=color, orient="h", fixed=int(rows[0]), start=int(cols.min()), end=int(cols.max()))
            continue
        if len(cols) == 1 and len(rows) >= 2:
            segments[color] = Segment(color=color, orient="v", fixed=int(cols[0]), start=int(rows.min()), end=int(rows.max()))
            continue
        raise AssertionError(f"color {color} does not form one horizontal or vertical segment")

    return segments


def _occlusion_graph(scene: np.ndarray, segments: dict[int, Segment]) -> dict[int, set[int]]:
    edges = {int(color): set() for color in segments}
    horizontals = [seg for seg in segments.values() if seg.orient == "h"]
    verticals = [seg for seg in segments.values() if seg.orient == "v"]

    crossing_count = 0
    for h_seg in horizontals:
        for v_seg in verticals:
            if not (h_seg.start <= v_seg.fixed <= h_seg.end):
                continue
            if not (v_seg.start <= h_seg.fixed <= v_seg.end):
                continue

            crossing_count += 1
            visible = int(scene[h_seg.fixed, v_seg.fixed])
            assert visible in {int(h_seg.color), int(v_seg.color)}, "crossing must show one of the two segment colors"
            if visible == int(h_seg.color):
                edges[int(h_seg.color)].add(int(v_seg.color))
            else:
                edges[int(v_seg.color)].add(int(h_seg.color))

    if len(segments) > 1:
        assert crossing_count > 0, "multiple segments need at least one crossing"
    return edges


def _unique_topological_order(edges: dict[int, set[int]]) -> list[int]:
    indegree = {int(node): 0 for node in edges}
    for src, dsts in edges.items():
        for dst in dsts:
            assert dst in indegree and src in indegree
            indegree[int(dst)] += 1

    ready = sorted(node for node, degree in indegree.items() if degree == 0)
    order: list[int] = []
    while ready:
        assert len(ready) == 1, "occlusion evidence must determine a unique total order"
        node = ready.pop()
        order.append(int(node))
        for dst in sorted(edges[node]):
            indegree[int(dst)] -= 1
            if indegree[int(dst)] == 0:
                ready.append(int(dst))
        ready.sort()

    assert len(order) == len(edges), "occlusion graph must be acyclic"
    return order


def _draw_segment(canvas: np.ndarray, segment: Segment, color: int) -> None:
    if segment.orient == "h":
        canvas[int(segment.fixed), int(segment.start) : int(segment.end) + 1] = int(color)
    else:
        canvas[int(segment.start) : int(segment.end) + 1, int(segment.fixed)] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the line-depth order from crossing occlusions and recolor the full segments by the right-side depth legend.

    Input structure:
    - The background is black.
    - The left part of the grid contains several 1-pixel-wide horizontal or vertical line segments, each in a unique non-black color.
    - Some horizontal/vertical pairs cross; at each crossing only the shallower line's color is visible, so the deeper line is missing that one pixel.
    - The right side contains a separated vertical legend of solid color blocks, ordered from shallowest at the top to deepest at the bottom.

    Rule:
    - Split off the right-side legend and read its colors from top to bottom.
    - For each scene color, recover the unique straight horizontal or vertical segment by taking the visible row/column and its visible span endpoints.
    - At every reconstructed horizontal/vertical crossing, whichever color is visible is above the other one; topologically sort these pairwise relations into the unique shallow-to-deep order.
    - Recolor depth rank k with the k-th legend color, then redraw the recovered full segments from shallow to deep so each previously hidden crossing pixel is restored to the deeper segment's new color.

    Key insight:
    - A one-pixel break at a crossing is evidence of occlusion, not of segment termination: reconstructing every full straight segment and chaining the local over/under clues yields the global depth order, which the legend then turns into the new color assignment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    legend_c0, legend_c1 = _legend_bounds(g)
    scene = g[:, :legend_c0]
    legend = g[:, legend_c0 : legend_c1 + 1]

    legend_colors = _extract_legend_colors(legend)
    segments = _extract_segments(scene)
    assert len(legend_colors) == len(segments), "legend size must match the number of segments"

    edges = _occlusion_graph(scene, segments)
    shallow_to_deep = _unique_topological_order(edges)
    recolor = dict(zip(shallow_to_deep, legend_colors))

    out = g.copy()
    out[:, :legend_c0] = BG
    scene_out = out[:, :legend_c0]
    for color in shallow_to_deep:
        _draw_segment(scene_out, segments[int(color)], recolor[int(color)])
    return out


SEGMENT_COUNT_CHOICES = [3, 4, 5, 6]
START_ORIENTATIONS = ["h", "v"]
POSITION_GAPS = [2, 3, 4]
TAIL_CHOICES = [1, 2, 3]
LEGEND_BLOCK_CHOICES = [1, 2]
LEGEND_GAP_CHOICES = [1, 2]
SEPARATOR_CHOICES = [1, 2, 3]
MAX_GENERATION_ATTEMPTS = 400


def _sample_positions(count: int) -> tuple[list[int], int]:
    assert count >= 1
    current = random.choice(TAIL_CHOICES)
    positions = [int(current)]
    for _ in range(count - 1):
        current += random.choice(POSITION_GAPS)
        positions.append(int(current))
    size = int(positions[-1] + random.choice(TAIL_CHOICES) + 1)
    return positions, size


def _absolute_segment(segment: Segment, row_offset: int, col_offset: int) -> Segment:
    if segment.orient == "h":
        return Segment(
            color=int(segment.color),
            orient="h",
            fixed=int(segment.fixed + row_offset),
            start=int(segment.start + col_offset),
            end=int(segment.end + col_offset),
        )
    return Segment(
        color=int(segment.color),
        orient="v",
        fixed=int(segment.fixed + col_offset),
        start=int(segment.start + row_offset),
        end=int(segment.end + row_offset),
    )


def _scene_segments(
    count: int,
    start_orient: str,
    colors: list[int],
) -> tuple[list[Segment], int, int]:
    orient_seq = [start_orient if i % 2 == 0 else ("v" if start_orient == "h" else "h") for i in range(count)]
    h_count = sum(orient == "h" for orient in orient_seq)
    v_count = count - h_count
    assert h_count >= 1 and v_count >= 1

    rows, scene_h = _sample_positions(h_count)
    cols, scene_w = _sample_positions(v_count)
    min_row = min(rows)
    max_row = max(rows)
    min_col = min(cols)
    max_col = max(cols)

    h_rows = random.sample(rows, len(rows))
    v_cols = random.sample(cols, len(cols))
    h_idx = 0
    v_idx = 0
    segments: list[Segment] = []
    for color, orient in zip(colors, orient_seq):
        if orient == "h":
            row = int(h_rows[h_idx])
            h_idx += 1
            start = random.randint(0, min_col - 1)
            end = random.randint(max_col + 1, scene_w - 1)
            segments.append(Segment(color=int(color), orient="h", fixed=row, start=int(start), end=int(end)))
        else:
            col = int(v_cols[v_idx])
            v_idx += 1
            start = random.randint(0, min_row - 1)
            end = random.randint(max_row + 1, scene_h - 1)
            segments.append(Segment(color=int(color), orient="v", fixed=col, start=int(start), end=int(end)))

    return segments, int(scene_h), int(scene_w)


def _paint_legend(grid: np.ndarray, colors: list[int], top: int, left: int, block: int, gap: int) -> None:
    for idx, color in enumerate(colors):
        r0 = int(top + idx * (block + gap))
        grid[r0 : r0 + block, left : left + block] = int(color)


def _render_input(
    shape: tuple[int, int],
    segments: list[Segment],
    legend_colors: list[int],
    scene_top: int,
    scene_left: int,
    legend_top: int,
    legend_left: int,
    legend_block: int,
    legend_gap: int,
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    _paint_legend(grid, legend_colors, legend_top, legend_left, legend_block, legend_gap)
    for segment in reversed(segments):
        _draw_segment(grid, _absolute_segment(segment, scene_top, scene_left), int(segment.color))
    return grid


def _render_expected_output(
    shape: tuple[int, int],
    segments: list[Segment],
    legend_colors: list[int],
    scene_top: int,
    scene_left: int,
    legend_top: int,
    legend_left: int,
    legend_block: int,
    legend_gap: int,
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    _paint_legend(out, legend_colors, legend_top, legend_left, legend_block, legend_gap)
    for segment, color in zip(segments, legend_colors):
        _draw_segment(out, _absolute_segment(segment, scene_top, scene_left), int(color))
    return out


def _generate_attempt() -> np.ndarray | None:
    count = random.choice(SEGMENT_COUNT_CHOICES)
    start_orient = random.choice(START_ORIENTATIONS)
    line_colors = random.sample(COLOR_POOL, count)
    legend_colors = random.sample(COLOR_POOL, count)
    segments, scene_h, scene_w = _scene_segments(count, start_orient, line_colors)

    legend_block = random.choice(LEGEND_BLOCK_CHOICES)
    legend_gap = random.choice(LEGEND_GAP_CHOICES)
    separator = random.choice(SEPARATOR_CHOICES)
    legend_h = count * legend_block + (count - 1) * legend_gap
    legend_w = legend_block

    min_h = max(scene_h, legend_h)
    min_w = scene_w + separator + legend_w
    if min_h > 30 or min_w > 30:
        return None

    grid_h = random.randint(min_h, 30)
    grid_w = random.randint(min_w, 30)
    scene_top = random.randint(0, grid_h - scene_h)
    legend_top = random.randint(0, grid_h - legend_h)
    scene_left = random.randint(0, grid_w - min_w)
    legend_left = scene_left + scene_w + separator

    inp = _render_input(
        (grid_h, grid_w),
        segments,
        legend_colors,
        scene_top,
        scene_left,
        legend_top,
        legend_left,
        legend_block,
        legend_gap,
    )
    expected = _render_expected_output(
        (grid_h, grid_w),
        segments,
        legend_colors,
        scene_top,
        scene_left,
        legend_top,
        legend_left,
        legend_block,
        legend_gap,
    )

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
