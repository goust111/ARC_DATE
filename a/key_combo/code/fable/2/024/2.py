from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import random
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    arc_utils_dir = str(_ARC_UTILS_DIR)
    if arc_utils_dir not in sys.path:
        sys.path.append(arc_utils_dir)

try:
    from arc_utils import bbox_coords, get_bg
except Exception:
    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])

    def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)


@dataclass(frozen=True)
class Obstacle:
    color: int
    r0: int
    c0: int
    r1: int
    c1: int
    markers: tuple[tuple[int, int], tuple[int, int]]


@dataclass(frozen=True)
class BeamSpec:
    color: int
    source_col: int
    lane: tuple[int, int]
    obstacle_colors: tuple[int, ...]


_BG = 0
_GEN_CALL_COUNT = 0
_SCENARIOS = (
    (2, (2, 1)),
    (3, (1, 2, 1)),
    (2, (3, 1)),
    (3, (2, 1, 2)),
)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _parse_scene(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[Obstacle], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    sources = [(int(g[0, c]), int(c)) for c in range(w) if int(g[0, c]) != bg]
    assert sources, "expected at least one source pixel on the top edge"
    source_cols = [c for _color, c in sources]
    assert all(source_cols[i + 1] - source_cols[i] > 1 for i in range(len(source_cols) - 1)), "top-edge sources must not touch"

    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r in range(1, h):
        for c in range(w):
            color = int(g[r, c])
            if color != bg:
                groups[color].append((int(r), int(c)))

    obstacles: list[Obstacle] = []
    marker_set: set[tuple[int, int]] = set()
    for color, coords_list in groups.items():
        assert len(coords_list) == 2, f"marker color {color} must appear exactly twice below the top row"
        coords = np.array(coords_list, dtype=int)
        r0, c0, r1, c1 = bbox_coords(coords)
        assert r0 < r1 and c0 < c1, "marker pair must define a proper diagonal rectangle"
        obstacles.append(
            Obstacle(
                color=int(color),
                r0=int(r0),
                c0=int(c0),
                r1=int(r1),
                c1=int(c1),
                markers=(coords_list[0], coords_list[1]),
            )
        )
        marker_set.update(coords_list)

    for i, ob_a in enumerate(obstacles):
        for ob_b in obstacles[i + 1 :]:
            assert not _boxes_overlap((ob_a.r0, ob_a.c0, ob_a.r1, ob_a.c1), (ob_b.r0, ob_b.c0, ob_b.r1, ob_b.c1)), "invisible rectangles must not overlap"

    obstacles.sort(key=lambda ob: (ob.r0, ob.c0, ob.color))
    return bg, sources, obstacles, marker_set


def _next_obstacle(obstacles: list[Obstacle], row: int, col: int) -> Obstacle | None:
    candidates = [ob for ob in obstacles if ob.r0 > row and ob.c0 <= col <= ob.c1]
    if not candidates:
        return None
    top = min(ob.r0 for ob in candidates)
    nearest = [ob for ob in candidates if ob.r0 == top]
    assert len(nearest) == 1, "the next blocking rectangle must be uniquely determined"
    return nearest[0]


def _trace_beam(height: int, source_col: int, obstacles: list[Obstacle]) -> tuple[list[tuple[int, int]], list[int]]:
    path: list[tuple[int, int]] = [(0, int(source_col))]
    hits: list[int] = []
    row = 0
    col = int(source_col)

    while True:
        ob = _next_obstacle(obstacles, row, col)
        if ob is None:
            for rr in range(row + 1, height):
                path.append((int(rr), int(col)))
            return path, hits

        bend_r = int(ob.r0 - 1)
        for rr in range(row + 1, bend_r + 1):
            path.append((int(rr), int(col)))

        left_distance = int(col - ob.c0)
        right_distance = int(ob.c1 - col)
        if left_distance <= right_distance:
            new_col = int(ob.c0 - 1)
        else:
            new_col = int(ob.c1 + 1)

        step = 1 if new_col > col else -1
        for cc in range(col + step, new_col + step, step):
            path.append((bend_r, int(cc)))

        hits.append(int(ob.color))
        row = bend_r
        col = new_col


def _paint_path_cell(
    out: np.ndarray,
    *,
    row: int,
    col: int,
    color: int,
    bg: int,
    marker_set: set[tuple[int, int]],
) -> None:
    cell = (int(row), int(col))
    if cell in marker_set:
        return
    current = int(out[row, col])
    assert current in (bg, color), "beam paths must not overlap each other"
    out[row, col] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw each top-edge source beam downward, detouring around every invisible rectangle implied by a same-color diagonal marker pair.

    Input structure:
    - The background grid has one or more isolated colored source pixels on the top edge.
    - Away from the top edge, each non-background color appears exactly twice as two isolated diagonal corner markers of one hidden solid rectangle.
    - The rectangles themselves are not drawn, and different hidden rectangles do not overlap.

    Rule:
    - Recover each hidden rectangle from the bounding box of its same-color marker pair.
    - For every source pixel, trace a beam in the source color straight downward.
    - Whenever the current beam column first meets a rectangle below it, turn on the row just above that rectangle and move horizontally to the nearer outside column; if both sides are equally near, choose the left outside column.
    - Continue downward from that new column and repeat for later rectangles until reaching the bottom edge.
    - Keep all marker pixels unchanged; the drawn beam may pass around them but never overwrite them.

    Key insight:
    - The visible marker pairs are only sparse residues of full solid obstacles, so the beam path is determined by first reconstructing every hidden rectangle and then applying the cascaded nearest-side detours in the grid's original row/column frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, _width = g.shape
    bg, sources, obstacles, marker_set = _parse_scene(g)

    out = g.copy()
    for color, source_col in sorted(sources, key=lambda item: item[1]):
        path, _hits = _trace_beam(height, source_col, obstacles)
        for row, col in path:
            _paint_path_cell(out, row=row, col=col, color=color, bg=bg, marker_set=marker_set)
    return out


def _lane_width(count: int) -> int:
    if count >= 3:
        return random.randint(7, 8)
    if count == 2:
        return random.randint(6, 8)
    return random.randint(5, 7)


def _build_mode_sequences(scenario_idx: int) -> list[tuple[str, ...]]:
    flip = random.random() < 0.5

    def swap(mode: str) -> str:
        if mode == "L":
            return "R"
        if mode == "R":
            return "L"
        return mode

    templates = {
        0: [("T", "R"), ("L",)],
        1: [("R",), ("L", "R"), ("L",)],
        2: [("T", "R", "L"), ("R",)],
        3: [("R", "L"), ("L",), ("L", "R")],
    }[int(scenario_idx)]
    if not flip:
        return [tuple(seq) for seq in templates]
    return [tuple(swap(mode) for mode in seq) for seq in templates]


def _sample_row_plan(count: int) -> tuple[list[int], list[int], int]:
    heights = [random.randint(2, 4) for _ in range(count)]
    gaps = [random.randint(2, 4) for _ in range(max(0, count - 1))]

    tops = [2]
    bottoms = [tops[0] + heights[0] - 1]
    for idx in range(1, count):
        top = bottoms[-1] + gaps[idx - 1]
        tops.append(int(top))
        bottoms.append(int(top + heights[idx] - 1))
    return heights, tops, int(bottoms[-1])


def _candidate_columns(
    lane_c0: int,
    lane_c1: int,
    current_col: int,
    mode: str,
    height: int,
) -> list[tuple[int, int, int]]:
    candidates: list[tuple[int, int, int]] = []
    for c0 in range(lane_c0 + 1, current_col + 1):
        for c1 in range(current_col, lane_c1):
            width = c1 - c0 + 1
            if width < 2 or width > 5:
                continue
            if height == 2 and width == 2:
                continue

            left_distance = current_col - c0
            right_distance = c1 - current_col
            if left_distance == right_distance:
                kind = "T"
                new_col = c0 - 1
            elif left_distance < right_distance:
                kind = "L"
                new_col = c0 - 1
            else:
                kind = "R"
                new_col = c1 + 1

            if kind == mode:
                candidates.append((int(c0), int(c1), int(new_col)))
    random.shuffle(candidates)
    return candidates


def _plan_columns_for_beam(
    lane: tuple[int, int],
    modes: tuple[str, ...],
    heights: list[int],
) -> tuple[int, list[tuple[int, int]]] | None:
    lane_c0, lane_c1 = lane
    source_choices = list(range(lane_c0 + 1, lane_c1))
    random.shuffle(source_choices)

    def dfs(idx: int, current_col: int) -> list[tuple[int, int]] | None:
        if idx == len(modes):
            return []
        for c0, c1, new_col in _candidate_columns(lane_c0, lane_c1, current_col, modes[idx], heights[idx]):
            tail = dfs(idx + 1, new_col)
            if tail is not None:
                return [(int(c0), int(c1))] + tail
        return None

    for source_col in source_choices:
        columns = dfs(0, int(source_col))
        if columns is not None:
            return int(source_col), columns
    return None


def _render_input(
    shape: tuple[int, int],
    beams: list[BeamSpec],
    obstacles: list[Obstacle],
) -> np.ndarray:
    grid = np.full(shape, _BG, dtype=int)
    for beam in beams:
        grid[0, beam.source_col] = int(beam.color)
    for ob in obstacles:
        for r, c in ob.markers:
            grid[r, c] = int(ob.color)
    return grid


def _render_expected_output(
    shape: tuple[int, int],
    beams: list[BeamSpec],
    obstacles: list[Obstacle],
) -> tuple[np.ndarray, dict[int, tuple[list[tuple[int, int]], list[int]]]]:
    grid = _render_input(shape, beams, obstacles)
    marker_set = {marker for ob in obstacles for marker in ob.markers}
    traces: dict[int, tuple[list[tuple[int, int]], list[int]]] = {}
    for beam in beams:
        beam_obstacles = [ob for ob in obstacles if ob.color in beam.obstacle_colors]
        beam_obstacles.sort(key=lambda ob: ob.r0)
        traces[int(beam.color)] = _trace_beam(shape[0], beam.source_col, beam_obstacles)
        for row, col in traces[int(beam.color)][0]:
            if (row, col) not in marker_set:
                grid[row, col] = int(beam.color)
    return grid, traces


def _validate_semantics(
    inp: np.ndarray,
    beams: list[BeamSpec],
    obstacles: list[Obstacle],
    expected: np.ndarray,
    traces: dict[int, tuple[list[tuple[int, int]], list[int]]],
) -> None:
    assert np.array_equal(main(inp), expected), "solver/output mismatch"

    all_path_cells: set[tuple[int, int]] = set()
    marker_set = {marker for ob in obstacles for marker in ob.markers}
    for beam in beams:
        path, hits = traces[int(beam.color)]
        assert tuple(hits) == beam.obstacle_colors, "beam must hit its assigned obstacles in order"
        assert marker_set.isdisjoint(path), "a beam must never cover a marker pixel"
        for cell in path:
            assert cell not in all_path_cells, "beam paths must be pairwise disjoint"
            all_path_cells.add(cell)

    for i, ob_a in enumerate(obstacles):
        for ob_b in obstacles[i + 1 :]:
            assert not _boxes_overlap((ob_a.r0, ob_a.c0, ob_a.r1, ob_a.c1), (ob_b.r0, ob_b.c0, ob_b.r1, ob_b.c1))

    for ob in obstacles:
        modified = inp.copy()
        for r, c in ob.markers:
            modified[r, c] = _BG
        changed = main(modified)
        assert not np.array_equal(changed, expected), "every obstacle must affect the output"


def _generate_attempt(scenario_idx: int, counts: tuple[int, ...]) -> np.ndarray | None:
    mode_sequences = _build_mode_sequences(scenario_idx)
    assert len(mode_sequences) == len(counts)

    lane_widths = [_lane_width(count) for count in counts]
    gap = 2
    margin = 1
    lanes: list[tuple[int, int]] = []
    cursor = margin
    for width in lane_widths:
        lanes.append((int(cursor), int(cursor + width - 1)))
        cursor += width + gap
    width = lanes[-1][1] + margin + 1
    if width > 30:
        return None

    row_specs = [_sample_row_plan(count) for count in counts]
    min_height = max(last_bottom + 2 for _heights, _tops, last_bottom in row_specs)
    height_lo = max(14, min_height)
    height_hi = min(28, min_height + 6)
    if height_lo > height_hi:
        return None
    height = random.randint(height_lo, height_hi)

    shifts = [random.randint(0, height - 2 - last_bottom) for _heights, _tops, last_bottom in row_specs]
    total_colors = len(counts) + sum(counts)
    if total_colors > 9:
        return None
    palette = random.sample(list(range(1, 10)), total_colors)
    source_colors = palette[: len(counts)]
    obstacle_colors = palette[len(counts) :]
    obstacle_color_idx = 0

    obstacles: list[Obstacle] = []
    beams: list[BeamSpec] = []
    marker_touch: set[tuple[int, int]] = set()

    for beam_idx, (count, lane, modes, row_spec, shift, source_color) in enumerate(
        zip(counts, lanes, mode_sequences, row_specs, shifts, source_colors)
    ):
        heights, base_tops, _last_bottom = row_spec
        plan = _plan_columns_for_beam(lane, modes, heights)
        if plan is None:
            return None
        source_col, columns = plan

        assigned_colors: list[int] = []
        for obs_idx in range(count):
            r0 = base_tops[obs_idx] + shift
            height_i = heights[obs_idx]
            r1 = r0 + height_i - 1
            c0, c1 = columns[obs_idx]
            if random.random() < 0.5:
                markers = ((r0, c0), (r1, c1))
            else:
                markers = ((r0, c1), (r1, c0))

            if any(max(abs(r - rr), abs(c - cc)) <= 1 for r, c in markers for rr, cc in marker_touch):
                return None

            obstacle_color = obstacle_colors[obstacle_color_idx]
            obstacle_color_idx += 1
            assigned_colors.append(int(obstacle_color))
            obstacles.append(
                Obstacle(
                    color=int(obstacle_color),
                    r0=int(r0),
                    c0=int(c0),
                    r1=int(r1),
                    c1=int(c1),
                    markers=markers,
                )
            )
            marker_touch.update(markers)

        beams.append(
            BeamSpec(
                color=int(source_color),
                source_col=int(source_col),
                lane=lane,
                obstacle_colors=tuple(assigned_colors),
            )
        )

    inp = _render_input((height, width), beams, obstacles)
    expected, traces = _render_expected_output((height, width), beams, obstacles)
    _validate_semantics(inp, beams, obstacles, expected, traces)
    return inp


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    for _ in range(400):
        scenario_idx = _GEN_CALL_COUNT % len(_SCENARIOS)
        _beam_count, counts = _SCENARIOS[scenario_idx]
        grid = _generate_attempt(scenario_idx, counts)
        _GEN_CALL_COUNT += 1
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-boulder detour input")
