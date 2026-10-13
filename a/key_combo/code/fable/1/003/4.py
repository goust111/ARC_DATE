from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np


ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import get_bg


UP = (-1, 0)
RIGHT = (0, 1)
DOWN = (1, 0)
LEFT = (0, -1)

SIDE_TO_DIR = {"top": UP, "right": RIGHT, "bottom": DOWN, "left": LEFT}
INWARD_DIR = {"top": DOWN, "right": LEFT, "bottom": UP, "left": RIGHT}
DIR_TO_SIDE = {value: key for key, value in SIDE_TO_DIR.items()}
ORTHOGONAL_DIRS = {
    UP: (LEFT, RIGHT),
    DOWN: (LEFT, RIGHT),
    LEFT: (UP, DOWN),
    RIGHT: (UP, DOWN),
}

PALETTE = tuple(range(10))
GEN_RETRIES = 400
MIN_GRID = 11
MAX_GRID = 22
MIN_ROUTE_POINTS = 2
MAX_ROUTE_POINTS = 5
MIN_SEGMENT_LEN = 2
MIN_FINAL_SEGMENT_LEN = 2
MAX_EXTRA_DISTRACTORS = 2


def _side_coords(side: str, h: int, w: int) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, c) for c in range(1, w - 1)]
    if side == "right":
        return [(r, w - 1) for r in range(1, h - 1)]
    if side == "bottom":
        return [(h - 1, c) for c in range(1, w - 1)]
    if side == "left":
        return [(r, 0) for r in range(1, h - 1)]
    raise ValueError(f"unknown side: {side}")


def _segment_cells(start: tuple[int, int], direction: tuple[int, int], steps: int) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = direction
    return [(r + dr * step, c + dc * step) for step in range(1, steps + 1)]


def _available_interior_steps(
    start: tuple[int, int], direction: tuple[int, int], h: int, w: int
) -> int:
    r, c = start
    dr, dc = direction
    steps = 0
    while True:
        r += dr
        c += dc
        if not (1 <= r < h - 1 and 1 <= c < w - 1):
            return steps
        steps += 1


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _parse_legend(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 5 and w >= 5

    bg = int(get_bg(g))
    corners = [int(g[0, 0]), int(g[0, w - 1]), int(g[h - 1, 0]), int(g[h - 1, w - 1])]
    assert all(color == bg for color in corners), "corners must be background"

    side_arrays = {side: [int(g[r, c]) for r, c in _side_coords(side, h, w)] for side in SIDE_TO_DIR}
    border_colors = [color for values in side_arrays.values() for color in values]
    counts = Counter(border_colors)
    start_candidates = [color for color, count in counts.items() if color != bg and count == 1]
    assert len(start_candidates) == 1, "expected one unique start color on the border"
    start_color = int(start_candidates[0])

    side_to_color: dict[str, int] = {}
    gap_side = None
    gap_pos = None
    for side, values in side_arrays.items():
        unique_values = set(values)
        assert bg not in unique_values, "border sides must not use background"
        if start_color in unique_values:
            assert gap_side is None, "expected exactly one gap side"
            assert values.count(start_color) == 1, "gap must occupy exactly one border cell"
            remaining = unique_values - {start_color}
            assert len(remaining) == 1, "gap side must otherwise be monochrome"
            gap_side = side
            side_to_color[side] = int(next(iter(remaining)))
            gap_index = values.index(start_color)
            gap_pos = _side_coords(side, h, w)[gap_index]
        else:
            assert len(unique_values) == 1, "non-gap sides must be monochrome"
            side_to_color[side] = int(next(iter(unique_values)))

    assert gap_side is not None and gap_pos is not None
    edge_colors = list(side_to_color.values())
    assert len(set(edge_colors)) == 4, "the four border side colors must be distinct"
    assert start_color not in edge_colors

    route_points: dict[tuple[int, int], int] = {}
    edge_color_set = set(edge_colors)
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            color = int(g[r, c])
            if color in edge_color_set:
                route_points[(r, c)] = color
            else:
                assert color in {bg}, "interior cells must be background or border-colored route points"

    color_to_dir = {side_to_color["top"]: UP, side_to_color["right"]: RIGHT, side_to_color["bottom"]: DOWN, side_to_color["left"]: LEFT}
    return {
        "bg": bg,
        "start_color": start_color,
        "gap_side": gap_side,
        "gap_pos": gap_pos,
        "side_to_color": side_to_color,
        "color_to_dir": color_to_dir,
        "route_points": route_points,
    }


def _trace_route(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    info = _parse_legend(g)
    h, w = g.shape

    pos = info["gap_pos"]
    direction = INWARD_DIR[info["gap_side"]]
    bg = int(info["bg"])
    route_points = info["route_points"]
    color_to_dir = info["color_to_dir"]

    path_bg: list[tuple[int, int]] = []
    visited_points: list[dict] = []
    seen_states = {(pos[0], pos[1], direction[0], direction[1])}

    for _ in range(g.size * 4 + 4):
        nr = pos[0] + direction[0]
        nc = pos[1] + direction[1]
        if not (1 <= nr < h - 1 and 1 <= nc < w - 1):
            return {
                **info,
                "path_bg": path_bg,
                "visited_points": visited_points,
                "terminal_pos": pos,
                "terminal_dir": direction,
            }

        pos = (nr, nc)
        cell = int(g[nr, nc])
        if cell == bg:
            path_bg.append(pos)
        else:
            assert pos in route_points, "ray hit an unexpected foreground cell"
            new_direction = color_to_dir[cell]
            visited_points.append(
                {"pos": pos, "color": cell, "in_dir": direction, "out_dir": new_direction}
            )
            direction = new_direction

        state = (pos[0], pos[1], direction[0], direction[1])
        assert state not in seen_states, "routing loop detected"
        seen_states.add(state)

    raise AssertionError("routing exceeded the maximum step budget")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route a single start ray through border-defined turn instructions and record its folded path.

    Input structure:
    - The outermost ring is a one-cell border whose four sides have four distinct colors; the corners are background.
    - Exactly one border cell is replaced by a unique start color, creating a gap that indicates the ray's entry side.
    - The interior contains only background plus isolated single-pixel route points whose colors are chosen from the four border side colors.

    Rule:
    - Read the border as a legend: a route point turns the ray toward the border side with the same color.
    - Launch the ray inward from the gap, recoloring every visited background cell to the start color.
    - When the ray enters a route point, keep that point's color unchanged, immediately update the ray direction from the legend, and stop once the next step would reach the border.

    Key insight:
    - The scattered colored pixels are not destinations; they are local turn instructions whose meanings are defined on the fly by the current border-color-to-side legend.
    """
    g = np.asarray(grid, dtype=int)
    trace = _trace_route(g)
    out = g.copy()
    for r, c in trace["path_bg"]:
        out[r, c] = int(trace["start_color"])
    return out


def _search_turns(
    rng: np.random.Generator,
    h: int,
    w: int,
    pos: tuple[int, int],
    direction: tuple[int, int],
    remaining_turns: int,
    occupied: set[tuple[int, int]],
) -> list[dict] | None:
    if remaining_turns == 0:
        final_steps = _available_interior_steps(pos, direction, h, w)
        if final_steps < MIN_FINAL_SEGMENT_LEN:
            return None
        final_seg = _segment_cells(pos, direction, final_steps)
        if any(cell in occupied for cell in final_seg):
            return None
        return []

    forward_steps = _available_interior_steps(pos, direction, h, w)
    candidates: list[tuple[int, tuple[int, int]]] = []
    next_min = MIN_FINAL_SEGMENT_LEN if remaining_turns == 1 else MIN_SEGMENT_LEN
    for steps in range(MIN_SEGMENT_LEN, forward_steps + 1):
        point = _segment_cells(pos, direction, steps)[-1]
        for new_direction in ORTHOGONAL_DIRS[direction]:
            if _available_interior_steps(point, new_direction, h, w) >= next_min:
                candidates.append((steps, new_direction))

    rng.shuffle(candidates)
    for steps, new_direction in candidates:
        seg = _segment_cells(pos, direction, steps)
        if any(cell in occupied for cell in seg):
            continue
        point = seg[-1]
        suffix = _search_turns(
            rng,
            h,
            w,
            point,
            new_direction,
            remaining_turns - 1,
            occupied | set(seg),
        )
        if suffix is not None:
            return [
                {
                    "pos": point,
                    "in_dir": direction,
                    "out_dir": new_direction,
                    "approach_steps": steps,
                }
            ] + suffix
    return None


def _build_path_metadata(
    h: int,
    w: int,
    gap_side: str,
    gap_pos: tuple[int, int],
    turns: list[dict],
) -> dict:
    pos = gap_pos
    direction = INWARD_DIR[gap_side]
    occupied = {gap_pos}
    segments: list[dict] = []

    for turn in turns:
        seg = _segment_cells(pos, direction, int(turn["approach_steps"]))
        assert seg and seg[-1] == turn["pos"]
        assert not any(cell in occupied for cell in seg)
        occupied |= set(seg)
        segments.append({"start": pos, "direction": direction, "cells": seg, "turn_pos": turn["pos"]})
        pos = turn["pos"]
        direction = turn["out_dir"]

    final_steps = _available_interior_steps(pos, direction, h, w)
    assert final_steps >= MIN_FINAL_SEGMENT_LEN
    final_seg = _segment_cells(pos, direction, final_steps)
    assert not any(cell in occupied for cell in final_seg)
    occupied |= set(final_seg)

    return {
        "segments": segments,
        "final_segment": {"start": pos, "direction": direction, "cells": final_seg},
        "occupied": occupied,
    }


def _aligned_distractor_cells(
    point: tuple[int, int],
    in_dir: tuple[int, int],
    h: int,
    w: int,
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    r, c = point
    dr, dc = in_dir
    distance = 0
    while True:
        distance += 1
        r += dr
        c += dc
        if not (1 <= r < h - 1 and 1 <= c < w - 1):
            break
        cell = (r, c)
        if cell in forbidden:
            break
        if distance >= 2:
            cells.append(cell)
    return cells


def _random_free_cell(
    rng: np.random.Generator,
    h: int,
    w: int,
    blocked: set[tuple[int, int]],
    points: set[tuple[int, int]],
) -> tuple[int, int] | None:
    candidates = [
        (r, c)
        for r in range(1, h - 1)
        for c in range(1, w - 1)
        if (r, c) not in blocked and (r, c) not in points and all(_chebyshev((r, c), p) >= 2 for p in points)
    ]
    if not candidates:
        return None
    return candidates[int(rng.integers(len(candidates)))]


def _swap_border_sides(
    grid: np.ndarray,
    side_a: str,
    color_a: int,
    side_b: str,
    color_b: int,
    gap_pos: tuple[int, int],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    for r, c in _side_coords(side_a, out.shape[0], out.shape[1]):
        if (r, c) != gap_pos:
            out[r, c] = int(color_b)
    for r, c in _side_coords(side_b, out.shape[0], out.shape[1]):
        if (r, c) != gap_pos:
            out[r, c] = int(color_a)
    return out


def _validate_generated_grid(grid: np.ndarray, meta: dict) -> bool:
    trace = _trace_route(grid)
    used_positions = [item["pos"] for item in meta["used_points"]]
    traced_positions = [item["pos"] for item in trace["visited_points"]]
    assert traced_positions == used_positions, "generated route order changed"
    assert len(traced_positions) >= MIN_ROUTE_POINTS

    for expected, actual in zip(meta["used_points"], trace["visited_points"]):
        assert expected["color"] == actual["color"]
        assert expected["in_dir"] == actual["in_dir"]
        assert expected["out_dir"] == actual["out_dir"]
        assert actual["in_dir"] != actual["out_dir"], "each visited route point must change direction"

    all_route_positions = set(trace["route_points"])
    distractor_positions = {item["pos"] for item in meta["distractors"]}
    assert distractor_positions, "need at least one distractor route point"
    assert distractor_positions <= all_route_positions
    assert any(pos not in traced_positions for pos in all_route_positions), "need at least one unvisited route point"

    point_colors = {int(grid[r, c]) for r, c in all_route_positions}
    assert point_colors == set(meta["side_to_color"].values()), "all four legend colors must appear among route points"

    out = main(grid)
    for r, c in trace["path_bg"]:
        assert int(out[r, c]) == int(trace["start_color"])
    for r, c in all_route_positions:
        assert int(out[r, c]) == int(grid[r, c])

    pivot = meta["used_points"][0]
    pivot_side = DIR_TO_SIDE[pivot["out_dir"]]
    other_side = next(side for side in SIDE_TO_DIR if side != pivot_side)
    swapped = _swap_border_sides(
        grid,
        pivot_side,
        meta["side_to_color"][pivot_side],
        other_side,
        meta["side_to_color"][other_side],
        meta["gap_pos"],
    )
    try:
        swapped_out = main(swapped)
    except AssertionError:
        swapped_out = None
    if swapped_out is not None:
        assert not np.array_equal(swapped_out, out), "border legend must affect the routed output"

    for point in meta["used_points"]:
        current_side = DIR_TO_SIDE[point["out_dir"]]
        alternate_side = next(side for side in SIDE_TO_DIR if side != current_side)
        mutated = np.array(grid, copy=True)
        mutated[point["pos"]] = int(meta["side_to_color"][alternate_side])
        try:
            alt_out = main(mutated)
        except AssertionError:
            continue
        assert not np.array_equal(alt_out, out), "each visited route point must affect the path"

    return True


def _gen_attempt(rng: np.random.Generator) -> tuple[np.ndarray, dict] | None:
    h = int(rng.integers(MIN_GRID, MAX_GRID + 1))
    w = int(rng.integers(MIN_GRID, MAX_GRID + 1))
    if min(h, w) < 7:
        return None

    colors = list(map(int, rng.choice(PALETTE, size=6, replace=False)))
    bg = colors[0]
    start_color = colors[1]
    border_palette = colors[2:]
    rng.shuffle(border_palette)
    side_to_color = {side: border_palette[i] for i, side in enumerate(SIDE_TO_DIR)}

    gap_side = str(rng.choice(list(SIDE_TO_DIR)))
    side_cells = _side_coords(gap_side, h, w)
    if len(side_cells) < 5:
        return None
    gap_pos = side_cells[int(rng.integers(1, len(side_cells) - 1))]

    chain_len = int(rng.choice(np.array([2, 3, 4, 5]), p=np.array([0.2, 0.35, 0.3, 0.15])))
    turns = _search_turns(rng, h, w, gap_pos, INWARD_DIR[gap_side], chain_len, {gap_pos})
    if turns is None:
        return None

    path_meta = _build_path_metadata(h, w, gap_side, gap_pos, turns)
    occupied = set(path_meta["occupied"])
    point_positions = {turn["pos"] for turn in turns}

    used_points = []
    for turn in turns:
        color = int(side_to_color[DIR_TO_SIDE[turn["out_dir"]]])
        used_points.append({**turn, "color": color})

    distractors: list[dict] = []
    aligned_candidates: list[tuple[tuple[int, int], dict]] = []
    for turn in turns:
        candidates = _aligned_distractor_cells(turn["pos"], turn["in_dir"], h, w, occupied | point_positions)
        for cell in candidates:
            if all(_chebyshev(cell, other) >= 2 for other in point_positions):
                aligned_candidates.append((cell, turn))
    if not aligned_candidates:
        return None

    missing_colors = [color for color in side_to_color.values() if color not in {item["color"] for item in used_points}]
    aligned_cell, _aligned_turn = aligned_candidates[int(rng.integers(len(aligned_candidates)))]
    aligned_color = int(missing_colors[0] if missing_colors else rng.choice(list(side_to_color.values())))
    distractors.append({"pos": aligned_cell, "color": aligned_color})
    point_positions.add(aligned_cell)

    for color in side_to_color.values():
        if color in {item["color"] for item in used_points} | {item["color"] for item in distractors}:
            continue
        cell = _random_free_cell(rng, h, w, occupied, point_positions)
        if cell is None:
            return None
        distractors.append({"pos": cell, "color": int(color)})
        point_positions.add(cell)

    extra_count = int(rng.integers(0, MAX_EXTRA_DISTRACTORS + 1))
    for _ in range(extra_count):
        cell = _random_free_cell(rng, h, w, occupied, point_positions)
        if cell is None:
            break
        distractors.append({"pos": cell, "color": int(rng.choice(list(side_to_color.values())))})
        point_positions.add(cell)

    grid = np.full((h, w), int(bg), dtype=int)
    for side, color in side_to_color.items():
        for r, c in _side_coords(side, h, w):
            grid[r, c] = int(color)
    grid[gap_pos] = int(start_color)

    for item in used_points:
        r, c = item["pos"]
        grid[r, c] = int(item["color"])
    for item in distractors:
        r, c = item["pos"]
        grid[r, c] = int(item["color"])

    meta = {
        "bg": bg,
        "start_color": start_color,
        "gap_side": gap_side,
        "gap_pos": gap_pos,
        "side_to_color": side_to_color,
        "used_points": used_points,
        "distractors": distractors,
    }

    if not _validate_generated_grid(grid, meta):
        return None
    return grid, meta


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        result = _gen_attempt(rng)
        if result is not None:
            grid, _meta = result
            return grid
    raise RuntimeError("failed to generate a valid routing-legend input")
