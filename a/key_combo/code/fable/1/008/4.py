from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, find_components, get_bg


PALETTE = tuple(range(1, 10))
GEN_BG = Color.BLACK
GEN_RETRIES = 400
GEN_EMITTER_COUNT = 4
GEN_ROW_RANGE = (20, 29)
GEN_COL_RANGE = (18, 29)
GEN_COL_GAP = 4
GEN_ROW_GAP = 2
GEN_INTERNAL_MARGIN = 2


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    h, w = map(int, g.shape)
    bg = int(get_bg(g))

    emitters = [(int(c), int(g[0, c])) for c in np.flatnonzero(g[0] != bg)]
    assert emitters, "expected at least one top-row emitter"
    assert len({color for _c, color in emitters}) == len(emitters), "emitters must have distinct colors"

    interior = g.copy()
    interior[0, :] = bg

    role_by_cell: dict[tuple[int, int], tuple[str, int]] = {}
    dyes: dict[tuple[int, int], int] = {}
    walls: list[dict] = []
    for comp in find_components(interior, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        color = int(comp["color"])
        area = int(comp["area"])
        r0, c0, r1, c1 = map(int, comp["bbox"])

        if area == 1:
            r, c = map(int, comp["coords"][0])
            dyes[(r, c)] = color
            role_by_cell[(r, c)] = ("dye", color)
            continue

        assert r0 == r1, "non-singleton foreground objects must be horizontal"
        assert area >= 2 and area == c1 - c0 + 1, "walls must be straight horizontal segments"
        walls.append({"row": r0, "c0": c0, "c1": c1, "color": color})
        for c in range(c0, c1 + 1):
            role_by_cell[(r0, c)] = ("wall", color)

    return {
        "grid": g,
        "bg": bg,
        "h": h,
        "w": w,
        "emitters": sorted(emitters),
        "dyes": dyes,
        "walls": sorted(walls, key=lambda wall: (wall["row"], wall["c0"])),
        "role_by_cell": role_by_cell,
    }


def _simulate(grid: np.ndarray, *, mode: str = "real") -> tuple[np.ndarray, list[dict]]:
    scene = _parse_scene(grid)
    g = scene["grid"]
    out = g.copy()
    traces: list[dict] = []

    for col, emitter_color in scene["emitters"]:
        current = int(emitter_color)
        stopped = False
        events: list[dict] = []

        for r in range(1, scene["h"]):
            info = scene["role_by_cell"].get((r, col))
            if info is None:
                out[r, col] = current
                continue

            kind, object_color = info
            object_color = int(object_color)

            if mode == "any_blocker":
                events.append(
                    {
                        "type": kind,
                        "row": r,
                        "object_color": object_color,
                        "current_before": current,
                        "current_after": current,
                        "stops": True,
                    }
                )
                stopped = True
                break

            if kind == "dye":
                new_color = object_color if mode == "real" else current
                events.append(
                    {
                        "type": "dye",
                        "row": r,
                        "object_color": object_color,
                        "current_before": current,
                        "current_after": new_color,
                        "stops": False,
                    }
                )
                current = new_color
                continue

            stops = object_color == current
            events.append(
                {
                    "type": "wall",
                    "row": r,
                    "object_color": object_color,
                    "current_before": current,
                    "current_after": current,
                    "stops": stops,
                }
            )
            if stops:
                stopped = True
                break

        traces.append(
            {
                "col": col,
                "emitter_color": int(emitter_color),
                "events": events,
                "stopped": stopped,
                "final_color": current,
            }
        )

    return out, traces


def main(grid: np.ndarray) -> np.ndarray:
    """Extend each top-row emitter downward, recoloring rays at dye pixels and stopping only on same-color walls.

    Input structure:
    - The background is black, and every non-background pixel on the top row is an emitter with a distinct color.
    - Inside the grid, isolated single pixels are dyes while horizontal monochrome segments of length at least two are walls.
    - Object role is determined by geometry: size-one components are dyes, and longer horizontal components are walls.

    Rule:
    - Copy the input, then let each emitter send a ray straight downward in its own column, initially carrying the emitter color.
    - Whenever the ray passes through a background cell, paint that cell with the ray's current color.
    - When the ray reaches a dye pixel, do not stop; keep the dye cell unchanged and continue downward with the dye color as the new current color.
    - When the ray reaches a wall cell, stop only if the wall color matches the ray's current color; otherwise pass through while leaving the wall unchanged.

    Key insight:
    - The same wall can block one ray and let another through because interception depends on the ray's current carried color, which may have changed earlier at a single-pixel dye.
    """
    return _simulate(grid, mode="real")[0]


def _sample_spaced(lo: int, hi: int, count: int, gap: int) -> list[int]:
    values = list(range(lo, hi + 1))
    for _ in range(300):
        chosen = sorted(random.sample(values, count))
        if all(b - a >= gap for a, b in zip(chosen, chosen[1:])):
            return chosen
    raise AssertionError("failed to sample spaced coordinates")


def _pick_color(*, exclude: set[int]) -> int:
    choices = [color for color in PALETTE if color not in exclude]
    assert choices, "expected at least one available color"
    return int(random.choice(choices))


def _single_lane_segment(col: int, width: int) -> tuple[int, int]:
    candidates = set()
    for length in (2, 3):
        for offset in range(length):
            c0 = col - offset
            c1 = c0 + length - 1
            if 0 <= c0 and c1 < width:
                candidates.add((c0, c1))
    assert candidates, "expected a wall segment through the target column"
    return random.choice(sorted(candidates))


def _shared_segment(left_col: int, right_col: int, width: int) -> tuple[int, int]:
    left_pad = random.randint(1, min(2, left_col))
    right_pad = random.randint(1, min(2, width - 1 - right_col))
    c0 = left_col - left_pad
    c1 = right_col + right_pad
    assert c1 - c0 + 1 >= 2
    return c0, c1


def _decoy_segment(width: int, emitter_cols: set[int]) -> tuple[int, int]:
    candidates = []
    for length in (2, 3, 4):
        for c0 in range(0, width - length + 1):
            c1 = c0 + length - 1
            if all(c not in emitter_cols for c in range(c0, c1 + 1)):
                candidates.append((c0, c1))
    assert candidates, "expected a decoy wall location"
    return random.choice(candidates)


def _decoy_col(width: int, emitter_cols: set[int]) -> int:
    candidates = [c for c in range(width) if c not in emitter_cols]
    assert candidates, "expected a decoy dye column"
    return int(random.choice(candidates))


def _place_dye(grid: np.ndarray, row: int, col: int, color: int) -> None:
    assert grid[row, col] == GEN_BG, "dye placement must land on background"
    if row > 0:
        assert grid[row - 1, col] == GEN_BG
    if row + 1 < grid.shape[0]:
        assert grid[row + 1, col] == GEN_BG
    if col > 0:
        assert grid[row, col - 1] == GEN_BG
    if col + 1 < grid.shape[1]:
        assert grid[row, col + 1] == GEN_BG
    grid[row, col] = int(color)


def _place_wall(grid: np.ndarray, row: int, c0: int, c1: int, color: int) -> None:
    assert c1 - c0 + 1 >= 2, "walls must have length at least two"
    assert np.all(grid[row, c0 : c1 + 1] == GEN_BG), "wall cells must be empty"
    if row > 0:
        assert np.all(grid[row - 1, c0 : c1 + 1] == GEN_BG)
    if row + 1 < grid.shape[0]:
        assert np.all(grid[row + 1, c0 : c1 + 1] == GEN_BG)
    grid[row, c0 : c1 + 1] = int(color)


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        out, traces = _simulate(grid, mode="real")
        any_blocker_out, _ = _simulate(grid, mode="any_blocker")
        no_recolor_out, _ = _simulate(grid, mode="no_recolor")
        scene = _parse_scene(grid)
    except AssertionError:
        return False

    if len(scene["emitters"]) != GEN_EMITTER_COUNT:
        return False
    if len(scene["dyes"]) < 4 or len(scene["walls"]) < 5:
        return False

    has_pass = any(
        event["type"] == "wall" and not event["stops"]
        for trace in traces
        for event in trace["events"]
    )
    has_stop = any(
        event["type"] == "wall" and event["stops"]
        for trace in traces
        for event in trace["events"]
    )
    has_dye_controlled_stop = any(
        event["type"] == "wall"
        and event["stops"]
        and event["current_before"] != trace["emitter_color"]
        for trace in traces
        for event in trace["events"]
    )
    reaches_bottom = any(not trace["stopped"] for trace in traces)

    if not (has_pass and has_stop and has_dye_controlled_stop and reaches_bottom):
        return False

    if np.array_equal(out, grid):
        return False
    if np.count_nonzero(out != grid) < 12:
        return False
    if np.array_equal(out, any_blocker_out):
        return False
    if np.array_equal(out, no_recolor_out):
        return False

    emitter_cols = {col for col, _color in scene["emitters"]}
    has_off_lane_dye = any(col not in emitter_cols for (_row, col) in scene["dyes"])
    has_off_lane_wall = any(
        all(not (wall["c0"] <= col <= wall["c1"]) for col in emitter_cols)
        for wall in scene["walls"]
    )
    if not (has_off_lane_dye and has_off_lane_wall):
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    try:
        height = random.randint(*GEN_ROW_RANGE)
        width = random.randint(*GEN_COL_RANGE)
        emitter_cols = _sample_spaced(
            GEN_INTERNAL_MARGIN,
            width - 1 - GEN_INTERNAL_MARGIN,
            GEN_EMITTER_COUNT,
            GEN_COL_GAP,
        )
        rows = _sample_spaced(2, height - 2, 9, GEN_ROW_GAP)
    except AssertionError:
        return None

    c0, c1, c2, c3 = emitter_cols
    (
        r_chain_dye,
        r_shared_wall,
        r_pass_wall,
        r_pass_dye,
        r_pass_stop,
        r_extra_a,
        r_extra_b,
        r_decoy_wall,
        r_decoy_dye,
    ) = rows

    emitter_colors = random.sample(PALETTE, GEN_EMITTER_COUNT)
    e0, e1, e2, e3 = map(int, emitter_colors)

    pass_wall_color = _pick_color(exclude={e2})
    pass_stop_color = _pick_color(exclude={e2, pass_wall_color})

    extra_mode = random.choice(("wall_then_dye", "dye_then_wall"))
    if extra_mode == "wall_then_dye":
        extra_wall_color = _pick_color(exclude={e3})
        extra_dye_color = _pick_color(exclude={e3})
    else:
        extra_dye_color = _pick_color(exclude={e3})
        extra_wall_color = _pick_color(exclude={extra_dye_color})

    grid = np.full((height, width), GEN_BG, dtype=int)
    for col, color in zip(emitter_cols, emitter_colors):
        grid[0, col] = int(color)

    emitter_col_set = set(emitter_cols)
    shared_c0, shared_c1 = _shared_segment(c0, c1, width)
    pass_c0, pass_c1 = _single_lane_segment(c2, width)
    stop_c0, stop_c1 = _single_lane_segment(c2, width)
    extra_c0, extra_c1 = _single_lane_segment(c3, width)
    decoy_c0, decoy_c1 = _decoy_segment(width, emitter_col_set)
    decoy_dye_col = _decoy_col(width, emitter_col_set)

    try:
        _place_dye(grid, r_chain_dye, c0, e1)
        _place_wall(grid, r_shared_wall, shared_c0, shared_c1, e1)

        _place_wall(grid, r_pass_wall, pass_c0, pass_c1, pass_wall_color)
        _place_dye(grid, r_pass_dye, c2, pass_stop_color)
        _place_wall(grid, r_pass_stop, stop_c0, stop_c1, pass_stop_color)

        if extra_mode == "wall_then_dye":
            _place_wall(grid, r_extra_a, extra_c0, extra_c1, extra_wall_color)
            _place_dye(grid, r_extra_b, c3, extra_dye_color)
        else:
            _place_dye(grid, r_extra_a, c3, extra_dye_color)
            _place_wall(grid, r_extra_b, extra_c0, extra_c1, extra_wall_color)

        _place_wall(grid, r_decoy_wall, decoy_c0, decoy_c1, _pick_color(exclude=set()))
        _place_dye(grid, r_decoy_dye, decoy_dye_col, _pick_color(exclude=set()))
    except AssertionError:
        return None

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid selective-wall ray puzzle")
