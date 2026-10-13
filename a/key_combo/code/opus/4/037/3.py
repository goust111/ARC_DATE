from __future__ import annotations

import importlib.util
from collections import defaultdict
from pathlib import Path
import random

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib.util.spec_from_file_location("_arc_utils_shared_037_3", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load arc_utils from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ARC_UTILS = _load_arc_utils()
Color = _ARC_UTILS.Color
in_bounds = _ARC_UTILS.in_bounds

BG = int(Color.BLACK)
MARKER = int(Color.BLUE)
OBSTACLE = int(Color.GREY)

DIR_TO_COLOR = {
    (-1, 1): int(Color.RED),     # up-right
    (-1, -1): int(Color.YELLOW), # up-left
    (1, 1): int(Color.GREEN),    # down-right
    (1, -1): int(Color.ORANGE),  # down-left
}

DIAGONALS = tuple(DIR_TO_COLOR)
GEN_ATTEMPTS = 400
GEN_SIZE_RANGE = (13, 18)
GEN_OBS_RANGE = (5, 10)


def _candidate_key(candidate: dict) -> tuple[int, int, int]:
    src_r, src_c = candidate["source"]
    return int(candidate["steps"]), int(src_r), int(src_c)


def _marker_list(grid: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(grid == MARKER)]


def _ray_trace(grid: np.ndarray, source: tuple[int, int], dr: int, dc: int) -> dict:
    h, w = grid.shape
    r, c = source
    steps = 0
    path: list[tuple[int, int, int]] = []
    while True:
        r += dr
        c += dc
        if not in_bounds((h, w), r, c):
            return {
                "source": source,
                "dir": (dr, dc),
                "color": DIR_TO_COLOR[(dr, dc)],
                "path": path,
                "stop_reason": "boundary",
                "stop_cell": None,
            }
        value = int(grid[r, c])
        if value in {MARKER, OBSTACLE}:
            return {
                "source": source,
                "dir": (dr, dc),
                "color": DIR_TO_COLOR[(dr, dc)],
                "path": path,
                "stop_reason": "marker" if value == MARKER else "obstacle",
                "stop_cell": (int(r), int(c)),
            }
        steps += 1
        path.append((int(r), int(c), int(steps)))


def _trace_scene(grid: np.ndarray) -> tuple[dict[tuple[int, int], dict], dict[tuple[int, int], list[dict]], list[dict]]:
    claims: dict[tuple[int, int], dict] = {}
    contenders: dict[tuple[int, int], list[dict]] = defaultdict(list)
    rays: list[dict] = []
    for source in _marker_list(grid):
        for dr, dc in DIAGONALS:
            ray = _ray_trace(grid, source, dr, dc)
            rays.append(ray)
            for r, c, steps in ray["path"]:
                candidate = {
                    "source": source,
                    "steps": int(steps),
                    "color": int(ray["color"]),
                    "dir": (dr, dc),
                }
                contenders[(r, c)].append(candidate)
                if (r, c) not in claims or _candidate_key(candidate) < _candidate_key(claims[(r, c)]):
                    claims[(r, c)] = candidate
    return claims, contenders, rays


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend colored diagonal rays from each blue marker until blockers stop them, resolving overlaps by nearer source.

    Input structure:
    - A black background grid contains 2 or 3 blue marker pixels and several gray obstacle pixels.
    - Markers are isolated single pixels; obstacles are also isolated single pixels.
    - All other cells are background.

    Rule:
    - Keep every blue marker and gray obstacle unchanged.
    - From each blue marker, extend four diagonal rays one step at a time: up-right is red, up-left is yellow, down-right is green, and down-left is orange.
    - A ray stops before the first obstacle, marker, or grid boundary.
    - If different markers claim the same cell, keep the color from the smaller source distance; if those distances tie, keep the ray from the marker with the smaller source row.

    Key insight:
    - The output is determined by combining local direction-to-color coding with global line-of-sight geometry: obstacles cut ray length, and crossed rays must be compared back to their sources.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, f"grid exceeds ARC size limit: {g.shape}"
    assert set(map(int, np.unique(g))) <= {BG, MARKER, OBSTACLE}, "unexpected input colors"

    markers = _marker_list(g)
    assert 2 <= len(markers) <= 3, "expected 2 or 3 blue markers"

    claims, _contenders, _rays = _trace_scene(g)
    out = g.copy()
    for (r, c), candidate in claims.items():
        out[r, c] = int(candidate["color"])
    return out


def _segment(source: tuple[int, int], dr: int, dc: int, steps: int) -> list[tuple[int, int]]:
    r0, c0 = source
    return [(int(r0 + dr * t), int(c0 + dc * t)) for t in range(1, int(steps) + 1)]


def _marker_spacing_ok(markers: list[tuple[int, int]]) -> bool:
    for i, (r0, c0) in enumerate(markers):
        for r1, c1 in markers[i + 1 :]:
            if abs(r0 - r1) + abs(c0 - c1) < 4:
                return False
    return True


def _enumerate_core_layouts(h: int, w: int, mode: str) -> list[dict]:
    layouts = []
    for family in ("right", "left"):
        for upper_steps in range(2, 5):
            for lower_steps in range(2, 5):
                if mode == "tie" and upper_steps != lower_steps:
                    continue
                if mode == "strict" and upper_steps == lower_steps:
                    continue

                row_lo = upper_steps + 1
                row_hi = h - lower_steps - 2
                if family == "right":
                    col_lo = max(upper_steps, lower_steps) + 1
                    col_hi = w - 3
                else:
                    col_lo = 2
                    col_hi = w - max(upper_steps, lower_steps) - 2
                if row_lo > row_hi or col_lo > col_hi:
                    continue

                for pr in range(row_lo, row_hi + 1):
                    for pc in range(col_lo, col_hi + 1):
                        if family == "right":
                            top_src = (pr - upper_steps, pc - upper_steps)
                            bot_src = (pr + lower_steps, pc - lower_steps)
                            top_dir = (1, 1)
                            bot_dir = (-1, 1)
                            top_extra = min(h - 1 - pr, w - 1 - pc)
                            bot_extra = min(pr, w - 1 - pc)
                        else:
                            top_src = (pr - upper_steps, pc + upper_steps)
                            bot_src = (pr + lower_steps, pc + lower_steps)
                            top_dir = (1, -1)
                            bot_dir = (-1, -1)
                            top_extra = min(h - 1 - pr, pc)
                            bot_extra = min(pr, pc)

                        markers = [top_src, bot_src]
                        if not _marker_spacing_ok(markers):
                            continue
                        layouts.append(
                            {
                                "family": family,
                                "mode": mode,
                                "pivot": (int(pr), int(pc)),
                                "top_source": top_src,
                                "bottom_source": bot_src,
                                "top_dir": top_dir,
                                "bottom_dir": bot_dir,
                                "top_steps": int(upper_steps),
                                "bottom_steps": int(lower_steps),
                                "top_extra": int(top_extra),
                                "bottom_extra": int(bot_extra),
                            }
                        )
    return layouts


def _protected_cells(layout: dict) -> set[tuple[int, int]]:
    protected = {tuple(layout["pivot"]), tuple(layout["top_source"]), tuple(layout["bottom_source"])}
    protected.update(_segment(layout["top_source"], *layout["top_dir"], layout["top_steps"]))
    protected.update(_segment(layout["bottom_source"], *layout["bottom_dir"], layout["bottom_steps"]))
    return protected


def _guaranteed_blockers(layout: dict) -> list[tuple[int, int]]:
    pr, pc = layout["pivot"]
    blockers = []
    if layout["top_extra"] >= 1:
        dr, dc = layout["top_dir"]
        blockers.append((int(pr + dr), int(pc + dc)))
    if layout["bottom_extra"] >= 1:
        dr, dc = layout["bottom_dir"]
        blockers.append((int(pr + dr), int(pc + dc)))
    return blockers


def _can_place_marker(cell: tuple[int, int], markers: list[tuple[int, int]], protected: set[tuple[int, int]]) -> bool:
    if cell in protected:
        return False
    r, c = cell
    if r in {mr for mr, _ in markers}:
        return False
    for mr, mc in markers:
        if abs(r - mr) + abs(c - mc) < 4:
            return False
    return True


def _add_optional_third_marker(
    h: int,
    w: int,
    markers: list[tuple[int, int]],
    protected: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    if random.random() < 0.35:
        return markers

    candidates = [
        (r, c)
        for r in range(1, h - 1)
        for c in range(1, w - 1)
        if _can_place_marker((r, c), markers, protected)
    ]
    random.shuffle(candidates)
    if not candidates:
        return markers
    return markers + [candidates[0]]


def _obstacle_ok(cell: tuple[int, int], markers: list[tuple[int, int]], forbidden: set[tuple[int, int]]) -> bool:
    if cell in forbidden:
        return False
    r, c = cell
    for mr, mc in markers:
        if max(abs(r - mr), abs(c - mc)) <= 1:
            return False
    return True


def _fill_obstacles(
    grid: np.ndarray,
    markers: list[tuple[int, int]],
    protected: set[tuple[int, int]],
    guaranteed: list[tuple[int, int]],
) -> None:
    forbidden = set(protected)
    for marker in markers:
        forbidden.add(marker)

    obstacle_count = random.randint(*GEN_OBS_RANGE)
    chosen: list[tuple[int, int]] = []

    guaranteed_choices = guaranteed[:]
    random.shuffle(guaranteed_choices)
    keep_count = 1 if len(guaranteed_choices) == 1 else random.choice([1, 2])
    for cell in guaranteed_choices[:keep_count]:
        if _obstacle_ok(cell, markers, forbidden):
            chosen.append(cell)
            forbidden.add(cell)

    pool = [
        (r, c)
        for r in range(grid.shape[0])
        for c in range(grid.shape[1])
        if _obstacle_ok((r, c), markers, forbidden)
    ]
    random.shuffle(pool)
    for cell in pool:
        if len(chosen) >= obstacle_count:
            break
        chosen.append(cell)
        forbidden.add(cell)

    if len(chosen) != obstacle_count:
        raise RuntimeError("failed to place enough obstacles")

    for r, c in chosen:
        grid[r, c] = OBSTACLE


def _build_candidate() -> np.ndarray:
    h = random.randint(*GEN_SIZE_RANGE)
    w = random.randint(*GEN_SIZE_RANGE)
    mode = random.choice(["strict", "tie"])
    layouts = _enumerate_core_layouts(h, w, mode)
    if not layouts:
        raise RuntimeError("no feasible core layouts")
    layout = random.choice(layouts)

    grid = np.full((h, w), BG, dtype=int)
    markers = [layout["top_source"], layout["bottom_source"]]
    protected = _protected_cells(layout)
    markers = _add_optional_third_marker(h, w, markers, protected)

    for r, c in markers:
        grid[r, c] = MARKER

    guaranteed = _guaranteed_blockers(layout)
    _fill_obstacles(grid, markers, protected, guaranteed)
    return grid


def _scene_metrics(grid: np.ndarray) -> dict:
    claims, contenders, rays = _trace_scene(grid)
    overlap_cells = {}
    has_strict = False
    has_tie = False

    for cell, cands in contenders.items():
        unique_sources = {cand["source"] for cand in cands}
        if len(unique_sources) < 2:
            continue
        overlap_cells[cell] = cands
        ordered = sorted(cands, key=_candidate_key)
        best_steps = ordered[0]["steps"]
        tied_best = [cand for cand in cands if cand["steps"] == best_steps]
        if len(tied_best) >= 2:
            if len({cand["source"][0] for cand in tied_best}) >= 2:
                has_tie = True
        else:
            has_strict = True

    painted_colors = {cand["color"] for cand in claims.values()}
    obstacle_blocked = [ray for ray in rays if ray["stop_reason"] == "obstacle"]
    return {
        "claims": claims,
        "rays": rays,
        "overlap_cells": overlap_cells,
        "has_overlap": bool(overlap_cells),
        "has_strict": has_strict,
        "has_tie": has_tie,
        "obstacle_blocked": obstacle_blocked,
        "painted_colors": painted_colors,
    }


def _validate_generated(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if set(map(int, np.unique(grid))) - {BG, MARKER, OBSTACLE}:
        return False

    markers = _marker_list(grid)
    if len(markers) not in {2, 3} or not _marker_spacing_ok(markers):
        return False
    if len({r for r, _ in markers}) != len(markers):
        return False

    obstacles = [tuple(map(int, rc)) for rc in np.argwhere(grid == OBSTACLE)]
    if not (GEN_OBS_RANGE[0] <= len(obstacles) <= GEN_OBS_RANGE[1]):
        return False
    for r, c in obstacles:
        for mr, mc in markers:
            if max(abs(r - mr), abs(c - mc)) <= 1:
                return False

    metrics = _scene_metrics(grid)
    if not metrics["has_overlap"]:
        return False
    if not metrics["obstacle_blocked"]:
        return False
    if len(metrics["painted_colors"]) < 3:
        return False
    if len(metrics["claims"]) < 12:
        return False

    # Require that distance priority is genuinely needed on some cell, and
    # often include the row tie-break branch as well.
    if not (metrics["has_strict"] or metrics["has_tie"]):
        return False

    out = main(grid)
    if out.shape != grid.shape or max(out.shape) > 30:
        return False
    if not np.array_equal(out[grid == MARKER], grid[grid == MARKER]):
        return False
    if not np.array_equal(out[grid == OBSTACLE], grid[grid == OBSTACLE]):
        return False
    if np.count_nonzero(out != grid) < 10:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            grid = _build_candidate()
        except RuntimeError:
            continue
        if _validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid diagonal-ray ARC input")
