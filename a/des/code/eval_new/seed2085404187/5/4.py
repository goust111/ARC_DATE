from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)

HORIZONTAL_SIDES = ("left", "right")
VERTICAL_SIDES = ("top", "bottom")
SIDE_ORDER = ("left", "right", "top", "bottom")

GEN_HEIGHT_RANGE = (10, 18)
GEN_WIDTH_RANGE = (10, 18)
GEN_TOTAL_ARROWS = (3, 6)
GEN_MAX_ARROWS_PER_SIDE = 2
GEN_SIDE_GAP = 3
GEN_EXTRA_OBSTACLES_MAX = 4
GEN_RETRIES = 400


def _background_color(grid: np.ndarray) -> int:
    flat = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(flat).argmax())


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _orthogonal_steps(direction: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    if direction[0] == 0:
        return (-1, 0), (1, 0)
    return (0, -1), (0, 1)


def _arrow_geometry(
    side: str, center: int, shape: tuple[int, int]
) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    height, width = shape
    if side == "left":
        cells = [(center - 1, 0), (center, 0), (center + 1, 0), (center, 1)]
        return cells, (center, 1), (0, 1)
    if side == "right":
        cells = [
            (center - 1, width - 1),
            (center, width - 1),
            (center + 1, width - 1),
            (center, width - 2),
        ]
        return cells, (center, width - 2), (0, -1)
    if side == "top":
        cells = [(0, center - 1), (0, center), (0, center + 1), (1, center)]
        return cells, (1, center), (1, 0)
    if side == "bottom":
        cells = [
            (height - 1, center - 1),
            (height - 1, center),
            (height - 1, center + 1),
            (height - 2, center),
        ]
        return cells, (height - 2, center), (-1, 0)
    raise AssertionError(f"unknown side: {side}")


def _make_arrow(side: str, center: int, color: int, shape: tuple[int, int]) -> dict[str, object]:
    cells, tip, direction = _arrow_geometry(side, center, shape)
    return {
        "side": side,
        "center": int(center),
        "color": int(color),
        "cells": cells,
        "tip": tip,
        "direction": direction,
    }


def _find_arrows(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    found: list[dict[str, object]] = []

    scans = {
        "left": range(1, height - 1),
        "right": range(1, height - 1),
        "top": range(1, width - 1),
        "bottom": range(1, width - 1),
    }

    for side in SIDE_ORDER:
        for center in scans[side]:
            cells, tip, direction = _arrow_geometry(side, int(center), g.shape)
            color = int(g[tip])
            if color == bg:
                continue
            if all(int(g[r, c]) == color for r, c in cells):
                found.append(
                    {
                        "side": side,
                        "center": int(center),
                        "color": color,
                        "cells": cells,
                        "tip": tip,
                        "direction": direction,
                    }
                )

    used: set[tuple[int, int]] = set()
    for arrow in found:
        for cell in arrow["cells"]:
            assert cell not in used, "arrow markers must not overlap"
            used.add(cell)

    return sorted(found, key=lambda arrow: (SIDE_ORDER.index(str(arrow["side"])), int(arrow["center"])))


def _simulate(
    grid: np.ndarray, collect_trace: bool = False
) -> tuple[np.ndarray, list[dict[str, object]], list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    arrows = _find_arrows(g, bg)
    assert arrows, "expected at least one edge arrow"

    arrow_cells = {cell for arrow in arrows for cell in arrow["cells"]}
    out = g.copy()
    traces: list[dict[str, object]] = []

    for arrow in arrows:
        color = int(arrow["color"])
        dr, dc = arrow["direction"]
        endpoint = tuple(arrow["tip"])
        ray_cells: list[tuple[int, int]] = []

        while True:
            nr = endpoint[0] + dr
            nc = endpoint[1] + dc
            if not _in_bounds(g.shape, nr, nc):
                blocker_kind = "edge"
                blocker = None
                break
            if int(out[nr, nc]) != bg:
                blocker = (nr, nc)
                if int(g[nr, nc]) == bg:
                    blocker_kind = "painted"
                elif (nr, nc) in arrow_cells:
                    blocker_kind = "arrow"
                else:
                    blocker_kind = "obstacle"
                break
            out[nr, nc] = color
            endpoint = (nr, nc)
            ray_cells.append(endpoint)

        diffusion: list[tuple[int, int]] = []
        for odr, odc in _orthogonal_steps((dr, dc)):
            tr = endpoint[0] + odr
            tc = endpoint[1] + odc
            if _in_bounds(g.shape, tr, tc) and int(out[tr, tc]) == bg:
                out[tr, tc] = color
                diffusion.append((tr, tc))

        if collect_trace:
            traces.append(
                {
                    "side": arrow["side"],
                    "center": int(arrow["center"]),
                    "tip": tuple(arrow["tip"]),
                    "endpoint": endpoint,
                    "ray_len": len(ray_cells),
                    "blocker_kind": blocker_kind,
                    "blocker": blocker,
                    "diffusion_count": len(diffusion),
                    "diffusion": diffusion,
                }
            )

    return out, traces, arrows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend edge-arrow stripes inward and add one-cell perpendicular blooms at each stripe tip.

    Input structure:
    - The background is black.
    - Interior cells may contain a few colored obstacle pixels.
    - The four borders may contain 4-pixel arrow markers: three same-color border
      pixels plus one inward tip pixel, rotated to point right, left, down, or up.

    Rule:
    - Detect every border arrow and process them in this fixed order: left-edge arrows
      from top to bottom, right-edge arrows from top to bottom, top-edge arrows from
      left to right, then bottom-edge arrows from left to right.
    - Each arrow launches a 1-cell-wide ray starting at its tip and continuing in the
      pointing direction until the next step would hit a non-background cell or leave
      the grid; only background cells are painted.
    - After a ray stops, its final ray cell spreads one more cell in the two
      perpendicular directions, again only into background cells. Obstacles and arrow
      markers are never overwritten.

    Key insight:
    - Later rays treat earlier painted stripes as solid blockers, so recovering the
      global processing order is essential; the tiny perpendicular bloom at every ray
      endpoint is the hidden second step.
    """
    out, _traces, _arrows = _simulate(np.asarray(grid, dtype=int), collect_trace=False)
    return out


def _pick_spaced_positions(limit: int, count: int, gap: int) -> list[int] | None:
    if count == 0:
        return []
    candidates = list(range(2, limit - 2))
    if len(candidates) < count:
        return None
    for _ in range(80):
        picks = sorted(random.sample(candidates, count))
        if all(b - a >= gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _arrow_path_before_input_block(
    grid: np.ndarray, arrow: dict[str, object]
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    r, c = arrow["tip"]
    dr, dc = arrow["direction"]
    while True:
        r += dr
        c += dc
        if not _in_bounds(grid.shape, r, c):
            break
        if int(grid[r, c]) != BG:
            break
        cells.append((r, c))
    return cells


def _path_to_crossing(
    arrow: dict[str, object], crossing: tuple[int, int]
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    r, c = arrow["tip"]
    dr, dc = arrow["direction"]
    while (r, c) != crossing:
        r += dr
        c += dc
        cells.append((r, c))
    return cells


def _choose_arrow_counts() -> dict[str, int]:
    target = random.randint(*GEN_TOTAL_ARROWS)
    counts = {side: 0 for side in SIDE_ORDER}
    counts[random.choice(HORIZONTAL_SIDES)] += 1
    counts[random.choice(VERTICAL_SIDES)] += 1
    while sum(counts.values()) < target:
        side = random.choice(SIDE_ORDER)
        if counts[side] < GEN_MAX_ARROWS_PER_SIDE:
            counts[side] += 1
    return counts


def _collect_open_cells(
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    obstacle_cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    height, width = shape
    cells: list[tuple[int, int]] = []
    for r in range(1, height - 1):
        for c in range(1, width - 1):
            if (r, c) in blocked:
                continue
            if any(abs(r - orow) + abs(c - ocol) == 1 for orow, ocol in obstacle_cells):
                continue
            cells.append((r, c))
    return cells


def _attempt_input() -> np.ndarray | None:
    height = random.randint(*GEN_HEIGHT_RANGE)
    width = random.randint(*GEN_WIDTH_RANGE)
    grid = np.full((height, width), BG, dtype=int)

    counts = _choose_arrow_counts()
    arrow_total = sum(counts.values())
    arrow_colors = random.sample(PALETTE, arrow_total)
    color_index = 0
    arrows: list[dict[str, object]] = []

    for side in SIDE_ORDER:
        limit = height if side in HORIZONTAL_SIDES else width
        centers = _pick_spaced_positions(limit, counts[side], GEN_SIDE_GAP)
        if centers is None:
            return None
        for center in centers:
            arrow = _make_arrow(side, center, arrow_colors[color_index], grid.shape)
            color_index += 1
            for r, c in arrow["cells"]:
                if int(grid[r, c]) != BG:
                    return None
                grid[r, c] = int(arrow["color"])
            arrows.append(arrow)

    horizontal = [arrow for arrow in arrows if arrow["side"] in HORIZONTAL_SIDES]
    vertical = [arrow for arrow in arrows if arrow["side"] in VERTICAL_SIDES]
    if not horizontal or not vertical:
        return None

    interaction_h = random.choice(horizontal)
    interaction_v = random.choice(vertical)
    crossing = (interaction_h["tip"][0], interaction_v["tip"][1])
    reserved_clear = set(_path_to_crossing(interaction_h, crossing))
    reserved_clear.update(_path_to_crossing(interaction_v, crossing))

    arrow_cells = {cell for arrow in arrows for cell in arrow["cells"]}
    blocked_for_obstacles = set(arrow_cells)
    blocked_for_obstacles.update(reserved_clear)
    obstacle_cells: set[tuple[int, int]] = set()

    stop_choices = [arrow for arrow in arrows if arrow not in (interaction_h, interaction_v)]
    if not stop_choices:
        stop_choices = arrows[:]
    random.shuffle(stop_choices)

    stop_arrow = None
    stop_cell = None
    stop_path: list[tuple[int, int]] = []
    for arrow in stop_choices:
        path = [cell for cell in _arrow_path_before_input_block(grid, arrow) if cell not in reserved_clear]
        if path:
            stop_arrow = arrow
            stop_path = path
            stop_cell = random.choice(path)
            break
    if stop_arrow is None or stop_cell is None:
        return None

    obstacle_palette = [color for color in PALETTE if color not in arrow_colors]
    if not obstacle_palette:
        obstacle_palette = list(PALETTE)

    grid[stop_cell] = random.choice(obstacle_palette)
    obstacle_cells.add(stop_cell)
    blocked_for_obstacles.add(stop_cell)

    stop_index = stop_path.index(stop_cell)
    endpoint = stop_arrow["tip"] if stop_index == 0 else stop_path[stop_index - 1]
    orth_candidates = []
    for odr, odc in _orthogonal_steps(stop_arrow["direction"]):
        target = (endpoint[0] + odr, endpoint[1] + odc)
        if (
            _in_bounds(grid.shape, *target)
            and target not in blocked_for_obstacles
            and target not in obstacle_cells
        ):
            orth_candidates.append(target)
    random.shuffle(orth_candidates)
    desired_blockers = random.choice((0, 1, 2))
    for target in orth_candidates[:desired_blockers]:
        grid[target] = random.choice(obstacle_palette)
        obstacle_cells.add(target)
        blocked_for_obstacles.add(target)

    extra_target = random.randint(0, GEN_EXTRA_OBSTACLES_MAX)
    open_cells = _collect_open_cells(grid.shape, blocked_for_obstacles, obstacle_cells)
    random.shuffle(open_cells)
    extras_added = 0
    for cell in open_cells:
        if extras_added >= extra_target:
            break
        grid[cell] = random.choice(obstacle_palette)
        obstacle_cells.add(cell)
        blocked_for_obstacles.add(cell)
        extras_added += 1

    try:
        out, traces, parsed_arrows = _simulate(grid, collect_trace=True)
    except AssertionError:
        return None

    if len(parsed_arrows) != len(arrows):
        return None
    if not any(trace["blocker_kind"] == "obstacle" for trace in traces):
        return None
    if not any(
        trace["side"] in VERTICAL_SIDES and trace["blocker_kind"] == "painted"
        for trace in traces
    ):
        return None
    if sum(int(trace["diffusion_count"]) for trace in traces) < 2:
        return None
    if not any(int(trace["ray_len"]) >= 2 for trace in traces):
        return None
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
