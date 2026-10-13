"""
Phase-turn particle ARC task.

Design notes kept explicit for the local codegen workflow:
- The corner 1x3 multicolor bar is the legend and defines the trail cycle A,B,C.
- The only non-legend singleton on the border is the start marker. Walls are
  interior solid rectangles with area >= 2, so parsing stays unambiguous.
- Accepted generated instances must show at least one collision-driven turn, so
  the phase-dependent turning rule is visually observable instead of collapsing
  to a trivial straight exit.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_HELPER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_HELPER_DIR) not in sys.path:
    sys.path.append(str(_HELPER_DIR))

from arc_utils import find_components, get_bg, in_bounds


PALETTE = list(range(10))
CARDINALS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
RECT_SHAPES = [(1, 2), (2, 1), (2, 2), (1, 3), (3, 1)]
GEN_RETRIES = 400


def _left(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return -dc, dr


def _right(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return dc, -dr


def _legend_cells(height: int, width: int, corner: str) -> list[tuple[int, int]]:
    if corner == "tl":
        return [(0, 0), (0, 1), (0, 2)]
    if corner == "tr":
        return [(0, width - 3), (0, width - 2), (0, width - 1)]
    if corner == "bl":
        return [(height - 1, 0), (height - 1, 1), (height - 1, 2)]
    if corner == "br":
        return [(height - 1, width - 3), (height - 1, width - 2), (height - 1, width - 1)]
    raise ValueError(f"unknown corner: {corner}")


def _legend_moat(cells: list[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    height, width = shape
    reserved: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    reserved.add((nr, nc))
    return reserved


def _edge_direction(start: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    r, c = start
    height, width = shape
    if r == 0:
        return 1, 0
    if r == height - 1:
        return -1, 0
    if c == 0:
        return 0, 1
    if c == width - 1:
        return 0, -1
    raise ValueError("start marker must lie on the border")


def _ordered_legend_colors(grid: np.ndarray, coords: np.ndarray) -> list[int]:
    points = [tuple(map(int, rc)) for rc in coords]
    rows = {r for r, _ in points}
    if len(rows) == 1:
        ordered = sorted(points, key=lambda rc: rc[1])
    else:
        ordered = sorted(points, key=lambda rc: rc[0])
    return [int(grid[r, c]) for r, c in ordered]


def _extract_task(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("input_grid must be 2D")
    bg = int(get_bg(g))
    height, width = g.shape

    multi_components = find_components(
        g,
        bg=bg,
        connectivity=8,
        monochromatic=False,
        return_format="dict",
    )
    legend_candidates = []
    for comp in multi_components:
        if int(comp["area"]) != 3:
            continue
        r0, c0, r1, c1 = map(int, comp["bbox"])
        box_h = r1 - r0 + 1
        box_w = c1 - c0 + 1
        if (box_h, box_w) not in {(1, 3), (3, 1)}:
            continue
        touches_corner = (
            (r0 == 0 and c0 == 0)
            or (r0 == 0 and c1 == width - 1)
            or (r1 == height - 1 and c0 == 0)
            or (r1 == height - 1 and c1 == width - 1)
        )
        if touches_corner:
            legend_candidates.append(comp)
    if len(legend_candidates) != 1:
        raise ValueError("expected exactly one corner 1x3 legend")

    legend_coords = legend_candidates[0]["coords"]
    legend_set = {tuple(map(int, rc)) for rc in legend_coords}
    legend_colors = _ordered_legend_colors(g, legend_coords)

    mono_components = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    start_candidates = []
    for comp in mono_components:
        if int(comp["area"]) != 1:
            continue
        cell = tuple(map(int, comp["coords"][0]))
        if cell in legend_set:
            continue
        r, c = cell
        if r in {0, height - 1} or c in {0, width - 1}:
            start_candidates.append(cell)
    if len(start_candidates) != 1:
        raise ValueError("expected exactly one non-legend edge singleton start marker")

    start = start_candidates[0]
    static_obstacles = (g != bg)
    static_obstacles[start] = False

    return {
        "grid": g,
        "bg": bg,
        "legend_colors": legend_colors,
        "legend_set": legend_set,
        "start": start,
        "direction": _edge_direction(start, g.shape),
        "static_obstacles": static_obstacles,
    }


def _simulate_task(task: dict) -> tuple[np.ndarray, dict]:
    grid = task["grid"]
    bg = int(task["bg"])
    colors = list(map(int, task["legend_colors"]))
    start_r, start_c = map(int, task["start"])
    dr, dc = map(int, task["direction"])

    out = grid.copy()
    blocked = task["static_obstacles"].copy()
    trail = np.zeros_like(grid, dtype=bool)

    phase = 0
    out[start_r, start_c] = colors[phase]
    blocked[start_r, start_c] = True
    trail[start_r, start_c] = True

    path = [(start_r, start_c)]
    wall_hits = 0
    trail_hits = 0
    turns = 0
    stop_reason = "exit"
    r, c = start_r, start_c

    while True:
        nr, nc = r + dr, c + dc
        if not in_bounds(out.shape, nr, nc):
            stop_reason = "exit"
            break

        if blocked[nr, nc]:
            if task["static_obstacles"][nr, nc]:
                wall_hits += 1
            else:
                trail_hits += 1

            if phase == 2:
                stop_reason = "phase_c_stop"
                break

            turn_dir = _left((dr, dc)) if phase == 0 else _right((dr, dc))
            tr, tc = r + turn_dir[0], c + turn_dir[1]
            if not in_bounds(out.shape, tr, tc):
                stop_reason = "turn_exits_grid"
                break
            if blocked[tr, tc]:
                stop_reason = "blocked_after_turn"
                break

            dr, dc = turn_dir
            nr, nc = tr, tc
            turns += 1

        r, c = nr, nc
        phase = (phase + 1) % 3
        out[r, c] = colors[phase]
        blocked[r, c] = True
        trail[r, c] = True
        path.append((r, c))

    changed = (out != grid)
    changed_count = int(changed.sum())
    trail_count = int(trail.sum())

    return out, {
        "path": path,
        "trail_mask": trail,
        "turns": turns,
        "wall_hits": wall_hits,
        "trail_hits": trail_hits,
        "stop_reason": stop_reason,
        "changed_count": changed_count,
        "trail_count": trail_count,
        "bg": bg,
    }


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Trace a color-cycling particle that turns according to its current phase.

    Input structure:
    - The background is the most common color.
    - A corner 1x3 legend bar gives the trail colors A, B, C in order.
    - A single non-legend border pixel is the start marker.
    - Other foreground objects are solid wall blocks that remain unchanged.

    Rule:
    - The particle enters from the marked border cell and paints visited cells
      with A, then B, then C, repeating every step.
    - If the cell ahead is blocked by a wall or by the already painted trail,
      the current phase decides the response: A turns left, B turns right,
      C stops immediately.
    - After a turn, if the new forward cell is still blocked or outside the
      grid, the particle stops. Leaving the grid also stops the motion.

    Key insight:
    - Geometry only tells you where collisions happen; the legend fixes the
      three-phase state machine that decides left turn, right turn, or stop.
      Because the trail becomes an obstacle, the same state machine can trap
      itself later in the run.
    """
    task = _extract_task(input_grid)
    out, _stats = _simulate_task(task)
    return out


def _start_options(shape: tuple[int, int], reserved: set[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    height, width = shape
    options: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for c in range(1, width - 1):
        start = (0, c)
        if start not in reserved and (1, c) not in reserved:
            options.append((start, (1, 0)))
        start = (height - 1, c)
        if start not in reserved and (height - 2, c) not in reserved:
            options.append((start, (-1, 0)))
    for r in range(1, height - 1):
        start = (r, 0)
        if start not in reserved and (r, 1) not in reserved:
            options.append((start, (0, 1)))
        start = (r, width - 1)
        if start not in reserved and (r, width - 2) not in reserved:
            options.append((start, (0, -1)))
    random.shuffle(options)
    return options


def _candidate_wall_rects(
    shape: tuple[int, int],
    blocker: tuple[int, int],
    forbidden: set[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    height, width = shape
    br, bc = blocker
    candidates: list[list[tuple[int, int]]] = []
    for rect_h, rect_w in RECT_SHAPES:
        for r0 in range(br - rect_h + 1, br + 1):
            for c0 in range(bc - rect_w + 1, bc + 1):
                if r0 < 0 or c0 < 0 or r0 + rect_h > height or c0 + rect_w > width:
                    continue
                cells = [(r, c) for r in range(r0, r0 + rect_h) for c in range(c0, c0 + rect_w)]
                if any(cell in forbidden for cell in cells):
                    continue
                if blocker in cells:
                    candidates.append(cells)
    return candidates


def _plan_turn(
    shape: tuple[int, int],
    pos: tuple[int, int],
    direction: tuple[int, int],
    step_count: int,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    wall_cells: set[tuple[int, int]],
    desired_phase: int,
) -> tuple | None:
    height, width = shape
    candidates = []
    wall_guard = _expand_cells4(shape, wall_cells)
    for travel in range(1, max(height, width)):
        if (step_count + travel - 1) % 3 != desired_phase:
            continue
        walk: list[tuple[int, int]] = []
        ok = True
        for dist in range(1, travel + 1):
            cell = (pos[0] + direction[0] * dist, pos[1] + direction[1] * dist)
            if not in_bounds(shape, *cell) or cell in occupied or cell in reserved or cell in wall_cells:
                ok = False
                break
            walk.append(cell)
        if not ok:
            continue

        last = walk[-1]
        blocker = (last[0] + direction[0], last[1] + direction[1])
        if not in_bounds(shape, *blocker) or blocker in occupied or blocker in reserved or blocker in wall_cells:
            continue

        new_direction = _left(direction) if desired_phase == 0 else _right(direction)
        turned = (last[0] + new_direction[0], last[1] + new_direction[1])
        if not in_bounds(shape, *turned) or turned in occupied or turned in reserved or turned in wall_cells:
            continue

        forbidden = occupied | reserved | wall_guard | set(walk) | {turned}
        rects = _candidate_wall_rects(shape, blocker, forbidden)
        if not rects:
            continue
        candidates.append((travel, walk, new_direction, turned, random.choice(rects)))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    top_band = candidates[: min(6, len(candidates))]
    return random.choice(top_band)


def _plan_phase_c_stop(
    shape: tuple[int, int],
    pos: tuple[int, int],
    direction: tuple[int, int],
    step_count: int,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    wall_cells: set[tuple[int, int]],
) -> tuple | None:
    height, width = shape
    candidates = []
    wall_guard = _expand_cells4(shape, wall_cells)
    for travel in range(1, max(height, width)):
        if (step_count + travel - 1) % 3 != 2:
            continue
        walk: list[tuple[int, int]] = []
        ok = True
        for dist in range(1, travel + 1):
            cell = (pos[0] + direction[0] * dist, pos[1] + direction[1] * dist)
            if not in_bounds(shape, *cell) or cell in occupied or cell in reserved or cell in wall_cells:
                ok = False
                break
            walk.append(cell)
        if not ok:
            continue

        blocker = (walk[-1][0] + direction[0], walk[-1][1] + direction[1])
        if not in_bounds(shape, *blocker) or blocker in occupied or blocker in reserved or blocker in wall_cells:
            continue

        forbidden = occupied | reserved | wall_guard | set(walk)
        rects = _candidate_wall_rects(shape, blocker, forbidden)
        if not rects:
            continue
        candidates.append((travel, walk, random.choice(rects)))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    top_band = candidates[: min(6, len(candidates))]
    return random.choice(top_band)


def _plan_trail_stop(
    shape: tuple[int, int],
    pos: tuple[int, int],
    direction: tuple[int, int],
    step_count: int,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    wall_cells: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    height, width = shape
    candidates = []
    for travel in range(1, max(height, width)):
        if (step_count + travel - 1) % 3 != 2:
            continue
        walk: list[tuple[int, int]] = []
        ok = True
        for dist in range(1, travel + 1):
            cell = (pos[0] + direction[0] * dist, pos[1] + direction[1] * dist)
            if not in_bounds(shape, *cell) or cell in occupied or cell in reserved or cell in wall_cells:
                ok = False
                break
            walk.append(cell)
        if not ok:
            continue
        blocker = (walk[-1][0] + direction[0], walk[-1][1] + direction[1])
        if blocker in occupied:
            candidates.append(walk)

    if not candidates:
        return None

    candidates.sort(key=len, reverse=True)
    top_band = candidates[: min(6, len(candidates))]
    return random.choice(top_band)


def _plan_exit(
    shape: tuple[int, int],
    pos: tuple[int, int],
    direction: tuple[int, int],
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    wall_cells: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    walk: list[tuple[int, int]] = []
    dist = 1
    while True:
        cell = (pos[0] + direction[0] * dist, pos[1] + direction[1] * dist)
        if not in_bounds(shape, *cell):
            break
        if cell in occupied or cell in reserved or cell in wall_cells:
            return None
        walk.append(cell)
        dist += 1
    return walk if walk else None


def _expand_cells4(shape: tuple[int, int], cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    expanded = set(cells)
    for r, c in list(cells):
        for dr, dc in CARDINALS:
            nr, nc = r + dr, c + dc
            if in_bounds(shape, nr, nc):
                expanded.add((nr, nc))
    return expanded


def _path_buffer(shape: tuple[int, int], path_cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return _expand_cells4(shape, path_cells)


def _place_extra_rectangles(
    shape: tuple[int, int],
    forbidden: set[tuple[int, int]],
    count: int,
) -> list[list[tuple[int, int]]]:
    height, width = shape
    used = set(forbidden)
    extras: list[list[tuple[int, int]]] = []
    for _ in range(count):
        rect_candidates = []
        shapes = RECT_SHAPES + [(2, 3), (3, 2)]
        random.shuffle(shapes)
        for rect_h, rect_w in shapes:
            for r0 in range(1, height - rect_h):
                for c0 in range(1, width - rect_w):
                    cells = [(r, c) for r in range(r0, r0 + rect_h) for c in range(c0, c0 + rect_w)]
                    if any(cell in used for cell in cells):
                        continue
                    rect_candidates.append(cells)
        if not rect_candidates:
            break
        chosen = random.choice(rect_candidates)
        extras.append(chosen)
        used.update(_expand_cells4(shape, set(chosen)))
    return extras


def _try_generate() -> np.ndarray | None:
    height = random.randint(12, 22)
    width = random.randint(12, 22)
    bg = random.choice(PALETTE)

    corner = random.choice(["tl", "tr", "bl", "br"])
    legend_pos = _legend_cells(height, width, corner)
    reserved = _legend_moat(legend_pos, (height, width))

    start_options = _start_options((height, width), reserved)
    if not start_options:
        return None
    start, direction = random.choice(start_options)

    occupied = {start}
    wall_cells: set[tuple[int, int]] = set()
    step_count = 1
    pos = start
    path_cells = {start}

    turns_target = random.randint(1, 3)
    for _ in range(turns_target):
        desired_phase = random.choice([0, 1])
        planned = _plan_turn(
            (height, width),
            pos,
            direction,
            step_count,
            occupied,
            reserved,
            wall_cells,
            desired_phase,
        )
        if planned is None:
            return None
        _travel, walk, direction, turned, rect = planned
        occupied.update(walk)
        occupied.add(turned)
        path_cells.update(walk)
        path_cells.add(turned)
        wall_cells.update(rect)
        step_count += len(walk) + 1
        pos = turned

    final_mode = random.choice(["exit", "phase_c_stop", "trail_stop"])
    if final_mode == "trail_stop":
        walk = _plan_trail_stop(
            (height, width),
            pos,
            direction,
            step_count,
            occupied,
            reserved,
            wall_cells,
        )
        if walk is None:
            final_mode = random.choice(["exit", "phase_c_stop"])
        else:
            occupied.update(walk)
            path_cells.update(walk)

    if final_mode == "phase_c_stop":
        planned_stop = _plan_phase_c_stop(
            (height, width),
            pos,
            direction,
            step_count,
            occupied,
            reserved,
            wall_cells,
        )
        if planned_stop is None:
            return None
        _travel, walk, rect = planned_stop
        occupied.update(walk)
        path_cells.update(walk)
        wall_cells.update(rect)
    elif final_mode == "exit":
        walk = _plan_exit((height, width), pos, direction, occupied, reserved, wall_cells)
        if walk is None:
            return None
        occupied.update(walk)
        path_cells.update(walk)

    wall_buffer = _expand_cells4((height, width), wall_cells)
    buffer_cells = _path_buffer((height, width), path_cells) | reserved | wall_buffer
    extra_rects = _place_extra_rectangles((height, width), buffer_cells, random.randint(1, 3))

    colors = [c for c in PALETTE if c != bg]
    random.shuffle(colors)
    if len(colors) < 4:
        return None
    legend_colors = colors[:3]
    start_color = colors[3]
    wall_colors = colors[4:]
    all_rects: list[list[tuple[int, int]]] = []
    # Rebuild connected wall components from the planned wall mask so each block gets one color.
    wall_mask = np.zeros((height, width), dtype=bool)
    for r, c in wall_cells:
        wall_mask[r, c] = True
    for coords in find_components(
        wall_mask.astype(int),
        bg=0,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    ):
        all_rects.append([tuple(map(int, rc)) for rc in coords["coords"]])
    all_rects.extend(extra_rects)

    if len(wall_colors) < len(all_rects):
        random.shuffle(colors)
        wall_colors = (wall_colors + colors)[: len(all_rects)]

    grid = np.full((height, width), bg, dtype=int)
    for (r, c), color in zip(legend_pos, legend_colors):
        grid[r, c] = color
    grid[start] = start_color
    for idx, rect in enumerate(all_rects):
        color = int(wall_colors[idx])
        for r, c in rect:
            grid[r, c] = color

    try:
        task = _extract_task(grid)
        out, stats = _simulate_task(task)
    except Exception:
        return None

    if stats["turns"] < 1:
        return None
    if stats["trail_count"] < 6:
        return None
    if stats["wall_hits"] < 1:
        return None
    if out.shape[0] > 30 or out.shape[1] > 30:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _try_generate()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
