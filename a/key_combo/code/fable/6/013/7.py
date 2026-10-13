from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg


BLACK = int(Color.BLACK)
MAX_GENERATION_ATTEMPTS = 800
HEIGHT_RANGE = (10, 19)
WIDTH_RANGE = (10, 19)

DIR_UP = (-1, 0)
DIR_DOWN = (1, 0)
DIR_LEFT = (0, -1)
DIR_RIGHT = (0, 1)
CARDINAL_DIRS = (DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT)

ORIENTATION_BY_LOCAL = {
    frozenset({(0, 0), (0, 1), (1, 0)}): {"corner_local": (0, 0), "hdir": 1, "vdir": 1},
    frozenset({(0, 0), (0, 1), (1, 1)}): {"corner_local": (0, 1), "hdir": -1, "vdir": 1},
    frozenset({(0, 0), (1, 0), (1, 1)}): {"corner_local": (1, 0), "hdir": 1, "vdir": -1},
    frozenset({(0, 1), (1, 0), (1, 1)}): {"corner_local": (1, 1), "hdir": -1, "vdir": -1},
}

ORIENTATION_SPECS = {
    "TL": {"offsets": ((0, 0), (0, 1), (1, 0)), "hdir": 1, "vdir": 1},
    "TR": {"offsets": ((0, 0), (0, -1), (1, 0)), "hdir": -1, "vdir": 1},
    "BL": {"offsets": ((0, 0), (0, 1), (-1, 0)), "hdir": 1, "vdir": -1},
    "BR": {"offsets": ((0, 0), (0, -1), (-1, 0)), "hdir": -1, "vdir": -1},
}


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    h, w = shape
    r, c = cell
    return 0 <= r < h and 0 <= c < w


def _is_horizontal(direction: tuple[int, int]) -> bool:
    return int(direction[1]) != 0


def _turn_direction(hdir: int, vdir: int, incoming: tuple[int, int]) -> tuple[int, int]:
    if _is_horizontal(incoming):
        return (int(vdir), 0)
    return (0, int(hdir))


def _steps_to_edge(shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]) -> int:
    h, w = shape
    r, c = start
    dr, dc = direction
    steps = 0
    while True:
        nr = r + dr
        nc = c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return steps
        r, c = nr, nc
        steps += 1


def _ray_cells(start: tuple[int, int], direction: tuple[int, int], steps: int) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = direction
    return [(r + dr * k, c + dc * k) for k in range(1, steps + 1)]


def _extract_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)

    ball_candidates: list[dict[str, object]] = []
    deflectors: list[dict[str, object]] = []
    obstacle_cells: set[tuple[int, int]] = set()

    for comp in components:
        area = int(comp["area"])
        color = int(comp["color"])
        coords = np.asarray(comp["coords"], dtype=int)
        bbox = tuple(map(int, comp["bbox"]))

        if area == 1:
            r, c = map(int, coords[0])
            ball_candidates.append({"pos": (r, c), "color": color})
            continue

        r0, c0, r1, c1 = bbox
        if area == 3 and (r1 - r0 + 1, c1 - c0 + 1) == (2, 2):
            local = frozenset((int(r) - r0, int(c) - c0) for r, c in coords)
            if local in ORIENTATION_BY_LOCAL:
                orient = ORIENTATION_BY_LOCAL[local]
                corner_local = orient["corner_local"]
                corner = (r0 + int(corner_local[0]), c0 + int(corner_local[1]))
                cells = {tuple(map(int, rc)) for rc in coords.tolist()}
                deflectors.append(
                    {
                        "color": color,
                        "cells": cells,
                        "corner": corner,
                        "hdir": int(orient["hdir"]),
                        "vdir": int(orient["vdir"]),
                    }
                )
                continue

        for cell in coords.tolist():
            obstacle_cells.add((int(cell[0]), int(cell[1])))

    assert len(ball_candidates) == 1, "expected exactly one ball"
    assert deflectors, "expected at least one L deflector"

    return {
        "grid": g,
        "bg": bg,
        "ball_pos": tuple(ball_candidates[0]["pos"]),
        "ball_color": int(ball_candidates[0]["color"]),
        "deflectors": deflectors,
        "obstacle_cells": obstacle_cells,
    }


def _scan_event(
    shape: tuple[int, int],
    current: tuple[int, int],
    direction: tuple[int, int],
    active_by_cell: dict[tuple[int, int], int],
    obstacle_cells: set[tuple[int, int]],
) -> tuple[str, object, list[tuple[int, int]], tuple[int, int]]:
    r, c = current
    dr, dc = direction
    traversed: list[tuple[int, int]] = []

    while True:
        nr = r + dr
        nc = c + dc
        if not (0 <= nr < shape[0] and 0 <= nc < shape[1]):
            return ("edge", None, traversed, (r, c))
        next_cell = (nr, nc)
        if next_cell in obstacle_cells:
            return ("obstacle", None, traversed, (r, c))
        if next_cell in active_by_cell:
            return ("deflector", int(active_by_cell[next_cell]), traversed, next_cell)
        traversed.append(next_cell)
        r, c = nr, nc


def _valid_initial_directions(
    shape: tuple[int, int],
    ball_pos: tuple[int, int],
    deflectors: list[dict[str, object]],
    obstacle_cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    by_cell = {}
    for idx, deflector in enumerate(deflectors):
        for cell in deflector["cells"]:
            by_cell[cell] = idx

    valid: list[tuple[int, int]] = []
    for direction in CARDINAL_DIRS:
        event_kind, _idx, _traversed, _stop_or_hit = _scan_event(
            shape, ball_pos, direction, by_cell, obstacle_cells
        )
        if event_kind == "deflector":
            valid.append(direction)
    return valid


def _solve(grid: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    scene = _extract_scene(grid)
    g = np.asarray(scene["grid"], dtype=int)
    bg = int(scene["bg"])
    ball_pos = tuple(scene["ball_pos"])
    ball_color = int(scene["ball_color"])
    deflectors = list(scene["deflectors"])
    obstacle_cells = set(scene["obstacle_cells"])

    initial_dirs = _valid_initial_directions(g.shape, ball_pos, deflectors, obstacle_cells)
    assert len(initial_dirs) == 1, "expected exactly one viable initial direction"
    direction = initial_dirs[0]

    out = g.copy()
    out[ball_pos] = BLACK

    active_ids = set(range(len(deflectors)))
    active_by_cell = {
        cell: idx
        for idx, deflector in enumerate(deflectors)
        for cell in deflector["cells"]
    }

    current = ball_pos
    used_ids: list[int] = []
    trail_cells = {ball_pos}

    while True:
        event_kind, event_payload, traversed, stop_or_hit = _scan_event(
            g.shape, current, direction, active_by_cell, obstacle_cells
        )

        for cell in traversed:
            out[cell] = BLACK
            trail_cells.add(cell)

        if event_kind != "deflector":
            final_pos = tuple(stop_or_hit)
            out[final_pos] = ball_color
            return out, {
                "bg": bg,
                "ball_color": ball_color,
                "ball_pos": ball_pos,
                "final_pos": final_pos,
                "initial_direction": direction,
                "used_deflectors": used_ids,
                "trail_cells": trail_cells,
                "obstacle_cells": obstacle_cells,
            }

        deflector_id = int(event_payload)
        deflector = deflectors[deflector_id]
        hit_cell = tuple(stop_or_hit)
        corner = tuple(deflector["corner"])

        for cell in deflector["cells"]:
            out[cell] = bg
            active_by_cell.pop(cell, None)
        active_ids.remove(deflector_id)
        used_ids.append(deflector_id)

        out[hit_cell] = BLACK
        out[corner] = BLACK
        trail_cells.add(hit_cell)
        trail_cells.add(corner)

        current = corner
        direction = _turn_direction(int(deflector["hdir"]), int(deflector["vdir"]), direction)


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Infer the unique hidden initial direction, then simulate the ball.

    The ball moves straight until it either reaches a boundary cell or would hit
    a solid obstacle. When it first touches a 3-cell 2x2 L deflector, that
    deflector is consumed, the ball jumps to the L's corner cell, and it leaves
    in the other arm's encoded direction. All traversed cells become black,
    except the final resting cell, which keeps the ball color.
    """
    return _solve(np.asarray(input_grid, dtype=int))[0]


def _matching_orientation_names(
    incoming: tuple[int, int], outgoing: tuple[int, int]
) -> list[str]:
    matches: list[str] = []
    for name, spec in ORIENTATION_SPECS.items():
        if _turn_direction(int(spec["hdir"]), int(spec["vdir"]), incoming) == outgoing:
            matches.append(name)
    return matches


def _first_hit_for_orientation(
    corner: tuple[int, int], incoming: tuple[int, int], spec: dict[str, object]
) -> tuple[int, int]:
    r, c = corner
    hdir = int(spec["hdir"])
    vdir = int(spec["vdir"])
    if incoming == DIR_RIGHT:
        return (r, c if hdir == 1 else c - 1)
    if incoming == DIR_LEFT:
        return (r, c + 1 if hdir == 1 else c)
    if incoming == DIR_DOWN:
        return (r if vdir == 1 else r - 1, c)
    assert incoming == DIR_UP
    return (r + 1 if vdir == 1 else r, c)


def _candidate_deflectors(
    shape: tuple[int, int],
    current: tuple[int, int],
    incoming: tuple[int, int],
    outgoing: tuple[int, int],
    object_occupied: set[tuple[int, int]],
    reserved_empty: set[tuple[int, int]],
    vanished_cells: set[tuple[int, int]],
    min_after_room: int,
) -> list[dict[str, object]]:
    h, w = shape
    matches = _matching_orientation_names(incoming, outgoing)
    candidates: list[dict[str, object]] = []

    for name in matches:
        spec = ORIENTATION_SPECS[name]
        for corner_r in range(h):
            for corner_c in range(w):
                corner = (corner_r, corner_c)
                cells = {(corner_r + dr, corner_c + dc) for dr, dc in spec["offsets"]}
                if any(not _in_bounds(shape, cell) for cell in cells):
                    continue
                if cells & object_occupied:
                    continue
                if cells & reserved_empty:
                    continue

                hit = _first_hit_for_orientation(corner, incoming, spec)
                if not _in_bounds(shape, hit):
                    continue

                if incoming == DIR_RIGHT:
                    if hit[0] != current[0] or hit[1] <= current[1]:
                        continue
                    steps = hit[1] - current[1]
                elif incoming == DIR_LEFT:
                    if hit[0] != current[0] or hit[1] >= current[1]:
                        continue
                    steps = current[1] - hit[1]
                elif incoming == DIR_DOWN:
                    if hit[1] != current[1] or hit[0] <= current[0]:
                        continue
                    steps = hit[0] - current[0]
                else:
                    if hit[1] != current[1] or hit[0] >= current[0]:
                        continue
                    steps = current[0] - hit[0]

                if steps < 2:
                    continue

                path_before = _ray_cells(current, incoming, steps - 1)
                blocked_now = object_occupied - vanished_cells
                if set(path_before) & blocked_now:
                    continue
                if set(path_before) & reserved_empty:
                    continue

                if _steps_to_edge(shape, corner, outgoing) < min_after_room:
                    continue

                candidates.append(
                    {
                        "name": name,
                        "corner": corner,
                        "cells": cells,
                        "path_before": path_before,
                        "outgoing": outgoing,
                    }
                )
    return candidates


def _place_rectangle_obstacle(
    grid: np.ndarray,
    rng: np.random.Generator,
    color: int,
    object_occupied: set[tuple[int, int]],
    reserved_empty: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    choices: list[tuple[int, int, int, int]] = []

    for height in (2, 3):
        for width in (2, 3, 4):
            if height > h or width > w:
                continue
            for r0 in range(h - height + 1):
                for c0 in range(w - width + 1):
                    cells = {
                        (r, c)
                        for r in range(r0, r0 + height)
                        for c in range(c0, c0 + width)
                    }
                    if cells & object_occupied:
                        continue
                    if cells & reserved_empty:
                        continue
                    choices.append((r0, c0, height, width))

    if not choices:
        return False

    r0, c0, height, width = choices[int(rng.integers(0, len(choices)))]
    for r in range(r0, r0 + height):
        for c in range(c0, c0 + width):
            grid[r, c] = int(color)
            object_occupied.add((r, c))
    return True


def _place_distractor_deflector(
    grid: np.ndarray,
    rng: np.random.Generator,
    color: int,
    object_occupied: set[tuple[int, int]],
    reserved_empty: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    choices: list[tuple[str, tuple[int, int], set[tuple[int, int]]]] = []

    for name, spec in ORIENTATION_SPECS.items():
        for corner_r in range(h):
            for corner_c in range(w):
                cells = {(corner_r + dr, corner_c + dc) for dr, dc in spec["offsets"]}
                if any(not _in_bounds((h, w), cell) for cell in cells):
                    continue
                if cells & object_occupied:
                    continue
                if cells & reserved_empty:
                    continue
                choices.append((name, (corner_r, corner_c), cells))

    if not choices:
        return False

    _name, _corner, cells = choices[int(rng.integers(0, len(choices)))]
    for r, c in cells:
        grid[r, c] = int(color)
        object_occupied.add((r, c))
    return True


def _build_path_instance(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(*HEIGHT_RANGE))
    width = int(rng.integers(*WIDTH_RANGE))
    shape = (height, width)

    bg = int(rng.choice(np.arange(1, 10)))
    palette = [color for color in range(1, 10) if color != bg]
    initial_turns = int(rng.integers(1, 4))
    obstacle_count = int(rng.integers(1, 3))
    distractor_count = int(rng.integers(0, 2))
    object_count = 1 + initial_turns + obstacle_count + distractor_count
    if object_count > len(palette):
        return None

    colors = [int(color) for color in rng.choice(palette, size=object_count, replace=False).tolist()]
    ball_color = colors[0]
    path_deflector_colors = colors[1 : 1 + initial_turns]
    next_color_idx = 1 + initial_turns
    obstacle_colors = colors[next_color_idx : next_color_idx + obstacle_count]
    next_color_idx += obstacle_count
    distractor_colors = colors[next_color_idx : next_color_idx + distractor_count]

    row_margin = 2
    col_margin = 2
    if height <= 2 * row_margin or width <= 2 * col_margin:
        return None

    ball_pos = (
        int(rng.integers(row_margin, height - row_margin)),
        int(rng.integers(col_margin, width - col_margin)),
    )
    initial_direction = CARDINAL_DIRS[int(rng.integers(0, len(CARDINAL_DIRS)))]

    object_occupied = {ball_pos}
    reserved_empty: set[tuple[int, int]] = set()
    vanished_cells: set[tuple[int, int]] = set()
    placements: list[dict[str, object]] = []

    current = ball_pos
    direction = initial_direction

    for turn_idx in range(initial_turns):
        if _is_horizontal(direction):
            outgoing_options = [DIR_UP, DIR_DOWN]
        else:
            outgoing_options = [DIR_LEFT, DIR_RIGHT]
        rng.shuffle(outgoing_options)

        chosen = None
        min_after_room = 2 if turn_idx < initial_turns - 1 else 1
        for outgoing in outgoing_options:
            candidates = _candidate_deflectors(
                shape,
                current,
                direction,
                outgoing,
                object_occupied,
                reserved_empty,
                vanished_cells,
                min_after_room,
            )
            if candidates:
                chosen = candidates[int(rng.integers(0, len(candidates)))]
                break
        if chosen is None:
            return None

        for cell in chosen["path_before"]:
            if cell not in vanished_cells:
                reserved_empty.add(cell)
        object_occupied |= set(chosen["cells"])
        vanished_cells |= set(chosen["cells"])
        placements.append(chosen)
        current = tuple(chosen["corner"])
        direction = tuple(chosen["outgoing"])

    final_steps = _steps_to_edge(shape, current, direction)
    if final_steps < 1:
        return None
    for cell in _ray_cells(current, direction, final_steps):
        if cell not in vanished_cells:
            reserved_empty.add(cell)

    grid = np.full(shape, bg, dtype=int)
    grid[ball_pos] = ball_color
    for color, placement in zip(path_deflector_colors, placements):
        for r, c in placement["cells"]:
            grid[r, c] = int(color)

    for color in obstacle_colors:
        if not _place_rectangle_obstacle(grid, rng, color, object_occupied, reserved_empty):
            return None

    for color in distractor_colors:
        if not _place_distractor_deflector(grid, rng, color, object_occupied, reserved_empty):
            return None

    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.shape[0] > 30 or g.shape[1] > 30:
        return False

    try:
        scene = _extract_scene(g)
        initial_dirs = _valid_initial_directions(
            g.shape,
            tuple(scene["ball_pos"]),
            list(scene["deflectors"]),
            set(scene["obstacle_cells"]),
        )
        if len(initial_dirs) != 1:
            return False

        out, details = _solve(g)
    except AssertionError:
        return False

    if out.ndim != 2 or out.shape != g.shape:
        return False

    ball_color = int(details["ball_color"])
    ball_count = int(np.count_nonzero(out == ball_color))
    if ball_count != 1:
        return False

    trail_cells = {tuple(cell) for cell in details["trail_cells"]}
    if not trail_cells:
        return False
    if int(out[tuple(details["final_pos"])]) != ball_color:
        return False

    black_cells = set(map(tuple, np.argwhere(out == BLACK).tolist()))
    if tuple(details["final_pos"]) in black_cells:
        return False
    if not trail_cells - {tuple(details["final_pos"])} <= black_cells:
        return False
    if len(details["used_deflectors"]) < 1:
        return False

    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_path_instance(rng)
        if grid is not None and _validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid dark-pinball ARC input")
