from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, in_bounds


BACKGROUND = Color.BLACK
OBSTACLE = Color.GREY
PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
GEN_RETRY_LIMIT = 500
EVEN_DISTANCES = (2, 4, 6)
LONG_EVEN_DISTANCES = (4, 6, 8)

Coord = tuple[int, int]


def _same_lattice(a: Coord, b: Coord) -> bool:
    return (a[0] - b[0]) % 2 == 0 and (a[1] - b[1]) % 2 == 0


def _exclusive_segment(a: Coord, b: Coord) -> list[Coord]:
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 1 if bc > ac else -1
        return [(ar, c) for c in range(ac + step, bc, step)]
    if ac == bc:
        step = 1 if br > ar else -1
        return [(r, ac) for r in range(ar + step, br, step)]
    raise ValueError("segment endpoints must share a row or column")


def _segment_clear(grid: np.ndarray, a: Coord, b: Coord) -> bool:
    return all(int(grid[r, c]) == BACKGROUND for r, c in _exclusive_segment(a, b))


def _parse_grid(grid: np.ndarray) -> tuple[np.ndarray, int, int, Coord]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    colors = set(map(int, np.unique(g))) - {BACKGROUND, OBSTACLE}
    assert len(colors) == 2, "expected exactly one sprite color and one marker color"

    counts = {color: int(np.count_nonzero(g == color)) for color in colors}
    sprite_candidates = [color for color, count in counts.items() if count == 1]
    assert len(sprite_candidates) == 1, "expected the sprite to be the unique singleton-colored pixel"

    sprite_color = int(sprite_candidates[0])
    marker_color = next(int(color) for color in colors if int(color) != sprite_color)
    start = tuple(map(int, np.argwhere(g == sprite_color)[0]))
    return g, sprite_color, marker_color, start


def _remaining_valid_markers(grid: np.ndarray, marker_color: int, start: Coord) -> list[Coord]:
    coords = [tuple(map(int, coord)) for coord in np.argwhere(grid == marker_color)]
    return [coord for coord in coords if _same_lattice(coord, start)]


def _aligned_targets(grid: np.ndarray, current: Coord, marker_color: int, start: Coord) -> list[tuple[Coord, int, bool]]:
    targets = []
    for coord in _remaining_valid_markers(grid, marker_color, start):
        if coord[0] != current[0] and coord[1] != current[1]:
            continue
        dist = abs(coord[0] - current[0]) + abs(coord[1] - current[1])
        targets.append((coord, dist, _segment_clear(grid, current, coord)))
    return targets


def _analyze_instance(grid: np.ndarray) -> dict:
    g, sprite_color, marker_color, start = _parse_grid(grid)
    invalid_markers = [
        tuple(map(int, coord))
        for coord in np.argwhere(g == marker_color)
        if not _same_lattice(tuple(map(int, coord)), start)
    ]

    state = g.copy()
    current = start
    path: list[Coord] = []
    steps: list[dict] = []

    while True:
        remaining_valid = _remaining_valid_markers(state, marker_color, start)
        if not remaining_valid:
            break

        aligned = _aligned_targets(state, current, marker_color, start)
        visible = [(coord, dist) for coord, dist, clear in aligned if clear]
        assert visible, "expected at least one visible valid marker at every step"

        visible.sort(key=lambda item: (item[1], item[0][0], item[0][1]))
        best_dist = visible[0][1]
        assert sum(1 for _coord, dist in visible if dist == best_dist) == 1, "nearest visible target must be unique"
        target = visible[0][0]

        steps.append(
            {
                "current": current,
                "target": target,
                "distance": best_dist,
                "axis": "row" if current[0] == target[0] else "col",
                "blocked_closer": any((not clear) and dist < best_dist for _coord, dist, clear in aligned),
            }
        )
        state[current] = BACKGROUND
        state[target] = BACKGROUND
        current = target
        path.append(target)

    state[current] = sprite_color
    valid_markers = [
        tuple(map(int, coord))
        for coord in np.argwhere(g == marker_color)
        if _same_lattice(tuple(map(int, coord)), start)
    ]
    grey_coords = [tuple(map(int, coord)) for coord in np.argwhere(g == OBSTACLE)]
    return {
        "input": g,
        "output": state,
        "sprite_color": sprite_color,
        "marker_color": marker_color,
        "start": start,
        "path": path,
        "steps": steps,
        "valid_markers": valid_markers,
        "invalid_markers": invalid_markers,
        "grey_coords": grey_coords,
        "has_blocked_closer": any(step["blocked_closer"] for step in steps),
        "has_aligned_invalid": any(
            (coord[0] == start[0] or coord[1] == start[1]) and _segment_clear(g, start, coord)
            for coord in invalid_markers
        ),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Move the sprite along its parity lattice and consume the visible nearest markers one by one.

    Input structure:
    - A black background contains one singleton sprite pixel of its own color, many marker pixels of a second color, and some fixed grey obstacle pixels.
    - Some markers lie on the same row/column parity lattice as the sprite start, while others sit on the other three parity lattices.

    Rule:
    - Only markers whose row parity and column parity both match the sprite start are valid targets.
    - Repeatedly look from the sprite's current cell along its row and column, keep only valid markers whose in-between cells are all black, and jump to the unique nearest such marker.
    - Each jumped-to marker is consumed and disappears; grey obstacles and parity-mismatched markers never move, and the sprite ends on the final consumed marker.

    Key insight:
    - Target eligibility is fixed by the start parity lattice, and grey blockers can hide a geometrically closer valid marker, so the eating order must be simulated step by step.
    """
    return _analyze_instance(grid)["output"]


def _segment_hits_occupied(start: Coord, end: Coord, occupied: set[Coord]) -> bool:
    return any(cell in occupied for cell in _exclusive_segment(start, end))


def _build_relative_chain() -> tuple[list[Coord], list[Coord]] | None:
    mode = random.choice(("hfirst", "vfirst"))
    hsign = random.choice((-1, 1))
    vsign = random.choice((-1, 1))

    if mode == "hfirst":
        dx = random.choice(LONG_EVEN_DISTANCES)
        dy_choices = [dist for dist in EVEN_DISTANCES if dist < dx]
        if not dy_choices:
            return None
        dy = random.choice(dy_choices)
        points = [(0, 0), (0, hsign * dx), (vsign * dy, hsign * dx), (vsign * dy, 0)]
        obstacles = [(vsign * (dy // 2), 0)]
        next_axis = "v"
    else:
        dy = random.choice(LONG_EVEN_DISTANCES)
        dx_choices = [dist for dist in EVEN_DISTANCES if dist < dy]
        if not dx_choices:
            return None
        dx = random.choice(dx_choices)
        points = [(0, 0), (vsign * dy, 0), (vsign * dy, hsign * dx), (0, hsign * dx)]
        obstacles = [(0, hsign * (dx // 2))]
        next_axis = "h"

    used = set(points) | set(obstacles)
    current = points[-1]
    extra_steps = random.randint(0, 3)

    for _ in range(extra_steps):
        candidates: list[Coord] = []
        for sign in (-1, 1):
            for dist in EVEN_DISTANCES:
                if next_axis == "h":
                    nxt = (current[0], current[1] + sign * dist)
                else:
                    nxt = (current[0] + sign * dist, current[1])
                if nxt in used:
                    continue
                if _segment_hits_occupied(current, nxt, used):
                    continue
                prior_alignments = sum(1 for point in points[:-1] if point[0] == nxt[0] or point[1] == nxt[1])
                if prior_alignments > 1:
                    continue
                candidates.append(nxt)
        if not candidates:
            break
        nxt = random.choice(candidates)
        points.append(nxt)
        used.add(nxt)
        current = nxt
        next_axis = "v" if next_axis == "h" else "h"

    return points, obstacles


def _translate_points(points: list[Coord], shift_r: int, shift_c: int) -> list[Coord]:
    return [(r + shift_r, c + shift_c) for r, c in points]


def _choose_canvas(points: list[Coord], obstacles: list[Coord]) -> tuple[int, int, int, int] | None:
    all_coords = np.array(points + obstacles, dtype=int)
    r0, c0, r1, c1 = bbox_coords(all_coords)
    shape_h = int(r1 - r0 + 1)
    shape_w = int(c1 - c0 + 1)

    min_h = max(shape_h + 2, 10)
    min_w = max(shape_w + 2, 10)
    max_h = min(30, shape_h + 10)
    max_w = min(30, shape_w + 10)
    if min_h > max_h or min_w > max_w:
        return None

    height = random.randint(min_h, max_h)
    width = random.randint(min_w, max_w)
    shift_r = random.randint(1 - r0, height - 2 - r1)
    shift_c = random.randint(1 - c0, width - 2 - c1)
    return height, width, shift_r, shift_c


def _choose_decoy(start: Coord, occupied: set[Coord], shape: tuple[int, int]) -> Coord | None:
    height, width = shape
    candidates: list[Coord] = []
    for dist in (1, 3, 5, 7):
        for sign in (-1, 1):
            row_decoy = (start[0], start[1] + sign * dist)
            col_decoy = (start[0] + sign * dist, start[1])
            for coord in (row_decoy, col_decoy):
                if not in_bounds(shape, coord[0], coord[1]):
                    continue
                if coord in occupied:
                    continue
                if any(cell in occupied for cell in _exclusive_segment(start, coord)):
                    continue
                candidates.append(coord)
    return random.choice(candidates) if candidates else None


def _path_cells(points: list[Coord]) -> set[Coord]:
    cells: set[Coord] = set(points)
    for a, b in zip(points, points[1:]):
        cells.update(_exclusive_segment(a, b))
    return cells


def _scatter_extras(
    grid: np.ndarray,
    forbidden: set[Coord],
    marker_color: int,
    start: Coord,
) -> None:
    height, width = grid.shape
    candidates = [(r, c) for r in range(height) for c in range(width) if (r, c) not in forbidden]
    random.shuffle(candidates)

    extra_obstacles = random.randint(0, min(6, len(candidates)))
    for _ in range(extra_obstacles):
        if not candidates:
            break
        coord = candidates.pop()
        grid[coord] = OBSTACLE
        forbidden.add(coord)

    random.shuffle(candidates)
    extra_invalid = random.randint(1, min(7, len(candidates)))
    placed = 0
    for coord in candidates:
        if _same_lattice(coord, start):
            continue
        if grid[coord] != BACKGROUND:
            continue
        grid[coord] = marker_color
        forbidden.add(coord)
        placed += 1
        if placed == extra_invalid:
            break


def _build_input() -> np.ndarray | None:
    chain_data = _build_relative_chain()
    if chain_data is None:
        return None
    relative_points, relative_obstacles = chain_data
    canvas = _choose_canvas(relative_points, relative_obstacles)
    if canvas is None:
        return None

    height, width, shift_r, shift_c = canvas
    points = _translate_points(relative_points, shift_r, shift_c)
    obstacles = _translate_points(relative_obstacles, shift_r, shift_c)
    start = points[0]
    valid_markers = points[1:]

    sprite_color, marker_color = random.sample(PALETTE, 2)
    grid = np.full((height, width), BACKGROUND, dtype=int)
    occupied = set(points) | set(obstacles) | _path_cells(points)

    grid[start] = sprite_color
    for coord in valid_markers:
        grid[coord] = marker_color
    for coord in obstacles:
        grid[coord] = OBSTACLE

    decoy = _choose_decoy(start, set(points) | set(obstacles), (height, width))
    if decoy is None:
        return None
    grid[decoy] = marker_color
    occupied.add(decoy)
    occupied.update(_exclusive_segment(start, decoy))

    _scatter_extras(grid, occupied, marker_color, start)
    return grid


def _rich_enough(info: dict) -> bool:
    if len(info["valid_markers"]) < 3:
        return False
    if len(info["invalid_markers"]) < 1 or len(info["grey_coords"]) < 1:
        return False
    if len(info["path"]) != len(info["valid_markers"]):
        return False
    if {step["axis"] for step in info["steps"]} != {"row", "col"}:
        return False
    if not info["has_blocked_closer"]:
        return False
    if not info["has_aligned_invalid"]:
        return False

    out = info["output"]
    sprite_color = info["sprite_color"]
    if int(np.count_nonzero(out == sprite_color)) != 1:
        return False

    for coord in info["invalid_markers"]:
        if int(out[coord]) != info["marker_color"]:
            return False
    for coord in info["grey_coords"]:
        if int(out[coord]) != OBSTACLE:
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _build_input()
        if grid is None:
            continue
        try:
            info = _analyze_instance(grid)
        except Exception:
            continue
        if _rich_enough(info):
            return grid
    raise RuntimeError("failed to generate a rich valid input")
