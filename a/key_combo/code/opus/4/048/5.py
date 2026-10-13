from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, in_bounds


BLACK = Color.BLACK
TEAL = Color.TEAL
OBSTACLE_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.GREY, Color.PINK, Color.ORANGE, Color.MAROON]

UP, RIGHT, DOWN, LEFT = range(4)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
BEAM_ORDER = [UP, RIGHT, DOWN, LEFT]


def _rot_cw(direction: int) -> int:
    return (int(direction) + 1) % 4


def _rot_ccw(direction: int) -> int:
    return (int(direction) - 1) % 4


def _turn(direction: int, area: int, mode: str) -> int:
    if mode == "parity":
        return _rot_cw(direction) if area % 2 == 1 else _rot_ccw(direction)
    if mode == "reverse":
        return _rot_ccw(direction) if area % 2 == 1 else _rot_cw(direction)
    if mode == "cw":
        return _rot_cw(direction)
    if mode == "ccw":
        return _rot_ccw(direction)
    raise ValueError(f"unknown turn mode: {mode}")


def _solid_teal_square(grid: np.ndarray) -> tuple[int, int, int, int]:
    teal_components = find_components(
        grid,
        bg=BLACK,
        connectivity=4,
        monochromatic=True,
        colors=[TEAL],
    )
    candidates = []
    for comp in teal_components:
        if comp["area"] != 4:
            continue
        r0, c0, r1, c1 = comp["bbox"]
        if r1 - r0 + 1 != 2 or c1 - c0 + 1 != 2:
            continue
        if np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == TEAL):
            candidates.append((r0, c0, r1, c1))
    assert len(candidates) == 1, "expected exactly one solid 2x2 teal source square"
    return candidates[0]


def _source_starts(source_bbox: tuple[int, int, int, int]) -> dict[int, tuple[int, int]]:
    r0, c0, r1, c1 = source_bbox
    # A 2x2 square has tied side-midpoints; use the four perimeter cells in clockwise order.
    return {
        UP: (r0, c0),
        RIGHT: (r0, c1),
        DOWN: (r1, c1),
        LEFT: (r1, c0),
    }


def _obstacle_data(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    colors = [int(v) for v in np.unique(grid) if int(v) not in {BLACK, TEAL}]
    components = find_components(
        grid,
        bg=BLACK,
        connectivity=4,
        monochromatic=True,
        colors=colors,
    )
    obstacles: list[dict] = []
    obstacle_at = np.full(grid.shape, -1, dtype=int)
    for idx, comp in enumerate(components):
        coords = comp["coords"]
        cells = [tuple(map(int, rc)) for rc in coords]
        for r, c in cells:
            obstacle_at[r, c] = idx
        obstacles.append(
            {
                "id": idx,
                "color": int(comp["color"]),
                "area": int(comp["area"]),
                "bbox": tuple(map(int, comp["bbox"])),
                "cells": cells,
            }
        )
    return obstacles, obstacle_at


def _simulate(
    grid: np.ndarray,
    *,
    mode: str = "parity",
    stop_after_first_hit: bool = False,
) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    source_bbox = _solid_teal_square(g)
    obstacles, obstacle_at = _obstacle_data(g)
    starts = _source_starts(source_bbox)
    out = g.copy()

    beam_paths: list[list[tuple[int, int]]] = [[] for _ in BEAM_ORDER]
    beam_hits: list[list[int]] = [[] for _ in BEAM_ORDER]

    for beam_index, direction in enumerate(BEAM_ORDER):
        r, c = starts[direction]
        d = direction
        while True:
            dr, dc = DIRS[d]
            nr, nc = r + dr, c + dc
            if not in_bounds(out.shape, nr, nc):
                break
            if out[nr, nc] == TEAL:
                break

            obstacle_id = int(obstacle_at[nr, nc])
            if obstacle_id != -1:
                beam_hits[beam_index].append(obstacle_id)
                if stop_after_first_hit:
                    break
                d = _turn(d, obstacles[obstacle_id]["area"], mode)
                tr, tc = r + DIRS[d][0], c + DIRS[d][1]
                if in_bounds(out.shape, tr, tc) and obstacle_at[tr, tc] != -1:
                    break
                continue

            if out[nr, nc] != BLACK:
                break
            out[nr, nc] = TEAL
            beam_paths[beam_index].append((nr, nc))
            r, c = nr, nc

    hit_ids = [hit for hits in beam_hits for hit in hits]
    hit_areas = [obstacles[obs_id]["area"] for obs_id in hit_ids]
    stats = {
        "beam_paths": beam_paths,
        "beam_hits": beam_hits,
        "hit_ids": hit_ids,
        "hit_areas": hit_areas,
        "path_cells": {cell for path in beam_paths for cell in path},
        "obstacles": obstacles,
        "source_bbox": source_bbox,
    }
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace four teal beams whose turn direction is encoded by each obstacle's own area parity.

    Input structure:
    - The grid has black background, one solid teal 2x2 source square, and several
      separated single-color obstacle blocks in other colors.
    - Each obstacle is a connected rectangle-like block whose total pixel count can be odd or even.
    - The source emits one width-1 beam per side; for the tied side-midpoints of a 2x2 square,
      use the four perimeter cells in up, right, down, left order.

    Rule:
    - Process the four beams in the order up, right, down, left.
    - A beam advances one cell at a time, painting only black cells teal.
    - If the next cell is part of an obstacle, do not paint that obstacle; instead turn in place:
      odd-area obstacle -> clockwise quarter turn, even-area obstacle -> counterclockwise quarter turn.
    - After turning, continue from the same cell. A beam stops when its next step would leave the grid,
      enter an already teal cell, or face another obstacle immediately after the turn.

    Key insight:
    - The obstacle itself supplies both pieces of information at once: it blocks the beam and its
      own area parity determines whether the beam turns clockwise or counterclockwise.
    """
    out, _stats = _simulate(np.asarray(grid, dtype=int), mode="parity")
    return out


def _rect_cells(top: int, left: int, height: int, width: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(int(top), int(top) + int(height)) for c in range(int(left), int(left) + int(width))]


def _can_place(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    return all(in_bounds(grid.shape, r, c) and grid[r, c] == BLACK for r, c in cells)


def _place_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _bar_candidates(intercept: tuple[int, int], approach_direction: int, length: int, shape: tuple[int, int] | None = None) -> list[list[tuple[int, int]]]:
    r, c = map(int, intercept)
    if shape is None:
        if approach_direction in {UP, DOWN}:
            shape = (1, int(length))
        else:
            shape = (int(length), 1)

    height, width = map(int, shape)
    candidates: list[list[tuple[int, int]]] = []
    if approach_direction in {UP, DOWN}:
        for left in range(c - width + 1, c + 1):
            candidates.append(_rect_cells(r, left, height, width))
    else:
        for top in range(r - height + 1, r + 1):
            candidates.append(_rect_cells(top, c, height, width))
    random.shuffle(candidates)
    return candidates


def _place_obstacle(
    grid: np.ndarray,
    intercept: tuple[int, int],
    approach_direction: int,
    parity: str,
    color: int,
) -> bool:
    if parity == "odd":
        lengths = [1, 3]
    else:
        lengths = [2, 4]
    random.shuffle(lengths)
    for length in lengths:
        for cells in _bar_candidates(intercept, approach_direction, length):
            if _can_place(grid, cells):
                _place_cells(grid, cells, color)
                return True
    return False


def _random_black_rect(
    grid: np.ndarray,
    *,
    height: int,
    width: int,
    forbidden: set[tuple[int, int]],
    color: int,
) -> bool:
    height = int(height)
    width = int(width)
    rows = list(range(grid.shape[0] - height + 1))
    cols = list(range(grid.shape[1] - width + 1))
    random.shuffle(rows)
    random.shuffle(cols)
    for r0 in rows:
        for c0 in cols:
            cells = _rect_cells(r0, c0, height, width)
            if any(cell in forbidden for cell in cells):
                continue
            if _can_place(grid, cells):
                _place_cells(grid, cells, color)
                return True
    return False


def _canonical_grid() -> np.ndarray | None:
    h = random.randint(15, 19)
    w = random.randint(18, 22)
    grid = np.full((h, w), BLACK, dtype=int)

    primary_len1 = random.randint(2, 4)
    primary_len2 = random.randint(2, 4)
    secondary_len1 = primary_len2 + random.randint(3, 5)

    source_row_lo = max(primary_len1 + 4, h // 2)
    source_row_hi = h - 4
    if source_row_lo > source_row_hi:
        return None
    sr = random.randint(source_row_lo, source_row_hi)
    sc = random.randint(2, 4)
    if sc + secondary_len1 + 3 >= w:
        return None

    center_r = (h - 1) / 2.0
    center_c = (w - 1) / 2.0
    if abs((sr + 0.5) - center_r) + abs((sc + 0.5) - center_c) < 2.0:
        return None

    grid[sr : sr + 2, sc : sc + 2] = TEAL

    obstacle_count = random.randint(3, 5)
    colors = random.sample(OBSTACLE_COLORS, k=obstacle_count)

    # Up beam: odd obstacle first, then even obstacle after turning right.
    intercept1 = (sr - primary_len1 - 1, sc)
    if not _place_obstacle(grid, intercept1, UP, "odd", colors[0]):
        return None

    turn_row = sr - primary_len1
    turn_col = sc + primary_len2 + 1
    intercept2 = (turn_row, turn_col)
    if not _place_obstacle(grid, intercept2, RIGHT, "even", colors[1]):
        return None

    # Right beam: first collision is even, so its first turn differs from the up beam.
    intercept3 = (sr, sc + secondary_len1 + 2)
    if not _place_obstacle(grid, intercept3, RIGHT, "even", colors[2]):
        return None

    core_out, core_stats = _simulate(grid)
    core_paths = core_stats["path_cells"]
    source_cells = set(_rect_cells(sr, sc, 2, 2))
    forbidden = set(core_paths) | source_cells
    for obs in core_stats["obstacles"]:
        forbidden.update(obs["cells"])

    extra_specs = [(1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 2)]
    for color in colors[3:]:
        random.shuffle(extra_specs)
        placed = False
        for rect_h, rect_w in extra_specs:
            if _random_black_rect(grid, height=rect_h, width=rect_w, forbidden=forbidden, color=color):
                placed = True
                break
        if not placed:
            return None

    grid = np.rot90(grid, k=random.randint(0, 3))
    return grid


def _valid_instance(grid: np.ndarray) -> bool:
    try:
        out, stats = _simulate(grid, mode="parity")
        reverse_out, _ = _simulate(grid, mode="reverse")
        cw_out, _ = _simulate(grid, mode="cw")
        ccw_out, _ = _simulate(grid, mode="ccw")
        first_hit_out, _ = _simulate(grid, mode="parity", stop_after_first_hit=True)
    except AssertionError:
        return False

    source_bbox = stats["source_bbox"]
    r0, c0, r1, c1 = source_bbox
    center_r = (grid.shape[0] - 1) / 2.0
    center_c = (grid.shape[1] - 1) / 2.0
    if abs(((r0 + r1) / 2.0) - center_r) + abs(((c0 + c1) / 2.0) - center_c) < 2.0:
        return False

    obstacles = stats["obstacles"]
    if not (3 <= len(obstacles) <= 5):
        return False

    hit_areas = stats["hit_areas"]
    if not hit_areas:
        return False
    if not any(area % 2 == 1 for area in hit_areas):
        return False
    if not any(area % 2 == 0 for area in hit_areas):
        return False
    if max(len(hits) for hits in stats["beam_hits"]) < 2:
        return False
    if sum(1 for hits in stats["beam_hits"] if hits) < 2:
        return False

    path_cells = list(stats["path_cells"])
    if len(path_cells) < 8:
        return False

    if np.array_equal(out, reverse_out):
        return False
    if np.array_equal(out, cw_out):
        return False
    if np.array_equal(out, ccw_out):
        return False
    if np.array_equal(out, first_hit_out):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _canonical_grid()
        if grid is not None and _valid_instance(grid):
            return grid
    raise RuntimeError("failed to generate a valid odd/even beam-turning ARC input")
