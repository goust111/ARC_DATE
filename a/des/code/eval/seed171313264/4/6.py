from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
SOURCE = 8
TARGET = 9
MIRROR_POOL = (1, 2, 3, 4, 6, 7)

DIRS: dict[str, tuple[int, int]] = {
    "U": (-1, 0),
    "R": (0, 1),
    "D": (1, 0),
    "L": (0, -1),
}
PERP: dict[str, tuple[str, str]] = {
    "U": ("L", "R"),
    "R": ("U", "D"),
    "D": ("L", "R"),
    "L": ("U", "D"),
}
OPPOSITE = {"U": "D", "R": "L", "D": "U", "L": "R"}

GEN_RETRIES = 500
GEN_MIRROR_COUNTS = (3, 4, 5)
GEN_SEGMENT_LEN_RANGE = (3, 8)
GEN_WALL_LEN_RANGE = (3, 7)
GEN_SIDE_MARGIN_RANGE = (1, 4)
GEN_TOP_MAIN_MARGIN_RANGE = (1, 4)

_START_DIR_BAG: list[str] = []
_MIRROR_COUNT_BAG: list[int] = []
_TURN_BAG: list[int] = []
_TARGET_OFFSET_BAG: list[int] = []


def _step(pos: tuple[int, int], direction: str, count: int = 1) -> tuple[int, int]:
    dr, dc = DIRS[direction]
    return int(pos[0] + dr * count), int(pos[1] + dc * count)


def _iter_block(top_left: tuple[int, int], height: int = 2, width: int = 2) -> list[tuple[int, int]]:
    r0, c0 = top_left
    return [(int(r0 + dr), int(c0 + dc)) for dr in range(height) for dc in range(width)]


def _find_block_top_left(grid: np.ndarray, color: int) -> tuple[int, int]:
    pts = np.argwhere(grid == int(color))
    assert int(pts.shape[0]) == 4, f"expected one 2x2 block of color {color}"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    assert r1 == r0 + 1 and c1 == c0 + 1, f"color {color} must form a 2x2 block"
    assert np.all(grid[r0 : r0 + 2, c0 : c0 + 2] == int(color))
    return r0, c0


def _legend_colors(grid: np.ndarray) -> list[int]:
    colors: list[int] = []
    seen: set[int] = set()
    for c in range(grid.shape[1]):
        value = int(grid[0, c])
        if value == BG or value in seen:
            continue
        seen.add(value)
        colors.append(value)
    return colors


def _extract_mirrors(grid: np.ndarray) -> tuple[dict[int, dict[str, object]], dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    legend = _legend_colors(grid)
    if legend:
        mirror_colors = legend
    else:
        mirror_colors = [
            int(color)
            for color in np.unique(grid[1:, :])
            if int(color) not in (BG, GREY, SOURCE, TARGET)
        ]

    mirror_info: dict[int, dict[str, object]] = {}
    cell_to_color: dict[tuple[int, int], int] = {}
    endpoint_to_color: dict[tuple[int, int], int] = {}

    for color in mirror_colors:
        pts = np.argwhere(grid[1:, :] == int(color))
        assert int(pts.shape[0]) >= 2, f"mirror color {color} must appear below legend"
        pts[:, 0] += 1
        rows = [int(r) for r in pts[:, 0]]
        cols = [int(c) for c in pts[:, 1]]
        cells = [(int(r), int(c)) for r, c in pts]
        if len(set(rows)) == 1:
            orientation = "h"
            left = (int(rows[0]), int(min(cols)))
            right = (int(rows[0]), int(max(cols)))
            endpoints = (left, right)
        else:
            assert len(set(cols)) == 1, f"mirror color {color} must be straight"
            orientation = "v"
            top = (int(min(rows)), int(cols[0]))
            bottom = (int(max(rows)), int(cols[0]))
            endpoints = (top, bottom)
        mirror_info[int(color)] = {
            "color": int(color),
            "orientation": orientation,
            "cells": set(cells),
            "endpoints": endpoints,
        }
        for cell in cells:
            cell_to_color[cell] = int(color)
        for cell in endpoints:
            endpoint_to_color[cell] = int(color)

    return mirror_info, cell_to_color, endpoint_to_color


def _locate_source_start(
    grid: np.ndarray,
    source_cells: set[tuple[int, int]],
    mirror_info: dict[int, dict[str, object]],
    endpoint_to_color: dict[tuple[int, int], int],
) -> tuple[tuple[int, int], str]:
    h, w = map(int, grid.shape)
    candidates: list[tuple[tuple[int, int], str]] = []

    for cell in sorted(source_cells):
        for direction in DIRS:
            gap = _step(cell, direction)
            hit = _step(gap, direction)
            if not (0 <= gap[0] < h and 0 <= gap[1] < w):
                continue
            if not (0 <= hit[0] < h and 0 <= hit[1] < w):
                continue
            if int(grid[gap]) != BG:
                continue
            if hit not in endpoint_to_color:
                continue
            color = endpoint_to_color[hit]
            orient = str(mirror_info[color]["orientation"])
            if direction in ("U", "D") and orient != "h":
                continue
            if direction in ("L", "R") and orient != "v":
                continue
            candidates.append((cell, direction))

    assert len(candidates) == 1, f"expected one source emission candidate, got {len(candidates)}"
    return candidates[0]


def _reflect(direction: str, hit_cell: tuple[int, int], info: dict[str, object]) -> str:
    orientation = str(info["orientation"])
    end0, end1 = info["endpoints"]  # type: ignore[misc]
    end0 = tuple(int(x) for x in end0)
    end1 = tuple(int(x) for x in end1)

    # Rasterized samples are generated so collisions happen at endpoints, which
    # makes the 90-degree turn direction unique. Unexpected side hits fall back
    # to a component reversal to stay consistent with the textual rule.
    if orientation == "h":
        if direction in ("U", "D"):
            if hit_cell == end0:
                return "L"
            if hit_cell == end1:
                return "R"
            return OPPOSITE[direction]
        return OPPOSITE[direction]

    if direction in ("L", "R"):
        if hit_cell == end0:
            return "U"
        if hit_cell == end1:
            return "D"
        return OPPOSITE[direction]
    return OPPOSITE[direction]


def _trace_solution(grid: np.ndarray) -> tuple[np.ndarray, list[int], bool]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    out = g.copy()
    out[0, :] = BG

    mirror_info, cell_to_color, endpoint_to_color = _extract_mirrors(g)
    source_tl = _find_block_top_left(g, SOURCE)
    target_tl = _find_block_top_left(g, TARGET)
    source_cells = set(_iter_block(source_tl))
    target_cells = set(_iter_block(target_tl))
    current, direction = _locate_source_start(g, source_cells, mirror_info, endpoint_to_color)
    collisions: list[int] = []
    color = GREY
    reached_target = False
    seen: set[tuple[int, int, str]] = set()

    for _ in range(h * w * 8):
        state = (int(current[0]), int(current[1]), direction)
        assert state not in seen, "beam path must not cycle"
        seen.add(state)

        nxt = _step(current, direction)
        nr, nc = nxt
        if not (0 <= nr < h and 0 <= nc < w):
            break
        if nxt in target_cells:
            for r, c in target_cells:
                out[r, c] = int(color)
            reached_target = True
            break
        if nxt in cell_to_color:
            mirror_color = cell_to_color[nxt]
            collisions.append(int(mirror_color))
            direction = _reflect(direction, nxt, mirror_info[mirror_color])
            color = int(mirror_color)
            continue
        if int(g[nr, nc]) == BG:
            out[nr, nc] = int(color)
            current = nxt
            continue
        break

    return out, collisions, reached_target


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a reflected beam from the azure source to the maroon target.

    The top row is only a color legend; the actual path is determined by the
    geometry of the mirror segments below it. The beam starts in the unique
    1-cell source-to-mirror gap, draws grey until the first collision, then
    adopts the color of each mirror it hits, and finally recolors the target.
    """
    out, _collisions, _reached = _trace_solution(np.asarray(grid, dtype=int))
    return out


def _draw_balanced_start_dir() -> str:
    global _START_DIR_BAG
    if not _START_DIR_BAG:
        _START_DIR_BAG = random.sample(["U", "R", "D", "L"], 4)
    return _START_DIR_BAG.pop()


def _draw_balanced_mirror_count() -> int:
    global _MIRROR_COUNT_BAG
    if not _MIRROR_COUNT_BAG:
        _MIRROR_COUNT_BAG = random.sample(list(GEN_MIRROR_COUNTS), len(GEN_MIRROR_COUNTS))
    return _MIRROR_COUNT_BAG.pop()


def _draw_balanced_turn() -> int:
    global _TURN_BAG
    if not _TURN_BAG:
        _TURN_BAG = random.sample([-1, 1, -1, 1], 4)
    return int(_TURN_BAG.pop())


def _draw_balanced_target_offset() -> int:
    global _TARGET_OFFSET_BAG
    if not _TARGET_OFFSET_BAG:
        _TARGET_OFFSET_BAG = random.sample([0, 1], 2)
    return int(_TARGET_OFFSET_BAG.pop())


def _source_top_left(emission: tuple[int, int], initial_dir: str, first_out: str) -> tuple[int, int]:
    r, c = emission
    if initial_dir == "U":
        return (int(r), int(c)) if first_out == "L" else (int(r), int(c - 1))
    if initial_dir == "D":
        return (int(r - 1), int(c)) if first_out == "L" else (int(r - 1), int(c - 1))
    if initial_dir == "R":
        return (int(r), int(c - 1)) if first_out == "U" else (int(r - 1), int(c - 1))
    return (int(r), int(c)) if first_out == "U" else (int(r - 1), int(c))


def _target_top_left(contact: tuple[int, int], final_dir: str, offset: int) -> tuple[int, int]:
    r, c = contact
    if final_dir == "U":
        return int(r - 1), int(c - offset)
    if final_dir == "D":
        return int(r), int(c - offset)
    if final_dir == "R":
        return int(r - offset), int(c)
    return int(r - offset), int(c - 1)


def _choice_from_turn(prev_dir: str, turn_sign: int) -> str:
    left_like, right_like = PERP[prev_dir]
    return left_like if int(turn_sign) < 0 else right_like


def _candidate_layout() -> tuple[np.ndarray, list[int]] | None:
    mirror_count = _draw_balanced_mirror_count()
    initial_dir = _draw_balanced_start_dir()

    directions = [initial_dir]
    for _ in range(mirror_count):
        directions.append(_choice_from_turn(directions[-1], _draw_balanced_turn()))

    lengths = [
        random.randint(GEN_SEGMENT_LEN_RANGE[0], GEN_SEGMENT_LEN_RANGE[1])
        for _ in range(mirror_count)
    ]

    corners = [(0, 0)]
    path_cells = {(0, 0)}
    pos = (0, 0)
    for seg_idx in range(1, mirror_count + 1):
        direction = directions[seg_idx]
        for _ in range(lengths[seg_idx - 1]):
            pos = _step(pos, direction)
            if pos in path_cells:
                return None
            path_cells.add(pos)
        if seg_idx < mirror_count:
            corners.append(pos)
    last_path = pos

    source_emission = _step(corners[0], OPPOSITE[initial_dir])
    source_tl = _source_top_left(source_emission, initial_dir, directions[1])
    source_cells = set(_iter_block(source_tl))
    if source_cells & path_cells:
        return None

    target_contact = _step(last_path, directions[-1])
    target_tl = _target_top_left(target_contact, directions[-1], _draw_balanced_target_offset())
    target_cells = set(_iter_block(target_tl))
    if target_cells & path_cells:
        return None
    if target_cells & source_cells:
        return None

    mirror_colors = random.sample(list(MIRROR_POOL), mirror_count)
    occupied = set(path_cells) | source_cells | target_cells
    mirrors: list[tuple[int, list[tuple[int, int]]]] = []

    for idx in range(mirror_count):
        corner = corners[idx]
        incoming = directions[idx]
        outgoing = directions[idx + 1]
        endpoint = _step(corner, incoming)
        wall_dir = OPPOSITE[outgoing]
        wall_len = random.randint(GEN_WALL_LEN_RANGE[0], GEN_WALL_LEN_RANGE[1])
        cells = [_step(endpoint, wall_dir, t) for t in range(wall_len)]
        if any(cell in occupied for cell in cells):
            return None
        occupied.update(cells)
        mirrors.append((mirror_colors[idx], cells))

    all_cells = set(occupied)
    min_r = min(r for r, _ in all_cells)
    max_r = max(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)
    max_c = max(c for _, c in all_cells)

    shift_r = 1 + random.randint(GEN_TOP_MAIN_MARGIN_RANGE[0], GEN_TOP_MAIN_MARGIN_RANGE[1]) - min_r
    shift_c = random.randint(GEN_SIDE_MARGIN_RANGE[0], GEN_SIDE_MARGIN_RANGE[1]) - min_c
    bottom_margin = random.randint(GEN_SIDE_MARGIN_RANGE[0], GEN_SIDE_MARGIN_RANGE[1])
    right_margin = random.randint(GEN_SIDE_MARGIN_RANGE[0], GEN_SIDE_MARGIN_RANGE[1])

    source_tl = (int(source_tl[0] + shift_r), int(source_tl[1] + shift_c))
    target_tl = (int(target_tl[0] + shift_r), int(target_tl[1] + shift_c))
    shifted_mirrors = [
        (int(color), [(int(r + shift_r), int(c + shift_c)) for r, c in cells])
        for color, cells in mirrors
    ]

    height = int(max_r + shift_r + 1 + bottom_margin)
    width = int(max_c + shift_c + 1 + right_margin)
    width = max(width, 2 * mirror_count + 3)

    grid = np.full((height, width), BG, dtype=int)
    for color, cells in shifted_mirrors:
        for r, c in cells:
            grid[r, c] = int(color)
    sr, sc = source_tl
    tr, tc = target_tl
    grid[sr : sr + 2, sc : sc + 2] = SOURCE
    grid[tr : tr + 2, tc : tc + 2] = TARGET

    legend_slots = random.sample(list(range(1, width - 1)), mirror_count)
    legend_slots.sort()
    for col, color in zip(legend_slots, mirror_colors):
        grid[0, int(col)] = int(color)

    return grid, mirror_colors


def _valid_input(grid: np.ndarray, collision_order: list[int]) -> bool:
    try:
        out, collisions, reached = _trace_solution(grid)
    except AssertionError:
        return False

    if not reached:
        return False
    if collisions != [int(c) for c in collision_order]:
        return False
    if not np.all(out[0, :] == BG):
        return False

    if _legend_colors(grid) != collision_order:
        return False

    mirror_info, _cell_to_color, endpoint_to_color = _extract_mirrors(grid)
    if len(mirror_info) != len(collision_order):
        return False
    if len(endpoint_to_color) != 2 * len(collision_order):
        return False

    try:
        _find_block_top_left(grid, SOURCE)
        _find_block_top_left(grid, TARGET)
    except AssertionError:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        candidate = _candidate_layout()
        if candidate is None:
            continue
        grid, collision_order = candidate
        if _valid_input(grid, collision_order):
            return grid
    raise RuntimeError("failed to generate a valid mirror-corridor input")
