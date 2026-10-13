from __future__ import annotations

import itertools
import random

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREEN = 3

GRID_MIN = 12
GRID_MAX = 16
MARKER_COUNT_CHOICES = (2, 2, 3, 3, 4)
MAX_GENERATION_ATTEMPTS = 500
MAX_ASSIGNMENT_ATTEMPTS = 80

Coord = tuple[int, int]


def _nearest_row_edge(r: int, h: int) -> int:
    return 0 if r < (h - 1 - r) else h - 1


def _nearest_col_edge(c: int, w: int) -> int:
    return 0 if c < (w - 1 - c) else w - 1


def _row_interval(r: int, h: int) -> tuple[int, int]:
    edge = _nearest_row_edge(r, h)
    return (min(r, edge), max(r, edge))


def _col_interval(c: int, w: int) -> tuple[int, int]:
    edge = _nearest_col_edge(c, w)
    return (min(c, edge), max(c, edge))


def _between(x: int, interval: tuple[int, int]) -> bool:
    return interval[0] <= x <= interval[1]


def _ray_mask(shape: tuple[int, int], r: int, c: int) -> np.ndarray:
    h, w = shape
    row_edge = _nearest_row_edge(r, h)
    col_edge = _nearest_col_edge(c, w)
    mask = np.zeros(shape, dtype=bool)
    mask[min(r, row_edge) : max(r, row_edge) + 1, c] = True
    mask[r, min(c, col_edge) : max(c, col_edge) + 1] = True
    return mask


def _marker_positions(grid: np.ndarray, color: int) -> list[Coord]:
    coords = np.argwhere(grid == color)
    return [(int(r), int(c)) for r, c in coords]


def _oracle_output(red_markers: list[Coord], blue_markers: list[Coord], shape: tuple[int, int]) -> np.ndarray:
    red_occ = np.zeros(shape, dtype=bool)
    blue_occ = np.zeros(shape, dtype=bool)

    for r, c in red_markers:
        red_occ |= _ray_mask(shape, r, c)
    for r, c in blue_markers:
        blue_occ |= _ray_mask(shape, r, c)

    out = np.full(shape, BACKGROUND, dtype=int)
    out[red_occ & ~blue_occ] = RED
    out[blue_occ & ~red_occ] = BLUE
    out[red_occ & blue_occ] = GREEN

    for r, c in red_markers:
        out[r, c] = RED
    for r, c in blue_markers:
        out[r, c] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend red and blue markers to their nearest edges, then color the per-cell overlap pattern.

    Input structure
    - The grid is a black background with isolated blue(1) and red(2) marker pixels.
    - Each marker defines one vertical and one horizontal ray by looking to the nearest boundary on each axis.
    - Only the marker colors participate; all other cells are background.

    Rule
    - For every red marker, fill its column from the marker to the nearer top/bottom edge and fill its row from the marker to the nearer left/right edge.
    - Do the same independently for every blue marker.
    - After all red and blue rays are generated, color red-only cells red(2), blue-only cells blue(1), and cells occupied by both colors green(3).
    - Finally restore the original marker pixels to their input colors even if both ray systems pass through those cells.

    Key insight
    - The output is not drawn from the markers directly: it comes from comparing two dynamically generated ray canvases and marking the cells where both canvases occupy the same position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))).issubset({BACKGROUND, BLUE, RED}), "input may only contain black, blue, and red"

    red_markers = _marker_positions(g, RED)
    blue_markers = _marker_positions(g, BLUE)
    return _oracle_output(red_markers, blue_markers, g.shape)


def _row_dir(r: int, h: int) -> str:
    return "U" if _nearest_row_edge(r, h) == 0 else "D"


def _col_dir(c: int, w: int) -> str:
    return "L" if _nearest_col_edge(c, w) == 0 else "R"


def _top_rows(h: int) -> list[int]:
    return [r for r in range(1, h - 1) if _nearest_row_edge(r, h) == 0]


def _bottom_rows(h: int) -> list[int]:
    return [r for r in range(1, h - 1) if _nearest_row_edge(r, h) == h - 1]


def _left_cols(w: int) -> list[int]:
    return [c for c in range(1, w - 1) if _nearest_col_edge(c, w) == 0]


def _right_cols(w: int) -> list[int]:
    return [c for c in range(1, w - 1) if _nearest_col_edge(c, w) == w - 1]


def _is_isolated(cell: Coord, placed: list[Coord]) -> bool:
    r, c = cell
    return all(max(abs(r - pr), abs(c - pc)) > 1 for pr, pc in placed)


def _quadrant_candidates(
    h: int,
    w: int,
    quadrant: str,
    used_rows: set[int],
    used_cols: set[int],
    placed: list[Coord],
) -> list[Coord]:
    row_pool = _top_rows(h) if quadrant[0] == "T" else _bottom_rows(h)
    col_pool = _left_cols(w) if quadrant[1] == "L" else _right_cols(w)
    candidates: list[Coord] = []
    for r in row_pool:
        if r in used_rows:
            continue
        for c in col_pool:
            if c in used_cols:
                continue
            cell = (r, c)
            if _is_isolated(cell, placed):
                candidates.append(cell)
    return candidates


def _place_positions(h: int, w: int, total_markers: int) -> list[Coord] | None:
    required = ["TL", "TR", "BL", "BR"]
    extras = [random.choice(required) for _ in range(total_markers - 4)]
    desired_quadrants = required + extras
    random.shuffle(desired_quadrants)

    placed: list[Coord] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    for quadrant in desired_quadrants:
        candidates = _quadrant_candidates(h, w, quadrant, used_rows, used_cols, placed)
        if not candidates:
            return None
        cell = random.choice(candidates)
        placed.append(cell)
        used_rows.add(cell[0])
        used_cols.add(cell[1])
    return placed


def _intersection_stats(
    red_markers: list[Coord],
    blue_markers: list[Coord],
    shape: tuple[int, int],
) -> dict[str, object]:
    h, w = shape
    marker_set = set(red_markers) | set(blue_markers)
    type_a: set[Coord] = set()
    type_b: set[Coord] = set()
    red_cross_ids: set[int] = set()
    blue_cross_ids: set[int] = set()

    for red_id, (rr, rc) in enumerate(red_markers):
        red_v = _row_interval(rr, h)
        red_h = _col_interval(rc, w)
        for blue_id, (br, bc) in enumerate(blue_markers):
            blue_v = _row_interval(br, h)
            blue_h = _col_interval(bc, w)

            cross_a = (br, rc)
            if _between(br, red_v) and _between(rc, blue_h) and cross_a not in marker_set:
                type_a.add(cross_a)
                red_cross_ids.add(red_id)
                blue_cross_ids.add(blue_id)

            cross_b = (rr, bc)
            if _between(bc, red_h) and _between(rr, blue_v) and cross_b not in marker_set:
                type_b.add(cross_b)
                red_cross_ids.add(red_id)
                blue_cross_ids.add(blue_id)

    out = _oracle_output(red_markers, blue_markers, shape)
    green_cells = {(int(r), int(c)) for r, c in np.argwhere(out == GREEN)}
    return {
        "out": out,
        "green_cells": green_cells,
        "type_a": type_a,
        "type_b": type_b,
        "red_cross_ids": red_cross_ids,
        "blue_cross_ids": blue_cross_ids,
        "non_bg": int(np.count_nonzero(out)),
        "red_only": int(np.count_nonzero(out == RED)),
        "blue_only": int(np.count_nonzero(out == BLUE)),
        "green_count": len(green_cells),
    }


def _valid_marker_layout(red_markers: list[Coord], blue_markers: list[Coord], shape: tuple[int, int]) -> bool:
    h, w = shape
    all_markers = red_markers + blue_markers
    if not (2 <= len(red_markers) <= 4 and 2 <= len(blue_markers) <= 4):
        return False
    if len(all_markers) != len(set(all_markers)):
        return False
    if len({r for r, _ in all_markers}) != len(all_markers):
        return False
    if len({c for _, c in all_markers}) != len(all_markers):
        return False
    if any(r <= 0 or r >= h - 1 or c <= 0 or c >= w - 1 for r, c in all_markers):
        return False
    if any(max(abs(r1 - r2), abs(c1 - c2)) <= 1 for (r1, c1), (r2, c2) in itertools.combinations(all_markers, 2)):
        return False
    if any(2 * r == h - 1 or 2 * c == w - 1 for r, c in all_markers):
        return False

    row_dirs = {_row_dir(r, h) for r, _ in all_markers}
    col_dirs = {_col_dir(c, w) for _, c in all_markers}
    if row_dirs != {"U", "D"} or col_dirs != {"L", "R"}:
        return False

    stats = _intersection_stats(red_markers, blue_markers, shape)
    if stats["green_cells"] != (stats["type_a"] | stats["type_b"]):
        return False
    if len(stats["type_a"]) == 0 or len(stats["type_b"]) == 0:
        return False
    if stats["green_count"] < 2 or stats["green_count"] > min(12, (h * w) // 8):
        return False
    if len(stats["red_cross_ids"]) < 2 or len(stats["blue_cross_ids"]) < 2:
        return False
    if stats["red_only"] <= len(red_markers) or stats["blue_only"] <= len(blue_markers):
        return False
    if stats["non_bg"] >= int(h * w * 0.42):
        return False

    inp = np.full(shape, BACKGROUND, dtype=int)
    for r, c in red_markers:
        inp[r, c] = RED
    for r, c in blue_markers:
        inp[r, c] = BLUE
    return np.array_equal(main(inp), stats["out"])


def _attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    n_red = random.choice(MARKER_COUNT_CHOICES)
    n_blue = random.choice(MARKER_COUNT_CHOICES)
    total_markers = n_red + n_blue

    if total_markers > min(h - 2, w - 2):
        return None

    positions = _place_positions(h, w, total_markers)
    if positions is None:
        return None

    assignments = list(itertools.combinations(range(total_markers), n_red))
    random.shuffle(assignments)
    for red_indices in assignments[:MAX_ASSIGNMENT_ATTEMPTS]:
        red_set = set(red_indices)
        red_markers = [positions[i] for i in range(total_markers) if i in red_set]
        blue_markers = [positions[i] for i in range(total_markers) if i not in red_set]
        if not _valid_marker_layout(red_markers, blue_markers, (h, w)):
            continue

        grid = np.full((h, w), BACKGROUND, dtype=int)
        for r, c in red_markers:
            grid[r, c] = RED
        for r, c in blue_markers:
            grid[r, c] = BLUE
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-difference input")
