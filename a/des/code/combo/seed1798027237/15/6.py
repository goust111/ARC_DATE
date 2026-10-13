from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


BG = 0
WALL = 5
RED = 2
BLUE = 1
EQUAL_LABEL = 3
MARKER_POOL = (4, 6, 7, 8, 9)

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIRS:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _is_straight_segment(coords: list[tuple[int, int]]) -> bool:
    rows = {r for r, _c in coords}
    cols = {c for _r, c in coords}
    if len(rows) == 1:
        cols_sorted = sorted(cols)
        return cols_sorted == list(range(cols_sorted[0], cols_sorted[-1] + 1))
    if len(cols) == 1:
        rows_sorted = sorted(rows)
        return rows_sorted == list(range(rows_sorted[0], rows_sorted[-1] + 1))
    return False


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _is_lookup_candidate(grid: np.ndarray, bg: int, r0: int, c0: int) -> bool:
    h, w = grid.shape
    if r0 + 3 > h or c0 + 2 > w:
        return False
    block = grid[r0 : r0 + 3, c0 : c0 + 2]
    if np.any(block == bg):
        return False
    for rr in range(r0 - 1, r0 + 4):
        for cc in range(c0 - 1, c0 + 3):
            if not _in_bounds(grid.shape, rr, cc):
                continue
            if r0 <= rr < r0 + 3 and c0 <= cc < c0 + 2:
                continue
            if int(grid[rr, cc]) != bg:
                return False
    return True


def _find_lookup_table(grid: np.ndarray, bg: int) -> tuple[tuple[int, int], np.ndarray]:
    candidates: list[tuple[int, int]] = []
    h, w = grid.shape
    for r0 in range(h - 2):
        for c0 in range(w - 1):
            if _is_lookup_candidate(grid, bg, r0, c0):
                candidates.append((int(r0), int(c0)))
    assert len(candidates) == 1, "expected exactly one isolated 3x2 lookup table"
    r0, c0 = candidates[0]
    return (r0, c0), np.array(grid[r0 : r0 + 3, c0 : c0 + 2], dtype=int)


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    (table_r0, table_c0), table = _find_lookup_table(g, bg)

    table_mask = np.zeros_like(g, dtype=bool)
    table_mask[table_r0 : table_r0 + 3, table_c0 : table_c0 + 2] = True

    work = g.copy()
    work[table_mask] = bg

    wall_colors: set[int] = set()
    towers: list[tuple[int, int, int]] = []
    for color in map(int, np.unique(work)):
        if color == bg:
            continue
        comps = _components(work == color)
        for comp in comps:
            if len(comp) > 1:
                assert _is_straight_segment(comp), "wall components must be straight segments"
                wall_colors.add(int(color))
            else:
                (r, c) = comp[0]
                towers.append((int(r), int(c), int(color)))

    assert len(wall_colors) == 1, "expected a single wall color"
    wall_color = next(iter(wall_colors))
    towers = [tower for tower in towers if tower[2] != wall_color]
    tower_colors = {color for _r, _c, color in towers}
    assert len(tower_colors) == 2, "expected exactly two tower colors"

    win_marker: dict[int, int] = {}
    equal_marker = None
    for row in range(3):
        label = int(table[row, 0])
        marker = int(table[row, 1])
        if label in tower_colors:
            win_marker[label] = marker
        else:
            equal_marker = marker
    assert len(win_marker) == 2 and equal_marker is not None, "lookup rows must encode both colors and equality"

    blocker_mask = (g == wall_color) | table_mask
    for r, c, _color in towers:
        blocker_mask[r, c] = True

    return {
        "bg": int(bg),
        "wall_color": int(wall_color),
        "towers": towers,
        "win_marker": win_marker,
        "equal_marker": int(equal_marker),
        "blocker_mask": blocker_mask,
    }


def _cast_rays(
    shape: tuple[int, int],
    towers: list[tuple[int, int, int]],
    blocker_mask: np.ndarray,
) -> dict[tuple[int, int], dict[int, int]]:
    hits: dict[tuple[int, int], dict[int, int]] = defaultdict(dict)
    for r0, c0, color in towers:
        for dr, dc in DIRS:
            r = int(r0)
            c = int(c0)
            dist = 0
            while True:
                r += int(dr)
                c += int(dc)
                dist += 1
                if not _in_bounds(shape, r, c):
                    break
                if blocker_mask[r, c]:
                    break
                prev = hits[(r, c)].get(int(color))
                if prev is None or dist < prev:
                    hits[(r, c)][int(color)] = int(dist)
    return hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit tower rays until a wall, the boundary, or another tower blocks them.
    At red/blue intersections, the shorter ray wins and the lookup table maps
    that relation to the output marker color.
    """
    g = np.asarray(grid, dtype=int)
    scene = _parse_scene(g)
    hits = _cast_rays(g.shape, scene["towers"], scene["blocker_mask"])

    out = g.copy()
    for (r, c), by_color in hits.items():
        if len(by_color) == 1:
            out[r, c] = int(next(iter(by_color)))
            continue
        assert len(by_color) == 2, "generator should avoid 3-way intersections"
        (color_a, dist_a), (color_b, dist_b) = list(by_color.items())
        if dist_a < dist_b:
            out[r, c] = int(scene["win_marker"][int(color_a)])
        elif dist_b < dist_a:
            out[r, c] = int(scene["win_marker"][int(color_b)])
        else:
            out[r, c] = int(scene["equal_marker"])
    return out


GRID_H_RANGE = (18, 26)
GRID_W_RANGE = (18, 26)
PAIR_COUNT_CHOICES = (1, 2)
PAIR_COUNT_WEIGHTS = (3, 2)
WALL_LEN_RANGE = (2, 3)
RAY_LEN_RANGE = (2, 4)
EXTRA_WALL_COUNT_RANGE = (0, 3)
GEN_RETRIES = 400
PAIR_TRIES = 200
EXTRA_WALL_TRIES = 80


def _segment_cells(r0: int, c0: int, dr: int, dc: int, length: int) -> list[tuple[int, int]]:
    return [(int(r0 + dr * i), int(c0 + dc * i)) for i in range(length)]


def _inflate(cells: set[tuple[int, int]], shape: tuple[int, int], radius: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - radius, r + radius + 1):
            for cc in range(c - radius, c + radius + 1):
                if _in_bounds(shape, rr, cc):
                    out.add((int(rr), int(cc)))
    return out


def _table_origin(h: int, w: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, w - 2
    if corner == "bl":
        return h - 3, 0
    if corner == "br":
        return h - 3, w - 2
    raise ValueError(corner)


def _place_lookup_table(grid: np.ndarray, markers: tuple[int, int, int], corner: str) -> set[tuple[int, int]]:
    r0, c0 = _table_origin(*grid.shape, corner)
    block = np.array(
        [
            [RED, markers[0]],
            [BLUE, markers[1]],
            [EQUAL_LABEL, markers[2]],
        ],
        dtype=int,
    )
    grid[r0 : r0 + 3, c0 : c0 + 2] = block
    cells = {(int(r), int(c)) for r in range(r0, r0 + 3) for c in range(c0, c0 + 2)}
    return cells


def _distance_pairs(outcome: str, h_color: int, v_color: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    lo, hi = RAY_LEN_RANGE
    for dh in range(lo, hi + 1):
        for dv in range(lo, hi + 1):
            if outcome == "equal" and dh == dv:
                pairs.append((int(dh), int(dv)))
            elif outcome == "red":
                red_len = dh if h_color == RED else dv
                blue_len = dv if h_color == RED else dh
                if red_len < blue_len:
                    pairs.append((int(dh), int(dv)))
            elif outcome == "blue":
                red_len = dh if h_color == RED else dv
                blue_len = dv if h_color == RED else dh
                if blue_len < red_len:
                    pairs.append((int(dh), int(dv)))
    return pairs


def _pair_blueprint(
    shape: tuple[int, int],
    cross: tuple[int, int],
    h_dir: int,
    v_dir: int,
    dh: int,
    dv: int,
    wall_len: int,
    h_color: int,
    v_color: int,
) -> dict | None:
    cr, cc = cross
    towers = [
        (int(cr), int(cc - dh * h_dir), int(h_color)),
        (int(cr - dv * v_dir), int(cc), int(v_color)),
    ]
    h_tower = towers[0][:2]
    v_tower = towers[1][:2]

    path_cells = {(int(cr), int(cc))}
    for step in range(1, dh):
        path_cells.add((int(h_tower[0]), int(h_tower[1] + step * h_dir)))
    for step in range(1, dv):
        path_cells.add((int(v_tower[0] + step * v_dir), int(v_tower[1])))

    wall_cells: set[tuple[int, int]] = set()
    wall_cells.update(_segment_cells(h_tower[0] - 1, h_tower[1], -1, 0, wall_len))
    wall_cells.update(_segment_cells(h_tower[0] + 1, h_tower[1], 1, 0, wall_len))
    wall_cells.update(_segment_cells(h_tower[0], h_tower[1] - h_dir, 0, -h_dir, wall_len))
    wall_cells.update(_segment_cells(cr, cc + h_dir, 0, h_dir, wall_len))

    wall_cells.update(_segment_cells(v_tower[0], v_tower[1] - 1, 0, -1, wall_len))
    wall_cells.update(_segment_cells(v_tower[0], v_tower[1] + 1, 0, 1, wall_len))
    wall_cells.update(_segment_cells(v_tower[0] - v_dir, v_tower[1], -v_dir, 0, wall_len))
    wall_cells.update(_segment_cells(cr + v_dir, cc, v_dir, 0, wall_len))

    occupied = wall_cells | {tower[:2] for tower in towers}
    if occupied & path_cells:
        return None
    all_cells = occupied | path_cells
    if any(not _in_bounds(shape, r, c) for r, c in all_cells):
        return None

    return {
        "towers": towers,
        "wall_cells": wall_cells,
        "path_cells": path_cells,
        "reserved": all_cells,
    }


def _add_extra_wall(
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    h, w = grid.shape
    for _ in range(EXTRA_WALL_TRIES):
        horizontal = random.choice([True, False])
        length = random.randint(2, 5)
        if horizontal:
            r = random.randint(0, h - 1)
            c = random.randint(0, w - length)
            cells = {(int(r), int(cc)) for cc in range(c, c + length)}
        else:
            r = random.randint(0, h - length)
            c = random.randint(0, w - 1)
            cells = {(int(rr), int(c)) for rr in range(r, r + length)}
        if cells & reserved:
            continue
        inflated = _inflate(cells, grid.shape)
        if inflated & reserved:
            continue
        return cells, inflated
    return None


def _wall_components_are_valid(grid: np.ndarray) -> bool:
    comps = _components(grid == WALL)
    return bool(comps) and all(_is_straight_segment(comp) for comp in comps)


def _count_intersections(grid: np.ndarray) -> int:
    scene = _parse_scene(grid)
    hits = _cast_rays(grid.shape, scene["towers"], scene["blocker_mask"])
    return sum(1 for by_color in hits.values() if len(by_color) == 2)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)

    marker_colors = tuple(random.sample(MARKER_POOL, 3))
    corner = random.choice(["tl", "tr", "bl", "br"])
    table_cells = _place_lookup_table(grid, marker_colors, corner)
    reserved = _inflate(table_cells, grid.shape)

    pair_count = random.choices(PAIR_COUNT_CHOICES, weights=PAIR_COUNT_WEIGHTS, k=1)[0]
    for _pair_idx in range(pair_count):
        placed = False
        for _ in range(PAIR_TRIES):
            wall_len = random.randint(*WALL_LEN_RANGE)
            h_color, v_color = random.sample([RED, BLUE], 2)
            outcome = random.choice(["red", "blue", "equal"])
            dh, dv = random.choice(_distance_pairs(outcome, h_color, v_color))
            h_dir = random.choice([-1, 1])
            v_dir = random.choice([-1, 1])

            row_margin = max(dv + wall_len, wall_len + 1) + 1
            col_margin = max(dh + wall_len, wall_len + 1) + 1
            if h <= 2 * row_margin or w <= 2 * col_margin:
                return None
            cross = (
                random.randint(row_margin, h - row_margin - 1),
                random.randint(col_margin, w - col_margin - 1),
            )
            blueprint = _pair_blueprint(
                grid.shape,
                cross,
                h_dir,
                v_dir,
                dh,
                dv,
                wall_len,
                h_color,
                v_color,
            )
            if blueprint is None:
                continue
            if blueprint["reserved"] & reserved:
                continue
            inflated = _inflate(blueprint["reserved"], grid.shape)
            if inflated & reserved:
                continue

            for r, c in blueprint["wall_cells"]:
                grid[r, c] = WALL
            for r, c, color in blueprint["towers"]:
                grid[r, c] = color
            reserved |= inflated
            placed = True
            break
        if not placed:
            return None

    for _ in range(random.randint(*EXTRA_WALL_COUNT_RANGE)):
        extra = _add_extra_wall(grid, reserved)
        if extra is None:
            continue
        cells, inflated = extra
        for r, c in cells:
            grid[r, c] = WALL
        reserved |= inflated

    if _most_common_color(grid) != BG:
        return None
    if not _wall_components_are_valid(grid):
        return None

    try:
        scene = _parse_scene(grid)
        hits = _cast_rays(grid.shape, scene["towers"], scene["blocker_mask"])
        if sum(1 for by_color in hits.values() if len(by_color) == 2) != pair_count:
            return None
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-tower puzzle input")
