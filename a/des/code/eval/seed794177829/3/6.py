import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7

BASE_COLORS = (BLUE, RED, YELLOW)
MIXED_LINE_COLOR = {
    frozenset((RED, BLUE)): MAGENTA,
    frozenset((RED, YELLOW)): ORANGE,
    frozenset((BLUE, YELLOW)): GREEN,
}
MIX_INPUT_PAIRS = ((RED, BLUE), (RED, YELLOW), (BLUE, YELLOW))

GRID_MIN = 24
GRID_MAX = 30
GEN_RETRIES = 400
PAIR_GAP_MIN = 6
EXTRA_TOWER_MAX = 3


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _line_color(color_a: int, color_b: int) -> int:
    if int(color_a) == int(color_b):
        return int(color_a)
    return int(MIXED_LINE_COLOR[frozenset((int(color_a), int(color_b)))])


def _find_towers(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    allowed = {int(bg), GREY, *BASE_COLORS}
    extras = {int(v) for v in np.unique(grid)} - allowed
    assert not extras, f"unexpected input colors: {sorted(extras)}"

    visited = np.zeros((h, w), dtype=bool)
    towers: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color not in BASE_COLORS or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            pts = np.asarray(coords, dtype=int)
            assert int(pts.shape[0]) == 9, "every tower must contain exactly 9 pixels"
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "every tower must have a 3x3 bbox"
            assert bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color)), "tower bbox must be solid"
            towers.append(
                {
                    "color": int(color),
                    "bbox": (r0, c0, r1, c1),
                    "center": (r0 + 1, c0 + 1),
                    "coords": pts,
                }
            )

    assert len(towers) >= 2, "expected at least two towers"

    tower_mask = np.zeros((h, w), dtype=bool)
    for tower in towers:
        pts = tower["coords"]
        tower_mask[pts[:, 0], pts[:, 1]] = True
    assert bool(np.array_equal(np.isin(grid, BASE_COLORS), tower_mask)), "base colors must only appear in towers"

    return sorted(towers, key=lambda tower: (tower["center"][0], tower["center"][1], tower["color"]))


def _pair_connections(towers: list[dict], wall_mask: np.ndarray) -> list[dict]:
    connections: list[dict] = []
    for idx_a in range(len(towers)):
        for idx_b in range(idx_a + 1, len(towers)):
            tower_a = towers[idx_a]
            tower_b = towers[idx_b]
            ar, ac = map(int, tower_a["center"])
            br, bc = map(int, tower_b["center"])

            if ar == br:
                orientation = "horizontal"
                left, right = sorted((ac, bc))
                cells = [(int(ar), int(c)) for c in range(int(left + 1), int(right))]
            elif ac == bc:
                orientation = "vertical"
                top, bottom = sorted((ar, br))
                cells = [(int(r), int(ac)) for r in range(int(top + 1), int(bottom))]
            else:
                continue

            blocked = any(bool(wall_mask[r, c]) for r, c in cells)
            connections.append(
                {
                    "orientation": orientation,
                    "visible": not blocked,
                    "cells": cells,
                    "line_color": _line_color(int(tower_a["color"]), int(tower_b["color"])),
                    "colors": (int(tower_a["color"]), int(tower_b["color"])),
                }
            )
    return connections


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect 3x3 signal towers when their center points share a row or column and
    no grey wall blocks the center-to-center path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    towers = _find_towers(g, bg)
    wall_mask = g == GREY

    tower_mask = np.zeros_like(wall_mask)
    for tower in towers:
        pts = tower["coords"]
        tower_mask[pts[:, 0], pts[:, 1]] = True

    assert not bool(np.any(tower_mask & wall_mask)), "walls may not overlap towers"

    out = g.copy()
    for connection in _pair_connections(towers, wall_mask):
        if not bool(connection["visible"]):
            continue
        line_color = int(connection["line_color"])
        for r, c in connection["cells"]:
            if bool(tower_mask[r, c]) or bool(wall_mask[r, c]):
                continue
            prev = int(out[r, c])
            assert prev in (int(bg), line_color), "conflicting visible links overlap"
            out[r, c] = line_color

    return out


def _paint_tower(grid: np.ndarray, center: tuple[int, int], color: int) -> None:
    r, c = map(int, center)
    grid[r - 1 : r + 2, c - 1 : c + 2] = int(color)


def _can_place_tower(grid: np.ndarray, center: tuple[int, int]) -> bool:
    h, w = grid.shape
    r, c = map(int, center)
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return False
    return bool(np.all(grid[r - 1 : r + 2, c - 1 : c + 2] == BLACK))


def _pick_two_with_gap(candidates: list[int], gap_min: int, rng: random.Random) -> tuple[int, int] | None:
    for _ in range(80):
        a, b = sorted(rng.sample(candidates, 2))
        if int(b - a) >= int(gap_min):
            return int(a), int(b)
    return None


def _random_pair_colors(rng: random.Random) -> tuple[int, int]:
    if rng.random() < 0.5:
        color = int(rng.choice(BASE_COLORS))
        return color, color
    return tuple(map(int, rng.choice(MIX_INPUT_PAIRS)))


def _add_segment(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    for r, c in cells:
        if int(grid[r, c]) != BLACK:
            return False
    for r, c in cells:
        grid[r, c] = GREY
    return True


def _block_horizontal_pair(
    grid: np.ndarray,
    row: int,
    left_col: int,
    right_col: int,
    row_bounds: tuple[int, int],
    rng: random.Random,
) -> None:
    gap_cols = list(range(int(left_col + 2), int(right_col - 1)))
    assert gap_cols

    for _ in range(40):
        if rng.random() < 0.5:
            col = int(rng.choice(gap_cols))
            top = int(rng.randint(row_bounds[0], row))
            bottom = int(rng.randint(row, row_bounds[1]))
            if bottom - top + 1 < 2:
                continue
            cells = [(int(rr), col) for rr in range(top, bottom + 1)]
        else:
            if len(gap_cols) == 1:
                cells = [(int(row), int(gap_cols[0]))]
            else:
                left = int(rng.choice(gap_cols[:-1]))
                right_choices = [c for c in gap_cols if int(c) > int(left)]
                if not right_choices:
                    continue
                right = int(rng.choice(right_choices))
                cells = [(int(row), int(cc)) for cc in range(left, right + 1)]
        if _add_segment(grid, cells):
            return
    raise AssertionError("failed to place horizontal blocker")


def _block_vertical_pair(
    grid: np.ndarray,
    top_row: int,
    bottom_row: int,
    col: int,
    col_bounds: tuple[int, int],
    rng: random.Random,
) -> None:
    gap_rows = list(range(int(top_row + 2), int(bottom_row - 1)))
    assert gap_rows

    for _ in range(40):
        if rng.random() < 0.5:
            if len(gap_rows) == 1:
                cells = [(int(gap_rows[0]), int(col))]
            else:
                top = int(rng.choice(gap_rows[:-1]))
                bottom_choices = [r for r in gap_rows if int(r) > int(top)]
                if not bottom_choices:
                    continue
                bottom = int(rng.choice(bottom_choices))
                cells = [(int(rr), int(col)) for rr in range(top, bottom + 1)]
        else:
            row = int(rng.choice(gap_rows))
            left = int(rng.randint(col_bounds[0], col))
            right = int(rng.randint(col, col_bounds[1]))
            if right - left + 1 < 2:
                continue
            cells = [(row, int(cc)) for cc in range(left, right + 1)]
        if _add_segment(grid, cells):
            return
    raise AssertionError("failed to place vertical blocker")


def _validate_scene(grid: np.ndarray) -> bool:
    try:
        bg = _get_bg(grid)
        towers = _find_towers(grid, bg)
        wall_mask = grid == GREY
        tower_mask = np.zeros(grid.shape, dtype=bool)
        for tower in towers:
            pts = tower["coords"]
            tower_mask[pts[:, 0], pts[:, 1]] = True
        connections = _pair_connections(towers, wall_mask)
        visible = [conn for conn in connections if bool(conn["visible"])]
        blocked = [conn for conn in connections if not bool(conn["visible"])]

        if not visible or not blocked:
            return False
        if not any(conn["orientation"] == "horizontal" and conn["visible"] for conn in connections):
            return False
        if not any(conn["orientation"] == "vertical" and conn["visible"] for conn in connections):
            return False
        if not any(conn["orientation"] == "horizontal" and not conn["visible"] for conn in connections):
            return False
        if not any(conn["orientation"] == "vertical" and not conn["visible"] for conn in connections):
            return False
        if not any(conn["visible"] and conn["colors"][0] == conn["colors"][1] for conn in connections):
            return False
        if not any(conn["visible"] and conn["colors"][0] != conn["colors"][1] for conn in connections):
            return False

        painted: dict[tuple[int, int], int] = {}
        for conn in visible:
            for r, c in conn["cells"]:
                if bool(tower_mask[r, c]) or bool(wall_mask[r, c]):
                    continue
                prev = painted.get((int(r), int(c)))
                if prev is not None and int(prev) != int(conn["line_color"]):
                    return False
                painted[(int(r), int(c))] = int(conn["line_color"])

        out = main(grid)
        if np.array_equal(out, grid):
            return False
        if int(np.count_nonzero(out != grid)) < 4:
            return False
        return True
    except AssertionError:
        return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = int(rng.randint(GRID_MIN, GRID_MAX))
    w = int(rng.randint(GRID_MIN, GRID_MAX))
    grid = np.full((h, w), BLACK, dtype=int)

    top_rows = list(range(2, h // 2 - 2))
    bottom_rows = list(range(h // 2 + 2, h - 2))
    left_cols = list(range(2, w // 2 - 2))
    right_cols = list(range(w // 2 + 2, w - 2))

    if len(top_rows) < 2 or len(bottom_rows) < 4 or len(left_cols) < 3 or len(right_cols) < 3:
        return None

    same_color = int(rng.choice(BASE_COLORS))
    mixed_colors = tuple(map(int, rng.choice(MIX_INPUT_PAIRS)))
    horizontal_same = bool(rng.random() < 0.5)
    visible_h_colors = (same_color, same_color) if horizontal_same else mixed_colors
    visible_v_colors = mixed_colors if horizontal_same else (same_color, same_color)
    blocked_h_colors = _random_pair_colors(rng)
    blocked_v_colors = _random_pair_colors(rng)

    visible_h_cols = _pick_two_with_gap(left_cols, PAIR_GAP_MIN, rng)
    blocked_h_cols = _pick_two_with_gap(right_cols, PAIR_GAP_MIN, rng)
    visible_v_rows = _pick_two_with_gap(bottom_rows, PAIR_GAP_MIN, rng)
    blocked_v_rows = _pick_two_with_gap(bottom_rows, PAIR_GAP_MIN, rng)
    if not visible_h_cols or not blocked_h_cols or not visible_v_rows or not blocked_v_rows:
        return None

    if set(visible_v_rows) & set(blocked_v_rows):
        return None

    visible_h_row, blocked_h_row = rng.sample(top_rows, 2)
    visible_v_col_choices = [c for c in left_cols if c not in visible_h_cols]
    blocked_v_col_choices = [c for c in right_cols if c not in blocked_h_cols]
    if not visible_v_col_choices or not blocked_v_col_choices:
        return None
    visible_v_col = int(rng.choice(visible_v_col_choices))
    blocked_v_col = int(rng.choice(blocked_v_col_choices))

    placements = [
        ((int(visible_h_row), int(visible_h_cols[0])), int(visible_h_colors[0])),
        ((int(visible_h_row), int(visible_h_cols[1])), int(visible_h_colors[1])),
        ((int(visible_v_rows[0]), int(visible_v_col)), int(visible_v_colors[0])),
        ((int(visible_v_rows[1]), int(visible_v_col)), int(visible_v_colors[1])),
        ((int(blocked_h_row), int(blocked_h_cols[0])), int(blocked_h_colors[0])),
        ((int(blocked_h_row), int(blocked_h_cols[1])), int(blocked_h_colors[1])),
        ((int(blocked_v_rows[0]), int(blocked_v_col)), int(blocked_v_colors[0])),
        ((int(blocked_v_rows[1]), int(blocked_v_col)), int(blocked_v_colors[1])),
    ]

    used_rows = {int(center[0]) for center, _ in placements}
    used_cols = {int(center[1]) for center, _ in placements}

    for center, color in placements:
        if not _can_place_tower(grid, center):
            return None
        _paint_tower(grid, center, color)

    top_band = (1, h // 2 - 1)
    right_band = (w // 2 + 1, w - 2)
    _block_horizontal_pair(grid, int(blocked_h_row), int(blocked_h_cols[0]), int(blocked_h_cols[1]), top_band, rng)
    _block_vertical_pair(grid, int(blocked_v_rows[0]), int(blocked_v_rows[1]), int(blocked_v_col), right_band, rng)

    for _ in range(int(rng.randint(0, EXTRA_TOWER_MAX + 1))):
        placed = False
        for _ in range(80):
            row_choices = [r for r in range(2, h - 2) if r not in used_rows]
            col_choices = [c for c in range(2, w - 2) if c not in used_cols]
            if not row_choices or not col_choices:
                break
            center = (int(rng.choice(row_choices)), int(rng.choice(col_choices)))
            if not _can_place_tower(grid, center):
                continue
            _paint_tower(grid, center, int(rng.choice(BASE_COLORS)))
            used_rows.add(int(center[0]))
            used_cols.add(int(center[1]))
            placed = True
            break
        if not placed:
            break

    return grid if _validate_scene(grid) else None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
