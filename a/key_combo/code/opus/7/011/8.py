from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
TEAL = 8
MAROON = 9

INPUT_H = 15
INPUT_W = 23
PANEL_H = 13
PANEL_W = 11
SEPARATOR_COL = 11

DIR_TO_DELTA = {
    RED: (1, 0),
    GREEN: (-1, 0),
    BLUE: (0, 1),
    YELLOW: (0, -1),
}
OPPOSITE_DIR = {
    RED: GREEN,
    GREEN: RED,
    BLUE: YELLOW,
    YELLOW: BLUE,
}

GEN_RETRIES = 400


def _parse_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h >= 4 and w >= 3 and w % 2 == 1, "expected row + two panels + separator"

    panel_h = h - 2
    panel_w = w // 2
    sep_col = panel_w

    left = g[1 : 1 + panel_h, :panel_w]
    right = g[1 : 1 + panel_h, sep_col + 1 :]
    directions = g[0, :panel_w]
    assert left.shape == right.shape == (panel_h, panel_w)
    assert all(int(color) in DIR_TO_DELTA for color in directions), "invalid direction color in header row"
    return left, right, directions


def _render_from_active(active: np.ndarray, directions: np.ndarray, *, stop_on_active: bool) -> np.ndarray:
    panel_h, panel_w = map(int, active.shape)
    out = np.zeros((panel_h, panel_w), dtype=int)
    out[active] = TEAL

    for r, c in map(tuple, np.argwhere(active)):
        dr, dc = DIR_TO_DELTA[int(directions[c])]
        rr = int(r + dr)
        cc = int(c + dc)
        while 0 <= rr < panel_h and 0 <= cc < panel_w:
            if stop_on_active and bool(active[rr, cc]):
                break
            out[rr, cc] = ORANGE
            rr += dr
            cc += dc

    out[active] = TEAL
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compute the NOR mask of the two panels, then fire per-column rays from the active NOR cells.

    Input structure:
    - The top row is a direction header over the left panel width: red(2)=down, green(3)=up, blue(1)=right, yellow(4)=left.
    - The middle rows contain two equal panels separated by one grey(5) column; panel pixels are black(0) or maroon(9).
    - A final blank row may appear below the panels and is not part of the comparison area.

    Rule:
    - Remove the grey separator and compare the two panels cell by cell.
    - In the output, mark a teal(8) cell exactly where both corresponding panel cells are black(0); all other cells stay black(0).
    - For each teal cell, read the direction color from its column in the header and extend an orange(7) ray from the adjacent cell in that direction.
    - Stop a ray when it would leave the output or when the next cell is another teal source; do not overwrite teal cells.

    Key insight:
    - The NOR result is self-referential: the same teal cells are both ray emitters and ray blockers, while the header row supplies the missing global direction information.
    """
    left, right, directions = _parse_panels(grid)
    active = (left == BACKGROUND) & (right == BACKGROUND)
    return _render_from_active(active, directions, stop_on_active=True)


def _has_positive_step(r: int, c: int, color: int) -> bool:
    dr, dc = DIR_TO_DELTA[int(color)]
    rr = int(r + dr)
    cc = int(c + dc)
    return 0 <= rr < PANEL_H and 0 <= cc < PANEL_W


def _ray_stop_kind(active: np.ndarray, directions: np.ndarray, r: int, c: int) -> str:
    dr, dc = DIR_TO_DELTA[int(directions[c])]
    rr = int(r + dr)
    cc = int(c + dc)
    painted = False
    while 0 <= rr < PANEL_H and 0 <= cc < PANEL_W:
        if bool(active[rr, cc]):
            return "collision"
        painted = True
        rr += dr
        cc += dc
    return "boundary" if painted else "none"


def _collision_source_count(active: np.ndarray, directions: np.ndarray) -> int:
    total = 0
    for r, c in map(tuple, np.argwhere(active)):
        if _ray_stop_kind(active, directions, int(r), int(c)) == "collision":
            total += 1
    return total


def _boundary_source_count(active: np.ndarray, directions: np.ndarray) -> int:
    total = 0
    for r, c in map(tuple, np.argwhere(active)):
        if _ray_stop_kind(active, directions, int(r), int(c)) == "boundary":
            total += 1
    return total


def _sample_direction_row() -> tuple[np.ndarray, int, int, int, int]:
    directions = [None] * PANEL_W

    right_col = random.randint(1, 3)
    left_col = random.randint(right_col + 4, 9)
    remaining = [c for c in range(PANEL_W) if c not in {right_col, left_col}]
    down_col, up_col = random.sample(remaining, 2)

    directions[right_col] = BLUE
    directions[left_col] = YELLOW
    directions[down_col] = RED
    directions[up_col] = GREEN

    for c in range(PANEL_W):
        if directions[c] is None:
            directions[c] = random.choice([BLUE, RED, GREEN, YELLOW])

    return np.array(directions, dtype=int), right_col, left_col, down_col, up_col


def _random_vertical_pair(avoid_rows: set[int]) -> tuple[int, int] | None:
    pairs = [
        (r0, r1)
        for r0 in range(PANEL_H)
        for r1 in range(r0 + 2, min(PANEL_H, r0 + 6))
        if r0 not in avoid_rows and r1 not in avoid_rows
    ]
    return None if not pairs else random.choice(pairs)


def _build_active_mask(
    directions: np.ndarray, right_col: int, left_col: int, down_col: int, up_col: int
) -> tuple[np.ndarray, set[tuple[int, int]]] | None:
    active = np.zeros((PANEL_H, PANEL_W), dtype=bool)
    row_counts = np.zeros(PANEL_H, dtype=int)

    row_h = random.randint(2, PANEL_H - 3)
    active[row_h, right_col] = True
    active[row_h, left_col] = True
    row_counts[row_h] = 2
    protected = {(row_h, c) for c in range(right_col, left_col + 1)}

    down_pair = _random_vertical_pair({row_h})
    if down_pair is None:
        return None
    d_top, d_bottom = down_pair
    active[d_top, down_col] = True
    active[d_bottom, down_col] = True
    row_counts[d_top] += 1
    row_counts[d_bottom] += 1
    protected.update((r, down_col) for r in range(d_top, d_bottom + 1))

    up_pair = _random_vertical_pair({row_h})
    if up_pair is None:
        return None
    u_top, u_bottom = up_pair
    active[u_top, up_col] = True
    active[u_bottom, up_col] = True
    row_counts[u_top] += 1
    row_counts[u_bottom] += 1
    if np.any(row_counts > 3):
        return None
    protected.update((r, up_col) for r in range(u_top, u_bottom + 1))

    target_count = random.randint(8, 15)
    candidates = [
        (r, c)
        for r in range(PANEL_H)
        for c in range(PANEL_W)
        if not active[r, c]
        and (r, c) not in protected
        and row_counts[r] < 3
        and _has_positive_step(r, c, int(directions[c]))
    ]
    random.shuffle(candidates)
    for r, c in candidates:
        if int(active.sum()) >= target_count:
            break
        if row_counts[r] >= 3:
            continue
        active[r, c] = True
        row_counts[r] += 1

    if not (8 <= int(active.sum()) <= 15):
        return None
    if int(row_counts.max()) > 3:
        return None
    if any(not np.any(active[:, directions == color]) for color in (BLUE, RED, GREEN, YELLOW)):
        return None
    if _collision_source_count(active, directions) < 2:
        return None
    if _boundary_source_count(active, directions) < 2:
        return None
    return active, protected


def _pick_extra_holes(
    active: np.ndarray, directions: np.ndarray, protected: set[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray] | None:
    left_black = active.copy()
    right_black = active.copy()

    candidates = [
        (r, c)
        for r in range(PANEL_H)
        for c in range(PANEL_W)
        if not active[r, c]
        and (r, c) not in protected
        and _has_positive_step(r, c, int(directions[c]))
    ]
    random.shuffle(candidates)

    left_extra = random.randint(4, 8)
    right_extra = random.randint(4, 8)
    if len(candidates) < left_extra + right_extra:
        return None

    for r, c in candidates[:left_extra]:
        left_black[r, c] = True
    for r, c in candidates[left_extra : left_extra + right_extra]:
        right_black[r, c] = True

    if np.array_equal(left_black, active) or np.array_equal(right_black, active):
        return None
    if np.any(left_black & right_black & ~active):
        return None
    return left_black, right_black


def _build_input_grid(left_black: np.ndarray, right_black: np.ndarray, directions: np.ndarray) -> np.ndarray:
    grid = np.zeros((INPUT_H, INPUT_W), dtype=int)
    grid[0, :PANEL_W] = directions
    grid[1 : 1 + PANEL_H, :PANEL_W] = MAROON
    grid[1 : 1 + PANEL_H, SEPARATOR_COL] = GREY
    grid[1 : 1 + PANEL_H, SEPARATOR_COL + 1 :] = MAROON
    grid[1 : 1 + PANEL_H, :PANEL_W][left_black] = BACKGROUND
    grid[1 : 1 + PANEL_H, SEPARATOR_COL + 1 :][right_black] = BACKGROUND
    return grid


def _validate_instance(grid: np.ndarray) -> bool:
    out = main(grid)
    left, right, directions = _parse_panels(grid)
    left_black = left == BACKGROUND
    right_black = right == BACKGROUND
    active = left_black & right_black

    if not np.array_equal(out, _render_from_active(active, directions, stop_on_active=True)):
        return False
    if not (8 <= int(active.sum()) <= 15):
        return False
    if int(active.sum(axis=1).max()) > 3:
        return False
    if set(map(int, directions.tolist())) != {BLUE, RED, GREEN, YELLOW}:
        return False
    if any(not np.any(active[:, directions == color]) for color in (BLUE, RED, GREEN, YELLOW)):
        return False
    if _collision_source_count(active, directions) < 2:
        return False
    if _boundary_source_count(active, directions) < 2:
        return False
    if int(np.count_nonzero(out == ORANGE)) < 6:
        return False

    opposite = np.array([OPPOSITE_DIR[int(color)] for color in directions], dtype=int)
    if np.array_equal(out, _render_from_active(active, opposite, stop_on_active=True)):
        return False
    if np.array_equal(out, _render_from_active(active, directions, stop_on_active=False)):
        return False
    if np.array_equal(out, _render_from_active(left_black, directions, stop_on_active=True)):
        return False
    if np.array_equal(out, _render_from_active(right_black, directions, stop_on_active=True)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    directions, right_col, left_col, down_col, up_col = _sample_direction_row()
    active_pack = _build_active_mask(directions, right_col, left_col, down_col, up_col)
    if active_pack is None:
        return None
    active, protected = active_pack

    hole_pack = _pick_extra_holes(active, directions, protected)
    if hole_pack is None:
        return None
    left_black, right_black = hole_pack

    grid = _build_input_grid(left_black, right_black, directions)
    return grid if _validate_instance(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid NOR-ray ARC input")
