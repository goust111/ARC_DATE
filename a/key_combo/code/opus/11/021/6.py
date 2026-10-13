from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
DIVIDER = 5

_GEN_RETRIES = 800
_VALID_COLORS = {BG, BLUE, RED, DIVIDER}


def _full_width_divider_rows(grid: np.ndarray) -> list[int]:
    return [int(r) for r in range(grid.shape[0]) if np.all(grid[r, :] == DIVIDER)]


def _find_divider(grid: np.ndarray) -> int:
    rows = _full_width_divider_rows(grid)
    assert len(rows) == 1, "expected exactly one full-width gray divider"
    return rows[0]


def _paint_vertical_segment(out: np.ndarray, c: int, r0: int, r1: int, color: int) -> None:
    lo = min(int(r0), int(r1))
    hi = max(int(r0), int(r1))
    out[lo : hi + 1, int(c)] = int(color)


def _paint_rays(grid: np.ndarray, divider: int, *, red_to_divider: bool, blue_to_boundary: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    H, _W = g.shape
    out = g.copy()

    for r, c in np.argwhere(g == RED):
        if red_to_divider:
            target_r = divider - 1 if r < divider else divider + 1
        else:
            target_r = 0 if r < divider else H - 1
        _paint_vertical_segment(out, int(c), int(r), int(target_r), RED)

    for r, c in np.argwhere(g == BLUE):
        if blue_to_boundary:
            target_r = 0 if r < divider else H - 1
        else:
            target_r = divider - 1 if r < divider else divider + 1
        _paint_vertical_segment(out, int(c), int(r), int(target_r), BLUE)

    out[divider, :] = DIVIDER
    return out


def _half_count(grid: np.ndarray, divider: int, half: str) -> int:
    if half == "top":
        patch = grid[:divider, :]
    else:
        patch = grid[divider + 1 :, :]
    return int(np.count_nonzero((patch != BG) & (patch != DIVIDER)))


def _mirror_winner(painted: np.ndarray, divider: int, winner: str) -> np.ndarray:
    out = painted.copy()
    H, _W = out.shape

    if winner == "top":
        for dst_r in range(divider + 1, H):
            src_r = 2 * divider - dst_r
            if 0 <= src_r < divider:
                out[dst_r, :] = painted[src_r, :]
    else:
        for dst_r in range(0, divider):
            src_r = 2 * divider - dst_r
            if divider < src_r < H:
                out[dst_r, :] = painted[src_r, :]

    out[divider, :] = DIVIDER
    return out


def _solve_details(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int, str, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= 30, "grid must fit ARC size limits"
    assert set(map(int, np.unique(g))) <= _VALID_COLORS, "unexpected colors in input"
    assert GREEN not in g, "green must not appear in the input"

    divider = _find_divider(g)
    H, _W = g.shape
    assert 3 <= divider <= H - 4, "divider must leave room on both sides"

    for color in (RED, BLUE):
        coords = np.argwhere(g == color)
        assert 2 <= len(coords) <= 4, f"expected 2-4 pixels of color {color}"
        top = int(np.sum(coords[:, 0] < divider))
        bottom = int(np.sum(coords[:, 0] > divider))
        assert top >= 1 and bottom >= 1, "both halves must contain red and blue"

    seeds = np.argwhere((g == RED) | (g == BLUE))
    for i, (r0, c0) in enumerate(seeds):
        assert abs(int(r0) - divider) >= 2, "seed must not touch the divider"
        for r1, c1 in seeds[i + 1 :]:
            assert max(abs(int(r0) - int(r1)), abs(int(c0) - int(c1))) >= 2, "seeds must be isolated"

    painted = _paint_rays(g, divider, red_to_divider=True, blue_to_boundary=True)
    U = _half_count(painted, divider, "top")
    D = _half_count(painted, divider, "bottom")
    assert U != D, "ray totals must not tie"
    winner = "top" if U > D else "bottom"
    out = _mirror_winner(painted, divider, winner)
    return painted, out, U, D, winner, divider


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend red and blue rays, then mirror the heavier half across the gray divider.

    Input structure:
    - The background is black.
    - Exactly one full-width horizontal gray line splits the grid into an upper half and a lower half.
    - Red and blue isolated pixels lie on both sides of the divider, and no seed touches the divider.

    Rule:
    - For each red pixel, draw a vertical red ray from that pixel toward the divider and stop one cell before the divider.
    - For each blue pixel, draw a vertical blue ray from that pixel toward the nearest outer boundary of its half and include the edge cell.
    - Count non-background, non-divider pixels in the upper half and lower half after all rays are drawn.
    - Mirror the half with the larger count across the divider onto the other half, while keeping the winner half and the gray divider unchanged.

    Key insight:
    - Color chooses each seed's ray direction, ray lengths determine each half's total painted weight, and that weight vote decides which completed half overwrites the other by reflection.
    """
    _painted, out, _U, _D, _winner, _divider = _solve_details(grid)
    return out


def _seed_profiles() -> list[tuple[tuple[int, int], tuple[int, int]]]:
    options = [(1, 1), (1, 2), (2, 1), (2, 2)]
    profiles: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for top in options:
        for bottom in options:
            if sum(top) == sum(bottom):
                profiles.append((top, bottom))
    return profiles


_PROFILES = _seed_profiles()


def _winner_compatible_divider(H: int, winner: str) -> int:
    center = H // 2
    if winner == "top":
        choices = [center, min(H - 5, center + 1), min(H - 5, center + 2)]
    else:
        choices = [center, max(4, center - 1), max(4, center - 2)]
    choices = sorted(set(int(x) for x in choices if 4 <= int(x) <= H - 5))
    return random.choice(choices)


def _candidate_rows(H: int, divider: int, half: str) -> list[int]:
    if half == "top":
        return list(range(0, divider - 1))
    return list(range(divider + 2, H))


def _can_place_seed(
    placed: list[tuple[int, int]],
    r: int,
    c: int,
    used_cols: set[int],
) -> bool:
    if c in used_cols:
        return False
    for rr, cc in placed:
        if max(abs(rr - r), abs(cc - c)) < 2:
            return False
    return True


def _place_half_seeds(
    grid: np.ndarray,
    divider: int,
    half: str,
    red_count: int,
    blue_count: int,
) -> bool:
    H, W = grid.shape
    rows = _candidate_rows(H, divider, half)
    cells = [(r, c) for r in rows for c in range(W)]
    random.shuffle(cells)

    used_cols: set[int] = set()
    placed: list[tuple[int, int]] = []

    for color, need in ((RED, red_count), (BLUE, blue_count)):
        made = 0
        for r, c in cells:
            if grid[r, c] != BG:
                continue
            if not _can_place_seed(placed, r, c, used_cols):
                continue
            grid[r, c] = color
            placed.append((r, c))
            used_cols.add(c)
            made += 1
            if made == need:
                break
        if made != need:
            return False
    return True


def _nontrivial_lengths(grid: np.ndarray, divider: int) -> tuple[bool, bool]:
    red_ok = False
    blue_ok = False
    H, _W = grid.shape
    for r, _c in np.argwhere(grid == RED):
        length = (divider - int(r)) if r < divider else (int(r) - divider)
        red_ok |= length >= 2
    for r, _c in np.argwhere(grid == BLUE):
        length = (int(r) + 1) if r < divider else (H - int(r))
        blue_ok |= length >= 2
    return red_ok, blue_ok


def _alternate_output(grid: np.ndarray, divider: int, *, all_to_divider: bool) -> np.ndarray:
    painted = _paint_rays(
        grid,
        divider,
        red_to_divider=True if all_to_divider else False,
        blue_to_boundary=False if all_to_divider else True,
    )
    U = _half_count(painted, divider, "top")
    D = _half_count(painted, divider, "bottom")
    if U == D:
        return painted
    winner = "top" if U > D else "bottom"
    return _mirror_winner(painted, divider, winner)


def _validate_generated_input(grid: np.ndarray, intended_winner: str) -> bool:
    try:
        painted, out, U, D, winner, divider = _solve_details(grid)
    except AssertionError:
        return False

    if winner != intended_winner:
        return False
    if abs(U - D) < 2:
        return False

    top_seeds = int(np.count_nonzero((grid[:divider, :] == RED) | (grid[:divider, :] == BLUE)))
    bottom_seeds = int(np.count_nonzero((grid[divider + 1 :, :] == RED) | (grid[divider + 1 :, :] == BLUE)))
    if top_seeds != bottom_seeds:
        return False

    red_ok, blue_ok = _nontrivial_lengths(grid, divider)
    if not (red_ok and blue_ok):
        return False

    if np.array_equal(out, painted):
        return False

    if np.array_equal(_alternate_output(grid, divider, all_to_divider=True), out):
        if np.array_equal(_alternate_output(grid, divider, all_to_divider=False), out):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.choice([11, 13, 15, 17, 19])
    W = random.randint(11, 18)
    intended_winner = random.choice(["top", "bottom"])
    divider = _winner_compatible_divider(H, intended_winner)
    (top_red, top_blue), (bottom_red, bottom_blue) = random.choice(_PROFILES)

    grid = np.zeros((H, W), dtype=int)
    grid[divider, :] = DIVIDER

    if not _place_half_seeds(grid, divider, "top", top_red, top_blue):
        return None
    if not _place_half_seeds(grid, divider, "bottom", bottom_red, bottom_blue):
        return None

    if not _validate_generated_input(grid, intended_winner):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
