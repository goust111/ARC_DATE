from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BLUE = 1
RED = 2
BLACK = 0

DIR_BY_COLOR = {
    3: (0, -1),   # left
    4: (1, 0),    # down
    6: (-1, 0),   # up
    8: (0, 1),    # right
}

BORDER_OFFSETS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)
DIAG_OFFSETS = ((-1, -1), (-1, 1), (1, -1), (1, 1))

GRID_HEIGHT_RANGE = (12, 19)
GRID_WIDTH_RANGE = (12, 19)
OBJECT_COUNT_CHOICES = (2, 3)
GEN_ATTEMPTS = 800
SCENE_ATTEMPTS = 150


@dataclass(frozen=True)
class Shell:
    r: int
    c: int
    color: int


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _border_cells_from_center(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in BORDER_OFFSETS]


def _footprint_cells(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]


def _diag_cells(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in DIAG_OFFSETS]


def _target_top_left(shell: Shell, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    dr, dc = DIR_BY_COLOR[shell.color]
    if (dr, dc) == (0, -1):
        return shell.r - 1, 0
    if (dr, dc) == (0, 1):
        return shell.r - 1, w - 3
    if (dr, dc) == (-1, 0):
        return 0, shell.c - 1
    return h - 3, shell.c - 1


def _target_border_cells(shell: Shell, shape: tuple[int, int]) -> list[tuple[int, int]]:
    top, left = _target_top_left(shell, shape)
    return [(top + dr, left + dc) for dr, dc in BORDER_OFFSETS]


def _sort_shells(shells: list[Shell]) -> list[Shell]:
    return sorted(shells, key=lambda shell: (shell.c, shell.r))


def _stamp_border(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _detect_shells(grid: np.ndarray) -> list[Shell]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    shells: list[Shell] = []
    covered: set[tuple[int, int]] = set()

    for r, c in np.argwhere(g == BLUE):
        r = int(r)
        c = int(c)
        if not (1 <= r < h - 1 and 1 <= c < w - 1):
            continue
        border_colors = {int(g[r + dr, c + dc]) for dr, dc in BORDER_OFFSETS}
        if len(border_colors) != 1:
            continue
        color = next(iter(border_colors))
        if color not in DIR_BY_COLOR:
            continue
        shell = Shell(r=r, c=c, color=color)
        shells.append(shell)
        for cell in _footprint_cells(r, c):
            if cell in covered:
                raise AssertionError("shells must not overlap")
            covered.add(cell)

    shells = _sort_shells(shells)
    assert 2 <= len(shells) <= 3, "expected 2 or 3 shell objects"
    assert len({shell.color for shell in shells}) == len(shells), "shell colors must be distinct"

    allowed = {BLACK}
    for shell in shells:
        allowed.add(shell.color)
    allowed.add(BLUE)
    assert set(map(int, np.unique(g))) <= allowed, "unexpected colors outside shell palette"

    expected_nonblack = set()
    for shell in shells:
        expected_nonblack.update(_border_cells_from_center(shell.r, shell.c))
        expected_nonblack.add((shell.r, shell.c))
        for rr, cc in _border_cells_from_center(shell.r, shell.c):
            assert int(g[rr, cc]) == shell.color
        assert int(g[shell.r, shell.c]) == BLUE

    actual_nonblack = {(int(r), int(c)) for r, c in np.argwhere(g != BLACK)}
    assert actual_nonblack == expected_nonblack, "foreground must be exactly the shells and centers"
    return shells


def _slide_shells_only(grid: np.ndarray, shells: list[Shell]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for shell in shells:
        for r, c in _border_cells_from_center(shell.r, shell.c):
            out[r, c] = BLACK
        _stamp_border(out, _target_border_cells(shell, out.shape), shell.color)
    return out


def _paint_rays(out: np.ndarray, shells: list[Shell]) -> None:
    for shell in shells:
        row = out[shell.r, :]
        row[row == BLACK] = BLUE
        col = out[:, shell.c]
        col[col == BLACK] = BLUE


def _paint_centers_red(out: np.ndarray, shells: list[Shell]) -> None:
    for shell in shells:
        out[shell.r, shell.c] = RED


def _paint_shell_diagonals(out: np.ndarray, shells: list[Shell]) -> None:
    for shell in shells:
        for rr, cc in _diag_cells(shell.r, shell.c):
            if _in_bounds(out.shape, rr, cc) and int(out[rr, cc]) in (BLACK, BLUE):
                out[rr, cc] = shell.color


def _solve_with_detected_shells(grid: np.ndarray, shells: list[Shell]) -> np.ndarray:
    out = _slide_shells_only(grid, shells)
    _paint_rays(out, shells)
    _paint_centers_red(out, shells)
    _paint_shell_diagonals(out, shells)
    return out


def _wrong_without_slide(grid: np.ndarray, shells: list[Shell]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    _paint_rays(out, shells)
    _paint_centers_red(out, shells)
    _paint_shell_diagonals(out, shells)
    return out


def _wrong_without_rays(grid: np.ndarray, shells: list[Shell]) -> np.ndarray:
    out = _slide_shells_only(grid, shells)
    _paint_centers_red(out, shells)
    _paint_shell_diagonals(out, shells)
    return out


def _wrong_without_diagonals(grid: np.ndarray, shells: list[Shell]) -> np.ndarray:
    out = _slide_shells_only(grid, shells)
    _paint_rays(out, shells)
    _paint_centers_red(out, shells)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each hollow shell to its color-coded edge, then fire rays from the exposed centers.

    Input structure:
    - The grid has a black background and contains 2 or 3 separate 3x3 shell objects.
    - Each object is a one-pixel-thick colored square border with a blue center pixel.
    - Shell colors come from green, yellow, magenta, and teal, and each color encodes one slide direction.

    Rule:
    - Detect every 3x3 shell; green slides left, yellow slides down, magenta slides up, and teal slides right.
    - Erase each shell border from its original position, then place that same 8-pixel border flush against the matching grid edge while keeping the blue center at its original location.
    - After all shells move, paint blue rays through every black cell in each center's full row and full column, recolor the centers red, and finally recolor the four diagonal neighbors with the shell color that originally surrounded that center.

    Key insight:
    - The center pixels stay fixed while the surrounding shells move away, so the output combines two coordinate systems: edge-aligned moved shells and center-anchored rays plus diagonal shell-color recall.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    shells = _detect_shells(g)
    return _solve_with_detected_shells(g, shells)


def _scene_valid(shape: tuple[int, int], shells: list[Shell]) -> bool:
    h, w = shape
    if len({shell.color for shell in shells}) != len(shells):
        return False
    if len({shell.r for shell in shells}) != len(shells):
        return False
    if len({shell.c for shell in shells}) != len(shells):
        return False

    original_footprints: list[set[tuple[int, int]]] = []
    diagonal_sets: list[set[tuple[int, int]]] = []
    target_borders: list[set[tuple[int, int]]] = []

    for shell in shells:
        if not (3 <= shell.r <= h - 4 and 3 <= shell.c <= w - 4):
            return False
        footprint = set(_footprint_cells(shell.r, shell.c))
        if any(not _in_bounds(shape, r, c) for r, c in footprint):
            return False
        diagonals = set(_diag_cells(shell.r, shell.c))
        if any(not _in_bounds(shape, r, c) for r, c in diagonals):
            return False
        target = set(_target_border_cells(shell, shape))
        if any(not _in_bounds(shape, r, c) for r, c in target):
            return False
        original_footprints.append(footprint)
        diagonal_sets.append(diagonals)
        target_borders.append(target)

    for i in range(len(shells)):
        for j in range(i + 1, len(shells)):
            if abs(shells[i].r - shells[j].r) <= 2:
                return False
            if abs(shells[i].c - shells[j].c) <= 2:
                return False
            if original_footprints[i] & original_footprints[j]:
                return False
            if target_borders[i] & target_borders[j]:
                return False
            if target_borders[i] & original_footprints[j]:
                return False
            if target_borders[j] & original_footprints[i]:
                return False

    for target in target_borders:
        for diagonals in diagonal_sets:
            if target & diagonals:
                return False

    return True


def _render_input(shape: tuple[int, int], shells: list[Shell]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for shell in shells:
        _stamp_border(grid, _border_cells_from_center(shell.r, shell.c), shell.color)
        grid[shell.r, shell.c] = BLUE
    return grid


def _semantic_checks(grid: np.ndarray, shells: list[Shell]) -> bool:
    out = main(grid)

    if np.array_equal(out, _wrong_without_slide(grid, shells)):
        return False
    if np.array_equal(out, _wrong_without_rays(grid, shells)):
        return False
    if np.array_equal(out, _wrong_without_diagonals(grid, shells)):
        return False

    total_ray_blue = 0
    for shell in shells:
        if int(out[shell.r, shell.c]) != RED:
            return False
        for rr, cc in _diag_cells(shell.r, shell.c):
            if int(out[rr, cc]) != shell.color:
                return False
        orthogonals = ((shell.r - 1, shell.c), (shell.r + 1, shell.c), (shell.r, shell.c - 1), (shell.r, shell.c + 1))
        for rr, cc in orthogonals:
            if int(out[rr, cc]) != BLUE:
                return False
        for rr, cc in _target_border_cells(shell, out.shape):
            if int(out[rr, cc]) != shell.color:
                return False
        total_ray_blue += int(np.count_nonzero(out[shell.r, :] == BLUE))
        total_ray_blue += int(np.count_nonzero(out[:, shell.c] == BLUE))

    if total_ray_blue <= 0:
        return False
    return True


def _candidate_centers(h: int, w: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(3, h - 3) for c in range(3, w - 3)]
    random.shuffle(cells)
    return cells


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(*GRID_HEIGHT_RANGE)
        w = random.randint(*GRID_WIDTH_RANGE)
        count = random.choice(OBJECT_COUNT_CHOICES)
        centers = _candidate_centers(h, w)

        for _scene_try in range(SCENE_ATTEMPTS):
            chosen_centers = random.sample(centers, count)
            chosen_colors = random.sample(list(DIR_BY_COLOR), count)
            shells = [Shell(r=r, c=c, color=color) for (r, c), color in zip(chosen_centers, chosen_colors)]
            shells = _sort_shells(shells)

            if not _scene_valid((h, w), shells):
                continue

            grid = _render_input((h, w), shells)
            if _detect_shells(grid) != shells:
                continue
            if not _semantic_checks(grid, shells):
                continue
            return grid

    raise RuntimeError("failed to generate a valid input")
