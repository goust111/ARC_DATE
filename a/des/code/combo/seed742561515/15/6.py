from __future__ import annotations

import random

import numpy as np


# Relative 3-pixel skeleton templates keyed by arrow direction.
_WINGS_BY_DIR: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]] = {
    (-1, 0): ((1, -1), (1, 1)),
    (1, 0): ((-1, -1), (-1, 1)),
    (0, -1): ((-1, 1), (1, 1)),
    (0, 1): ((-1, -1), (1, -1)),
    (-1, -1): ((1, 0), (0, 1)),
    (-1, 1): ((1, 0), (0, -1)),
    (1, -1): ((-1, 0), (0, 1)),
    (1, 1): ((-1, 0), (0, -1)),
}

_DIRS = tuple(_WINGS_BY_DIR)
_TEMPLATES = {
    direction: frozenset(((0, 0),) + wings)
    for direction, wings in _WINGS_BY_DIR.items()
}
_DIR_BY_TEMPLATE = {template: direction for direction, template in _TEMPLATES.items()}

_PALETTE = tuple(range(1, 10))
_RNG = np.random.default_rng()

_GRID_H_RANGE = (10, 19)
_GRID_W_RANGE = (10, 19)
_MAX_PLACEMENT_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)


def _skeleton_cells(tip: tuple[int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    tr, tc = map(int, tip)
    cells = {(tr, tc)}
    for dr, dc in _WINGS_BY_DIR[direction]:
        cells.add((tr + int(dr), tc + int(dc)))
    return cells


def _shaft_cell(tip: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int]:
    tr, tc = map(int, tip)
    dr, dc = map(int, direction)
    return tr - dr, tc - dc


def _ray_steps_to_boundary(
    shape: tuple[int, int], tip: tuple[int, int], direction: tuple[int, int]
) -> int:
    h, w = shape
    r, c = map(int, tip)
    dr, dc = map(int, direction)
    steps = 0
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return steps
        steps += 1


def _output_cells(
    shape: tuple[int, int], tip: tuple[int, int], direction: tuple[int, int]
) -> set[tuple[int, int]]:
    cells = set(_skeleton_cells(tip, direction))
    shaft = _shaft_cell(tip, direction)
    assert _in_bounds(shape, *shaft), "shaft cell must be inside the grid"
    cells.add(shaft)

    tr, tc = map(int, tip)
    dr, dc = map(int, direction)
    for step in range(1, 5):
        rr = tr + step * dr
        cc = tc + step * dc
        if not _in_bounds(shape, rr, cc):
            break
        cells.add((rr, cc))
    return cells


def _dilate8(points: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    for r, c in points:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r) + dr
                cc = int(c) + dc
                if _in_bounds(shape, rr, cc):
                    blocked.add((rr, cc))
    return blocked


def _identify_arrow(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    pts = {tuple(map(int, pt)) for pt in coords}
    assert len(pts) == 3, "each arrow must have exactly three pixels"

    matches: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for tip in pts:
        tr, tc = tip
        rel = frozenset((r - tr, c - tc) for r, c in pts)
        direction = _DIR_BY_TEMPLATE.get(rel)
        if direction is not None:
            matches.append((tip, direction))

    assert len(matches) == 1, "expected one valid arrow template match"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify each 3-pixel monochrome arrow skeleton, add one shaft pixel,
    then emit a 4-pixel ray in the pointing direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for color in sorted(int(v) for v in np.unique(g) if int(v) != 0):
        pts = [tuple(map(int, rc)) for rc in np.argwhere(g == int(color))]
        tip, direction = _identify_arrow(pts)

        shaft = _shaft_cell(tip, direction)
        assert _in_bounds(g.shape, *shaft), "valid input must keep the shaft in bounds"
        out[shaft] = int(color)

        tr, tc = tip
        dr, dc = direction
        for step in range(1, 5):
            rr = tr + step * dr
            cc = tc + step * dc
            if not _in_bounds(g.shape, rr, cc):
                break
            out[rr, cc] = int(color)

    return out


def _paint_arrow(
    grid: np.ndarray, color: int, tip: tuple[int, int], direction: tuple[int, int]
) -> None:
    for r, c in _skeleton_cells(tip, direction):
        grid[r, c] = int(color)


def _branch_pattern(n_arrows: int) -> list[bool]:
    if n_arrows == 1:
        return [bool(_RNG.integers(0, 2))]

    mode = int(_RNG.integers(0, 3))
    if mode == 0:
        return [False] * n_arrows
    if mode == 1:
        return [True] * n_arrows

    flags = [False] * n_arrows
    n_truncated = int(_RNG.integers(1, n_arrows))
    for idx in _RNG.choice(n_arrows, size=n_truncated, replace=False):
        flags[int(idx)] = True
    return flags


def _candidate_tips(
    shape: tuple[int, int],
    direction: tuple[int, int],
    want_truncated: bool,
    forbidden: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], set[tuple[int, int]]]]:
    h, w = shape
    candidates: list[tuple[tuple[int, int], set[tuple[int, int]]]] = []

    for r in range(h):
        for c in range(w):
            tip = (int(r), int(c))
            skeleton = _skeleton_cells(tip, direction)
            shaft = _shaft_cell(tip, direction)
            if not all(_in_bounds(shape, rr, cc) for rr, cc in skeleton | {shaft}):
                continue

            steps = _ray_steps_to_boundary(shape, tip, direction)
            if want_truncated and steps >= 4:
                continue
            if not want_truncated and steps < 4:
                continue

            footprint = _output_cells(shape, tip, direction)
            if footprint & forbidden:
                continue
            candidates.append((tip, footprint))

    random.shuffle(candidates)
    return candidates


def _validate_grid(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        return False
    if g.size == 0:
        return False
    if int(np.count_nonzero(g == 0)) <= int(np.count_nonzero(g)):
        return False

    colors = [int(v) for v in np.unique(g) if int(v) != 0]
    if not colors:
        return False

    blocked: set[tuple[int, int]] = set()
    for color in colors:
        pts = [tuple(map(int, rc)) for rc in np.argwhere(g == color)]
        if len(pts) != 3:
            return False
        try:
            tip, direction = _identify_arrow(pts)
        except AssertionError:
            return False

        if set(pts) != _skeleton_cells(tip, direction):
            return False
        try:
            footprint = _output_cells(g.shape, tip, direction)
        except AssertionError:
            return False
        if footprint & blocked:
            return False
        blocked |= _dilate8(footprint, g.shape)

    return True


def _attempt_generate() -> np.ndarray | None:
    h = int(_RNG.integers(*_GRID_H_RANGE))
    w = int(_RNG.integers(*_GRID_W_RANGE))
    area = h * w

    max_arrows = min(len(_PALETTE), max(2, area // 55))
    n_arrows = int(_RNG.integers(2, max_arrows + 1))
    colors = [int(v) for v in _RNG.choice(_PALETTE, size=n_arrows, replace=False)]
    directions = [tuple(map(int, _DIRS[i])) for i in _RNG.choice(len(_DIRS), size=n_arrows, replace=False)]
    trunc_flags = _branch_pattern(n_arrows)

    specs = [
        {
            "color": color,
            "direction": direction,
            "truncated": bool(truncated),
        }
        for color, direction, truncated in zip(colors, directions, trunc_flags)
    ]
    specs.sort(
        key=lambda spec: len(
            _candidate_tips((h, w), spec["direction"], spec["truncated"], set())
        )
    )

    grid = np.zeros((h, w), dtype=int)
    blocked: set[tuple[int, int]] = set()

    for spec in specs:
        candidates = _candidate_tips(
            (h, w), spec["direction"], spec["truncated"], blocked
        )
        if not candidates:
            return None

        pick_idx = int(_RNG.integers(0, len(candidates)))
        tip, footprint = candidates[pick_idx]
        _paint_arrow(grid, spec["color"], tip, spec["direction"])
        blocked |= _dilate8(footprint, (h, w))

    if not _validate_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_PLACEMENT_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow puzzle")
