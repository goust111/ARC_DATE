from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


_BORDER_OFFSETS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

_PAIR_COUNT_CHOICES = (3, 4)
_BAND_HEIGHT = 7
_WIDTH_MIN = 18
_WIDTH_MAX = 27
_MIN_CENTER_GAP = 7
_GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _tower_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _find_towers(grid: np.ndarray) -> tuple[list[dict[str, object]], int, set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    towers: list[dict[str, object]] = []

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            border_vals = [int(g[r + dr, c + dc]) for dr, dc in _BORDER_OFFSETS]
            frame = border_vals[0]
            if frame == bg or any(v != frame for v in border_vals):
                continue
            center_color = int(g[r, c])
            if center_color == frame:
                continue
            towers.append(
                {
                    "center": (r, c),
                    "frame": frame,
                    "center_color": center_color,
                    "solid": center_color != bg,
                    "cells": _tower_cells((r, c)),
                }
            )

    occupied: set[tuple[int, int]] = set()
    for tower in towers:
        cells = tower["cells"]
        assert isinstance(cells, set)
        assert occupied.isdisjoint(cells), "tower frames must not overlap"
        occupied.update(cells)

    assert towers, "expected at least one tower"
    groups: dict[int, list[dict[str, object]]] = defaultdict(list)
    for tower in towers:
        groups[int(tower["frame"])].append(tower)
    assert all(len(group) == 2 for group in groups.values()), "each frame color must appear exactly twice"

    return towers, bg, occupied


def _ordered_pair(group: list[dict[str, object]]) -> tuple[dict[str, object], dict[str, object]]:
    assert len(group) == 2
    a, b = group
    ac = a["center"]
    bc = b["center"]
    assert isinstance(ac, tuple) and isinstance(bc, tuple)
    return (a, b) if (ac[1], ac[0]) <= (bc[1], bc[0]) else (b, a)


def _segment_cells(
    start: tuple[int, int],
    end: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    r1, c1 = start
    r2, c2 = end
    horizontal: list[tuple[int, int]] = []
    vertical: list[tuple[int, int]] = []

    cur_r, cur_c = r1, c1
    if c1 != c2:
        step_c = 1 if c2 > c1 else -1
        while cur_c != c2:
            cur_c += step_c
            cell = (cur_r, cur_c)
            if cell not in blocked:
                horizontal.append(cell)
    if r1 != r2:
        step_r = 1 if r2 > r1 else -1
        while cur_r != r2:
            cur_r += step_r
            cell = (cur_r, cur_c)
            if cell not in blocked:
                vertical.append(cell)

    return horizontal, vertical


def _paint_solid(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _paint_dashed(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for idx, (r, c) in enumerate(cells):
        if idx % 2 == 0:
            grid[r, c] = color


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    towers, _bg, occupied = _find_towers(g)
    groups: dict[int, list[dict[str, object]]] = defaultdict(list)
    for tower in towers:
        groups[int(tower["frame"])].append(tower)

    pairs = []
    for group in groups.values():
        left, right = _ordered_pair(group)
        lc = left["center"]
        rc = right["center"]
        assert isinstance(lc, tuple) and isinstance(rc, tuple)
        key = (min(lc[0], rc[0]), min(lc[1], rc[1]), max(lc[0], rc[0]), max(lc[1], rc[1]))
        pairs.append((key, left, right))
    pairs.sort(key=lambda item: item[0])

    out = g.copy()
    for _key, left, right in pairs:
        start = left["center"]
        end = right["center"]
        assert isinstance(start, tuple) and isinstance(end, tuple)
        color = int(left["frame"])
        horizontal, vertical = _segment_cells(start, end, occupied)
        left_solid = bool(left["solid"])
        right_solid = bool(right["solid"])

        if left_solid and right_solid:
            _paint_solid(out, horizontal + vertical, color)
        elif not left_solid and not right_solid:
            _paint_dashed(out, horizontal + vertical, color)
        else:
            _paint_solid(out, horizontal, color)
            _paint_dashed(out, vertical, color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-frame 3x3 towers and connect their centers with horizontal-then-vertical L paths.

    The line color is the shared frame color. Both-solid pairs draw solid paths, both-hollow
    pairs draw a dashed path starting from the left tower, and mixed pairs draw a solid
    horizontal segment plus a dashed vertical segment. Tower pixels are never overwritten.
    """
    return _solve(grid)


def _draw_tower(grid: np.ndarray, center: tuple[int, int], frame: int, center_color: int) -> None:
    r, c = center
    grid[r - 1 : r + 2, c - 1 : c + 2] = frame
    grid[r, c] = center_color


def _pick_branches(rng: random.Random, pair_count: int) -> list[str]:
    branches = ["solid", "hollow", "mixed"]
    if pair_count == 4:
        branches.append(rng.choice(branches))
    rng.shuffle(branches)
    return branches


def _pick_modes(rng: random.Random, pair_count: int) -> list[str]:
    modes = [None] * pair_count  # type: ignore[list-item]
    first = rng.randrange(pair_count)
    second_choices = [idx for idx in range(pair_count) if idx != first]
    second = rng.choice(second_choices)
    modes[first] = "down"
    modes[second] = "up"
    options = ("flat_top", "flat_bottom", "down", "up")
    for idx in range(pair_count):
        if modes[idx] is None:
            modes[idx] = rng.choice(options)
    return [str(mode) for mode in modes]


def _mode_rows(band_top: int, mode: str) -> tuple[int, int]:
    upper = band_top + 1
    lower = band_top + 5
    if mode == "flat_top":
        return upper, upper
    if mode == "flat_bottom":
        return lower, lower
    if mode == "down":
        return upper, lower
    if mode == "up":
        return lower, upper
    raise ValueError(f"unknown mode: {mode}")


def _choose_columns(rng: random.Random, width: int) -> tuple[int, int]:
    left = rng.randint(1, width - 2 - _MIN_CENTER_GAP)
    right = rng.randint(left + _MIN_CENTER_GAP, width - 2)
    return left, right


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.choice(_PAIR_COUNT_CHOICES)
    top_margin = rng.randint(1, 2)
    bottom_margin = rng.randint(1, 2)
    height = top_margin + pair_count * _BAND_HEIGHT + bottom_margin
    width = rng.randint(_WIDTH_MIN, _WIDTH_MAX)

    if width - 2 - _MIN_CENTER_GAP < 1:
        return None

    colors = list(range(10))
    rng.shuffle(colors)
    bg = colors.pop()
    frame_colors = [colors.pop() for _ in range(pair_count)]
    solid_colors = colors[:]
    if not solid_colors:
        return None

    branches = _pick_branches(rng, pair_count)
    modes = _pick_modes(rng, pair_count)
    for idx, branch in enumerate(branches):
        if branch == "mixed" and modes[idx].startswith("flat"):
            modes[idx] = rng.choice(("down", "up"))

    grid = np.full((height, width), bg, dtype=int)
    for idx, (frame, branch, mode) in enumerate(zip(frame_colors, branches, modes)):
        band_top = top_margin + idx * _BAND_HEIGHT
        left_row, right_row = _mode_rows(band_top, mode)
        left_col, right_col = _choose_columns(rng, width)

        if branch == "solid":
            left_solid = True
            right_solid = True
        elif branch == "hollow":
            left_solid = False
            right_solid = False
        else:
            left_solid, right_solid = rng.choice(((True, False), (False, True)))

        left_center = rng.choice(solid_colors) if left_solid else bg
        right_center = rng.choice(solid_colors) if right_solid else bg

        _draw_tower(grid, (left_row, left_col), frame, left_center)
        _draw_tower(grid, (right_row, right_col), frame, right_center)

    try:
        towers, found_bg, _occupied = _find_towers(grid)
        if len(towers) != 2 * pair_count or found_bg != bg:
            return None
        _ = _solve(grid)
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {_GEN_RETRIES} attempts")
