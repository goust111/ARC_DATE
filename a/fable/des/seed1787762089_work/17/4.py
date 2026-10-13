from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


def _mirror_col(axis: int, col: int) -> int:
    return 2 * axis - col


def _infer_axis_and_singletons(
    grid: np.ndarray,
) -> tuple[int, int, list[tuple[int, int, int]], int, set[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, _w = g.shape
    bg = int(get_bg(g))

    pair_midpoints: list[int] = []
    singletons: list[tuple[int, int, int]] = []
    pair_count = 0
    pair_colors: set[int] = set()

    for r in range(h):
        row = g[r]
        colors = [int(color) for color in np.unique(row) if int(color) != bg]
        for color in colors:
            cols = np.flatnonzero(row == color)
            assert cols.size <= 2, "each row/color may contain at most two pixels"
            if cols.size == 2:
                c0, c1 = int(cols[0]), int(cols[1])
                assert (c0 + c1) % 2 == 0, "pair midpoint must be a visible grid column"
                midpoint = (c0 + c1) // 2
                assert midpoint not in (c0, c1), "the hidden axis cannot host a pair pixel"
                pair_midpoints.append(midpoint)
                pair_count += 1
                pair_colors.add(color)
            elif cols.size == 1:
                singletons.append((r, int(cols[0]), color))

    assert pair_midpoints, "expected at least one defining pair"
    axis = pair_midpoints[0]
    assert all(mid == axis for mid in pair_midpoints), "all pairs must share one midpoint"
    assert np.all(g[:, axis] == bg), "the inferred axis must stay invisible"

    return bg, axis, singletons, pair_count, pair_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the hidden vertical symmetry axis from same-row color pairs and mirror every singleton across it.
    """
    g = np.asarray(grid, dtype=int)
    bg, axis, singletons, _pair_count, _pair_colors = _infer_axis_and_singletons(g)
    h, w = g.shape

    out = g.copy()
    for r, c, color in singletons:
        mirrored = _mirror_col(axis, c)
        assert 0 <= mirrored < w and 0 <= r < h
        assert c != axis, "singleton markers cannot sit on the hidden axis"
        assert out[r, mirrored] == bg, "mirrored singleton target must be empty"
        out[r, mirrored] = color

    return out


def _singleton_candidates(row: np.ndarray, axis: int, side: str) -> list[int]:
    w = int(row.shape[0])
    if side == "L":
        cols = range(axis)
    else:
        cols = range(axis + 1, w)

    candidates: list[int] = []
    for col in cols:
        mirrored = _mirror_col(axis, col)
        if not (0 <= mirrored < w):
            continue
        if row[col] != BG or row[mirrored] != BG:
            continue
        candidates.append(int(col))
    return candidates


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 18)
    w = random.randint(18, 24)
    axis = random.randint(3, w - 4)
    max_radius = min(axis, w - 1 - axis)
    if max_radius < 2:
        return None

    grid = np.full((h, w), BG, dtype=int)

    pair_count = random.randint(2, min(6, h))
    singleton_count = random.randint(2, min(7, h))

    pair_rows = random.sample(range(h), pair_count)
    singleton_rows = random.sample(range(h), singleton_count)

    pair_palette_size = random.randint(2, 4)
    pair_palette = random.sample(PALETTE, pair_palette_size)
    random.shuffle(pair_rows)

    row_pair_color: dict[int, int] = {}
    used_pair_colors: list[int] = []
    for idx, row in enumerate(pair_rows):
        color = pair_palette[idx] if idx < len(pair_palette) else random.choice(pair_palette)
        radius = random.randint(1, max_radius)
        grid[row, axis - radius] = color
        grid[row, axis + radius] = color
        row_pair_color[row] = color
        used_pair_colors.append(color)

    used_pair_color_set = set(used_pair_colors)

    sides = ["L", "R"] + [random.choice(["L", "R"]) for _ in range(singleton_count - 2)]
    random.shuffle(sides)

    reuse_count = random.randint(1, singleton_count - 1)
    reuse_flags = [True] * reuse_count + [False] * (singleton_count - reuse_count)
    random.shuffle(reuse_flags)

    fresh_palette = [color for color in PALETTE if color not in used_pair_color_set]
    assert fresh_palette

    for row, side, reuse in zip(singleton_rows, sides, reuse_flags):
        blocked = {row_pair_color[row]} if row in row_pair_color else set()

        if reuse:
            color_choices = [color for color in used_pair_color_set if color not in blocked]
            if not color_choices:
                color_choices = [color for color in PALETTE if color not in blocked]
        else:
            color_choices = [color for color in fresh_palette if color not in blocked]
            if not color_choices:
                color_choices = [color for color in PALETTE if color not in blocked]

        candidates = _singleton_candidates(grid[row], axis, side)
        if not candidates or not color_choices:
            return None

        col = random.choice(candidates)
        grid[row, col] = random.choice(color_choices)

    return grid


def _validate_generated(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert 14 <= g.shape[0] <= 18
    assert 18 <= g.shape[1] <= 24

    bg, axis, singletons, pair_count, pair_colors = _infer_axis_and_singletons(g)
    assert bg == BG
    assert pair_count >= 2
    assert len(singletons) >= 2
    assert np.all(g[:, axis] == BG)

    sides = {"L" if c < axis else "R" for _r, c, _color in singletons}
    assert sides == {"L", "R"}, "singletons should appear on both sides over time"

    singleton_colors = {color for _r, _c, color in singletons}
    assert singleton_colors & pair_colors, "include reused pair colors among singletons"
    assert singleton_colors - pair_colors, "include fresh singleton-only colors too"

    out = main(g)
    assert out.shape == g.shape
    assert np.all(out[g != bg] == g[g != bg]), "main must preserve existing pixels"
    assert int(np.count_nonzero(out != g)) == len(singletons)

    for r, c, color in singletons:
        mirrored = _mirror_col(axis, c)
        assert out[r, mirrored] == color


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is None:
            continue
        _validate_generated(grid)
        return grid
    raise RuntimeError("failed to generate a valid hidden-axis mirror puzzle")
