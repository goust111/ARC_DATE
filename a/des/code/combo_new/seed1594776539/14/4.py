from __future__ import annotations

from collections import Counter

import numpy as np


BG = 0
PALETTE = np.array([1, 2, 3, 4, 5, 6, 8, 9], dtype=int)
SCENARIOS = ("tie", "unequal", "mixed")
GRID_SIDE_RANGE = (8, 19)
MAX_STRIPES_PER_ORIENTATION = 5
ASSIGNMENT_TRIES = 200
GENERATION_TRIES = 400
_RNG = np.random.default_rng()


def _dominant_non_bg(values: np.ndarray) -> int:
    fg = values[values != BG]
    assert fg.size > 0, "expected at least one foreground pixel"
    colors, counts = np.unique(fg, return_counts=True)
    winners = colors[counts == counts.max()]
    assert len(winners) == 1, "stripe identity must have a unique majority color"
    return int(winners[0])


def _recover_stripes(g: np.ndarray) -> tuple[np.ndarray, list[int], np.ndarray, list[int]]:
    h, w = map(int, g.shape)
    row_fill = np.count_nonzero(g != BG, axis=1)
    col_fill = np.count_nonzero(g != BG, axis=0)

    row_positions = np.flatnonzero(row_fill == w)
    col_positions = np.flatnonzero(col_fill == h)
    assert 0 < len(row_positions) < h, "expected some but not all rows to be full-width stripes"
    assert 0 < len(col_positions) < w, "expected some but not all columns to be full-height stripes"

    row_colors = [_dominant_non_bg(g[r, :]) for r in row_positions]
    col_colors = [_dominant_non_bg(g[:, c]) for c in col_positions]
    return row_positions.astype(int), row_colors, col_positions.astype(int), col_colors


def _stripe_totals(row_colors: list[int], col_colors: list[int]) -> dict[int, int]:
    totals = Counter(map(int, row_colors))
    totals.update(map(int, col_colors))
    return {int(color): int(count) for color, count in totals.items()}


def _render_ranked(
    shape: tuple[int, int],
    row_positions: np.ndarray,
    row_colors: list[int],
    col_positions: np.ndarray,
    col_colors: list[int],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    totals = _stripe_totals(row_colors, col_colors)

    for count in sorted(set(totals.values()), reverse=True):
        for c, color in zip(col_positions, col_colors):
            if totals[int(color)] == count:
                out[:, int(c)] = int(color)
        for r, color in zip(row_positions, row_colors):
            if totals[int(color)] == count:
                out[int(r), :] = int(color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the row and column stripes, then redraw them so rarer colors sit above common ones.

    Input structure:
    - The background is black.
    - Some one-pixel-wide horizontal stripes occupy whole rows, and some one-pixel-wide vertical stripes occupy whole columns.
    - Each row-column crossing shows only one visible color, because one stripe is on top at that intersection in the input.

    Rule:
    - Detect horizontal stripes as rows that are entirely non-black, and vertical stripes as columns that are entirely non-black.
    - Recover each stripe's true color by taking the unique most frequent non-black color along that full row or full column.
    - Count how many stripes belong to each color across both orientations.
    - Redraw the scene from bottom to top by painting colors with larger stripe counts first; when two colors have the same count, paint vertical stripes before horizontal stripes so the horizontal stripe wins that tie.

    Key insight:
    - The visible crossing colors are unreliable, but a full-row or full-column stripe still leaves a majority-color signature that reveals its identity before the correct depth order is recomputed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    row_positions, row_colors, col_positions, col_colors = _recover_stripes(g)
    return _render_ranked(g.shape, row_positions, row_colors, col_positions, col_colors)


def _sample_spaced_positions(length: int, count: int, rng: np.random.Generator) -> np.ndarray:
    base = np.sort(rng.choice(length - count + 1, size=count, replace=False))
    return (base + np.arange(count)).astype(int)


def _visible_branch_flags(row_colors: list[int], col_colors: list[int], totals: dict[int, int]) -> tuple[bool, bool]:
    equal_visible = False
    unequal_visible = False
    for row_color in row_colors:
        for col_color in col_colors:
            if int(row_color) == int(col_color):
                continue
            if totals[int(row_color)] == totals[int(col_color)]:
                equal_visible = True
            else:
                unequal_visible = True
    return equal_visible, unequal_visible


def _sample_color_assignments(
    nh: int,
    nv: int,
    palette: np.ndarray,
    scenario: str,
    rng: np.random.Generator,
) -> tuple[list[int], list[int]] | None:
    for _ in range(ASSIGNMENT_TRIES):
        row_colors = list(map(int, rng.choice(palette, size=nh, replace=True)))
        col_colors = list(map(int, rng.choice(palette, size=nv, replace=True)))
        totals = _stripe_totals(row_colors, col_colors)
        if len(totals) < 2:
            continue

        equal_visible, unequal_visible = _visible_branch_flags(row_colors, col_colors, totals)
        if scenario == "tie" and not equal_visible:
            continue
        if scenario == "unequal" and not unequal_visible:
            continue
        if scenario == "mixed" and not (equal_visible and unequal_visible):
            continue
        return row_colors, col_colors
    return None


def _render_random_input(
    shape: tuple[int, int],
    row_positions: np.ndarray,
    row_colors: list[int],
    col_positions: np.ndarray,
    col_colors: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    inp = np.full(shape, BG, dtype=int)

    for r, color in zip(row_positions, row_colors):
        inp[int(r), :] = int(color)
    for c, color in zip(col_positions, col_colors):
        inp[:, int(c)] = int(color)

    for r, row_color in zip(row_positions, row_colors):
        for c, col_color in zip(col_positions, col_colors):
            if int(row_color) == int(col_color):
                inp[int(r), int(c)] = int(row_color)
            elif int(rng.integers(0, 2)) == 0:
                inp[int(r), int(c)] = int(row_color)
            else:
                inp[int(r), int(c)] = int(col_color)

    return inp


def _gen_attempt() -> np.ndarray | None:
    rng = _RNG
    h = int(rng.integers(*GRID_SIDE_RANGE))
    w = int(rng.integers(*GRID_SIDE_RANGE))

    nh_max = min(MAX_STRIPES_PER_ORIENTATION, (h - 1) // 2)
    nv_max = min(MAX_STRIPES_PER_ORIENTATION, (w - 1) // 2)
    if nh_max < 1 or nv_max < 1:
        return None

    nh = int(rng.integers(1, nh_max + 1))
    nv = int(rng.integers(1, nv_max + 1))
    if nh + nv < 3:
        return None

    scenario = str(rng.choice(SCENARIOS))
    min_colors = 3 if scenario == "mixed" else 2
    max_colors = min(4, nh + nv, len(PALETTE))
    if min_colors > max_colors:
        return None

    palette = rng.choice(PALETTE, size=int(rng.integers(min_colors, max_colors + 1)), replace=False)
    assignments = _sample_color_assignments(nh, nv, palette, scenario, rng)
    if assignments is None:
        return None
    row_colors, col_colors = assignments

    row_positions = _sample_spaced_positions(h, nh, rng)
    col_positions = _sample_spaced_positions(w, nv, rng)
    expected = _render_ranked((h, w), row_positions, row_colors, col_positions, col_colors)
    inp = _render_random_input((h, w), row_positions, row_colors, col_positions, col_colors, rng)

    if np.array_equal(inp, expected):
        return None

    totals = _stripe_totals(row_colors, col_colors)
    equal_visible, unequal_visible = _visible_branch_flags(row_colors, col_colors, totals)
    if scenario == "tie" and not equal_visible:
        return None
    if scenario == "unequal" and not unequal_visible:
        return None
    if scenario == "mixed" and not (equal_visible and unequal_visible):
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid stripe-overlap input")
