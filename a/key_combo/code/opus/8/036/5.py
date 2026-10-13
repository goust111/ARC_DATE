import itertools
import random

import numpy as np


BLACK = 0
GREY = 5
NON_GREY_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 300


def _split_panels(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    H, W = g.shape
    separators = [c for c in range(W) if np.all(g[:, c] == GREY)]
    assert len(separators) == 3, "expected exactly three grey separator columns"

    widths = [
        separators[0],
        separators[1] - separators[0] - 1,
        separators[2] - separators[1] - 1,
        W - separators[2] - 1,
    ]
    assert len(set(widths)) == 1 and widths[0] >= 1, "panels must have equal positive width"
    panel_w = widths[0]

    expected = [panel_w, 2 * panel_w + 1, 3 * panel_w + 2]
    assert separators == expected, "separator placement must evenly split four panels"

    panels = []
    start = 0
    for i in range(4):
        panels.append(g[:, start : start + panel_w].copy())
        start += panel_w
        if i < 3:
            start += 1
    return panels


def _seed_columns(panel: np.ndarray) -> list[tuple[int, int]]:
    H, W = panel.shape
    seeds = [(c, int(panel[H - 1, c])) for c in range(W) if int(panel[H - 1, c]) != BLACK]
    assert 2 <= len(seeds) <= 3, "expected 2-3 bottom-row seeds"
    colors = [color for _c, color in seeds]
    assert len(set(colors)) == len(colors), "seed colors must be distinct"
    assert all(color != GREY for color in colors), "grey is reserved for separators"
    return seeds


def _band_height(panel: np.ndarray, col: int, color: int) -> int:
    H = panel.shape[0]
    assert int(panel[H - 1, col]) == color
    r = H - 1
    while r >= 0 and int(panel[r, col]) == color:
        r -= 1
    height = H - 1 - r
    assert height >= 1
    if r >= 0:
        assert np.all(panel[: r + 1, col] == BLACK), "band must be continuous from the bottom"
    return height


def _validate_panel_contents(panel: np.ndarray, seeds: list[tuple[int, int]]) -> None:
    H, W = panel.shape
    seed_map = dict(seeds)
    occupied_cols = set(seed_map)
    for c in range(W):
        column = panel[:, c]
        if c not in occupied_cols:
            assert np.all(column == BLACK), "unseeded columns must stay black"
            continue
        color = seed_map[c]
        assert int(column[H - 1]) == color
        height = _band_height(panel, c, color)
        mask = np.zeros(H, dtype=bool)
        mask[H - height :] = True
        assert np.all(column[mask] == color), "filled band cells must match the seed color"
        assert np.all(column[~mask] == BLACK), "cells above the band must stay black"


def _solve_heights(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    panels = _split_panels(grid)
    first_three = panels[:3]
    template = panels[3]

    seeds = _seed_columns(template)
    for panel in first_three:
        assert _seed_columns(panel) == seeds, "all panels must share the same seeds"

    for panel in panels:
        _validate_panel_contents(panel, seeds)

    predicted = []
    H, W = template.shape
    out = np.zeros((H, W), dtype=int)
    for col, color in seeds:
        assert _band_height(template, col, color) == 1, "the fourth panel must be an empty template"
        heights = [_band_height(panel, col, color) for panel in first_three]
        d1 = heights[1] - heights[0]
        d2 = heights[2] - heights[1]
        assert d1 >= 1 and d1 == d2, "heights must grow by a constant positive increment"
        next_height = heights[2] + d1
        assert next_height <= H, "predicted band must fit inside the panel"
        out[H - next_height :, col] = color
        predicted.append((col, color, next_height))

    return out, predicted


def main(grid: np.ndarray) -> np.ndarray:
    """Predict the next panel by extrapolating each colored vertical band's growth rate.

    Input structure:
    - The grid is split into four equal-width horizontal panels by three full-height grey separator columns.
    - In every panel, the background is black and the bottom row contains 2-3 colored seed pixels in fixed columns.
    - In the first three panels, each seed has a same-color vertical band growing upward from the bottom; in the fourth panel only the bottom seeds remain.

    Rule:
    - Measure each seed column's contiguous same-color band height in panels 1, 2, and 3, counting the seed itself.
    - For each color, verify the height increases by the same positive amount from panel 1 to 2 and from panel 2 to 3.
    - Extrapolate one more step to get the panel-4 height, then draw that band from the bottom upward in the output panel.

    Key insight:
    - The output heights are determined by per-color temporal growth rates across the first three frames, not by propagating existing pixels inside the blank template panel.
    """
    out, _ = _solve_heights(np.asarray(grid, dtype=int))
    return out


def _choose_columns(panel_w: int, count: int, rng: random.Random) -> list[int]:
    candidates = []
    cols = range(panel_w)
    for choice in itertools.combinations(cols, count):
        if all(b - a >= 2 for a, b in zip(choice, choice[1:])):
            candidates.append(choice)
    assert candidates, "panel width must allow spaced seed columns"
    return list(rng.choice(candidates))


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray, dict]:
    H = rng.randint(8, 12)
    panel_w = rng.randint(5, 6)
    seed_count = rng.choice((2, 3))
    columns = _choose_columns(panel_w, seed_count, rng)
    colors = rng.sample(NON_GREY_COLORS, seed_count)

    max_increment = min(3, (H - 1) // 3)
    assert max_increment >= 2
    high_increment = rng.randint(2, max_increment)

    increments = [1, high_increment]
    if seed_count == 3:
        increments.append(rng.choice((1, high_increment)))
    rng.shuffle(increments)

    reach_top = rng.random() < 0.35
    top_index = rng.randrange(seed_count) if reach_top else None

    starts = []
    heights4 = []
    for i, inc in enumerate(increments):
        min_start = 1
        max_start = H - 3 * inc
        assert max_start >= min_start
        if i == top_index:
            start = max_start
        else:
            start = rng.randint(min_start, max_start)
        starts.append(start)
        heights4.append(start + 3 * inc)

    if len(set(increments)) < 2:
        raise AssertionError("need at least two distinct growth rates")
    if max(increments) == 1:
        raise AssertionError("need a color that grows by more than one cell")
    if len(set(heights4)) < 2:
        raise AssertionError("predicted heights must not all collapse to the same value")

    panels = [np.zeros((H, panel_w), dtype=int) for _ in range(4)]
    for frame in range(3):
        for col, color, start, inc in zip(columns, colors, starts, increments):
            height = start + frame * inc
            panels[frame][H - height :, col] = color
    for col, color in zip(columns, colors):
        panels[3][H - 1, col] = color

    grid = np.zeros((H, panel_w * 4 + 3), dtype=int)
    for i, panel in enumerate(panels):
        start = i * (panel_w + 1)
        grid[:, start : start + panel_w] = panel
        if i < 3:
            grid[:, start + panel_w] = GREY

    out = np.zeros((H, panel_w), dtype=int)
    for col, color, height in zip(columns, colors, heights4):
        out[H - height :, col] = color

    meta = {
        "H": H,
        "panel_w": panel_w,
        "columns": tuple(columns),
        "colors": tuple(colors),
        "starts": tuple(starts),
        "increments": tuple(increments),
        "heights4": tuple(heights4),
    }
    return grid, out, meta


def _validate_generated(grid: np.ndarray, expected: np.ndarray, meta: dict) -> None:
    solved, predicted = _solve_heights(grid)
    assert np.array_equal(solved, expected), "generated case does not match solver"

    heights4 = [height for _col, _color, height in predicted]
    assert tuple(heights4) == meta["heights4"]
    assert len(set(meta["increments"])) >= 2
    assert max(meta["increments"]) >= 2
    assert len(set(meta["heights4"])) >= 2

    H = meta["H"]
    for start, inc in zip(meta["starts"], meta["increments"]):
        h1 = start
        h2 = start + inc
        h3 = start + 2 * inc
        h4 = start + 3 * inc
        assert 1 <= h1 < h2 < h3 < h4 <= H

    seed_cols = meta["columns"]
    assert all(b - a >= 2 for a, b in zip(seed_cols, seed_cols[1:]))
    assert expected.shape == (meta["H"], meta["panel_w"])


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            grid, expected, meta = _build_case(rng)
            _validate_generated(grid, expected, meta)
            return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input")
