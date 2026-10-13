from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
COLORS = tuple(range(1, 10))
STATE_BOTH_EMPTY = 0
STATE_BOTH_FULL = 1
STATE_LEFT_ONLY = 2
STATE_RIGHT_ONLY = 3
PANEL_DIMS = tuple((h, w) for h in range(3, 6) for w in range(3, 6))
GENERATION_ATTEMPTS = 500


def _single_panel_color(panel: np.ndarray) -> int:
    colors = set(map(int, panel.ravel())) - {BACKGROUND}
    assert len(colors) == 1, "each panel must contain exactly one foreground color"
    return int(next(iter(colors)))


def _derive_masks(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    height, width = map(int, g.shape)
    assert width % 2 == 0, "input width must split into two equal panels"

    panel_w = width // 2
    assert height * height <= 30 and panel_w * panel_w <= 30, "expanded output must fit ARC bounds"

    left = g[:, :panel_w]
    right = g[:, panel_w:]
    color_a = _single_panel_color(left)
    color_b = _single_panel_color(right)
    assert color_a != color_b, "panel colors must be distinct"

    assert np.all((left == BACKGROUND) | (left == color_a)), "left panel has stray colors"
    assert np.all((right == BACKGROUND) | (right == color_b)), "right panel has stray colors"
    assert not np.array_equal(left != BACKGROUND, right != BACKGROUND), "panel patterns must differ"

    left_fg = left == color_a
    right_fg = right == color_b
    m0 = (~left_fg) & (~right_fg)
    m1 = left_fg & right_fg
    left_only = left_fg & (~right_fg)
    right_only = (~left_fg) & right_fg

    assert np.any(m0), "common-empty mask must be non-empty"
    assert np.any(m1), "common-filled mask must be non-empty"
    assert np.any(left_only) and np.any(right_only), "both one-sided differences must be present"
    return m0, m1, color_a, color_b


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cross-stamp two derived overlap masks from aligned left and right panels into a block array.

    Input structure:
    - A black-background grid has even width and splits along the center into two equal panels.
    - The left panel contains only one non-black foreground color A, and the right panel
      contains only one different non-black foreground color B.
    - The two aligned panel patterns differ, and their aligned cells include common-empty,
      common-filled, left-only, and right-only cases.

    Rule:
    - Align the two panels cell by cell and derive M0 from cells where both panels are
      black and M1 from cells where both panels contain foreground.
    - Make an output array of panel-height by panel-width slots, with each slot the full
      panel size, so the output shape is (panel_height squared, panel_width squared).
    - For every M0 slot, stamp the M1 shape in color B; for every M1 slot, stamp the
      M0 shape in color A; all left-only and right-only slots remain black.

    Key insight:
    - The visible panel drawings are not the stamps. The stamps and slot selectors are
      the invisible common-empty and common-filled masks produced only after aligning
      the two panels, and their colors are taken from the opposite panel.
    """
    m0, m1, color_a, color_b = _derive_masks(grid)
    panel_h, panel_w = map(int, m0.shape)
    out = np.full((panel_h * panel_h, panel_w * panel_w), BACKGROUND, dtype=int)

    for r in range(panel_h):
        for c in range(panel_w):
            block = out[r * panel_h : (r + 1) * panel_h, c * panel_w : (c + 1) * panel_w]
            if bool(m0[r, c]):
                block[m1] = color_b
            elif bool(m1[r, c]):
                block[m0] = color_a

    return out


def _state_grid_to_input(states: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
    height, width = map(int, states.shape)
    left = np.zeros((height, width), dtype=int)
    right = np.zeros((height, width), dtype=int)

    left[(states == STATE_BOTH_FULL) | (states == STATE_LEFT_ONLY)] = int(color_a)
    right[(states == STATE_BOTH_FULL) | (states == STATE_RIGHT_ONLY)] = int(color_b)
    return np.concatenate([left, right], axis=1)


def _mask_has_spread(mask: np.ndarray) -> bool:
    rows, cols = np.where(mask)
    return len(set(map(int, rows))) >= 2 and len(set(map(int, cols))) >= 2


def _gen_count_plan(cell_count: int) -> tuple[int, int, int, int] | None:
    for _ in range(80):
        k0 = random.randint(2, min(8, cell_count - 5))
        k1 = random.randint(2, min(8, cell_count - k0 - 3))
        remaining = cell_count - k0 - k1
        if remaining < 2:
            continue
        left_only = random.randint(1, remaining - 1)
        right_only = remaining - left_only
        if left_only < 1 or right_only < 1:
            continue

        filled = 2 * k0 * k1
        total_output_cells = cell_count * cell_count
        if filled > int(total_output_cells * 0.45):
            continue
        return k0, k1, left_only, right_only
    return None


def _gen_states(height: int, width: int) -> np.ndarray | None:
    count_plan = _gen_count_plan(height * width)
    if count_plan is None:
        return None

    k0, k1, left_only, right_only = count_plan
    flat = (
        [STATE_BOTH_EMPTY] * k0
        + [STATE_BOTH_FULL] * k1
        + [STATE_LEFT_ONLY] * left_only
        + [STATE_RIGHT_ONLY] * right_only
    )
    random.shuffle(flat)
    states = np.array(flat, dtype=int).reshape(height, width)

    m0 = states == STATE_BOTH_EMPTY
    m1 = states == STATE_BOTH_FULL
    left = (states == STATE_BOTH_FULL) | (states == STATE_LEFT_ONLY)
    right = (states == STATE_BOTH_FULL) | (states == STATE_RIGHT_ONLY)

    if not (_mask_has_spread(m0) and _mask_has_spread(m1)):
        return None
    if np.array_equal(left, right):
        return None
    if np.array_equal(m0, np.flipud(m1)) or np.array_equal(m0, np.fliplr(m1)):
        return None
    return states


def _expected_from_states(states: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
    height, width = map(int, states.shape)
    m0 = states == STATE_BOTH_EMPTY
    m1 = states == STATE_BOTH_FULL
    out = np.zeros((height * height, width * width), dtype=int)

    for r, c in np.argwhere(m0):
        rr = int(r) * height
        cc = int(c) * width
        out[rr : rr + height, cc : cc + width][m1] = int(color_b)
    for r, c in np.argwhere(m1):
        rr = int(r) * height
        cc = int(c) * width
        out[rr : rr + height, cc : cc + width][m0] = int(color_a)
    return out


def _gen_attempt() -> np.ndarray | None:
    height, width = random.choice(PANEL_DIMS)
    states = _gen_states(height, width)
    if states is None:
        return None

    color_a, color_b = random.sample(COLORS, 2)
    inp = _state_grid_to_input(states, color_a, color_b)
    expected = _expected_from_states(states, color_a, color_b)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if inp.shape[0] > 30 or inp.shape[1] > 30 or expected.shape[0] > 30 or expected.shape[1] > 30:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dual-panel derived-mask input")
