from __future__ import annotations

import random

import numpy as np


BG = 0
ARC_COLORS = tuple(range(1, 10))
GEN_PANEL_H_RANGE = (4, 9)
GEN_WIDTH_RANGE = (6, 16)
GEN_ATTEMPTS = 300


def _panel_slices(height: int) -> tuple[slice, slice, slice, int, int]:
    """Return the three panel row slices and the two divider row indices."""
    assert (int(height) - 2) % 3 == 0, "height must be 3 panel heights plus two dividers"
    panel_h = (int(height) - 2) // 3
    assert panel_h > 0, "panels must be non-empty"
    d0 = panel_h
    d1 = 2 * panel_h + 1
    return (
        slice(0, panel_h),
        slice(panel_h + 1, 2 * panel_h + 1),
        slice(2 * panel_h + 2, 3 * panel_h + 2),
        d0,
        d1,
    )


def _is_solid_nonblack_row(row: np.ndarray) -> bool:
    return bool(row.size and int(row[0]) != BG and np.all(row == int(row[0])))


def _panel_color(panel: np.ndarray) -> int:
    colors = sorted(int(c) for c in np.unique(panel) if int(c) != BG)
    assert len(colors) == 1, "each panel must have exactly one foreground color"
    return int(colors[0])


def _parse_panels(grid: np.ndarray) -> tuple[list[np.ndarray], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be two-dimensional"
    h, w = map(int, g.shape)
    assert h <= 30 and w <= 30, "ARC grids must fit within 30x30"

    s0, s1, s2, d0, d1 = _panel_slices(h)
    assert _is_solid_nonblack_row(g[d0, :]), "first divider row must be full-width"
    assert _is_solid_nonblack_row(g[d1, :]), "second divider row must be full-width"

    panels = [g[s0, :], g[s1, :], g[s2, :]]
    colors = [_panel_color(panel) for panel in panels]
    assert len(set(colors)) == 3, "the three panel foreground colors must be distinct"
    return panels, colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Vote across three stacked panels and paint each cell with the unique odd panel's color.

    Input structure:
    - A black-background grid is split by two full-width divider rows into three
      equal-height panels with the same width.
    - Each panel contains black plus exactly one foreground color, and the three
      panel foreground colors are all distinct.

    Rule:
    - Remove the divider rows and compare the three panels at matching panel-local
      coordinates.
    - If exactly one panel has foreground at a coordinate, output that panel's color.
    - If exactly two panels have foreground at a coordinate, output the missing
      panel's color.
    - If all three panels agree, either all foreground or all background, output black.

    Key insight:
    - The result is not a union, intersection, XOR, or majority vote: the colored
      signature belongs to the one panel whose on/off state differs from the other two.
    """
    panels, colors = _parse_panels(grid)
    masks = [panel == color for panel, color in zip(panels, colors)]
    stack = np.stack(masks, axis=0)
    counts = np.sum(stack, axis=0)

    out = np.full(panels[0].shape, BG, dtype=int)
    for idx, color in enumerate(colors):
        is_unique_lit = (counts == 1) & stack[idx]
        is_unique_dark = (counts == 2) & (~stack[idx])
        out[is_unique_lit | is_unique_dark] = int(color)
    return out


# --------------- generator ---------------


def _codes_from_panels(panels: list[np.ndarray], colors: list[int]) -> np.ndarray:
    codes = np.zeros(panels[0].shape, dtype=int)
    for idx, (panel, color) in enumerate(zip(panels, colors)):
        codes[panel == int(color)] |= 1 << idx
    return codes


def _oracle_from_codes(codes: np.ndarray, colors: list[int]) -> np.ndarray:
    out = np.full(codes.shape, BG, dtype=int)
    for idx, color in enumerate(colors):
        out[codes == (1 << idx)] = int(color)
        out[codes == (7 ^ (1 << idx))] = int(color)
    return out


def _sample_code_grid(panel_h: int, width: int) -> np.ndarray:
    """Build sparse per-cell 3-bit membership codes, forcing every semantic branch."""
    cells = [(r, c) for r in range(panel_h) for c in range(width)]
    random.shuffle(cells)
    required = list(range(8))
    random.shuffle(required)

    codes = np.zeros((panel_h, width), dtype=int)
    for code, (r, c) in zip(required, cells):
        codes[r, c] = int(code)

    area = panel_h * width
    max_pattern_cells = max(len(required), min(area, round(area * 0.28)))
    target_pattern_cells = random.randint(len(required), int(max_pattern_cells))
    remaining = cells[len(required):]
    random.shuffle(remaining)

    extra_choices = (1, 2, 3, 4, 5, 6, 7)
    extra_weights = (3, 3, 2, 3, 2, 2, 1)
    for r, c in remaining[: target_pattern_cells - len(required)]:
        codes[r, c] = random.choices(extra_choices, weights=extra_weights, k=1)[0]
    return codes


def _render_input(codes: np.ndarray, colors: list[int], divider_color: int) -> np.ndarray:
    panel_h, width = map(int, codes.shape)
    height = 3 * panel_h + 2
    grid = np.full((height, width), BG, dtype=int)

    first_divider = panel_h
    second_divider = 2 * panel_h + 1
    grid[first_divider, :] = int(divider_color)
    grid[second_divider, :] = int(divider_color)

    panel_starts = (0, panel_h + 1, 2 * panel_h + 2)
    for idx, color in enumerate(colors):
        panel = grid[panel_starts[idx] : panel_starts[idx] + panel_h, :]
        panel[(codes & (1 << idx)) != 0] = int(color)
    return grid


def _has_no_confusing_full_panel_line(codes: np.ndarray) -> bool:
    for idx in range(3):
        mask = (codes & (1 << idx)) != 0
        if np.any(np.all(mask, axis=1)):
            return False
    return True


def _semantic_checks(grid: np.ndarray, codes: np.ndarray, colors: list[int]) -> bool:
    """Reject samples that fail the visible rule coverage, uniqueness, or oracle checks."""
    if max(grid.shape) > 30:
        return False
    if not _has_no_confusing_full_panel_line(codes):
        return False

    present_codes = set(int(x) for x in np.unique(codes))
    if present_codes != set(range(8)):
        return False

    expected = _oracle_from_codes(codes, colors)
    nonblack = int(np.count_nonzero(expected))
    area = int(expected.size)
    if nonblack < 6 or nonblack > max(6, round(area * 0.55)):
        return False
    if len(set(int(x) for x in expected.ravel() if int(x) != BG)) != 3:
        return False

    for idx, color in enumerate(colors):
        unique_lit = codes == (1 << idx)
        unique_dark = codes == (7 ^ (1 << idx))
        if not (bool(np.any(unique_lit)) and bool(np.any(unique_dark))):
            return False
        if not (
            bool(np.any(expected[unique_lit] == int(color)))
            and bool(np.any(expected[unique_dark] == int(color)))
        ):
            return False

    try:
        panels, parsed_colors = _parse_panels(grid)
        parsed_codes = _codes_from_panels(panels, parsed_colors)
        actual = main(grid)
    except AssertionError:
        return False

    return (
        parsed_colors == colors
        and np.array_equal(parsed_codes, codes)
        and np.array_equal(actual, expected)
    )


def _gen_attempt() -> np.ndarray | None:
    panel_h = random.randint(*GEN_PANEL_H_RANGE)
    width = random.randint(*GEN_WIDTH_RANGE)

    divider_color = random.choice(ARC_COLORS)
    colors = random.sample([c for c in ARC_COLORS if c != divider_color], 3)
    codes = _sample_code_grid(panel_h, width)
    grid = _render_input(codes, colors, divider_color)

    if _semantic_checks(grid, codes, colors):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid odd-one-out triple-panel input")
