from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BACKGROUND = 0
PALETTE = list(range(1, 10))
GEN_RETRIES = 300
PANEL_COUNT_RANGE = (2, 4)
EXTRA_ZONE_RANGE = (1, 3)


def _panel_count_from_colors(grid: np.ndarray) -> int:
    colors = [int(color) for color in np.unique(grid) if int(color) != BACKGROUND]
    return len(colors)


def _extract_panels(grid: np.ndarray) -> tuple[list[np.ndarray], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    panel_count = _panel_count_from_colors(g)
    assert 2 <= panel_count <= 4, "expected 2 to 4 distinct panel colors"

    total_h, width = map(int, g.shape)
    usable_h = total_h - (panel_count - 1)
    assert usable_h > 0 and usable_h % panel_count == 0, "stack height is incompatible with equal-size panels"
    panel_h = usable_h // panel_count

    panels: list[np.ndarray] = []
    panel_colors: list[int] = []
    for idx in range(panel_count):
        r0 = idx * (panel_h + 1)
        r1 = r0 + panel_h
        panel = g[r0:r1, :]
        assert panel.shape == (panel_h, width), "panel crop has inconsistent shape"
        colors = [int(color) for color in np.unique(panel) if int(color) != BACKGROUND]
        assert len(colors) == 1, "each panel must contain exactly one foreground color"
        panels.append(panel)
        panel_colors.append(colors[0])
        if idx < panel_count - 1:
            sep_row = g[r1, :]
            assert np.all(sep_row == BACKGROUND), "panels must be separated by a single black row"

    assert len(set(panel_colors)) == panel_count, "panel foreground colors must be unique"
    return panels, panel_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay aligned panel masks and recolor each cell by coverage depth.

    Input structure:
    - The grid is a vertical stack of 2 to 4 equal-size panels with a single black separator row between adjacent panels.
    - Each panel is a black-background mask that uses exactly one unique foreground color.
    - Panels are compared cell-by-cell after aligning them to the same top-left corner.

    Rule:
    - Split the stack into its equal-size panels and record their foreground colors from top to bottom.
    - For each output cell, count how many panel masks cover that aligned position.
    - Leave uncovered cells black; if the coverage count is k >= 1, color that cell with the foreground color of the k-th panel from the top.

    Key insight:
    - The visible color depends only on the number of covering layers, so coverage depth acts like a pointer to a panel index rather than letting the topmost mask win.
    """
    panels, panel_colors = _extract_panels(grid)
    panel_h, panel_w = map(int, panels[0].shape)

    coverage = np.zeros((panel_h, panel_w), dtype=int)
    for panel in panels:
        coverage += (panel != BACKGROUND).astype(int)

    out = np.zeros((panel_h, panel_w), dtype=int)
    for depth, color in enumerate(panel_colors, start=1):
        out[coverage == depth] = int(color)
    return out


def _shape_mask(kind: str, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if kind == "rect":
        mask[:, :] = True
    elif kind == "hbar":
        row = random.randrange(height)
        mask[row, :] = True
    elif kind == "vbar":
        col = random.randrange(width)
        mask[:, col] = True
    elif kind == "plus":
        mask[height // 2, :] = True
        mask[:, width // 2] = True
    elif kind == "L":
        orientation = random.choice(("tl", "tr", "bl", "br"))
        if orientation == "tl":
            mask[0, :] = True
            mask[:, 0] = True
        elif orientation == "tr":
            mask[0, :] = True
            mask[:, -1] = True
        elif orientation == "bl":
            mask[-1, :] = True
            mask[:, 0] = True
        else:
            mask[-1, :] = True
            mask[:, -1] = True
    else:
        raise ValueError(f"unknown shape kind: {kind}")

    assert np.any(mask), "shape mask must be non-empty"
    return mask


def _zone_spec(panel_count: int) -> list[tuple[int, ...]]:
    specs: list[tuple[int, ...]] = []

    lower_only = random.randint(1, panel_count - 1)
    specs.append((lower_only,))

    if panel_count >= 3:
        prefix = tuple(range(2))
        pair_choices = [combo for combo in combinations(range(panel_count), 2) if combo != prefix]
        specs.append(random.choice(pair_choices or list(combinations(range(panel_count), 2))))
    else:
        specs.append((0, 1))

    for depth in range(3, panel_count + 1):
        combos = list(combinations(range(panel_count), depth))
        specs.append(random.choice(combos))

    if not any(0 in spec and len(spec) == 1 for spec in specs):
        specs.append((0,))

    extra_count = random.randint(*EXTRA_ZONE_RANGE)
    for _ in range(extra_count):
        depth = random.randint(1, panel_count)
        combos = list(combinations(range(panel_count), depth))
        specs.append(random.choice(combos))

    covered_panels = {idx for spec in specs for idx in spec}
    for idx in range(panel_count):
        if idx not in covered_panels:
            specs.append((idx,))

    random.shuffle(specs)
    return specs


def _sample_zone_shape(panel_h: int, panel_w: int) -> tuple[np.ndarray, int, int]:
    choices: list[tuple[str, int, int]] = []

    for height in range(1, min(4, panel_h) + 1):
        for width in range(1, min(5, panel_w) + 1):
            choices.append(("rect", height, width))
            if width >= 2:
                choices.append(("hbar", height, width))
            if height >= 2:
                choices.append(("vbar", height, width))
            if height >= 2 and width >= 2:
                choices.append(("L", height, width))
            if height >= 3 and width >= 3:
                choices.append(("plus", height, width))

    kind, height, width = random.choice(choices)
    return _shape_mask(kind, height, width), height, width


def _can_place(occupied: np.ndarray, r0: int, c0: int, height: int, width: int) -> bool:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(occupied.shape[0], r0 + height + 1)
    cc1 = min(occupied.shape[1], c0 + width + 1)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _mark_occupied(occupied: np.ndarray, r0: int, c0: int, height: int, width: int) -> None:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(occupied.shape[0], r0 + height + 1)
    cc1 = min(occupied.shape[1], c0 + width + 1)
    occupied[rr0:rr1, cc0:cc1] = True


def _place_zones(panel_h: int, panel_w: int, specs: list[tuple[int, ...]]) -> list[tuple[np.ndarray, tuple[int, ...], int, int]]:
    occupied = np.zeros((panel_h, panel_w), dtype=bool)
    placed: list[tuple[np.ndarray, tuple[int, ...], int, int]] = []

    order = list(specs)
    random.shuffle(order)
    for spec in sorted(order, key=len, reverse=True):
        shape, height, width = _sample_zone_shape(panel_h, panel_w)
        candidates = [
            (r0, c0)
            for r0 in range(panel_h - height + 1)
            for c0 in range(panel_w - width + 1)
            if _can_place(occupied, r0, c0, height, width)
        ]
        if not candidates:
            raise ValueError("failed to place a zone")
        r0, c0 = random.choice(candidates)
        _mark_occupied(occupied, r0, c0, height, width)
        placed.append((shape, spec, r0, c0))

    return placed


def _build_masks(panel_count: int, panel_h: int, panel_w: int) -> list[np.ndarray]:
    masks = [np.zeros((panel_h, panel_w), dtype=bool) for _ in range(panel_count)]
    specs = _zone_spec(panel_count)
    zones = _place_zones(panel_h, panel_w, specs)

    for shape, spec, r0, c0 in zones:
        height, width = map(int, shape.shape)
        for panel_idx in spec:
            window = masks[panel_idx][r0 : r0 + height, c0 : c0 + width]
            window |= shape

    return masks


def _stack_panels(panel_masks: list[np.ndarray], panel_colors: list[int]) -> np.ndarray:
    panel_h, panel_w = map(int, panel_masks[0].shape)
    total_h = len(panel_masks) * panel_h + (len(panel_masks) - 1)
    grid = np.zeros((total_h, panel_w), dtype=int)

    for idx, (mask, color) in enumerate(zip(panel_masks, panel_colors)):
        r0 = idx * (panel_h + 1)
        panel = np.zeros((panel_h, panel_w), dtype=int)
        panel[mask] = int(color)
        grid[r0 : r0 + panel_h, :] = panel

    return grid


def _expected_output(panel_masks: list[np.ndarray], panel_colors: list[int]) -> np.ndarray:
    coverage = np.zeros_like(panel_masks[0], dtype=int)
    for mask in panel_masks:
        coverage += mask.astype(int)

    out = np.zeros_like(coverage)
    for depth, color in enumerate(panel_colors, start=1):
        out[coverage == depth] = int(color)
    return out


def _generator_attempt() -> np.ndarray | None:
    panel_count = random.randint(*PANEL_COUNT_RANGE)
    max_panel_h = (30 - (panel_count - 1)) // panel_count
    panel_h = random.randint(5, min(10, max_panel_h))
    panel_w = random.randint(max(9, panel_count * 3), 16)
    panel_colors = random.sample(PALETTE, panel_count)

    try:
        panel_masks = _build_masks(panel_count, panel_h, panel_w)
    except ValueError:
        return None

    if not all(mask.any() for mask in panel_masks):
        return None

    coverage = np.zeros((panel_h, panel_w), dtype=int)
    for mask in panel_masks:
        coverage += mask.astype(int)

    if not np.any(coverage == 0):
        return None
    if not np.any(coverage == panel_count):
        return None

    lower_only_cells = np.zeros((panel_h, panel_w), dtype=bool)
    for idx in range(1, panel_count):
        lower_only_cells |= panel_masks[idx]
    lower_only_cells &= ~panel_masks[0]
    lower_only_cells &= coverage == 1
    if not np.any(lower_only_cells):
        return None

    grid = _stack_panels(panel_masks, panel_colors)
    expected = _expected_output(panel_masks, panel_colors)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generator_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
