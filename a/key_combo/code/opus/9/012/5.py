from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
PANEL_HEIGHT = 3
INPUT_HEIGHT = 3 * PANEL_HEIGHT
COLOR_POOL = (1, 2, 3, 4, 7)
LAYER_ORDER = ("top", "middle", "bottom")

WIDTH_RANGE = (10, 18)
FINAL_COLOR_COUNT_RANGE = (3, 5)
GENERATION_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _composite(top: np.ndarray, middle: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    out = middle.copy()
    top_mask = top != BACKGROUND
    bottom_mask = bottom != BACKGROUND
    out[top_mask] = top[top_mask]
    out[bottom_mask] = bottom[bottom_mask]
    return out


def _count_colors(grid: np.ndarray) -> Counter[int]:
    values = [int(v) for v in grid.flat if int(v) != BACKGROUND]
    return Counter(values)


def _bars_from_counts(counts: Counter[int]) -> np.ndarray:
    assert counts, "expected at least one non-background color"
    ordered = sorted(counts.items(), key=lambda item: item[1])
    out = np.full((len(ordered), max(counts.values())), BACKGROUND, dtype=int)
    for r, (color, count) in enumerate(ordered):
        out[r, :count] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite three stacked panels and convert the resulting color frequencies into sorted horizontal bars.

    Input structure:
    - The grid is 9 rows high and is split into three consecutive 3-row panels: top, middle, and bottom.
    - The background is black.
    - Each panel contains non-black pixels using one or two colors chosen from blue, red, green, yellow, and orange.

    Rule:
    - Treat the middle panel as the base 3xW image.
    - Overlay every non-black pixel from the top panel onto the same relative position.
    - Overlay every non-black pixel from the bottom panel last, so the bottom panel has highest priority.
    - Count the non-black colors in the composed 3xW result.
    - Sort the colors by increasing pixel count and output one left-aligned bar per color, padded with black to the maximum count.

    Key insight:
    - The histogram must be computed after resolving layer conflicts by panel priority; counting the raw input panels directly gives the wrong frequencies.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] == INPUT_HEIGHT, "expected exactly three 3-row panels"
    assert 1 <= g.shape[1] <= 30, "invalid width"

    bg = _get_bg(g)
    assert bg == BACKGROUND, "expected black background"

    top = g[:PANEL_HEIGHT, :]
    middle = g[PANEL_HEIGHT : 2 * PANEL_HEIGHT, :]
    bottom = g[2 * PANEL_HEIGHT :, :]

    for panel in (top, middle, bottom):
        colors = _colors_present(panel, exclude={BACKGROUND})
        assert 1 <= len(colors) <= 2, "each panel must use one or two foreground colors"
        assert all(color in COLOR_POOL for color in colors), "unexpected panel color"

    composed = _composite(top, middle, bottom)
    counts = _count_colors(composed)
    assert len(counts) >= 3, "expected at least three colors after composition"
    assert len(set(counts.values())) == len(counts), "color frequencies must be unique"
    return _bars_from_counts(counts)


def _sample_group_sizes(n_colors: int) -> list[int]:
    if n_colors == 3:
        return [1, 1, 1]
    if n_colors == 4:
        sizes = [2, 1, 1]
        random.shuffle(sizes)
        return sizes
    if n_colors == 5:
        sizes = [2, 2, 1]
        random.shuffle(sizes)
        return sizes
    raise AssertionError("unsupported color count")


def _assign_visible_colors(colors: list[int]) -> tuple[dict[str, tuple[int, ...]], dict[int, str]]:
    shuffled = colors[:]
    random.shuffle(shuffled)
    sizes = _sample_group_sizes(len(colors))
    layers = list(LAYER_ORDER)
    random.shuffle(layers)

    groups: dict[str, tuple[int, ...]] = {}
    source_layer: dict[int, str] = {}
    start = 0
    for layer, size in zip(layers, sizes):
        block = tuple(shuffled[start : start + size])
        start += size
        groups[layer] = block
        for color in block:
            source_layer[color] = layer
    assert set(groups) == set(LAYER_ORDER)
    return groups, source_layer


def _extend_panel_sets(groups: dict[str, tuple[int, ...]], colors: list[int]) -> dict[str, tuple[int, ...]]:
    panel_sets = {layer: set(group) for layer, group in groups.items()}
    singletons = [layer for layer in LAYER_ORDER if len(panel_sets[layer]) == 1]
    random.shuffle(singletons)

    if singletons and random.random() < 0.7:
        share_count = random.randint(1, len(singletons))
        for layer in singletons[:share_count]:
            candidates = [color for color in colors if color not in panel_sets[layer]]
            if candidates:
                panel_sets[layer].add(random.choice(candidates))

    return {layer: tuple(sorted(panel_sets[layer])) for layer in LAYER_ORDER}


def _sample_final_counts(n_colors: int, total_cells: int) -> list[int]:
    target_lo = max(n_colors + 4, total_cells // 3)
    target_hi = total_cells - 4
    upper = min(14, total_cells - (n_colors - 1))
    for _ in range(300):
        counts = sorted(random.sample(range(1, upper + 1), n_colors))
        total = sum(counts)
        if target_lo <= total <= target_hi:
            return counts
    raise AssertionError("failed to sample distinct final counts")


def _pick_hidden_color(choices: tuple[int, ...], forbidden: int) -> int:
    candidates = [color for color in choices if color != forbidden]
    return int(random.choice(candidates or list(choices)))


def _place_final_grid(
    width: int,
    colors: list[int],
    counts: list[int],
    source_layer: dict[int, str],
) -> tuple[np.ndarray, dict[tuple[int, int], int], dict[tuple[int, int], str]]:
    cells = [(r, c) for r in range(PANEL_HEIGHT) for c in range(width)]
    random.shuffle(cells)

    final_grid = np.zeros((PANEL_HEIGHT, width), dtype=int)
    cell_color: dict[tuple[int, int], int] = {}
    cell_source: dict[tuple[int, int], str] = {}

    cursor = 0
    items = list(zip(colors, counts))
    random.shuffle(items)
    for color, count in items:
        for cell in cells[cursor : cursor + count]:
            final_grid[cell] = int(color)
            cell_color[cell] = int(color)
            cell_source[cell] = source_layer[int(color)]
        cursor += count

    return final_grid, cell_color, cell_source


def _build_panels(
    width: int,
    panel_sets: dict[str, tuple[int, ...]],
    cell_color: dict[tuple[int, int], int],
    cell_source: dict[tuple[int, int], str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    top = np.zeros((PANEL_HEIGHT, width), dtype=int)
    middle = np.zeros((PANEL_HEIGHT, width), dtype=int)
    bottom = np.zeros((PANEL_HEIGHT, width), dtype=int)
    layer_grids = {"top": top, "middle": middle, "bottom": bottom}

    for cell, color in cell_color.items():
        layer_grids[cell_source[cell]][cell] = int(color)

    top_visible = [cell for cell, layer in cell_source.items() if layer == "top"]
    bottom_visible = [cell for cell, layer in cell_source.items() if layer == "bottom"]

    top_override_count = random.randint(1, len(top_visible))
    for cell in random.sample(top_visible, top_override_count):
        middle[cell] = _pick_hidden_color(panel_sets["middle"], cell_color[cell])

    bottom_override_count = random.randint(1, len(bottom_visible))
    for cell in random.sample(bottom_visible, bottom_override_count):
        mode = random.choice(("top", "middle", "both"))
        if mode in ("top", "both"):
            top[cell] = _pick_hidden_color(panel_sets["top"], cell_color[cell])
        if mode in ("middle", "both"):
            middle[cell] = _pick_hidden_color(panel_sets["middle"], cell_color[cell])
        if top[cell] == BACKGROUND and middle[cell] == BACKGROUND:
            middle[cell] = _pick_hidden_color(panel_sets["middle"], cell_color[cell])

    return top, middle, bottom


def _expected_output(grid: np.ndarray) -> np.ndarray:
    top = grid[:PANEL_HEIGHT, :]
    middle = grid[PANEL_HEIGHT : 2 * PANEL_HEIGHT, :]
    bottom = grid[2 * PANEL_HEIGHT :, :]
    composed = _composite(top, middle, bottom)
    return _bars_from_counts(_count_colors(composed))


def _is_semantically_valid(
    grid: np.ndarray,
    groups: dict[str, tuple[int, ...]],
    expected: np.ndarray,
) -> bool:
    if grid.shape[0] != INPUT_HEIGHT or max(grid.shape) > 30:
        return False

    top = grid[:PANEL_HEIGHT, :]
    middle = grid[PANEL_HEIGHT : 2 * PANEL_HEIGHT, :]
    bottom = grid[2 * PANEL_HEIGHT :, :]
    panels = {"top": top, "middle": middle, "bottom": bottom}

    for layer in LAYER_ORDER:
        colors = _colors_present(panels[layer], exclude={BACKGROUND})
        if not (1 <= len(colors) <= 2):
            return False
        if any(color not in COLOR_POOL for color in colors):
            return False
        if not set(colors).issubset(set(groups[layer]) | set(groups[layer])):
            return False

    middle_only = middle.copy()
    top_over_middle = _composite(top, middle, np.zeros_like(bottom))
    composed = _composite(top, middle, bottom)
    final_counts = _count_colors(composed)
    raw_counts = _count_colors(grid)

    if len(final_counts) < 3 or len(set(final_counts.values())) != len(final_counts):
        return False
    if np.array_equal(middle_only, top_over_middle):
        return False
    if np.array_equal(top_over_middle, composed):
        return False
    if np.count_nonzero(grid) <= np.count_nonzero(composed):
        return False
    if sum(raw_counts[color] != final_counts.get(color, 0) for color in final_counts) < 2:
        return False

    top_override = np.any((top != BACKGROUND) & (middle != BACKGROUND) & (top != middle) & (bottom == BACKGROUND))
    bottom_override = np.any((bottom != BACKGROUND) & (((top != BACKGROUND) & (top != bottom)) | ((middle != BACKGROUND) & (middle != bottom))))
    middle_visible = np.any((middle != BACKGROUND) & (top == BACKGROUND) & (bottom == BACKGROUND))
    top_visible = np.any((top != BACKGROUND) & (bottom == BACKGROUND))
    bottom_visible = np.any(bottom != BACKGROUND)
    if not (top_override and bottom_override and middle_visible and top_visible and bottom_visible):
        return False

    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _generate_case() -> tuple[np.ndarray, np.ndarray, dict[str, tuple[int, ...]]] | None:
    width = random.randint(*WIDTH_RANGE)
    n_colors = random.randint(*FINAL_COLOR_COUNT_RANGE)
    colors = random.sample(list(COLOR_POOL), n_colors)
    groups, source_layer = _assign_visible_colors(colors)
    panel_sets = _extend_panel_sets(groups, colors)
    counts = _sample_final_counts(n_colors, PANEL_HEIGHT * width)

    _, cell_color, cell_source = _place_final_grid(width, colors, counts, source_layer)
    top, middle, bottom = _build_panels(width, panel_sets, cell_color, cell_source)
    grid = np.vstack([top, middle, bottom])
    expected = _expected_output(grid)
    return grid, expected, panel_sets


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        case = _generate_case()
        if case is None:
            continue
        grid, expected, panel_sets = case
        if _is_semantically_valid(grid, panel_sets, expected):
            return grid
    raise RuntimeError("failed to generate a valid layered-frequency ARC input")
