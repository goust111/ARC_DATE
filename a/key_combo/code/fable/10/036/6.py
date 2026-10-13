from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
GEN_RETRIES = 400
PANEL_H_RANGE = (4, 8)
PANEL_W_RANGE = (4, 8)
VISIBLE_FILL_RANGE = (0.45, 0.75)

_CROSS_KEYS = ("top", "left", "center", "right", "bottom")
_CROSS_REGIONS = {
    "top": (0, 1),
    "left": (1, 0),
    "center": (1, 1),
    "right": (1, 2),
    "bottom": (2, 1),
}
_CORNER_REGIONS = ((0, 0), (0, 2), (2, 2), (2, 0))


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _region_bounds(size: int, separators: list[int]) -> list[tuple[int, int]]:
    assert len(separators) == 2
    a, b = separators
    bounds = [(0, a), (a + 1, b), (b + 1, size)]
    lengths = [hi - lo for lo, hi in bounds]
    assert all(length > 0 for length in lengths)
    assert lengths[0] == lengths[1] == lengths[2]
    return bounds


def _find_separator_lines(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    h, w = grid.shape
    row_candidates = [r for r in range(h) if int(grid[r, 0]) != bg and np.all(grid[r, :] == grid[r, 0])]
    col_candidates = [c for c in range(w) if int(grid[0, c]) != bg and np.all(grid[:, c] == grid[0, c])]
    sep_colors = {int(grid[r, 0]) for r in row_candidates} & {int(grid[0, c]) for c in col_candidates}
    assert len(sep_colors) == 1, "expected one separator color shared by full rows and full columns"
    sep = next(iter(sep_colors))
    sep_rows = [r for r in row_candidates if np.all(grid[r, :] == sep)]
    sep_cols = [c for c in col_candidates if np.all(grid[:, c] == sep)]
    assert len(sep_rows) == 2 and len(sep_cols) == 2, "expected two separator rows and two separator columns"
    assert sep_rows[0] + 1 < sep_rows[1]
    assert sep_cols[0] + 1 < sep_cols[1]
    return sep, sep_rows, sep_cols


def _parse_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)
    sep, sep_rows, sep_cols = _find_separator_lines(g, bg)
    row_bounds = _region_bounds(g.shape[0], sep_rows)
    col_bounds = _region_bounds(g.shape[1], sep_cols)
    panel_h = row_bounds[0][1] - row_bounds[0][0]
    panel_w = col_bounds[0][1] - col_bounds[0][0]

    def region(rc: tuple[int, int]) -> np.ndarray:
        rr, cc = rc
        r0, r1 = row_bounds[rr]
        c0, c1 = col_bounds[cc]
        return g[r0:r1, c0:c1]

    panels_by_color: dict[int, np.ndarray] = {}
    for key in _CROSS_KEYS:
        patch = region(_CROSS_REGIONS[key]).copy()
        colors = [int(v) for v in np.unique(patch) if int(v) != bg]
        assert len(colors) == 1, "each cross panel must use exactly one foreground color"
        color = colors[0]
        assert color != sep
        assert color not in panels_by_color, "cross-panel colors must be distinct"
        panels_by_color[color] = patch

    panel_colors = set(panels_by_color)
    corner_colors = []
    for rc in _CORNER_REGIONS:
        patch = region(rc)
        coords = np.argwhere(patch != bg)
        assert len(coords) == 1, "each corner region must contain exactly one key dot"
        color = int(patch[coords[0][0], coords[0][1]])
        assert color in panel_colors, "corner dots must use panel colors"
        corner_colors.append(color)
    assert len(set(corner_colors)) == 4, "corner key colors must be distinct"

    missing = list(panel_colors - set(corner_colors))
    assert len(missing) == 1, "exactly one panel color must be absent from the corners"
    layer_order = corner_colors + missing
    return {
        "bg": bg,
        "sep": sep,
        "panel_h": panel_h,
        "panel_w": panel_w,
        "panels_by_color": panels_by_color,
        "layer_order": layer_order,
    }


def _compose_from_order(panels_by_color: dict[int, np.ndarray], layer_order: list[int], bg: int) -> np.ndarray:
    sample = next(iter(panels_by_color.values()))
    out = np.full(sample.shape, bg, dtype=int)
    for color in reversed(layer_order):
        panel = panels_by_color[color]
        mask = panel != bg
        out[mask] = panel[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the five cross panels into one panel using the clockwise corner-key color order as top-to-bottom layers.

    Input structure:
    - Two full separator rows and two full separator columns split the grid into nine equal regions.
    - The top, left, center, right, and bottom regions each contain one transparent-background panel in a unique foreground color.
    - Each corner region contains exactly one single-pixel key dot, and those four key colors are four of the five panel colors.

    Rule:
    - Read the corner key colors clockwise starting at the top-left corner; these are layers 1 to 4 from top to bottom.
    - The one panel color not used by any corner key becomes layer 5, the bottom layer.
    - Overlay the five aligned panels onto one region-sized canvas, treating background as transparent and painting from bottom to top.

    Key insight:
    - The corner dots are only a legend: their clockwise color cycle determines layer priority, and the unmentioned panel color is meaningful because it is forced to the bottom.
    """
    parsed = _parse_input(grid)
    return _compose_from_order(parsed["panels_by_color"], parsed["layer_order"], parsed["bg"])


def _sample_winners(panel_h: int, panel_w: int) -> tuple[np.ndarray, dict[str, list[tuple[int, int]] | tuple[int, int]]]:
    n = panel_h * panel_w
    coords = [(r, c) for r in range(panel_h) for c in range(panel_w)]
    random.shuffle(coords)

    winners = np.full((panel_h, panel_w), -1, dtype=int)
    designated: dict[str, list[tuple[int, int]] | tuple[int, int]] = {}

    solo = coords[:5]
    adj = coords[5:9]
    extra_top = coords[9]
    extra_bottom = coords[10]
    designated["solo"] = solo
    designated["adj"] = adj
    designated["extra_top"] = extra_top
    designated["extra_bottom"] = extra_bottom

    for layer, cell in enumerate(solo):
        winners[cell] = layer
    for layer, cell in enumerate(adj):
        winners[cell] = layer
    winners[extra_top] = 0
    winners[extra_bottom] = 4

    base_counts = [3, 2, 2, 2, 2]
    min_visible = max(sum(base_counts), int(np.ceil(VISIBLE_FILL_RANGE[0] * n)))
    max_visible = min(n - 1, int(np.floor(VISIBLE_FILL_RANGE[1] * n)))
    assert min_visible <= max_visible
    visible_total = random.randint(min_visible, max_visible)

    counts = base_counts[:]
    extra_visible = visible_total - sum(base_counts)
    weighted_layers = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    for _ in range(extra_visible):
        counts[random.choice(weighted_layers)] += 1

    remaining = coords[11:]
    labels = []
    for layer, count in enumerate(counts):
        labels.extend([layer] * (count - base_counts[layer]))
    labels.extend([-1] * (n - visible_total))
    random.shuffle(labels)
    for cell, label in zip(remaining, labels):
        winners[cell] = label

    return winners, designated


def _derive_presence(winners: np.ndarray, designated: dict[str, list[tuple[int, int]] | tuple[int, int]]) -> np.ndarray:
    panel_h, panel_w = winners.shape
    presence = np.zeros((5, panel_h, panel_w), dtype=bool)

    solo = {tuple(cell): layer for layer, cell in enumerate(designated["solo"])}
    adj = {tuple(cell): layer for layer, cell in enumerate(designated["adj"])}
    extra_top = tuple(designated["extra_top"])
    extra_bottom = tuple(designated["extra_bottom"])

    for r in range(panel_h):
        for c in range(panel_w):
            layer = int(winners[r, c])
            if layer < 0:
                continue
            presence[layer, r, c] = True
            cell = (r, c)
            if cell in solo or cell == extra_top or cell == extra_bottom:
                continue
            if cell in adj:
                upper = adj[cell]
                assert layer == upper
                presence[upper + 1, r, c] = True
                if upper == 0:
                    presence[2:, r, c] = True
                else:
                    for lower in range(upper + 2, 5):
                        if random.random() < 0.35:
                            presence[lower, r, c] = True
                continue
            for lower in range(layer + 1, 5):
                prob = max(0.18, 0.42 - 0.08 * (lower - layer - 1))
                if random.random() < prob:
                    presence[lower, r, c] = True
    return presence


def _render_input(
    presence: np.ndarray,
    panel_colors_by_key: dict[str, int],
    corner_order: list[int],
    sep_color: int,
) -> np.ndarray:
    _, panel_h, panel_w = presence.shape
    grid = np.full((panel_h * 3 + 2, panel_w * 3 + 2), BG, dtype=int)
    sep_rows = [panel_h, 2 * panel_h + 1]
    sep_cols = [panel_w, 2 * panel_w + 1]
    grid[sep_rows, :] = sep_color
    grid[:, sep_cols] = sep_color

    row_bounds = [(0, panel_h), (panel_h + 1, 2 * panel_h + 1), (2 * panel_h + 2, 3 * panel_h + 2)]
    col_bounds = [(0, panel_w), (panel_w + 1, 2 * panel_w + 1), (2 * panel_w + 2, 3 * panel_w + 2)]

    color_to_layer = {color: idx for idx, color in enumerate(corner_order + [c for c in panel_colors_by_key.values() if c not in corner_order])}
    for key in _CROSS_KEYS:
        rr, cc = _CROSS_REGIONS[key]
        r0, r1 = row_bounds[rr]
        c0, c1 = col_bounds[cc]
        color = panel_colors_by_key[key]
        layer = color_to_layer[color]
        patch = np.where(presence[layer], color, BG)
        grid[r0:r1, c0:c1] = patch

    for region_rc, color in zip(_CORNER_REGIONS, corner_order):
        rr, cc = region_rc
        r0, r1 = row_bounds[rr]
        c0, c1 = col_bounds[cc]
        dr = random.randrange(r1 - r0)
        dc = random.randrange(c1 - c0)
        grid[r0 + dr, c0 + dc] = color

    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        parsed = _parse_input(grid)
    except Exception:
        return False

    bg = parsed["bg"]
    panels_by_color = parsed["panels_by_color"]
    order = parsed["layer_order"]
    actual = _compose_from_order(panels_by_color, order, bg)

    if not np.array_equal(main(grid), actual):
        return False

    if len(set(order)) != 5:
        return False

    visible_colors = {int(v) for v in np.unique(actual) if int(v) != bg}
    if visible_colors != set(order):
        return False

    presence = np.stack([panels_by_color[color] != bg for color in order], axis=0)
    winner = np.full(actual.shape, -1, dtype=int)
    for layer in range(5):
        mask = presence[layer] & (winner == -1)
        winner[mask] = layer

    for layer in range(5):
        if int(np.count_nonzero(winner == layer)) < 2:
            return False

    for layer in range(4):
        mask = presence[layer] & presence[layer + 1]
        if layer > 0:
            mask &= ~np.any(presence[:layer], axis=0)
        if not np.any(mask):
            return False

    if not np.any(presence[4] & np.any(presence[:4], axis=0)):
        return False

    alt_without_bottom = np.full(actual.shape, bg, dtype=int)
    for color in reversed(order[:4]):
        panel = panels_by_color[color]
        mask = panel != bg
        alt_without_bottom[mask] = panel[mask]
    if np.array_equal(actual, alt_without_bottom):
        return False

    for perm in itertools.permutations(order):
        if list(perm) == order:
            continue
        alt = _compose_from_order(panels_by_color, list(perm), bg)
        if np.array_equal(actual, alt):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    panel_h = random.randint(*PANEL_H_RANGE)
    panel_w = random.randint(*PANEL_W_RANGE)

    palette = random.sample(range(1, 10), 6)
    sep_color = palette[0]
    panel_colors = palette[1:]

    winners, designated = _sample_winners(panel_h, panel_w)
    presence = _derive_presence(winners, designated)

    panel_colors_shuffled = panel_colors[:]
    random.shuffle(panel_colors_shuffled)
    panel_colors_by_key = {key: color for key, color in zip(_CROSS_KEYS, panel_colors_shuffled)}

    layer_order = panel_colors[:]
    random.shuffle(layer_order)
    corner_order = layer_order[:4]

    grid = _render_input(presence, panel_colors_by_key, corner_order, sep_color)
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 6")
