from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

PANEL_COLORS = [BLUE, RED, GREEN, YELLOW, PINK, ORANGE, TEAL, MAROON]
GEN_RETRIES = 400


def _non_separator_segments(sep_cols: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for c, is_sep in enumerate(sep_cols.tolist()):
        if not is_sep and start is None:
            start = c
        elif is_sep and start is not None:
            segments.append((start, c))
            start = None
    if start is not None:
        segments.append((start, len(sep_cols)))
    return segments


def _overlay_in_order(panels: dict[int, np.ndarray], front_to_back: list[int]) -> np.ndarray:
    sample = next(iter(panels.values()))
    out = np.zeros_like(sample)
    for color in reversed(front_to_back):
        layer = panels[color]
        mask = layer != BLACK
        out[mask] = layer[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite equal-sized transparent color panels using the left legend column as the depth order.

    Input structure:
    - The grid has a left legend column containing isolated colored pixels on black.
    - Full-height grey separator columns split the legend from several equal-sized pattern panels.
    - Each panel uses exactly one non-black foreground color with black cells acting as transparency.

    Rule:
    - Read the legend colors from top to bottom; that sequence gives the layers from front to back.
    - Match each legend color to the panel that uses that same foreground color.
    - Starting from the backmost legend color and moving upward, overlay every panel's non-black cells into one panel-sized output.

    Key insight:
    - The panel positions do not encode depth; the separate legend column is the real control signal that reorders the layers before compositing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_cols = np.all(g == GREY, axis=0)
    assert np.any(sep_cols), "expected grey separator columns"
    segments = _non_separator_segments(sep_cols)
    assert len(segments) >= 4, "expected one legend segment plus multiple panel segments"

    legend_c0, legend_c1 = segments[0]
    legend = g[:, legend_c0:legend_c1]
    legend_coords = np.argwhere(legend != BLACK)
    assert legend_coords.shape[0] >= 3, "expected multiple legend markers"
    order = np.argsort(legend_coords[:, 0], kind="stable")
    legend_coords = legend_coords[order]
    assert len(set(map(int, legend_coords[:, 0]))) == legend_coords.shape[0], "legend rows must be unique"
    legend_colors = [int(legend[r, c]) for r, c in legend_coords]
    assert GREY not in legend_colors, "legend markers must not use the separator color"
    assert len(set(legend_colors)) == len(legend_colors), "legend colors must be unique"

    panels: dict[int, np.ndarray] = {}
    panel_widths = set()
    for c0, c1 in segments[1:]:
        panel = g[:, c0:c1]
        panel_widths.add(c1 - c0)
        colors = sorted(set(map(int, panel.flat)) - {BLACK})
        assert len(colors) == 1, "each panel must contain exactly one foreground color"
        color = colors[0]
        assert color != GREY, "panel color must differ from the separator color"
        assert color not in panels, "panel colors must be unique"
        panels[color] = panel

    assert len(panel_widths) == 1, "all panels must have the same width"
    assert set(panels) == set(legend_colors), "legend colors must match the panel colors"
    return _overlay_in_order(panels, legend_colors)


def _draw_rectangle(mask: np.ndarray) -> None:
    h, w = mask.shape
    rect_h = random.randint(2, h)
    rect_w = random.randint(2, w)
    r0 = random.randint(0, h - rect_h)
    c0 = random.randint(0, w - rect_w)
    mask[r0 : r0 + rect_h, c0 : c0 + rect_w] = True


def _draw_frame(mask: np.ndarray) -> None:
    h, w = mask.shape
    if h < 3 or w < 3:
        _draw_rectangle(mask)
        return
    rect_h = random.randint(3, h)
    rect_w = random.randint(3, w)
    r0 = random.randint(0, h - rect_h)
    c0 = random.randint(0, w - rect_w)
    r1 = r0 + rect_h - 1
    c1 = c0 + rect_w - 1
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True


def _draw_plus(mask: np.ndarray) -> None:
    h, w = mask.shape
    r = random.randint(1, h - 2) if h >= 3 else random.randint(0, h - 1)
    c = random.randint(1, w - 2) if w >= 3 else random.randint(0, w - 1)
    up = random.randint(1, r + 1)
    down = random.randint(1, h - r)
    left = random.randint(1, c + 1)
    right = random.randint(1, w - c)
    mask[r - up + 1 : r + down, c] = True
    mask[r, c - left + 1 : c + right] = True


def _draw_l(mask: np.ndarray) -> None:
    h, w = mask.shape
    corner_r = random.randint(0, h - 2)
    corner_c = random.randint(0, w - 2)
    vert_len = random.randint(2, h - corner_r)
    hor_len = random.randint(2, w - corner_c)
    orientation = random.choice(["dr", "dl", "ur", "ul"])
    if orientation == "dr":
        mask[corner_r : corner_r + vert_len, corner_c] = True
        mask[corner_r, corner_c : corner_c + hor_len] = True
    elif orientation == "dl":
        mask[corner_r : corner_r + vert_len, corner_c + hor_len - 1] = True
        mask[corner_r, corner_c : corner_c + hor_len] = True
    elif orientation == "ur":
        mask[corner_r + vert_len - 1 : corner_r + vert_len, corner_c : corner_c + hor_len] = True
        mask[corner_r : corner_r + vert_len, corner_c] = True
    else:
        mask[corner_r + vert_len - 1 : corner_r + vert_len, corner_c : corner_c + hor_len] = True
        mask[corner_r : corner_r + vert_len, corner_c + hor_len - 1] = True


def _draw_diagonal(mask: np.ndarray) -> None:
    h, w = mask.shape
    length = random.randint(3, min(h, w))
    r0 = random.randint(0, h - length)
    c0 = random.randint(0, w - length)
    if random.random() < 0.5:
        for k in range(length):
            mask[r0 + k, c0 + k] = True
    else:
        for k in range(length):
            mask[r0 + k, c0 + length - 1 - k] = True


def _draw_bars(mask: np.ndarray) -> None:
    h, w = mask.shape
    row = random.randint(0, h - 1)
    col = random.randint(0, w - 1)
    c0 = random.randint(0, w - 2)
    c1 = random.randint(c0 + 1, w - 1)
    r0 = random.randint(0, h - 2)
    r1 = random.randint(r0 + 1, h - 1)
    mask[row, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, col] = True


def _random_mask(h: int, w: int) -> np.ndarray:
    drawers = [_draw_rectangle, _draw_frame, _draw_plus, _draw_l, _draw_diagonal, _draw_bars]
    for _ in range(80):
        mask = np.zeros((h, w), dtype=bool)
        draw_count = random.randint(1, 2)
        for drawer in random.sample(drawers, draw_count):
            drawer(mask)
        area = int(mask.sum())
        if area < 4 or area > int(h * w * 0.5):
            continue
        rows = np.any(mask, axis=1).sum()
        cols = np.any(mask, axis=0).sum()
        if rows < 2 or cols < 2:
            continue
        return mask
    raise RuntimeError("failed to build a panel mask")


def _sample_legend_rows(height: int, count: int) -> list[int]:
    assert height >= 2 * count - 1, "not enough rows for isolated legend markers"
    bases = sorted(random.sample(range(height - count + 1), count))
    return [base + idx for idx, base in enumerate(bases)]


def _mask_dict(colors: list[int], masks: list[np.ndarray]) -> dict[int, np.ndarray]:
    panels: dict[int, np.ndarray] = {}
    for color, mask in zip(colors, masks):
        panel = np.zeros(mask.shape, dtype=int)
        panel[mask] = color
        panels[color] = panel
    return panels


def _visible_colors(output: np.ndarray) -> set[int]:
    return set(map(int, output[output != BLACK]))


def _build_candidate() -> tuple[np.ndarray, np.ndarray] | None:
    panel_count = random.randint(3, 5)
    min_height = max(7, 2 * panel_count - 1)
    height = random.randint(min_height, 12)
    max_panel_width = (29 - panel_count) // panel_count
    if max_panel_width < 4:
        return None
    panel_width = random.randint(4, max_panel_width)

    colors = random.sample(PANEL_COLORS, panel_count)
    legend_order = random.sample(colors, panel_count)
    panel_order = random.sample(colors, panel_count)
    if panel_order == legend_order:
        return None

    masks = [_random_mask(height, panel_width) for _ in range(panel_count)]
    panels = _mask_dict(colors, masks)
    expected = _overlay_in_order(panels, legend_order)
    if _visible_colors(expected) != set(colors):
        return None

    overlap = sum(int(np.sum(masks[i] & masks[j])) for i in range(panel_count) for j in range(i + 1, panel_count))
    if overlap < 2:
        return None

    union_density = float(np.count_nonzero(np.any(np.stack(masks, axis=0), axis=0))) / float(height * panel_width)
    if not (0.18 <= union_density <= 0.7):
        return None

    left_to_right_front = _overlay_in_order(panels, panel_order)
    right_to_left_front = _overlay_in_order(panels, list(reversed(panel_order)))
    if np.array_equal(expected, left_to_right_front) or np.array_equal(expected, right_to_left_front):
        return None

    legend_rows = _sample_legend_rows(height, panel_count)
    total_width = 1 + panel_count * panel_width + panel_count
    grid = np.zeros((height, total_width), dtype=int)
    grid[:, 1] = GREY
    for row, color in zip(legend_rows, legend_order):
        grid[row, 0] = color

    col = 2
    for idx, color in enumerate(panel_order):
        grid[:, col : col + panel_width] = panels[color]
        col += panel_width
        if idx != panel_count - 1:
            grid[:, col] = GREY
            col += 1

    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_candidate()
        if built is None:
            continue
        grid, _expected = built
        return grid
    raise RuntimeError("failed to generate a valid legend-overlay input")
