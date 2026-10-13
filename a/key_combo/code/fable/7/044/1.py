from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
GREY = 5
MAX_GENERATION_ATTEMPTS = 300


def _non_separator_runs(separator_mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for c, is_separator in enumerate(separator_mask.tolist()):
        if not is_separator and start is None:
            start = c
        elif is_separator and start is not None:
            runs.append((start, c))
            start = None
    if start is not None:
        runs.append((start, len(separator_mask)))
    return runs


def _column_majority(column: np.ndarray) -> int:
    counts = Counter(map(int, column.tolist()))
    ranked = counts.most_common()
    if len(ranked) > 1:
        assert ranked[0][1] > ranked[1][1], f"column has no unique majority: {counts}"
    return int(ranked[0][0])


def _band_order(band: np.ndarray, allowed_colors: set[int]) -> list[int]:
    majority_colors = [_column_majority(band[:, c]) for c in range(band.shape[1])]
    assert set(majority_colors) <= allowed_colors, "band columns must majority-match panel colors"

    order: list[int] = []
    for color in majority_colors:
        if not order or order[-1] != color:
            order.append(color)

    assert len(order) == len(allowed_colors), "compressed band order must list each panel color once"
    assert set(order) == allowed_colors, "band order must cover all panel colors"
    return order


def _find_layout(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int]], list[int], list[int]]:
    height, width = grid.shape
    candidates: list[tuple[int, int, list[tuple[int, int]], list[int], list[int]]] = []

    for sep_row in range(1, height - 1):
        if not np.all(grid[sep_row] == grid[sep_row, 0]):
            continue

        separator_color = int(grid[sep_row, 0])
        top = grid[:sep_row]
        band = grid[sep_row + 1 :]
        if top.size == 0 or band.size == 0:
            continue

        separator_cols = np.all(top == separator_color, axis=0)
        if not separator_cols.any():
            continue

        panel_spans = _non_separator_runs(separator_cols)
        if len(panel_spans) < 2:
            continue

        widths = {c1 - c0 for c0, c1 in panel_spans}
        if len(widths) != 1:
            continue

        panel_colors: list[int] = []
        valid = True
        for c0, c1 in panel_spans:
            panel = top[:, c0:c1]
            colors = {int(v) for v in np.unique(panel)} - {BLACK, separator_color}
            if len(colors) != 1:
                valid = False
                break
            panel_colors.append(colors.pop())

        if not valid or len(set(panel_colors)) != len(panel_colors):
            continue

        try:
            order = _band_order(band, set(panel_colors))
        except AssertionError:
            continue

        candidates.append((sep_row, separator_color, panel_spans, panel_colors, order))

    assert candidates, "failed to locate panel/band layout"
    candidates.sort(key=lambda item: (item[0], len(item[2])), reverse=True)
    return candidates[0]


def _overlay_panels(color_to_panel: dict[int, np.ndarray], order: list[int]) -> np.ndarray:
    sample_panel = next(iter(color_to_panel.values()))
    out = np.zeros_like(sample_panel)
    filled = np.zeros(sample_panel.shape, dtype=bool)

    for color in order:
        panel = color_to_panel[color]
        mask = (panel != BLACK) & (~filled)
        out[mask] = panel[mask]
        filled |= mask

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the upper mask panels using the compressed left-to-right color order hidden in the noisy bottom band.

    Input structure:
    - The upper region is split by uniform separator columns into equal-width panels.
    - Each panel has black background plus exactly one distinct foreground color mask.
    - A uniform separator row splits the panels from a lower multicolor band with jagged vertical boundaries.

    Rule:
    - For each column of the lower band, take that column's majority color.
    - Compress adjacent columns with the same majority color to recover a left-to-right color order.
    - Use that color order as the front-to-back panel priority, treating black as transparent, and output one overlaid panel.

    Key insight:
    - The noisy lower strip is a hidden layer-order legend: its coarse color-band sequence, not the panel positions, determines which mask is visible on top.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_row, _separator_color, panel_spans, panel_colors, order = _find_layout(g)
    top = g[:sep_row]

    color_to_panel: dict[int, np.ndarray] = {}
    for (c0, c1), color in zip(panel_spans, panel_colors):
        panel = top[:, c0:c1].copy()
        color_to_panel[color] = panel

    return _overlay_panels(color_to_panel, order)


def _neighbors4(r: int, c: int, height: int, width: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            out.append((nr, nc))
    return out


def _shared_core(height: int, width: int, rng: random.Random) -> list[tuple[int, int]]:
    row = rng.randrange(max(1, height // 3), min(height, (2 * height + 2) // 3))
    col = rng.randrange(max(1, width // 3), min(width, (2 * width + 2) // 3))
    core = [(row, col)]
    target_len = rng.randint(2, min(4, height + width))
    while len(core) < target_len:
        r, c = core[-1]
        options = [cell for cell in _neighbors4(r, c, height, width) if cell not in core]
        if not options:
            break
        core.append(rng.choice(options))
    return core


def _grow_mask(
    height: int,
    width: int,
    target_area: int,
    shared_cells: list[tuple[int, int]],
    rng: random.Random,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    frontier: set[tuple[int, int]] = set()

    for r, c in shared_cells:
        mask[r, c] = True
    for r, c in shared_cells:
        frontier.update(cell for cell in _neighbors4(r, c, height, width) if not mask[cell])

    while int(mask.sum()) < target_area:
        if not frontier:
            frontier = {
                cell
                for r, c in np.argwhere(mask)
                for cell in _neighbors4(int(r), int(c), height, width)
                if not mask[cell]
            }
        assert frontier, "mask growth frontier unexpectedly empty"

        choices = list(frontier)
        cell = rng.choice(choices)
        frontier.remove(cell)
        mask[cell] = True
        for neighbor in _neighbors4(cell[0], cell[1], height, width):
            if not mask[neighbor]:
                frontier.add(neighbor)

    return mask


def _sample_partition(total: int, count: int, minimum: int, rng: random.Random) -> list[int]:
    assert total >= count * minimum
    parts = [minimum] * count
    for _ in range(total - count * minimum):
        parts[rng.randrange(count)] += 1
    rng.shuffle(parts)
    return parts


def _render_band(height: int, widths: list[int], order: list[int], rng: random.Random) -> np.ndarray:
    band = np.zeros((height, sum(widths)), dtype=int)
    start = 0
    for color, width in zip(order, widths):
        band[:, start : start + width] = color
        start += width

    max_intrusion = max(1, (height - 1) // 2)
    boundary = widths[0]
    for i in range(len(widths) - 1):
        left_color = order[i]
        right_color = order[i + 1]
        left_reach = rng.randint(1, min(2, widths[i] - 1))
        right_reach = rng.randint(1, min(2, widths[i + 1] - 1))

        for offset in range(1, left_reach + 1):
            col = boundary - offset
            intrusion = rng.randint(1, max_intrusion)
            if rng.random() < 0.5:
                band[:intrusion, col] = right_color
            else:
                band[height - intrusion :, col] = right_color

        for offset in range(right_reach):
            col = boundary + offset
            intrusion = rng.randint(1, max_intrusion)
            if rng.random() < 0.5:
                band[:intrusion, col] = left_color
            else:
                band[height - intrusion :, col] = left_color

        boundary += widths[i + 1]

    return band


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[int], list[int]]:
    panel_count = rng.randint(3, 4)
    max_panel_width = (30 - (panel_count - 1)) // panel_count
    panel_width = rng.randint(5, max_panel_width)
    panel_height = rng.randint(5, 9)
    band_height = rng.randint(5, 8)

    total_width = panel_count * panel_width + (panel_count - 1)
    assert total_width <= 30
    assert panel_height + 1 + band_height <= 30

    palette = [color for color in range(1, 10) if color != GREY]
    panel_colors = rng.sample(palette, panel_count)
    order = panel_colors[:]
    while order == panel_colors:
        rng.shuffle(order)

    shared_cells = _shared_core(panel_height, panel_width, rng)
    panels: list[np.ndarray] = []
    panel_area = panel_height * panel_width
    for _color in panel_colors:
        target_area = rng.randint(max(len(shared_cells) + 2, panel_area // 5), max(len(shared_cells) + 3, panel_area // 3))
        mask = _grow_mask(panel_height, panel_width, target_area, shared_cells, rng)
        panel = np.zeros((panel_height, panel_width), dtype=int)
        panel[mask] = _color
        panels.append(panel)

    top = np.zeros((panel_height, total_width), dtype=int)
    for i, panel in enumerate(panels):
        c0 = i * (panel_width + 1)
        top[:, c0 : c0 + panel_width] = panel
        if i < panel_count - 1:
            top[:, c0 + panel_width] = GREY

    band_widths = _sample_partition(total_width, panel_count, 3, rng)
    band = _render_band(band_height, band_widths, order, rng)
    separator_row = np.full((1, total_width), GREY, dtype=int)
    grid = np.vstack([top, separator_row, band])

    color_to_panel = {color: panel for color, panel in zip(panel_colors, panels)}
    expected = _overlay_panels(color_to_panel, order)
    return grid, expected, panels, panel_colors, order


def _is_valid_case(
    grid: np.ndarray,
    expected: np.ndarray,
    panels: list[np.ndarray],
    panel_colors: list[int],
    order: list[int],
) -> bool:
    try:
        if not np.array_equal(main(grid), expected):
            return False
    except Exception:
        return False

    visible_colors = set(map(int, np.unique(expected))) - {BLACK}
    if visible_colors != set(panel_colors):
        return False

    coverage = np.zeros(expected.shape, dtype=int)
    for panel in panels:
        coverage += (panel != BLACK).astype(int)
    if int(np.sum(coverage >= 2)) < 2:
        return False

    if order == panel_colors:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = random.Random()
        grid, expected, panels, panel_colors, order = _build_case(rng)
        if _is_valid_case(grid, expected, panels, panel_colors, order):
            return grid
    raise RuntimeError("failed to generate a valid input")
