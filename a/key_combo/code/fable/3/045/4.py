from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
LAYER_COUNT = 4
GEN_RETRIES = 600


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = {int(v) for v in np.unique(grid)}
    if exclude is not None:
        colors -= {int(v) for v in exclude}
    return sorted(colors)


def _parse_input(grid: np.ndarray) -> tuple[int, list[np.ndarray], list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height > 0 and width >= 3, "expected a body, a blank separator, and a legend column"
    assert height % LAYER_COUNT == 0, "body must be split into four equal-height bands"

    separator_values = np.unique(g[:, -2])
    assert len(separator_values) == 1, "separator column must be entirely background"
    bg = int(separator_values[0])

    band_h = height // LAYER_COUNT
    body_w = width - 2
    assert band_h > 0 and body_w > 0, "expected nonempty layer masks"

    body = g[:, :body_w]
    bands = [body[i * band_h : (i + 1) * band_h, :] for i in range(LAYER_COUNT)]
    band_colors: list[int] = []
    for band in bands:
        colors = _colors_present(band, exclude={bg})
        assert len(colors) == 1, "each band must contain exactly one foreground color"
        color = int(colors[0])
        assert np.all((band == bg) | (band == color)), "each band must be a monochrome mask"
        band_colors.append(color)

    assert len(set(band_colors)) == LAYER_COUNT, "band colors must be distinct"

    legend = [int(v) for v in g[:, -1] if int(v) != bg]
    assert 1 <= len(legend) <= LAYER_COUNT, "legend must list at least one layer color"
    assert len(set(legend)) == len(legend), "legend colors must not repeat"
    assert set(legend).issubset(set(band_colors)), "legend colors must come from the layer bands"
    return bg, bands, band_colors, legend


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collapse four same-size color mask bands according to the side legend order, deleting absent layers.

    Input structure:
    - The main body is divided into four equal-height horizontal bands, each a same-size monochrome mask on the background.
    - The four mask colors are distinct.
    - Immediately to the right of the body is one blank separator column, followed by one legend column containing a top-to-bottom list of some band colors.

    Rule:
    - Return one grid with the height and width of a single band.
    - For each local position, consult the legend colors from top to bottom and output the first listed layer whose band has foreground at that position.
    - Band colors absent from the legend are ignored completely, even where their mask has foreground.
    - Positions with no foreground in any listed layer remain background.

    Key insight:
    - The layer priority is encoded by the side legend, not by the natural top-to-bottom order of the four bands; the same legend also filters out missing layers.
    """
    bg, bands, band_colors, legend = _parse_input(grid)
    color_to_band = {color: band for color, band in zip(band_colors, bands)}

    out = np.full(bands[0].shape, bg, dtype=int)
    for color in legend:
        band = color_to_band[int(color)]
        mask = (band != bg) & (out == bg)
        out[mask] = int(color)
    return out


def _composite(masks: list[np.ndarray], colors: list[int], order: list[int]) -> np.ndarray:
    out = np.full(masks[0].shape, BG, dtype=int)
    for idx in order:
        fill = masks[idx] & (out == BG)
        out[fill] = int(colors[idx])
    return out


def _force_cell(
    masks: list[np.ndarray],
    blocked: list[set[tuple[int, int]]],
    cell: tuple[int, int],
    on_layers: tuple[int, ...],
) -> None:
    allowed = set(map(int, on_layers))
    r, c = cell
    for idx in range(LAYER_COUNT):
        if idx in allowed:
            masks[idx][r, c] = True
        else:
            blocked[idx].add(cell)


def _inversion_pair(order: list[int]) -> tuple[int, int] | None:
    for a_pos in range(len(order)):
        for b_pos in range(a_pos + 1, len(order)):
            earlier = int(order[a_pos])
            later = int(order[b_pos])
            if earlier > later:
                return earlier, later
    return None


def _build_masks(shape: tuple[int, int], legend_order: list[int]) -> list[np.ndarray]:
    cells = [(r, c) for r in range(shape[0]) for c in range(shape[1])]
    active = set(map(int, legend_order))
    absent = [idx for idx in range(LAYER_COUNT) if idx not in active]

    pair = _inversion_pair(legend_order)
    if pair is None:
        pair = (int(legend_order[0]), int(legend_order[1]))

    specs: list[tuple[int, ...]] = [tuple(pair)]
    specs.extend((idx,) for idx in legend_order)
    specs.extend((idx,) for idx in absent)
    forced_cells = random.sample(cells, len(specs))

    masks = [np.zeros(shape, dtype=bool) for _ in range(LAYER_COUNT)]
    blocked: list[set[tuple[int, int]]] = [set() for _ in range(LAYER_COUNT)]
    for cell, on_layers in zip(forced_cells, specs):
        _force_cell(masks, blocked, cell, on_layers)

    area = shape[0] * shape[1]
    min_count = max(3, area // 8)
    max_count = max(min_count + 1, min(area - 1, int(round(area * 0.34))))

    for idx, mask in enumerate(masks):
        current = int(mask.sum())
        candidates = [cell for cell in cells if cell not in blocked[idx] and not bool(mask[cell])]
        if not candidates:
            continue
        high = min(max_count, current + len(candidates))
        low = min(high, max(min_count, current + 1))
        target = random.randint(low, high)
        random.shuffle(candidates)
        for r, c in candidates[: target - current]:
            mask[r, c] = True

    return masks


def _choose_legend_order() -> list[int]:
    count = random.choice((2, 2, 3, 3))
    chosen = sorted(random.sample(range(LAYER_COUNT), count))
    if random.random() < 0.25:
        return chosen

    order = chosen.copy()
    for _ in range(20):
        random.shuffle(order)
        if order != chosen:
            return order.copy()
    return chosen


def _legend_rows(total_h: int, count: int) -> list[int]:
    if count == 1:
        return [min(1, total_h - 1)]
    max_gap = max(1, min(3, (total_h - 1) // (count - 1)))
    gap = random.randint(1, max_gap)
    max_start = total_h - 1 - gap * (count - 1)
    start = random.randint(0, min(3, max_start))
    return [start + i * gap for i in range(count)]


def _render_input(
    masks: list[np.ndarray],
    colors: list[int],
    legend_order: list[int],
) -> np.ndarray:
    band_h, body_w = masks[0].shape
    total_h = band_h * LAYER_COUNT
    grid = np.full((total_h, body_w + 2), BG, dtype=int)

    for idx, color in enumerate(colors):
        band = grid[idx * band_h : (idx + 1) * band_h, :body_w]
        band[masks[idx]] = int(color)

    legend_col = body_w + 1
    for row, idx in zip(_legend_rows(total_h, len(legend_order)), legend_order):
        grid[row, legend_col] = int(colors[idx])
    return grid


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    masks: list[np.ndarray],
    colors: list[int],
    legend_order: list[int],
) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False

    active = set(map(int, legend_order))
    absent = [idx for idx in range(LAYER_COUNT) if idx not in active]
    active_stack = np.zeros(masks[0].shape, dtype=int)
    for idx in active:
        active_stack += masks[idx].astype(int)

    if not np.any(active_stack >= 2):
        return False

    for idx in range(LAYER_COUNT):
        others = np.zeros(masks[0].shape, dtype=bool)
        for j in range(LAYER_COUNT):
            if j != idx:
                others |= masks[j]
        if not np.any(masks[idx] & ~others):
            return False

    for idx in active:
        if not np.any(expected == int(colors[idx])):
            return False
    for idx in absent:
        if np.any(expected == int(colors[idx])):
            return False

    if int(np.count_nonzero(expected != BG)) < len(active) + 2:
        return False
    if float(np.mean(expected != BG)) > 0.55:
        return False

    if legend_order != sorted(legend_order):
        natural = _composite(masks, colors, sorted(active))
        if np.array_equal(natural, expected):
            return False

    try:
        bg, _bands, parsed_colors, parsed_legend = _parse_input(inp)
        actual = main(inp)
    except Exception:
        return False

    if bg != BG:
        return False
    if parsed_colors != [int(c) for c in colors]:
        return False
    if parsed_legend != [int(colors[idx]) for idx in legend_order]:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    band_h = random.randint(3, 6)
    body_w = random.randint(6, 13)
    colors = random.sample(PALETTE, LAYER_COUNT)
    legend_order = _choose_legend_order()
    masks = _build_masks((band_h, body_w), legend_order)
    expected = _composite(masks, colors, legend_order)
    inp = _render_input(masks, colors, legend_order)

    if _semantic_checks(inp, expected, masks, colors, legend_order):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid legend-ordered overlay input")
