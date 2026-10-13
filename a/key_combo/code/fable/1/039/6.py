from __future__ import annotations

from collections import Counter

import numpy as np


MAX_GENERATION_ATTEMPTS = 400
GRID_H_RANGE = (15, 23)
GRID_W_RANGE = (15, 23)
LEGEND_LEN_RANGE = (2, 4)
ANCHOR_TEMPLATES = (
    ((0, -1), (0, 1)),
    ((-1, 0), (1, 0)),
    ((-1, -1), (1, 1)),
    ((-1, 1), (1, -1)),
    ((0, -2), (0, 0), (0, 2)),
    ((-1, -1), (1, 1), (-1, 2)),
    ((-1, 1), (1, -1), (2, 1)),
)
MIN_INTERFERENCE_CELLS = 4
MAX_FILL_FRACTION = 0.42


def get_bg(grid: np.ndarray) -> int:
    """Return the most common grid color."""
    return int(Counter(map(int, np.asarray(grid).reshape(-1))).most_common(1)[0][0])


def _legend_candidates(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, tuple[int, ...]]]:
    counts = Counter(map(int, grid.reshape(-1)))
    height, width = map(int, grid.shape)
    candidates: list[tuple[int, int, int, tuple[int, ...]]] = []
    for row in (0, height - 1):
        col = 0
        while col < width:
            color = int(grid[row, col])
            if color == bg or counts[color] != 1:
                col += 1
                continue
            start = col
            colors: list[int] = []
            seen: set[int] = set()
            while col < width:
                color = int(grid[row, col])
                if color == bg or counts[color] != 1 or color in seen:
                    break
                colors.append(color)
                seen.add(color)
                col += 1
            candidates.append((len(colors), row, start, tuple(colors)))
    return candidates


def _infer_roles(grid: np.ndarray) -> tuple[int, int, int, int, tuple[int, ...], np.ndarray]:
    bg = get_bg(grid)
    candidates = _legend_candidates(grid, bg)
    assert candidates, "expected a legend strip on the top or bottom edge"
    max_len = max(length for length, _row, _start, _colors in candidates)
    best = [candidate for candidate in candidates if candidate[0] == max_len]
    assert len(best) == 1, "legend strip must be uniquely identifiable"
    _length, legend_row, legend_start, legend_colors = best[0]
    legend_set = set(legend_colors)

    legend_mask = np.zeros_like(grid, dtype=bool)
    legend_mask[legend_row, legend_start : legend_start + len(legend_colors)] = True
    non_bg_outside_legend = set(map(int, grid[(grid != bg) & ~legend_mask].reshape(-1)))
    assert len(non_bg_outside_legend) == 1, "expected exactly one anchor color outside the legend"
    anchor_color = next(iter(non_bg_outside_legend))
    assert anchor_color not in legend_set, "anchor color must differ from all legend colors"

    anchors = np.argwhere((grid == anchor_color) & ~legend_mask)
    assert anchors.shape[0] >= 2, "expected multiple anchor pixels"
    return bg, anchor_color, legend_row, legend_start, legend_colors, anchors


def _nearest_chebyshev(shape: tuple[int, int], anchors: np.ndarray) -> np.ndarray:
    rr = np.arange(shape[0], dtype=int)[:, None]
    cc = np.arange(shape[1], dtype=int)[None, :]
    distance_layers = [
        np.maximum(np.abs(rr - int(r)), np.abs(cc - int(c)))
        for r, c in anchors
    ]
    return np.minimum.reduce(distance_layers)


def _render_output(
    shape: tuple[int, int],
    bg: int,
    anchor_color: int,
    anchors: np.ndarray,
    legend_row: int,
    legend_start: int,
    legend_colors: tuple[int, ...],
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    distance = _nearest_chebyshev(shape, anchors)
    for ring_index, color in enumerate(legend_colors, start=1):
        out[distance == 2 * ring_index - 1] = int(color)
    out[anchors[:, 0], anchors[:, 1]] = int(anchor_color)
    out[legend_row, legend_start : legend_start + len(legend_colors)] = np.array(legend_colors, dtype=int)
    return out


def _render_independent_stamp(
    shape: tuple[int, int],
    bg: int,
    anchor_color: int,
    anchors: np.ndarray,
    legend_row: int,
    legend_start: int,
    legend_colors: tuple[int, ...],
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    best_odd = np.full(shape, 2 * len(legend_colors) + 2, dtype=int)
    rr = np.arange(shape[0], dtype=int)[:, None]
    cc = np.arange(shape[1], dtype=int)[None, :]
    max_distance = 2 * len(legend_colors) - 1
    for r, c in anchors:
        dist = np.maximum(np.abs(rr - int(r)), np.abs(cc - int(c)))
        odd_mask = (dist % 2 == 1) & (dist <= max_distance)
        best_odd[odd_mask] = np.minimum(best_odd[odd_mask], dist[odd_mask])
    for ring_index, color in enumerate(legend_colors, start=1):
        out[best_odd == 2 * ring_index - 1] = int(color)
    out[anchors[:, 0], anchors[:, 1]] = int(anchor_color)
    out[legend_row, legend_start : legend_start + len(legend_colors)] = np.array(legend_colors, dtype=int)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Color odd Chebyshev-distance rings from the nearest anchor set while preserving the legend strip.

    Input structure:
    - The background is dominant.
    - Several isolated anchor pixels of one shared color are scattered in the field.
    - A horizontal legend strip lies on the top or bottom edge; its colors are all distinct and differ from both the background and the anchor color.

    Rule:
    - For every non-legend cell, compute the minimum Chebyshev distance to any anchor pixel.
    - If that minimum distance is 1, 3, 5, ... and matches the k-th odd radius within the legend length, paint the cell with the k-th legend color.
    - Leave anchor pixels and legend cells unchanged; all other cells stay background.

    Key insight:
    - The rings are not stamped independently around each anchor: every cell first belongs to the nearest-source distance field, and only the odd distance levels of that shared field are made visible.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, anchor_color, legend_row, legend_start, legend_colors, anchors = _infer_roles(g)
    return _render_output(
        g.shape,
        bg,
        anchor_color,
        anchors,
        int(legend_row),
        legend_start,
        tuple(int(color) for color in legend_colors),
    )


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _sample_anchors(rng: np.random.Generator, height: int, width: int) -> np.ndarray | None:
    template = ANCHOR_TEMPLATES[int(rng.integers(len(ANCHOR_TEMPLATES)))]
    min_dr = min(dr for dr, _dc in template)
    max_dr = max(dr for dr, _dc in template)
    min_dc = min(dc for _dr, dc in template)
    max_dc = max(dc for _dr, dc in template)

    row_low = 2 - min_dr
    row_high = height - 3 - max_dr
    col_low = 2 - min_dc
    col_high = width - 3 - max_dc
    if row_low > row_high or col_low > col_high:
        return None

    center_r = int(rng.integers(row_low, row_high + 1))
    center_c = int(rng.integers(col_low, col_high + 1))
    anchors = np.array([(center_r + dr, center_c + dc) for dr, dc in template], dtype=int)
    for i in range(len(anchors)):
        for j in range(i + 1, len(anchors)):
            if _chebyshev(tuple(anchors[i]), tuple(anchors[j])) < 2:
                return None
    return anchors


def _validate_generated(grid: np.ndarray, legend_row: int, legend_start: int, legend_colors: tuple[int, ...], anchor_color: int, anchors: np.ndarray) -> bool:
    bg = get_bg(grid)
    legend_mask = np.zeros_like(grid, dtype=bool)
    legend_mask[legend_row, legend_start : legend_start + len(legend_colors)] = True
    anchor_mask = np.zeros_like(grid, dtype=bool)
    anchor_mask[anchors[:, 0], anchors[:, 1]] = True

    try:
        solved = main(grid)
    except AssertionError:
        return False

    expected = _render_output(
        grid.shape,
        bg,
        anchor_color,
        anchors,
        legend_row,
        legend_start,
        legend_colors,
    )
    if not np.array_equal(solved, expected):
        return False

    non_legend = ~legend_mask
    for color in legend_colors:
        if not np.any((solved == int(color)) & non_legend):
            return False

    independent = _render_independent_stamp(
        grid.shape,
        bg,
        anchor_color,
        anchors,
        legend_row,
        legend_start,
        legend_colors,
    )
    suppressed = (independent != solved) & (independent != bg) & (solved == bg) & non_legend & ~anchor_mask
    if int(np.count_nonzero(suppressed)) < MIN_INTERFERENCE_CELLS:
        return False

    added = (solved != bg) & non_legend & ~anchor_mask
    if int(np.count_nonzero(added)) / float(grid.size) > MAX_FILL_FRACTION:
        return False

    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(*GRID_H_RANGE))
    width = int(rng.integers(*GRID_W_RANGE))
    max_legend_len = min(LEGEND_LEN_RANGE[1], width - 2)
    if max_legend_len < LEGEND_LEN_RANGE[0]:
        return None
    legend_len = int(rng.integers(LEGEND_LEN_RANGE[0], max_legend_len + 1))

    palette = rng.choice(np.arange(10, dtype=int), size=legend_len + 2, replace=False)
    bg = int(palette[0])
    anchor_color = int(palette[1])
    legend_colors = tuple(int(color) for color in palette[2:])

    grid = np.full((height, width), bg, dtype=int)
    legend_row = 0 if rng.random() < 0.5 else height - 1
    legend_start = int(rng.integers(0, width - legend_len + 1))
    grid[legend_row, legend_start : legend_start + legend_len] = np.array(legend_colors, dtype=int)

    anchors = _sample_anchors(rng, height, width)
    if anchors is None:
        return None
    grid[anchors[:, 0], anchors[:, 1]] = anchor_color

    if not _validate_generated(grid, legend_row, legend_start, legend_colors, anchor_color, anchors):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid interference-ripple input")
