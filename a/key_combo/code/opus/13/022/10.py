from __future__ import annotations

from collections import Counter
import random

import numpy as np


MAX_GENERATION_ATTEMPTS = 400
BACKGROUND = 0
COLOR_POOL = tuple(range(1, 10))


def get_bg(grid: np.ndarray) -> int:
    """Return the most common color in the grid."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as coord arrays."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _classify_objects(
    grid: np.ndarray,
    bg: int,
) -> tuple[list[int], dict[int, tuple[int, int]], dict[int, int]]:
    h, w = grid.shape
    legend: list[int] | None = None
    markers: dict[int, tuple[int, int]] = {}
    bar_heights: dict[int, int] = {}

    for coords in components_mask(grid != bg):
        area = int(coords.shape[0])
        r0, c0, r1, c1 = bbox_coords(coords)
        comp_colors = {int(grid[r, c]) for r, c in coords}
        height = r1 - r0 + 1
        width = c1 - c0 + 1

        if r0 == 0 and height == 1 and area == width:
            assert legend is None, "expected a single top legend"
            cols = sorted(int(c) for _r, c in coords)
            assert cols == list(range(c0, c1 + 1)), "legend must be contiguous"
            legend = [int(grid[0, c]) for c in range(c0, c1 + 1)]
            assert len(legend) >= 3, "legend must contain at least three colors"
            assert len(set(legend)) == len(legend), "legend colors must be distinct"
            continue

        if area == 1:
            (r, c), = coords.tolist()
            color = int(grid[r, c])
            assert 0 < r < h - 1, "marker must be interior"
            assert color not in markers, "marker colors must be unique"
            markers[color] = (int(r), int(c))
            continue

        if width == 1 and area == height and r1 == h - 1:
            assert len(comp_colors) == 1, "bar must have one color"
            color = int(next(iter(comp_colors)))
            assert color not in bar_heights, "bar colors must be unique"
            bar_heights[color] = int(height)
            continue

        raise AssertionError("unexpected foreground component shape")

    assert legend is not None, "missing legend"
    legend_set = set(legend)
    assert set(markers) == legend_set, "marker colors must match legend colors"
    assert set(bar_heights) == legend_set, "bar colors must match legend colors"
    assert len(set(bar_heights.values())) == len(bar_heights), "bar heights must be distinct"
    assert max(bar_heights.values()) > min(bar_heights.values()), "need at least two heights"
    return legend, markers, bar_heights


def _expanded_row(width: int, legend: list[int], anchor_idx: int, marker_c: int, bg: int) -> np.ndarray:
    row = np.full(width, bg, dtype=int)
    start_c = int(marker_c) - int(anchor_idx)
    for offset, color in enumerate(legend):
        cc = start_c + offset
        if 0 <= cc < width:
            row[cc] = int(color)
    return row


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each isolated marker into the legend row, then stack the expanded rows by the height of the same-colored vertical bar.

    Input structure:
    - The background is black and the top row contains one contiguous horizontal legend made of distinct non-background colors.
    - The middle area contains one isolated single-pixel marker for each legend color.
    - The bottom area contains one bottom-aligned one-cell-wide vertical bar for each legend color, with all bar heights distinct.

    Rule:
    - Read the top legend sequence from left to right.
    - For each isolated marker, copy that full legend onto the marker row so the legend position with the marker's color stays on the marker column, clipping past the left or right border if needed.
    - Match each marker color to the bottom bar of the same color, sort the expanded legend rows by bar height from tallest to shortest, and stack those rows tightly from top to bottom.

    Key insight:
    - The same color serves two different roles at once: it selects the anchor position inside the legend for expansion, and it also points to a vertical bar whose height determines the row order in the final stack.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    legend, markers, bar_heights = _classify_objects(g, bg)
    width = int(g.shape[1])
    color_to_anchor = {int(color): idx for idx, color in enumerate(legend)}

    rows: list[tuple[int, int, np.ndarray]] = []
    for color, (r, c) in markers.items():
        expanded = _expanded_row(width, legend, color_to_anchor[color], c, bg)
        rows.append((bar_heights[color], r, expanded))

    rows.sort(key=lambda item: (-int(item[0]), int(item[1])))
    return np.vstack([row for _height, _r, row in rows])


def _spaced_columns(width: int, count: int) -> list[int]:
    base = sorted(random.sample(range(width - count + 1), count))
    return [int(value + idx) for idx, value in enumerate(base)]


def _marker_column_candidates(width: int, legend_len: int, anchor_idx: int, clip_mode: str) -> list[int]:
    if clip_mode == "none":
        return list(range(anchor_idx, width - legend_len + anchor_idx + 1))
    if clip_mode == "left":
        return list(range(anchor_idx))
    if clip_mode == "right":
        start = width - legend_len + anchor_idx + 1
        return list(range(start, width))
    raise ValueError(f"unknown clip mode: {clip_mode}")


def _choose_clip_modes(legend: list[int]) -> dict[int, str]:
    colors = legend[:]
    modes = {int(color): "none" for color in colors}
    clip_count = random.choices([0, 1, 2], weights=[1, 3, 2], k=1)[0]
    random.shuffle(colors)
    for color in colors:
        options = []
        anchor_idx = legend.index(color)
        if anchor_idx > 0:
            options.append("left")
        if anchor_idx < len(legend) - 1:
            options.append("right")
        if not options:
            continue
        modes[int(color)] = random.choice(options)
        clip_count -= 1
        if clip_count == 0:
            break
    return modes


def _build_expected(
    width: int,
    legend: list[int],
    markers: dict[int, tuple[int, int]],
    bar_heights: dict[int, int],
) -> np.ndarray:
    color_to_anchor = {int(color): idx for idx, color in enumerate(legend)}
    rows = []
    for color, (r, c) in markers.items():
        row = _expanded_row(width, legend, color_to_anchor[color], c, BACKGROUND)
        rows.append((bar_heights[color], r, row))
    rows.sort(key=lambda item: (-int(item[0]), int(item[1])))
    return np.vstack([row for _height, _r, row in rows])


def _gen_attempt() -> np.ndarray | None:
    k = random.randint(3, 5)
    h = random.randint(max(14, k + 9), 18)
    w = random.randint(max(11, 2 * k + 1), 16)

    legend = random.sample(COLOR_POOL, k)
    heights = random.sample(range(2, 7), k)
    max_height = max(heights)

    marker_rows = list(range(2, h - max_height - 1))
    if len(marker_rows) < k:
        return None
    marker_rows = sorted(random.sample(marker_rows, k))

    bar_columns = _spaced_columns(w, k)
    marker_order = legend[:]
    random.shuffle(marker_order)
    bar_order = legend[:]
    random.shuffle(bar_order)
    clip_modes = _choose_clip_modes(legend)

    grid = np.full((h, w), BACKGROUND, dtype=int)
    grid[0, :k] = np.asarray(legend, dtype=int)

    bar_heights: dict[int, int] = {}
    for color, bar_height, col in zip(bar_order, heights, bar_columns):
        grid[h - bar_height :, col] = int(color)
        bar_heights[int(color)] = int(bar_height)

    markers: dict[int, tuple[int, int]] = {}
    color_to_anchor = {int(color): idx for idx, color in enumerate(legend)}
    for row, color in zip(marker_rows, marker_order):
        anchor_idx = color_to_anchor[int(color)]
        preferred = _marker_column_candidates(w, k, anchor_idx, clip_modes[int(color)])
        fallback = _marker_column_candidates(w, k, anchor_idx, "none")
        candidates = preferred or fallback
        if not candidates:
            return None
        marker_c = int(random.choice(candidates))
        grid[row, marker_c] = int(color)
        markers[int(color)] = (int(row), marker_c)

    if len(set(markers.values())) != k:
        return None

    expected = _build_expected(w, legend, markers, bar_heights)
    if expected.shape != (k, w):
        return None
    if np.count_nonzero(expected[0]) == 0:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
