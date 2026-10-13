from __future__ import annotations

from collections import Counter
import random

import numpy as np


Span = tuple[int, int]


def _false_runs(mask: np.ndarray) -> list[Span]:
    spans: list[Span] = []
    start: int | None = None
    for idx, is_true in enumerate(mask.astype(bool).tolist()):
        if not is_true and start is None:
            start = idx
        elif is_true and start is not None:
            spans.append((start, idx - 1))
            start = None
    if start is not None:
        spans.append((start, len(mask) - 1))
    return spans


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    top = max(counts.values())
    winners = [color for color, count in counts.items() if count == top]
    assert len(winners) == 1, "expected a unique dominant color"
    return int(winners[0])


def _decode_partition(grid: np.ndarray) -> tuple[int, list[Span], list[Span], np.ndarray, np.ndarray]:
    candidates = []
    for color in map(int, np.unique(grid)):
        row_sep = np.all(grid == color, axis=1)
        col_sep = np.all(grid == color, axis=0)
        if int(row_sep.sum()) < 2 or int(col_sep.sum()) < 2:
            continue

        row_spans = _false_runs(row_sep)
        col_spans = _false_runs(col_sep)
        if not row_spans or not col_spans:
            continue

        rows = len(row_spans)
        cols = len(col_spans)
        if any(r1 - r0 + 1 != rows for r0, r1 in row_spans):
            continue
        if any(c1 - c0 + 1 != cols for c0, c1 in col_spans):
            continue
        candidates.append((color, row_spans, col_spans, row_sep, col_sep))

    assert len(candidates) == 1, "expected exactly one valid separator layout"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the occupied-region map, find the self-descriptive miniature region, and use it as a recoloring decree.

    Input structure:
    - Full separator rows and separator columns divide the grid into R by C equal regions.
    - Each region interior has size R by C, matching the macro region layout.
    - Some regions contain sparse colored dots while others are completely empty.

    Rule:
    - Build the macro occupancy map by marking which regions contain any dot.
    - Among the occupied regions, find the unique one whose own R by C dot mask exactly matches that macro map.
    - Treat that region as a color legend: its pixel at (i, j) gives the target color for region (i, j), and every dot in that region is recolored to that color.

    Key insight:
    - One region is a self-portrait of the whole partition map, and that self-portrait simultaneously serves as the per-region recoloring instruction table.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _sep, row_spans, col_spans, row_sep, col_sep = _decode_partition(g)
    non_separator = g[~row_sep][:, ~col_sep]
    bg = _unique_mode(non_separator)

    rows = len(row_spans)
    cols = len(col_spans)
    occupancy = np.zeros((rows, cols), dtype=bool)
    regions: list[list[np.ndarray]] = []

    for i, (r0, r1) in enumerate(row_spans):
        region_row: list[np.ndarray] = []
        for j, (c0, c1) in enumerate(col_spans):
            region = g[r0 : r1 + 1, c0 : c1 + 1]
            assert region.shape == (rows, cols), "region size must match macro layout"
            occupancy[i, j] = bool(np.any(region != bg))
            region_row.append(region)
        regions.append(region_row)

    matches: list[tuple[int, int]] = []
    for i in range(rows):
        for j in range(cols):
            if not occupancy[i, j]:
                continue
            if np.array_equal(regions[i][j] != bg, occupancy):
                matches.append((i, j))

    assert len(matches) == 1, "expected exactly one self-descriptive miniature"
    mini_r, mini_c = matches[0]
    legend = regions[mini_r][mini_c]

    out = g.copy()
    for i, (r0, r1) in enumerate(row_spans):
        for j, (c0, c1) in enumerate(col_spans):
            if not occupancy[i, j]:
                continue
            target = int(legend[i, j])
            assert target != bg, "occupied regions must map to visible target colors"
            region_mask = g[r0 : r1 + 1, c0 : c1 + 1] != bg
            out_region = out[r0 : r1 + 1, c0 : c1 + 1]
            out_region[region_mask] = target
    return out


PALETTE = list(range(10))
SIZE_CHOICES = [2, 3, 4]
MAX_GENERATION_ATTEMPTS = 400


def _random_mask(rows: int, cols: int, count: int, *, forbidden: set[bytes] | None = None) -> np.ndarray:
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    forbidden = forbidden or set()
    for _ in range(200):
        chosen = random.sample(cells, count)
        mask = np.zeros((rows, cols), dtype=bool)
        for r, c in chosen:
            mask[r, c] = True
        if mask.tobytes() not in forbidden:
            return mask
    raise RuntimeError("failed to sample a distinct mask")


def _sample_target_colors(count: int, palette: list[int]) -> list[int]:
    if count == 1:
        return [random.choice(palette)]

    distinct = random.sample(palette, k=min(len(palette), random.randint(2, min(4, len(palette)))))
    colors = [random.choice(distinct) for _ in range(count)]
    if len(set(colors)) == 1:
        colors[0] = distinct[0]
        colors[-1] = distinct[-1]
    return colors


def _region_spans(rows: int, cols: int) -> tuple[list[Span], list[Span]]:
    row_spans = []
    for i in range(rows):
        start = 1 + i * (rows + 1)
        row_spans.append((start, start + rows - 1))

    col_spans = []
    for j in range(cols):
        start = 1 + j * (cols + 1)
        col_spans.append((start, start + cols - 1))
    return row_spans, col_spans


def _paint_mask(region: np.ndarray, mask: np.ndarray, colors: np.ndarray) -> None:
    region[mask] = colors[mask]


def _build_instance() -> tuple[np.ndarray, np.ndarray]:
    rows = random.choice(SIZE_CHOICES)
    cols = random.choice(SIZE_CHOICES)
    total_regions = rows * cols
    max_occupied = max(2, min(total_regions - 1, (total_regions + 1) // 2))
    occupied_count = random.randint(2, max_occupied)

    color_pool = random.sample(PALETTE, k=10)
    sep = color_pool[0]
    bg = color_pool[1]
    dot_palette = [color for color in PALETTE if color not in {sep, bg}]
    assert len(dot_palette) >= 2

    flat_indices = random.sample(range(total_regions), occupied_count)
    occupancy = np.zeros((rows, cols), dtype=bool)
    for idx in flat_indices:
        occupancy[idx // cols, idx % cols] = True

    occupied_positions = [(idx // cols, idx % cols) for idx in flat_indices]
    mini_r, mini_c = random.choice(occupied_positions)
    distractor_choices = [pos for pos in occupied_positions if pos != (mini_r, mini_c)]
    distractor_r, distractor_c = random.choice(distractor_choices)

    target_colors = np.full((rows, cols), bg, dtype=int)
    chosen_targets = _sample_target_colors(occupied_count, dot_palette)
    for (r, c), color in zip(occupied_positions, chosen_targets):
        target_colors[r, c] = int(color)

    height = rows * rows + rows + 1
    width = cols * cols + cols + 1
    grid = np.full((height, width), sep, dtype=int)
    expected = np.full_like(grid, sep)

    row_spans, col_spans = _region_spans(rows, cols)
    region_masks: dict[tuple[int, int], np.ndarray] = {}

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            grid[r0 : r1 + 1, c0 : c1 + 1] = bg
            expected[r0 : r1 + 1, c0 : c1 + 1] = bg

    miniature_mask = occupancy.copy()
    forbidden = {miniature_mask.tobytes()}

    for i, (r0, r1) in enumerate(row_spans):
        for j, (c0, c1) in enumerate(col_spans):
            if not occupancy[i, j]:
                continue

            if (i, j) == (mini_r, mini_c):
                mask = miniature_mask
                colors = target_colors.copy()
            elif (i, j) == (distractor_r, distractor_c):
                mask = _random_mask(rows, cols, occupied_count, forbidden=forbidden)
                colors = np.full((rows, cols), bg, dtype=int)
                for rr, cc in np.argwhere(mask):
                    colors[int(rr), int(cc)] = random.choice(dot_palette)
            else:
                count = random.randint(1, max(1, occupied_count - 1))
                mask = _random_mask(rows, cols, count, forbidden=forbidden)
                colors = np.full((rows, cols), bg, dtype=int)
                dot_cells = [tuple(map(int, cell)) for cell in np.argwhere(mask)]
                alt_palette = [color for color in dot_palette if color != int(target_colors[i, j])]
                for rr, cc in dot_cells:
                    colors[rr, cc] = random.choice(alt_palette or dot_palette)

            region_masks[(i, j)] = mask.copy()
            region = grid[r0 : r1 + 1, c0 : c1 + 1]
            _paint_mask(region, mask, colors)

            expected_region = expected[r0 : r1 + 1, c0 : c1 + 1]
            expected_region[mask] = int(target_colors[i, j])

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp, out = _build_instance()
        try:
            solved = main(inp)
        except Exception:
            continue
        if np.array_equal(solved, out):
            return inp
    raise RuntimeError("failed to generate a valid input")
