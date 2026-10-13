from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
CHANNEL_COLORS = (5, 8)
SEED_PALETTE = (1, 2, 3, 4, 6, 7, 9)
COUNT_OPTIONS = (
    (1, 2, 3),
    (1, 2, 4),
    (1, 3, 4),
    (2, 3, 4),
)
GRID_SIZE_RANGE = (12, 24)
MAX_GENERATION_ATTEMPTS = 400


def _colors_present(g: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    excluded = exclude or set()
    return [int(color) for color in np.unique(g) if int(color) not in excluded]


def _dominant_color(g: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(g, dtype=int).reshape(-1)))
    color, _count = max(counts.items(), key=lambda item: item[1])
    return int(color)


def _first_occurrence(g: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(g == color)
    assert len(coords) > 0
    return tuple(map(int, coords[0]))


def _line_score(mask: np.ndarray) -> int:
    horiz = int(np.sum(mask[:, 1:] & mask[:, :-1]))
    vert = int(np.sum(mask[1:, :] & mask[:-1, :]))
    return horiz + vert


def _channel_color(g: np.ndarray, bg: int) -> int:
    colors = _colors_present(g, exclude={bg})
    assert colors, "expected at least one foreground color"

    ranked = []
    for color in colors:
        mask = g == color
        ranked.append(
            (
                -int(np.sum(mask)),
                -_line_score(mask),
                _first_occurrence(g, color),
                int(color),
            )
        )
    ranked.sort()
    return int(ranked[0][3])


def _cell_features(mask: np.ndarray, r: int, c: int) -> tuple[bool, bool]:
    h, w = mask.shape
    left = c > 0 and bool(mask[r, c - 1])
    right = c + 1 < w and bool(mask[r, c + 1])
    up = r > 0 and bool(mask[r - 1, c])
    down = r + 1 < h and bool(mask[r + 1, c])
    return left or right, up or down


def _cell_category(mask: np.ndarray, r: int, c: int) -> str:
    horizontal, vertical = _cell_features(mask, r, c)
    assert horizontal or vertical, "seed must lie on a channel"
    if horizontal and vertical:
        return "cross"
    if horizontal:
        return "horizontal"
    return "vertical"


def _growth_step(mask: np.ndarray, r: int, c: int) -> tuple[int, int]:
    horizontal, vertical = _cell_features(mask, r, c)
    assert horizontal or vertical, "seed must lie on a channel"
    if horizontal:
        return 0, 1
    return 1, 0


def _trace_band(mask: np.ndarray, start: tuple[int, int], length: int) -> list[tuple[int, int]]:
    r, c = map(int, start)
    dr, dc = _growth_step(mask, r, c)
    h, w = mask.shape

    cells = [(r, c)]
    nr, nc = r + dr, c + dc
    while len(cells) < length and 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]):
        cells.append((int(nr), int(nc)))
        nr += dr
        nc += dc
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each colored seed into a same-color band whose length equals that color's total seed count.

    Input structure:
    - The background is black and the remaining pixels lie on a one-pixel-wide network
      of horizontal and vertical channels.
    - One foreground color forms the visible channel cells, while other foreground colors
      appear as single-pixel seeds placed on top of those channels.
    - Channels may cross; seeds can lie on horizontal-only cells, vertical-only cells,
      or crossing cells.

    Rule:
    - Count how many seeds use each seed color, and use that count as the growth length
      for every seed of that color.
    - A seed on a horizontal channel grows rightward; a seed on a vertical channel grows
      downward; a seed on a crossing also grows rightward.
    - Growth includes the seed cell itself and continues only across existing channel
      cells, stopping early at the channel end without turning.
    - Process seed colors from smaller occurrence count to larger occurrence count; later
      colors overwrite earlier bands where they overlap.

    Key insight:
    - The growth length comes from the global frequency of a seed color, while the growth
      direction comes from local channel geometry rather than from the color identity itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(_dominant_color(g))
    channel = int(_channel_color(g, bg))
    seed_colors = _colors_present(g, exclude={bg, channel})
    assert seed_colors, "expected at least one seed color"

    channel_mask = g != bg
    color_to_seeds: dict[int, list[tuple[int, int]]] = {}
    color_to_count: dict[int, int] = {}
    for color in seed_colors:
        coords = [tuple(map(int, coord)) for coord in np.argwhere(g == color)]
        assert coords, "seed colors must appear at least once"
        color_to_seeds[int(color)] = coords
        color_to_count[int(color)] = len(coords)

    ordered_colors = sorted(
        seed_colors,
        key=lambda color: (color_to_count[int(color)], _first_occurrence(g, int(color))),
    )

    out = g.copy()
    for color in ordered_colors:
        band_length = color_to_count[int(color)]
        for seed in color_to_seeds[int(color)]:
            for r, c in _trace_band(channel_mask, seed, band_length):
                out[r, c] = int(color)

    return out


def _all_channel_cells(
    row_lines: list[int],
    col_lines: list[int],
    c0: int,
    c1: int,
    r0: int,
    r1: int,
) -> list[tuple[int, int]]:
    cells = set()
    for r in row_lines:
        for c in range(c0, c1 + 1):
            cells.add((int(r), int(c)))
    for c in col_lines:
        for r in range(r0, r1 + 1):
            cells.add((int(r), int(c)))
    return sorted(cells)


def _build_channel_grid(
    height: int,
    width: int,
    channel_color: int,
    row_lines: list[int],
    col_lines: list[int],
    c0: int,
    c1: int,
    r0: int,
    r1: int,
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for r in row_lines:
        grid[r, c0 : c1 + 1] = channel_color
    for c in col_lines:
        grid[r0 : r1 + 1, c] = channel_color
    return grid


def _not_adjacent(candidate: tuple[int, int], placed: list[tuple[int, int]]) -> bool:
    cr, cc = candidate
    for pr, pc in placed:
        if abs(cr - pr) + abs(cc - pc) <= 1:
            return False
    return True


def _available_length(mask: np.ndarray, cell: tuple[int, int]) -> int:
    return len(_trace_band(mask, cell, 1000))


def _band_cells(mask: np.ndarray, seeds: list[tuple[int, int]], length: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for seed in seeds:
        cells.update(_trace_band(mask, seed, length))
    return cells


def _find_overlap_pair(
    row_lines: list[int],
    col_lines: list[int],
    c1: int,
    high_count: int,
    mid_count: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    col_set = set(col_lines)
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in row_lines:
        late_cols = [c for c in range(c1 - mid_count + 2, c1 + 1) if c not in col_set]
        if not late_cols:
            continue
        for c_cross in col_lines:
            if c1 - c_cross + 1 < high_count:
                continue
            for c_h in late_cols:
                if c_h <= c_cross:
                    continue
                if c_h - c_cross <= 1:
                    continue
                if c_h - c_cross + 1 <= high_count:
                    pairs.append(((int(r), int(c_cross)), (int(r), int(c_h))))
    if not pairs:
        return None
    return random.choice(pairs)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_SIZE_RANGE)
    width = random.randint(*GRID_SIZE_RANGE)

    if height < 12 or width < 12:
        return None

    r0 = random.randint(1, height - 8)
    r1 = random.randint(r0 + 6, height - 2)
    c0 = random.randint(1, width - 8)
    c1 = random.randint(c0 + 6, width - 2)

    inner_rows = list(range(r0 + 1, r1))
    inner_cols = list(range(c0 + 1, c1))
    if len(inner_rows) < 2 or len(inner_cols) < 2:
        return None

    row_count = random.randint(2, min(4, len(inner_rows)))
    col_count = random.randint(2, min(4, len(inner_cols)))
    row_lines = sorted(random.sample(inner_rows, row_count))
    col_lines = sorted(random.sample(inner_cols, col_count))

    channel_color = int(random.choice(CHANNEL_COLORS))
    base = _build_channel_grid(height, width, channel_color, row_lines, col_lines, c0, c1, r0, r1)
    mask = base != BG

    counts = random.choice(COUNT_OPTIONS)
    low_count, mid_count, high_count = map(int, counts)
    low_color, mid_color, high_color = random.sample([c for c in SEED_PALETTE if c != channel_color], 3)

    overlap_pair = _find_overlap_pair(row_lines, col_lines, c1, high_count, mid_count)
    if overlap_pair is None:
        return None
    cross_seed, horizontal_seed = overlap_pair

    v_candidates = [
        (r, c)
        for c in col_lines
        for r in range(r0, r1 + 1)
        if r not in row_lines and _available_length(mask, (r, c)) >= low_count
    ]
    placed = [cross_seed, horizontal_seed]
    v_candidates = [cell for cell in v_candidates if cell not in placed and _not_adjacent(cell, placed)]
    if not v_candidates:
        return None
    vertical_seed = random.choice(v_candidates)

    assignments: dict[int, list[tuple[int, int]]] = {
        int(low_color): [vertical_seed],
        int(mid_color): [horizontal_seed],
        int(high_color): [cross_seed],
    }
    color_to_target = {
        int(low_color): low_count,
        int(mid_color): mid_count,
        int(high_color): high_count,
    }

    all_cells = _all_channel_cells(row_lines, col_lines, c0, c1, r0, r1)
    used = [vertical_seed, horizontal_seed, cross_seed]

    fill_order = [high_color, mid_color, low_color]
    for color in fill_order:
        while len(assignments[int(color)]) < color_to_target[int(color)]:
            candidates = [
                cell
                for cell in all_cells
                if cell not in used and _not_adjacent(cell, used)
            ]
            if not candidates:
                return None
            candidate = random.choice(candidates)
            assignments[int(color)].append(candidate)
            used.append(candidate)

    inp = base.copy()
    for color, seeds in assignments.items():
        for r, c in seeds:
            inp[r, c] = int(color)

    seed_colors = [int(low_color), int(mid_color), int(high_color)]
    seed_counts = {color: len(assignments[color]) for color in seed_colors}
    assert len(set(seed_counts.values())) == len(seed_counts)

    categories = [_cell_category(mask, r, c) for r, c in used]
    if not {"horizontal", "vertical", "cross"}.issubset(set(categories)):
        return None

    growth_lengths = {color: _band_cells(mask, assignments[color], seed_counts[color]) for color in seed_colors}
    if not (growth_lengths[int(low_color)] & growth_lengths[int(high_color)] or growth_lengths[int(mid_color)] & growth_lengths[int(high_color)]):
        return None

    trunc_flags = []
    full_flags = []
    for color, seeds in assignments.items():
        band_length = seed_counts[int(color)]
        for seed in seeds:
            available = _available_length(mask, seed)
            trunc_flags.append(available < band_length)
            full_flags.append(available >= band_length)
    if not any(trunc_flags) or not any(full_flags):
        return None

    non_bg = _colors_present(inp, exclude={BG})
    channel_candidate = _channel_color(inp, BG)
    if channel_candidate != channel_color:
        return None
    if int(np.sum(inp == channel_color)) <= max(seed_counts.values()):
        return None
    if len(non_bg) != 4:
        return None

    try:
        out = main(inp)
    except AssertionError:
        return None
    if out.shape != inp.shape or not np.any(out != inp):
        return None
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
