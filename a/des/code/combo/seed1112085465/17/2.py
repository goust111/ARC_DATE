import random

import numpy as np


BG_COLOR = 8
GEN_MAX_ATTEMPTS = 300
GEN_HEIGHT_RANGE = (9, 18)
GEN_WIDTH_RANGE = (10, 20)
GEN_COLOR_COUNT_RANGE = (2, 5)
GEN_EXTRA_SEEDS_MAX = 5
GEN_SIDES = ("top", "bottom")
GEN_OVERLAP_PROB = 0.5
NON_BG_COLORS = tuple(color for color in range(10) if color != BG_COLOR)
CROSS_OFFSETS = (
    (-2, 0),
    (-1, 0),
    (0, -2),
    (0, -1),
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (2, 0),
)


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _cross_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(r + dr, c + dc) for dr, dc in CROSS_OFFSETS}


def _find_legend(
    grid: np.ndarray, bg: int
) -> tuple[int, np.ndarray, list[int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, np.ndarray, list[int], np.ndarray]] = []

    for row in (0, h - 1):
        cols = np.flatnonzero(g[row] != bg)
        if cols.size < 2:
            continue
        if not np.array_equal(cols, np.arange(cols[0], cols[-1] + 1)):
            continue
        colors = [int(v) for v in g[row, cols]]
        if len(set(colors)) != len(colors):
            continue

        legend_mask = np.zeros((h, w), dtype=bool)
        legend_mask[row, cols] = True
        remaining = g[~legend_mask]
        remaining_non_bg = remaining[remaining != bg]
        if remaining_non_bg.size == 0:
            continue
        remaining_colors = {int(v) for v in remaining_non_bg}
        if remaining_colors != set(colors):
            continue
        if any(np.count_nonzero((g == color) & ~legend_mask) == 0 for color in colors):
            continue

        candidates.append((row, cols.astype(int), colors, legend_mask))

    assert len(candidates) == 1, "expected exactly one horizontal legend on the top or bottom edge"
    return candidates[0]


def _render_output(
    shape: tuple[int, int], bg: int, color_order: list[int], seeds_by_color: dict[int, list[tuple[int, int]]]
) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), bg, dtype=int)
    for color in color_order:
        for r, c in seeds_by_color[color]:
            for dr, dc in CROSS_OFFSETS:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each seed into a radius-2 cross, using the edge legend's left-to-right
    order as the paint priority. Later legend colors overwrite earlier ones.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    _legend_row, _legend_cols, color_order, legend_mask = _find_legend(g, bg)

    seeds_by_color: dict[int, list[tuple[int, int]]] = {}
    non_legend_non_bg = np.argwhere((g != bg) & ~legend_mask)
    assert non_legend_non_bg.size > 0, "expected at least one interior seed"

    for color in color_order:
        coords = np.argwhere((g == color) & ~legend_mask)
        assert coords.size > 0, "each legend color must mark at least one seed"
        seeds_by_color[color] = [(int(r), int(c)) for r, c in coords]

    seen_colors = {int(g[r, c]) for r, c in non_legend_non_bg}
    assert seen_colors == set(color_order), "all seed colors must be listed in the legend"

    return _render_output(g.shape, bg, color_order, seeds_by_color)


def _seed_row_limits(height: int, legend_row: int) -> tuple[int, int]:
    low = 2
    high = height - 3
    if legend_row == 0:
        low = max(low, 3)
    else:
        high = min(high, height - 4)
    return low, high


def _allowed_centers(height: int, width: int, legend_row: int) -> list[tuple[int, int]]:
    row_low, row_high = _seed_row_limits(height, legend_row)
    if row_low > row_high or width < 5:
        return []
    return [(r, c) for r in range(row_low, row_high + 1) for c in range(2, width - 2)]


def _seed_counts(rng: random.Random, color_count: int) -> list[int]:
    counts = [1] * color_count
    extra_budget = rng.randint(0, min(GEN_EXTRA_SEEDS_MAX, color_count * 2))
    for _ in range(extra_budget):
        available = [idx for idx, count in enumerate(counts) if count < 3]
        if not available:
            break
        counts[rng.choice(available)] += 1
    return counts


def _pick_disjoint_centers(
    rng: random.Random,
    candidates: list[tuple[int, int]],
    count: int,
    forbidden_cells: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]] | None:
    order = list(candidates)
    rng.shuffle(order)
    occupied = set() if forbidden_cells is None else set(forbidden_cells)
    chosen: list[tuple[int, int]] = []

    for center in order:
        cells = _cross_cells(center)
        if occupied & cells:
            continue
        chosen.append(center)
        occupied |= cells
        if len(chosen) == count:
            return chosen
    return None


def _overlap_pairs(
    candidates: list[tuple[int, int]], axis: str
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for i, first in enumerate(candidates):
        for second in candidates[i + 1 :]:
            if axis == "horizontal":
                aligned = first[0] == second[0] and 2 <= abs(first[1] - second[1]) <= 4
            else:
                aligned = first[1] == second[1] and 2 <= abs(first[0] - second[0]) <= 4
            if aligned:
                pairs.append((first, second))
    return pairs


def _place_seeds(
    rng: random.Random,
    candidates: list[tuple[int, int]],
    slots: list[int],
    force_overlap: bool,
) -> dict[int, list[tuple[int, int]]] | None:
    placed: list[tuple[int, int] | None] = [None] * len(slots)

    if force_overlap:
        distinct_pairs = [(i, j) for i in range(len(slots)) for j in range(i + 1, len(slots)) if slots[i] != slots[j]]
        if not distinct_pairs:
            return None
        rng.shuffle(distinct_pairs)

        axes = ["horizontal", "vertical"]
        rng.shuffle(axes)
        pair_candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for axis in axes:
            pair_candidates.extend(_overlap_pairs(candidates, axis))
        if not pair_candidates:
            return None
        rng.shuffle(pair_candidates)

        for slot_a, slot_b in distinct_pairs:
            for center_a, center_b in pair_candidates:
                used_cells = _cross_cells(center_a) | _cross_cells(center_b)
                remaining_count = len(slots) - 2
                remainder = [c for c in candidates if c not in (center_a, center_b)]
                extra_centers = _pick_disjoint_centers(rng, remainder, remaining_count, forbidden_cells=used_cells)
                if extra_centers is None:
                    continue
                placed[slot_a] = center_a
                placed[slot_b] = center_b
                extra_iter = iter(extra_centers)
                for idx in range(len(slots)):
                    if placed[idx] is None:
                        placed[idx] = next(extra_iter)
                break
            if all(center is not None for center in placed):
                break
        if any(center is None for center in placed):
            return None
    else:
        chosen = _pick_disjoint_centers(rng, candidates, len(slots))
        if chosen is None:
            return None
        placed = chosen

    seeds_by_color: dict[int, list[tuple[int, int]]] = {}
    for color, center in zip(slots, placed):
        seeds_by_color.setdefault(color, []).append(center)
    return seeds_by_color


def _gen_attempt(
    rng: random.Random, legend_side: str, force_overlap: bool
) -> np.ndarray | None:
    color_count = rng.randint(*GEN_COLOR_COUNT_RANGE)
    height = rng.randint(*GEN_HEIGHT_RANGE)
    width = rng.randint(max(GEN_WIDTH_RANGE[0], color_count), GEN_WIDTH_RANGE[1])
    legend_row = 0 if legend_side == "top" else height - 1
    legend_colors = rng.sample(NON_BG_COLORS, color_count)

    candidates = _allowed_centers(height, width, legend_row)
    if not candidates:
        return None

    counts = _seed_counts(rng, color_count)
    slots = [color for color, count in zip(legend_colors, counts) for _ in range(count)]
    if len(candidates) < len(slots):
        return None

    seeds_by_color = _place_seeds(rng, candidates, slots, force_overlap=force_overlap)
    if seeds_by_color is None:
        return None

    grid = np.full((height, width), BG_COLOR, dtype=int)
    legend_start = rng.randint(0, width - color_count)
    grid[legend_row, legend_start : legend_start + color_count] = legend_colors
    for color, centers in seeds_by_color.items():
        for r, c in centers:
            grid[r, c] = color

    expected = _render_output(grid.shape, BG_COLOR, legend_colors, seeds_by_color)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    legend_side = rng.choice(GEN_SIDES)
    force_overlap = rng.random() < GEN_OVERLAP_PROB
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng, legend_side=legend_side, force_overlap=force_overlap)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
