from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BG = 0
DIVIDER = 5
MAROON = 9
FREE_COLORS = (1, 2, 3, 4, 6, 7, 8)
DIRS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

GEN_RETRIES = 800


def _majority_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _find_divider(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for c in range(w):
        col = grid[:, c]
        if np.all(col == col[0]) and int(col[0]) != bg:
            candidates.append((c, int(col[0])))

    assert candidates, "expected a full-height divider"
    grey_cols = [(c, color) for c, color in candidates if color == DIVIDER]
    if len(grey_cols) == 1:
        return grey_cols[0]
    assert len(candidates) == 1, "divider should be unique"
    return candidates[0]


def _extract_templates(grid: np.ndarray, divider_col: int, bg: int, divider_color: int) -> dict[int, tuple[tuple[int, int], ...]]:
    h, _ = grid.shape
    left = grid[:, :divider_col]
    mapping: dict[int, tuple[tuple[int, int], ...]] = {}

    for r in range(1, h - 1):
        for c in range(1, divider_col - 1):
            color = int(left[r, c])
            if color in (bg, divider_color, MAROON):
                continue
            assert color not in mapping, "legend center colors must be unique"

            dirs: list[tuple[int, int]] = []
            for dr, dc in DIRS8:
                v = int(left[r + dr, c + dc])
                assert v in (bg, MAROON), "legend neighbors must be background or maroon"
                if v == MAROON:
                    dirs.append((dr, dc))
            assert dirs, "every legend entry must encode at least one direction"
            mapping[color] = tuple(dirs)

    assert mapping, "expected at least one legend entry"
    return mapping


def _ordered_seeds(grid: np.ndarray, divider_col: int, bg: int, reverse: bool = False) -> list[tuple[int, int, int]]:
    right = grid[:, divider_col + 1 :]
    seeds = [
        (int(r), int(c + divider_col + 1), int(right[r, c]))
        for r, c in np.argwhere(right != bg)
    ]
    seeds.sort(key=lambda item: (item[1], item[0]), reverse=reverse)
    return seeds


def _apply_growth(
    grid: np.ndarray,
    divider_col: int,
    bg: int,
    mapping: dict[int, tuple[tuple[int, int], ...]],
    reverse: bool = False,
) -> np.ndarray:
    out = np.array(grid, copy=True)
    h, w = out.shape
    seeds = _ordered_seeds(out, divider_col, bg, reverse=reverse)

    for r, c, color in seeds:
        assert color in mapping, "every seed color must appear in the legend"
        for dr, dc in mapping[color]:
            nr, nc = r + dr, c + dc
            while 0 <= nr < h and 0 <= nc < w and int(out[nr, nc]) == bg:
                out[nr, nc] = color
                nr += dr
                nc += dc

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow colored rays on the right canvas by decoding direction templates from the left legend.

    Input structure:
    - A black background grid is split by one full-height grey divider.
    - Left of the divider are spaced 3x3 legend patches. Each patch has one unique center color,
      and maroon neighbors around that center encode growth directions.
    - Right of the divider are isolated single-pixel seeds whose colors match legend centers.

    Rule:
    - For each seed, look up the matching legend center color.
    - For every maroon offset in that legend patch, extend a same-colored ray from the seed in
      that direction until it would leave the grid or hit an already non-background cell.
    - Seeds are processed from smaller column to larger column, so earlier growth blocks later rays.
    """
    g = np.asarray(grid, dtype=int)
    bg = _majority_color(g)
    divider_col, divider_color = _find_divider(g, bg)
    mapping = _extract_templates(g, divider_col, bg, divider_color)
    return _apply_growth(g, divider_col, bg, mapping, reverse=False)


ALL_DIRECTION_SETS = [
    combo
    for size in (1, 2, 3, 4)
    for combo in combinations(DIRS8, size)
]


def _legend_slots(height: int, left_width: int) -> list[tuple[int, int]]:
    rows = list(range(1, height - 1, 4))
    cols = list(range(1, left_width - 1, 4))
    return [(r, c) for c in cols for r in rows]


def _sample_direction_sets(count: int, need_order_pair: bool) -> list[tuple[tuple[int, int], ...]] | None:
    for _ in range(300):
        chosen = random.sample(ALL_DIRECTION_SETS, count)
        has_cardinal = any(any(abs(dr) + abs(dc) == 1 for dr, dc in dirs) for dirs in chosen)
        has_diagonal = any(any(abs(dr) + abs(dc) == 2 for dr, dc in dirs) for dirs in chosen)
        if not (has_cardinal and has_diagonal):
            continue
        if need_order_pair and _find_order_pair(chosen) is None:
            continue
        return chosen
    return None


def _find_order_pair(direction_sets: list[tuple[tuple[int, int], ...]]) -> tuple[int, int, tuple[int, int], tuple[int, int]] | None:
    for i, dirs_a in enumerate(direction_sets):
        for j, dirs_b in enumerate(direction_sets):
            if i == j:
                continue
            if (1, 1) in dirs_a and (-1, 0) in dirs_b:
                return i, j, (1, 1), (-1, 0)
            if (-1, 1) in dirs_a and (1, 0) in dirs_b:
                return i, j, (-1, 1), (1, 0)
    return None


def _paint_legend(
    grid: np.ndarray,
    colors: list[int],
    direction_sets: list[tuple[tuple[int, int], ...]],
    slots: list[tuple[int, int]],
) -> dict[int, tuple[tuple[int, int], ...]]:
    mapping: dict[int, tuple[tuple[int, int], ...]] = {}
    for color, dirs, (r, c) in zip(colors, direction_sets, slots, strict=True):
        patch = grid[r - 1 : r + 2, c - 1 : c + 2]
        assert patch.shape == (3, 3)
        assert np.all(patch == BG)
        patch[1, 1] = color
        for dr, dc in dirs:
            patch[1 + dr, 1 + dc] = MAROON
        mapping[color] = dirs
    return mapping


def _chebyshev_far_enough(pos: tuple[int, int], others: list[tuple[int, int]]) -> bool:
    r, c = pos
    return all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in others)


def _seed_has_open_ray(
    pos: tuple[int, int],
    dirs: tuple[tuple[int, int], ...],
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
) -> bool:
    h, w = shape
    r, c = pos
    for dr, dc in dirs:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in occupied:
            return True
    return False


def _random_seed_positions(
    height: int,
    width: int,
    divider_col: int,
    seed_colors: list[int],
    mapping: dict[int, tuple[tuple[int, int], ...]],
) -> list[tuple[int, int, int]] | None:
    right_cols = list(range(divider_col + 2, width - 1))
    if len(right_cols) < len(seed_colors):
        return None

    chosen_cols = sorted(random.sample(right_cols, len(seed_colors)))
    placed: list[tuple[int, int, int]] = []
    occupied: set[tuple[int, int]] = {(r, divider_col) for r in range(height)}

    for color, col in zip(seed_colors, chosen_cols, strict=True):
        candidates = list(range(1, height - 1))
        random.shuffle(candidates)
        found = False
        for row in candidates:
            pos = (row, col)
            if not _chebyshev_far_enough(pos, [(r, c) for r, c, _ in placed]):
                continue
            if not _seed_has_open_ray(pos, mapping[color], occupied, (height, width)):
                continue
            placed.append((row, col, color))
            occupied.add((row, col))
            found = True
            break
        if not found:
            return None

    return placed


def _targeted_order_seeds(
    height: int,
    width: int,
    divider_col: int,
    seed_colors: list[int],
    mapping: dict[int, tuple[tuple[int, int], ...]],
    pair: tuple[int, int, tuple[int, int], tuple[int, int]],
    legend_colors: list[int],
) -> list[tuple[int, int, int]] | None:
    idx_a, idx_b, dir_a, _dir_b = pair
    color_a = legend_colors[idx_a]
    color_b = legend_colors[idx_b]
    colors = seed_colors[:]

    try:
        pos_a = colors.index(color_a)
        colors[0], colors[pos_a] = colors[pos_a], colors[0]
        pos_b = colors.index(color_b)
        colors[1], colors[pos_b] = colors[pos_b], colors[1]
    except ValueError:
        return None

    right_cols = list(range(divider_col + 2, width - 1))
    if len(right_cols) < len(colors):
        return None

    delta_max = min(4, len(right_cols) - 1)
    if delta_max < 2:
        return None
    delta = random.randint(2, delta_max)
    ca_index = random.randint(0, len(right_cols) - delta - 1)
    ca = right_cols[ca_index]
    cb = right_cols[ca_index + delta]

    if dir_a == (1, 1):
        ra_min = 1
        ra_max = height - 3 - delta
        if ra_min > ra_max:
            return None
        ra = random.randint(ra_min, ra_max)
        cross_r = ra + delta
        rb_min = cross_r + 2
        rb_max = height - 2
        if rb_min > rb_max:
            return None
        rb = random.randint(rb_min, rb_max)
    else:
        ra_min = 1 + delta
        ra_max = height - 2
        if ra_min > ra_max:
            return None
        ra = random.randint(ra_min, ra_max)
        cross_r = ra - delta
        rb_min = 1
        rb_max = cross_r - 2
        if rb_min > rb_max:
            return None
        rb = random.randint(rb_min, rb_max)

    placed = [(ra, ca, colors[0]), (rb, cb, colors[1])]
    occupied: set[tuple[int, int]] = {(r, divider_col) for r in range(height)}
    occupied.update((r, c) for r, c, _ in placed)

    if not _seed_has_open_ray((ra, ca), mapping[colors[0]], occupied - {(ra, ca)}, (height, width)):
        return None
    if not _seed_has_open_ray((rb, cb), mapping[colors[1]], occupied - {(rb, cb)}, (height, width)):
        return None

    remaining_cols = [c for c in right_cols if c not in (ca, cb)]
    if len(remaining_cols) < len(colors) - 2:
        return None
    extra_cols = sorted(random.sample(remaining_cols, len(colors) - 2))

    for color, col in zip(colors[2:], extra_cols, strict=True):
        candidates = list(range(1, height - 1))
        random.shuffle(candidates)
        found = False
        for row in candidates:
            pos = (row, col)
            if not _chebyshev_far_enough(pos, [(r, c) for r, c, _ in placed]):
                continue
            if not _seed_has_open_ray(pos, mapping[color], occupied, (height, width)):
                continue
            placed.append((row, col, color))
            occupied.add((row, col))
            found = True
            break
        if not found:
            return None

    placed.sort(key=lambda item: (item[1], item[0]))
    return placed


def _seed_colors_for_sample(legend_colors: list[int], count: int) -> list[int]:
    colors = legend_colors[:]
    while len(colors) < count:
        colors.append(random.choice(legend_colors))
    random.shuffle(colors)
    return colors


def _build_input() -> np.ndarray | None:
    height = random.randint(13, 21)
    left_width = random.randint(7, 11)
    right_width = random.randint(9, 14)
    width = left_width + 1 + right_width

    slots = _legend_slots(height, left_width)
    max_legends = min(len(slots), len(FREE_COLORS), 5)
    if max_legends < 2:
        return None

    legend_count = random.randint(2, max_legends)
    want_order_effect = random.random() < 0.5
    direction_sets = _sample_direction_sets(legend_count, need_order_pair=want_order_effect)
    if direction_sets is None:
        return None

    legend_colors = random.sample(list(FREE_COLORS), legend_count)
    chosen_slots = random.sample(slots, legend_count)

    grid = np.full((height, width), BG, dtype=int)
    grid[:, left_width] = DIVIDER
    mapping = _paint_legend(grid, legend_colors, direction_sets, chosen_slots)

    max_seeds = min(6, right_width - 2)
    if max_seeds < legend_count:
        return None
    seed_count = random.randint(legend_count, max_seeds)
    seed_colors = _seed_colors_for_sample(legend_colors, seed_count)

    if want_order_effect:
        pair = _find_order_pair(direction_sets)
        assert pair is not None
        seeds = _targeted_order_seeds(height, width, left_width, seed_colors, mapping, pair, legend_colors)
    else:
        seeds = _random_seed_positions(height, width, left_width, seed_colors, mapping)
    if seeds is None:
        return None

    for r, c, color in seeds:
        assert int(grid[r, c]) == BG
        grid[r, c] = color

    out_forward = _apply_growth(grid, left_width, BG, mapping, reverse=False)
    out_reverse = _apply_growth(grid, left_width, BG, mapping, reverse=True)
    order_effect = not np.array_equal(out_forward, out_reverse)
    if want_order_effect != order_effect:
        return None

    changed = int(np.sum(out_forward != grid))
    if changed < max(4, seed_count):
        return None

    if not np.array_equal(main(grid), out_forward):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 4")
