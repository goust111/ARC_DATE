from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _parse_layout(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 5 and g.shape[1] >= 7

    bg = _get_bg(g)
    gap_row = g.shape[0] - 2
    legend_row = g.shape[0] - 1
    assert np.all(g[gap_row] == bg), "expected a blank separator row above the legend"
    assert np.any(g[legend_row] != bg), "expected a non-empty legend row"

    main = g[:gap_row]
    candidates = []
    for c in range(g.shape[1]):
        col = main[:, c]
        if col[0] != bg and np.all(col == col[0]) and g[gap_row, c] == bg and g[legend_row, c] == bg:
            candidates.append(c)
    assert len(candidates) == 1, "expected one solid separator column in the main region"

    sep = candidates[0]
    assert 2 * sep + 1 == g.shape[1], "separator must split two equal-width halves"
    sep_color = int(main[0, sep])

    side_mask = np.ones(main.shape, dtype=bool)
    side_mask[:, sep] = False
    assert not np.any(main[side_mask] == sep_color), "separator color must be confined to the divider"
    return main, g[legend_row], sep, bg, sep_color


def _parse_legend_pairs(legend: np.ndarray, bg: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    c = 0
    width = legend.shape[0]
    while c < width:
        if int(legend[c]) == bg:
            c += 1
            continue
        assert c + 1 < width and int(legend[c + 1]) != bg, "legend symbols must come in adjacent pairs"
        assert c == 0 or int(legend[c - 1]) == bg, "legend pairs must be background-separated"
        assert c + 2 == width or int(legend[c + 2]) == bg, "legend pairs must be exactly two cells wide"
        strong = int(legend[c])
        weak = int(legend[c + 1])
        assert strong != weak, "a legend relation must compare two different colors"
        pairs.append((strong, weak))
        c += 2
    assert len(pairs) >= 3, "expected multiple legend relations"
    return pairs


def _dominance_order(colors: set[int], pairs: list[tuple[int, int]]) -> tuple[list[int], dict[int, int], dict[int, set[int]]]:
    succ = {color: set() for color in colors}
    pred = {color: set() for color in colors}
    for strong, weak in pairs:
        assert strong in colors and weak in colors, "legend colors must all appear in the folded scene"
        if weak not in succ[strong]:
            succ[strong].add(weak)
            pred[weak].add(strong)

    remaining = set(colors)
    order: list[int] = []
    while remaining:
        ready = [color for color in remaining if not (pred[color] & remaining)]
        assert len(ready) == 1, "legend must imply a unique total order"
        current = ready[0]
        order.append(current)
        remaining.remove(current)

    reach: dict[int, set[int]] = {}
    for color in colors:
        seen: set[int] = set()
        todo = deque([color])
        while todo:
            node = todo.popleft()
            for nxt in succ[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    todo.append(nxt)
        reach[color] = seen

    for i, strong in enumerate(order):
        for weak in order[i + 1 :]:
            assert weak in reach[strong], "legend relations must connect into a full dominance chain"

    return order, {color: idx for idx, color in enumerate(order)}, reach


def main(grid: np.ndarray) -> np.ndarray:
    """Fold across the vertical separator and resolve color clashes by the legend's transitive dominance order.

    Input structure:
    - The upper main region contains a full-height non-background vertical separator with equal-width left and right halves.
    - Below the main region is one all-background row, and the bottom row is a legend made of background-separated adjacent color pairs.
    - Each legend pair means "left color defeats right color", and the full set of pairs links all scene colors into one strict order.

    Rule:
    - Remove the separator, blank row, and legend.
    - Mirror the right half horizontally onto the left-half coordinate system.
    - Keep a lone foreground color when only one side has it, keep the color when both sides agree, and when two different colors overlap choose the stronger color from the legend's transitive closure.

    Key insight:
    - The legend does not map colors directly to outputs; it defines a dominance chain whose indirect comparisons are needed to arbitrate most fold conflicts.
    """
    main_region, legend, sep, bg, _sep_color = _parse_layout(grid)
    left = main_region[:, :sep]
    right = np.fliplr(main_region[:, sep + 1 :])

    pairs = _parse_legend_pairs(legend, bg)
    scene_colors = set(map(int, np.unique(main_region))) - {bg}
    scene_colors.discard(int(main_region[0, sep]))
    legend_colors = {color for pair in pairs for color in pair}
    assert scene_colors == legend_colors, "legend must cover exactly the foreground colors used in the scene"

    _order, rank, _reach = _dominance_order(scene_colors, pairs)

    out = np.full(left.shape, bg, dtype=int)
    for r in range(left.shape[0]):
        for c in range(left.shape[1]):
            a = int(left[r, c])
            b = int(right[r, c])
            if a == bg:
                out[r, c] = b
            elif b == bg or a == b:
                out[r, c] = a
            else:
                out[r, c] = a if rank[a] < rank[b] else b
    return out


BG = 0
PALETTE = tuple(range(1, 10))
COLOR_COUNT_CHOICES = (4, 5)
HALF_WIDTH_MIN = 7
HALF_WIDTH_MAX = 12
MAIN_HEIGHT_MIN = 7
MAIN_HEIGHT_MAX = 11
MAX_GENERATION_ATTEMPTS = 400


def _random_composition(total: int, parts: int, rng: random.Random) -> list[int]:
    values = [0] * parts
    for _ in range(total):
        values[rng.randrange(parts)] += 1
    return values


def _side_min_width(pair_count: int) -> int:
    if pair_count == 0:
        return 0
    return 3 * pair_count - 1


def _side_pair_starts(pair_count: int, side_width: int, rng: random.Random) -> list[int]:
    if pair_count == 0:
        return []
    extra = side_width - _side_min_width(pair_count)
    assert extra >= 0
    extras = _random_composition(extra, pair_count + 1, rng)
    starts = []
    cursor = extras[0]
    for idx in range(pair_count):
        starts.append(cursor)
        cursor += 2
        if idx < pair_count - 1:
            cursor += 1 + extras[idx + 1]
    return starts


def _build_legend(display_pairs: list[tuple[int, int]], total_width: int, sep: int, rng: random.Random) -> np.ndarray:
    pair_count = len(display_pairs)
    feasible_left_counts = [
        left_count
        for left_count in range(1, pair_count)
        if _side_min_width(left_count) <= sep and _side_min_width(pair_count - left_count) <= sep
    ]
    assert feasible_left_counts, "legend pairs must fit on both sides of the separator"
    left_count = rng.choice(feasible_left_counts)
    left_pairs = display_pairs[:left_count]
    right_pairs = display_pairs[left_count:]

    legend = np.full(total_width, BG, dtype=int)
    for pairs, offset in ((left_pairs, 0), (right_pairs, sep + 1)):
        starts = _side_pair_starts(len(pairs), sep, rng)
        for (strong, weak), start in zip(pairs, starts):
            legend[offset + start] = strong
            legend[offset + start + 1] = weak
    return legend


def _planned_counts(color_count: int, cell_count: int, rng: random.Random) -> tuple[int, int, int, int, int]:
    direct = rng.randint(1, 3)
    indirect = rng.randint(2, 4)
    same = rng.randint(1, 2)
    left_only = rng.randint(color_count, color_count + 2)
    right_only = rng.randint(color_count, color_count + 2)
    if direct + indirect + same + left_only + right_only >= cell_count:
        return 0, 0, 0, 0, 0
    return direct, indirect, same, left_only, right_only


def _fill_scene(
    height: int,
    width: int,
    order: list[int],
    direct_pairs: list[tuple[int, int]],
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray] | None:
    left = np.full((height, width), BG, dtype=int)
    right = np.full((height, width), BG, dtype=int)
    total = height * width
    direct_count, indirect_count, same_count, left_only_count, right_only_count = _planned_counts(len(order), total, rng)
    if direct_count == 0:
        return None

    all_pairs = [(order[i], order[j]) for i in range(len(order)) for j in range(i + 1, len(order))]
    direct_set = set(direct_pairs)
    indirect_pairs = [pair for pair in all_pairs if pair not in direct_set]
    if not indirect_pairs:
        return None

    coords = [(r, c) for r in range(height) for c in range(width)]
    rng.shuffle(coords)
    needed = direct_count + indirect_count + same_count + left_only_count + right_only_count
    if needed >= len(coords):
        return None

    cursor = 0
    direct_cells = coords[cursor : cursor + direct_count]
    cursor += direct_count
    indirect_cells = coords[cursor : cursor + indirect_count]
    cursor += indirect_count
    same_cells = coords[cursor : cursor + same_count]
    cursor += same_count
    left_only_cells = coords[cursor : cursor + left_only_count]
    cursor += left_only_count
    right_only_cells = coords[cursor : cursor + right_only_count]

    used_colors: set[int] = set()
    conflict_winners = []
    winner_sides = []

    conflict_cells = direct_cells + indirect_cells
    for idx, (r, c) in enumerate(conflict_cells):
        choices = direct_pairs if idx < len(direct_cells) else indirect_pairs
        strong, weak = rng.choice(choices)
        strong_on_left = idx % 2 == 0
        if strong_on_left:
            left[r, c], right[r, c] = strong, weak
            winner_sides.append("left")
        else:
            left[r, c], right[r, c] = weak, strong
            winner_sides.append("right")
        conflict_winners.append(strong)
        used_colors.update((strong, weak))

    for r, c in same_cells:
        color = rng.choice(order)
        left[r, c] = color
        right[r, c] = color
        used_colors.add(color)

    solo_targets = list(left_only_cells) + list(right_only_cells)
    rng.shuffle(solo_targets)
    missing = [color for color in order if color not in used_colors]
    if len(missing) > len(solo_targets):
        return None

    for idx, color in enumerate(missing):
        r, c = solo_targets[idx]
        if (r, c) in left_only_cells:
            left[r, c] = color
        else:
            right[r, c] = color

    for r, c in left_only_cells:
        if left[r, c] == BG:
            left[r, c] = rng.choice(order)
    for r, c in right_only_cells:
        if right[r, c] == BG:
            right[r, c] = rng.choice(order)

    if len(set(conflict_winners)) < 2 or len(set(winner_sides)) < 2:
        return None
    return left, right


def _has_unique_separator(main_region: np.ndarray, sep: int, sep_color: int) -> bool:
    for c in range(main_region.shape[1]):
        col = main_region[:, c]
        if c == sep:
            if not (np.all(col == sep_color) and sep_color != BG):
                return False
            continue
        if col[0] != BG and np.all(col == col[0]):
            return False
    return True


def _semantic_check(grid: np.ndarray) -> bool:
    try:
        main_region, legend, sep, bg, sep_color = _parse_layout(grid)
        left = main_region[:, :sep]
        right = np.fliplr(main_region[:, sep + 1 :])
        pairs = _parse_legend_pairs(legend, bg)
        order, rank, _reach = _dominance_order(set(map(int, np.unique(left[left != bg]))) | set(map(int, np.unique(right[right != bg]))), pairs)
        direct_set = set(pairs)

        assert _has_unique_separator(main_region, sep, sep_color)
        assert np.any((left != bg) & (right == bg))
        assert np.any((left == bg) & (right != bg))
        assert np.any((left == right) & (left != bg))

        direct_conflicts = 0
        indirect_conflicts = 0
        winner_sides = set()
        winning_colors = set()
        for r in range(left.shape[0]):
            for c in range(left.shape[1]):
                a = int(left[r, c])
                b = int(right[r, c])
                if a == bg or b == bg or a == b:
                    continue
                strong, weak = (a, b) if rank[a] < rank[b] else (b, a)
                winning_colors.add(strong)
                winner_sides.add("left" if strong == a else "right")
                if (strong, weak) in direct_set:
                    direct_conflicts += 1
                else:
                    indirect_conflicts += 1

        assert direct_conflicts >= 1
        assert indirect_conflicts >= 2
        assert indirect_conflicts > direct_conflicts
        assert len(winner_sides) == 2
        assert len(winning_colors) >= 2

        solved = main(grid)
        assert not np.array_equal(solved, left)
        assert not np.array_equal(solved, right)
        assert len(set(map(int, np.unique(solved))) - {bg}) >= 2
        assert solved.shape == left.shape
        assert solved.shape[0] <= 30 and solved.shape[1] <= 30
        assert grid.shape[0] <= 30 and grid.shape[1] <= 30
        _ = order
        return True
    except AssertionError:
        return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    color_count = rng.choice(COLOR_COUNT_CHOICES)
    sampled = rng.sample(PALETTE, color_count + 1)
    order = sampled[:-1]
    rng.shuffle(order)
    sep_color = sampled[-1]

    display_pairs = [(order[i], order[i + 1]) for i in range(color_count - 1)]
    extra_options = [(order[i], order[j]) for i in range(color_count) for j in range(i + 2, color_count)]
    if extra_options and rng.random() < 0.45:
        display_pairs.append(rng.choice(extra_options))
    rng.shuffle(display_pairs)

    pair_count = len(display_pairs)
    feasible_widths = []
    for half_width in range(HALF_WIDTH_MIN, HALF_WIDTH_MAX + 1):
        total_width = 2 * half_width + 1
        if total_width > 30:
            continue
        if any(
            _side_min_width(left_count) <= half_width and _side_min_width(pair_count - left_count) <= half_width
            for left_count in range(1, pair_count)
        ):
            feasible_widths.append(half_width)
    if not feasible_widths:
        return None

    width = rng.choice(feasible_widths)
    height = rng.randint(MAIN_HEIGHT_MIN, MAIN_HEIGHT_MAX)
    sides = _fill_scene(height, width, order, display_pairs, rng)
    if sides is None:
        return None
    left, right = sides

    sep_col = np.full((height, 1), sep_color, dtype=int)
    main_region = np.concatenate([left, sep_col, np.fliplr(right)], axis=1)
    if not _has_unique_separator(main_region, width, sep_color):
        return None

    legend = _build_legend(display_pairs, main_region.shape[1], width, rng)
    grid = np.vstack(
        [
            main_region,
            np.full((1, main_region.shape[1]), BG, dtype=int),
            legend.reshape(1, -1),
        ]
    )

    if not _semantic_check(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fold-dominance puzzle input")
