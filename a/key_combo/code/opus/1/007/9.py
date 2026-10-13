from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np


COLORS = (1, 2, 3, 4)
MASK = 5
SIZES = (9, 11, 13)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
ALL_PAIRS = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
MAX_GENERATION_ATTEMPTS = 500

_PAIR_ORDER: list[tuple[int, int]] = []
_GEN_CALL_COUNT = 0


def _swap_color(color: int, left: int, right: int) -> int:
    color = int(color)
    if color == int(left):
        return int(right)
    if color == int(right):
        return int(left)
    return color


def _partner(size: int, row: int, col: int) -> tuple[int, int]:
    return int(size - 1 - row), int(size - 1 - col)


def _canonical_cells(size: int) -> list[tuple[int, int]]:
    return [
        (row, col)
        for row in range(size)
        for col in range(size)
        if (row, col) < _partner(size, row, col)
    ]


def _solve(grid: np.ndarray, use_swap: bool, remove_legend: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height == width, "expected a square grid"
    assert width >= 2, "legend needs two cells"
    assert height % 2 == 1, "expected an odd side length"
    assert set(map(int, np.unique(g))) <= {1, 2, 3, 4, 5}, "unexpected colors"

    left = int(g[0, 0])
    right = int(g[0, 1])
    assert left in COLORS and right in COLORS and left != right, "legend must define two distinct colors"

    if use_swap:
        mapper = lambda color: _swap_color(color, left, right)
    else:
        mapper = lambda color: int(color)

    out = g.copy()
    center = height // 2

    for row, col in np.argwhere(out == MASK):
        row = int(row)
        col = int(col)
        prow, pcol = _partner(height, row, col)
        if (row, col) == (prow, pcol):
            continue
        partner_value = int(out[prow, pcol])
        assert partner_value != MASK, "every non-center mask must have a visible rotation partner"
        out[row, col] = mapper(partner_value)

    if int(out[center, center]) == MASK:
        neighbor_values = [int(out[center + drow, center + dcol]) for drow, dcol in DIR4]
        assert all(value != MASK for value in neighbor_values), "center context must be fully visible"
        nonswap_values = [value for value in neighbor_values if value not in {left, right}]
        counts = Counter(nonswap_values)
        assert counts, "center mask must be inferred from non-swap neighbors"
        best = max(counts.values())
        winners = [color for color, count in counts.items() if count == best]
        assert len(winners) == 1 and best >= 3, "center mask must have a unique local majority"
        out[center, center] = int(winners[0])

    if remove_legend:
        for row, col in ((0, 0), (0, 1)):
            prow, pcol = _partner(height, row, col)
            partner_value = int(out[prow, pcol])
            assert partner_value != MASK, "legend partners must stay visible"
            out[row, col] = mapper(partner_value)

    assert not np.any(out == MASK), "all masks must be resolved"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a 180-degree rotational anti-symmetry pattern using a two-cell swap legend.

    Input structure:
    - The grid is an odd-sized square whose cells use colors 1, 2, 3, 4 and the mask color 5.
    - The two cells at (0,0) and (0,1) form a legend that declares a swap pair A<->B.
    - Away from masks, the hidden target image follows 180-degree rotational anti-symmetry:
      swap-pair colors rotate into each other, while the other two colors rotate to themselves.
    - The legend cells themselves are temporary markers that must be restored to their real pattern colors.

    Rule:
    - Read the legend colors A and B from the top-left two cells.
    - For every masked cell, look at its 180-degree rotation partner.
    - If the visible partner is A, fill the mask with B; if it is B, fill the mask with A;
      otherwise copy the partner color unchanged.
    - If the center cell is masked, it is the only self-partner orbit and is guaranteed to be a
      non-swap color revealed by a strict majority among its four orthogonal neighbors.
    - After all masks are filled, remove the legend by restoring (0,0) and (0,1) from their
      rotation partners using the same anti-symmetry rule.

    Key insight:
    - The pattern is not ordinary rotational symmetry: the legend changes the orbit rule from
      “same color” to “swap within one designated pair, otherwise stay the same”.
    """
    g = np.asarray(grid, dtype=int)
    out = _solve(g, use_swap=True, remove_legend=True)

    size = int(out.shape[0])
    left = int(g[0, 0])
    right = int(g[0, 1])
    center = size // 2
    assert int(out[center, center]) not in {left, right}, "the self-orbit center cannot use swap colors"

    for row in range(size):
        for col in range(size):
            prow, pcol = _partner(size, row, col)
            expected = _swap_color(int(out[row, col]), left, right)
            assert int(out[prow, pcol]) == expected, "final grid must satisfy the anti-symmetry orbit rule"

    return out


def _plain_rotation_completion(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, use_swap=False, remove_legend=True)


def _keep_legend_completion(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, use_swap=True, remove_legend=False)


def _seed_colors(left: int, right: int, fixed_colors: tuple[int, int]) -> list[int]:
    fixed_a, fixed_b = fixed_colors
    colors = [left, right, left, right, fixed_a, fixed_b]
    extras = (left, right, left, right, fixed_a, fixed_b)
    for _ in range(random.randint(1, 3)):
        colors.append(int(random.choice(extras)))
    random.shuffle(colors)
    return colors


def _pick_seeds(size: int, seed_colors: list[int]) -> list[tuple[int, int, int]] | None:
    candidates = _canonical_cells(size)
    for min_dist in (6, 5, 4, 3, 2, 1, 0):
        pool = candidates[:]
        random.shuffle(pool)
        seeds: list[tuple[int, int, int]] = []
        for color in seed_colors:
            chosen: tuple[int, int] | None = None
            for row, col in pool:
                if all(abs(row - srow) + abs(col - scol) >= min_dist for srow, scol, _ in seeds):
                    chosen = (row, col)
                    break
            if chosen is None:
                break
            pool.remove(chosen)
            seeds.append((int(chosen[0]), int(chosen[1]), int(color)))
        if len(seeds) == len(seed_colors):
            return seeds
    return None


def _build_output(size: int, left: int, right: int) -> np.ndarray | None:
    fixed_a, fixed_b = [color for color in COLORS if color not in (left, right)]
    seeds = _pick_seeds(size, _seed_colors(left, right, (fixed_a, fixed_b)))
    if seeds is None:
        return None

    out = np.zeros((size, size), dtype=int)
    for row, col in _canonical_cells(size):
        best = min(
            range(len(seeds)),
            key=lambda i: (abs(row - seeds[i][0]) + abs(col - seeds[i][1]), i),
        )
        color = int(seeds[best][2])
        prow, pcol = _partner(size, row, col)
        out[row, col] = color
        out[prow, pcol] = _swap_color(color, left, right)

    center = size // 2
    support_color = int(random.choice((fixed_a, fixed_b)))
    out[center, center] = support_color
    for drow, dcol in DIR4:
        out[center + drow, center + dcol] = support_color

    target00 = int(random.choice([color for color in COLORS if color != left]))
    target01 = int(random.choice([color for color in COLORS if color != right]))
    out[0, 0] = target00
    out[size - 1, size - 1] = _swap_color(target00, left, right)
    out[0, 1] = target01
    out[size - 1, size - 2] = _swap_color(target01, left, right)

    counts = Counter(map(int, out.ravel()))
    min_pixels = 6 if size == 9 else 8
    if any(counts[color] < min_pixels for color in COLORS):
        return None
    if counts[left] < min_pixels + 2 or counts[right] < min_pixels + 2:
        return None

    for row in range(size):
        for col in range(size):
            prow, pcol = _partner(size, row, col)
            if int(out[prow, pcol]) != _swap_color(int(out[row, col]), left, right):
                return None

    if int(out[center, center]) in {left, right}:
        return None
    if {int(out[center + drow, center + dcol]) for drow, dcol in DIR4} != {int(out[center, center])}:
        return None

    return out


def _orbit_buckets(out: np.ndarray, left: int, right: int) -> tuple[list[tuple[tuple[int, int], tuple[int, int]]], list[tuple[tuple[int, int], tuple[int, int]]]]:
    size = int(out.shape[0])
    blocked = {(0, 0), (0, 1), (size - 1, size - 1), (size - 1, size - 2)}
    swap_orbits: list[tuple[tuple[int, int], tuple[int, int]]] = []
    fixed_orbits: list[tuple[tuple[int, int], tuple[int, int]]] = []

    for row, col in _canonical_cells(size):
        if (row, col) in blocked:
            continue
        prow, pcol = _partner(size, row, col)
        if (prow, pcol) in blocked:
            continue
        orbit = ((row, col), (prow, pcol))
        if int(out[row, col]) in {left, right}:
            swap_orbits.append(orbit)
        else:
            fixed_orbits.append(orbit)

    return swap_orbits, fixed_orbits


def _pick_masks(out: np.ndarray, left: int, right: int, want_center_mask: bool) -> list[tuple[int, int]] | None:
    size = int(out.shape[0])
    fixed_colors = [color for color in COLORS if color not in (left, right)]
    min_masks = math.ceil(0.10 * size * size)
    max_masks = math.floor(0.25 * size * size)
    max_target = min(max_masks, min_masks + 8)
    target_masks = random.randint(min_masks, max_target)

    swap_orbits, fixed_orbits = _orbit_buckets(out, left, right)
    noncenter_target = target_masks - int(want_center_mask)

    min_each = 3
    if noncenter_target < 2 * min_each:
        min_each = noncenter_target // 2
    if min_each < 2:
        return None

    min_swap = min_each
    max_swap = min(noncenter_target - min_each, len(swap_orbits))
    if len(swap_orbits) < min_swap or len(fixed_orbits) < min_each or max_swap < min_swap:
        return None

    swap_count = random.randint(min_swap, max_swap)
    fixed_count = noncenter_target - swap_count
    if fixed_count < min_each or fixed_count > len(fixed_orbits):
        return None

    masks: list[tuple[int, int]] = []
    visible_swap_colors: list[int] = []
    visible_fixed_colors: list[int] = []

    for cell_a, cell_b in random.sample(swap_orbits, swap_count):
        if random.random() < 0.5:
            masks.append(cell_a)
            visible_swap_colors.append(int(out[cell_b]))
        else:
            masks.append(cell_b)
            visible_swap_colors.append(int(out[cell_a]))

    for cell_a, cell_b in random.sample(fixed_orbits, fixed_count):
        if random.random() < 0.5:
            masks.append(cell_a)
            visible_fixed_colors.append(int(out[cell_b]))
        else:
            masks.append(cell_b)
            visible_fixed_colors.append(int(out[cell_a]))

    if {left, right} - set(visible_swap_colors):
        return None
    if set(fixed_colors) - set(visible_fixed_colors):
        return None

    if want_center_mask:
        center = size // 2
        support_color = int(out[center, center])
        if support_color in {left, right}:
            return None
        if {int(out[center + drow, center + dcol]) for drow, dcol in DIR4} != {support_color}:
            return None
        masks.append((center, center))

    return masks


def _make_input(out: np.ndarray, left: int, right: int, masks: list[tuple[int, int]]) -> np.ndarray:
    inp = out.copy()
    inp[0, 0] = int(left)
    inp[0, 1] = int(right)
    for row, col in masks:
        inp[row, col] = MASK
    return inp


def _pair_for_call(call_index: int) -> tuple[int, int]:
    global _PAIR_ORDER
    slot = call_index % len(ALL_PAIRS)
    if slot == 0 or not _PAIR_ORDER:
        _PAIR_ORDER = list(ALL_PAIRS)
        random.shuffle(_PAIR_ORDER)
    pair = _PAIR_ORDER[slot]
    return pair if random.random() < 0.5 else pair[::-1]


def _gen_attempt(left: int, right: int, want_center_mask: bool) -> np.ndarray | None:
    size = int(random.choice(SIZES))
    out = _build_output(size, left, right)
    if out is None:
        return None

    masks = _pick_masks(out, left, right, want_center_mask)
    if masks is None:
        return None

    inp = _make_input(out, left, right, masks)
    if set(map(int, np.unique(inp))) - {MASK} != set(COLORS):
        return None

    actual = main(inp)
    if not np.array_equal(actual, out):
        return None
    if np.array_equal(_plain_rotation_completion(inp), out):
        return None
    if np.array_equal(_keep_legend_completion(inp), out):
        return None

    visible_swap_pairs = 0
    for row, col in _canonical_cells(size):
        prow, pcol = _partner(size, row, col)
        if MASK in {int(inp[row, col]), int(inp[prow, pcol])}:
            continue
        if int(actual[row, col]) in {left, right} and int(actual[prow, pcol]) in {left, right}:
            if int(actual[row, col]) != int(actual[prow, pcol]):
                visible_swap_pairs += 1
    if visible_swap_pairs < 4:
        return None

    if int(actual[0, 0]) == left and int(actual[0, 1]) == right:
        return None
    if want_center_mask and int(inp[size // 2, size // 2]) != MASK:
        return None

    return inp


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    call_index = int(_GEN_CALL_COUNT)
    left, right = _pair_for_call(call_index)
    want_center_mask = call_index % 3 == 1

    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(left, right, want_center_mask)
        if grid is not None:
            _GEN_CALL_COUNT += 1
            return grid

    raise RuntimeError("failed to generate a valid anti-symmetry swap-orbit input")
