from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND = 0
STONE = 5
AVAILABLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 300


def _background_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.flat))
    return int(counts.most_common(1)[0][0])


def _legend_pairs(grid: np.ndarray, bg: int) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for c in range(grid.shape[1]):
        top = int(grid[0, c])
        bottom = int(grid[1, c])
        if top != bg and bottom != bg:
            pairs.add((top, bottom))
    return pairs


def _reduce_column(top_to_bottom: list[int], pairs: set[tuple[int, int]]) -> list[int]:
    stack: list[int] = []
    for color in reversed(top_to_bottom):
        stack.append(int(color))
        while len(stack) >= 2 and (stack[-1], stack[-2]) in pairs:
            stack.pop()
            stack.pop()
    return stack


def main(grid: np.ndarray) -> np.ndarray:
    """Let legend-defined directed pairs annihilate after each main-area column settles under gravity.

    Input structure:
    - The top two rows form a legend band of vertical color pairs, followed by one separator row.
    - Each occupied legend column encodes one directed annihilation pair: top color X over bottom color Y.
    - Below the separator, the main area contains sparse single cells and some grey stone cells on a background.

    Rule:
    - Preserve the legend band and the separator row exactly.
    - In each main-area column, drop all non-background cells straight downward while preserving their top-to-bottom order.
    - After settling, repeatedly remove any adjacent vertical pair whose upper and lower colors match a legend pair, then let higher cells fall again, until the column is stable.

    Key insight:
    - The process is completely column-local: the legend decodes directed top-over-bottom kill pairs, and each settled column reduces by repeated adjacent-pair cancellation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 4

    bg = _background_color(g)
    pairs = _legend_pairs(g, bg)
    assert pairs, "expected at least one legend pair"

    out = g.copy()
    out[3:, :] = bg

    for c in range(g.shape[1]):
        top_to_bottom = [int(g[r, c]) for r in range(3, g.shape[0]) if int(g[r, c]) != bg]
        reduced = _reduce_column(top_to_bottom, pairs)
        for depth, color in enumerate(reduced):
            out[g.shape[0] - 1 - depth, c] = int(color)

    return out


def _sample_spaced_rows(main_h: int, count: int, require_bottom_gap: bool = True) -> list[int] | None:
    if count == 0:
        return []
    last_cap = main_h - 2 if require_bottom_gap else main_h - 1
    if last_cap < 0:
        return None

    rows: list[int] = []

    def place(idx: int, start: int) -> bool:
        remaining = count - idx
        max_here = last_cap - 2 * (remaining - 1)
        if start > max_here:
            return False
        options = list(range(start, max_here + 1))
        random.shuffle(options)
        for row in options:
            rows.append(int(row))
            if idx + 1 == count or place(idx + 1, row + 2):
                return True
            rows.pop()
        return False

    return rows.copy() if place(0, 0) else None


def _chain_sequence(chosen_pairs: list[tuple[int, int]]) -> list[int]:
    tops = [top for top, _bottom in chosen_pairs]
    bottoms = [bottom for _top, bottom in reversed(chosen_pairs)]
    return tops + bottoms


def _extra_sequence(pairs: list[tuple[int, int]]) -> list[int]:
    pair = random.choice(pairs)
    other_colors = [color for p in pairs for color in p if color not in pair]
    pattern = random.choice(["singleton", "kill", "reverse", "stone", "drop"])
    if pattern == "singleton":
        return [random.choice(pair)]
    if pattern == "kill":
        return [pair[0], pair[1]]
    if pattern == "reverse":
        return [pair[1], pair[0]]
    if pattern == "stone":
        return [pair[0], STONE, pair[1]]
    payload = random.choice(other_colors) if other_colors else pair[1]
    return [payload, pair[0], pair[1]]


def _legend_columns(width: int, pair_count: int) -> list[int] | None:
    candidates = list(range(1, width - 1, 2))
    if len(candidates) < pair_count:
        return None
    return sorted(random.sample(candidates, pair_count))


def _generate_candidate() -> np.ndarray | None:
    pair_count = random.randint(2, 3)
    pair_colors = random.sample(AVAILABLE_COLORS, 2 * pair_count)
    pairs = [(pair_colors[2 * i], pair_colors[2 * i + 1]) for i in range(pair_count)]

    remaining = [color for color in AVAILABLE_COLORS if color not in pair_colors]
    separator_color = random.choice(remaining)

    chain_len = random.randint(2, min(3, pair_count))
    chosen_for_chain = random.sample(pairs, chain_len)
    reverse_pair = random.choice(pairs)
    mandatory_sequences = [
        _chain_sequence(chosen_for_chain),
        [reverse_pair[1], reverse_pair[0]],
    ]

    stone_pair = random.choice(pairs)
    mandatory_sequences.append([stone_pair[0], STONE, stone_pair[1]])

    drop_pair = random.choice(pairs)
    payload_choices = [color for p in pairs for color in p if color not in drop_pair]
    if not payload_choices:
        return None
    mandatory_sequences.append([random.choice(payload_choices), drop_pair[0], drop_pair[1]])

    extra_count = random.randint(0, 3)
    for _ in range(extra_count):
        mandatory_sequences.append(_extra_sequence(pairs))

    width = random.randint(max(8, len(mandatory_sequences) + 1), 14)
    legend_cols = _legend_columns(width, pair_count)
    if legend_cols is None:
        return None

    max_seq_len = max(len(seq) for seq in mandatory_sequences)
    main_h = random.randint(max(9, 2 * max_seq_len), 15)
    height = 3 + main_h

    active_cols = random.sample(range(width), len(mandatory_sequences))
    random.shuffle(mandatory_sequences)

    grid = np.full((height, width), BACKGROUND, dtype=int)
    for c, (top, bottom) in zip(legend_cols, pairs):
        grid[0, c] = int(top)
        grid[1, c] = int(bottom)
    grid[2, :] = int(separator_color)

    for c, seq in zip(active_cols, mandatory_sequences):
        rows = _sample_spaced_rows(main_h, len(seq), require_bottom_gap=True)
        if rows is None:
            return None
        for row, color in zip(rows, seq):
            grid[3 + row, c] = int(color)

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    if not np.array_equal(out[:3, :], grid[:3, :]):
        return None

    reduced_sizes = []
    for c in range(width):
        seq = [int(grid[r, c]) for r in range(3, height) if int(grid[r, c]) != BACKGROUND]
        reduced_sizes.append((len(seq), len(_reduce_column(seq, set(pairs)))))

    if not any(before - after >= 2 for before, after in reduced_sizes):
        return None
    if not any(STONE in grid[3:, c] for c in range(width)):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC puzzle 8")
