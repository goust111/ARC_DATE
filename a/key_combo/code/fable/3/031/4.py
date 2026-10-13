from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
NON_BG_COLORS = list(range(1, 10))
MAX_GENERATION_ATTEMPTS = 1000


def _get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).reshape(-1)
    return int(Counter(map(int, values)).most_common(1)[0][0])


def _iter_non_bg_runs(row: np.ndarray, bg: int):
    width = int(row.shape[0])
    c = 0
    while c < width:
        while c < width and int(row[c]) == bg:
            c += 1
        if c >= width:
            break
        start = c
        while c < width and int(row[c]) != bg:
            c += 1
        yield start, c - 1


def _draw_bar(
    grid: np.ndarray,
    row: int,
    center_col: int,
    length: int,
    color: int,
    inset_color: int | None = None,
) -> None:
    half = length // 2
    c0 = center_col - half
    c1 = center_col + half
    assert 0 <= row < grid.shape[0]
    assert 0 <= c0 <= c1 < grid.shape[1]
    grid[row, c0 : c1 + 1] = int(color)
    if inset_color is not None:
        grid[row, center_col] = int(inset_color)


def _parse_bars(grid: np.ndarray, bg: int) -> tuple[list[dict], dict]:
    bars: list[dict] = []
    pivot: dict | None = None

    for r in range(grid.shape[0]):
        row = grid[r]
        for c0, c1 in _iter_non_bg_runs(row, bg):
            values = np.asarray(row[c0 : c1 + 1], dtype=int)
            length = int(values.size)
            assert length % 2 == 1 and length >= 1

            mid = length // 2
            unique = sorted(set(map(int, values)))
            if len(unique) == 1:
                bars.append(
                    {
                        "row": int(r),
                        "center_col": int(c0 + mid),
                        "length": length,
                        "base_color": unique[0],
                        "is_pivot": False,
                    }
                )
                continue

            assert len(unique) == 2
            outer = np.concatenate((values[:mid], values[mid + 1 :]))
            assert outer.size > 0 and len(set(map(int, outer))) == 1
            base_color = int(outer[0])
            inset_color = int(values[mid])
            assert inset_color != base_color

            bar = {
                "row": int(r),
                "center_col": int(c0 + mid),
                "length": length,
                "base_color": base_color,
                "inset_color": inset_color,
                "is_pivot": True,
            }
            assert pivot is None
            pivot = bar
            bars.append(bar)

    assert bars and pivot is not None
    lengths = [int(bar["length"]) for bar in bars]
    assert len(lengths) == len(set(lengths))
    return bars, pivot


def _infer_full_lengths(lengths: list[int]) -> tuple[list[int], int]:
    observed = sorted(map(int, lengths), reverse=True)
    assert all(length % 2 == 1 for length in observed)

    max_len = observed[0]
    min_len = observed[-1]
    full = list(range(max_len, min_len - 1, -2))
    assert len(full) == len(observed) + 1

    missing = [length for length in full if length not in set(observed)]
    assert len(missing) == 1
    assert missing[0] not in {full[0], full[-1]}
    return full, int(missing[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered odd-length horizontal bars into a centered pyramid around the inset-marked pivot bar.

    Input structure:
    - A black background contains several disconnected one-row horizontal bars with distinct odd lengths.
    - The full set of lengths belongs to one step-2 arithmetic progression with exactly one internal length missing.
    - Exactly one bar has a differently colored center pixel; that inset marks the pivot bar and also gives the missing layer color.

    Rule:
    - Keep the pivot bar fixed at its original row and center column, preserving its inset pixel.
    - Sort the complete length progression from longest to shortest so those ranks become the pyramid layers from bottom to top.
    - Use the pivot length rank to recover every target row, move each other observed bar to its rank row centered on the pivot column, and draw the missing length as a centered bar in the inset color.

    Key insight:
    - The inset pivot is not necessarily the bottom layer; its center column is the alignment axis, its length rank fixes the whole pyramid's vertical offset, and its inset color fills the single missing arithmetic term.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = BG
    bars, pivot = _parse_bars(g, bg)
    full_lengths, missing_length = _infer_full_lengths([bar["length"] for bar in bars])
    bars_by_length = {int(bar["length"]): bar for bar in bars}

    pivot_index = full_lengths.index(int(pivot["length"]))
    pivot_row = int(pivot["row"])
    center_col = int(pivot["center_col"])
    inset_color = int(pivot["inset_color"])

    out = np.full_like(g, bg)
    for index, length in enumerate(full_lengths):
        row = pivot_row + pivot_index - index
        assert 0 <= row < out.shape[0]

        if length == missing_length:
            _draw_bar(out, row, center_col, length, inset_color)
            continue

        bar = bars_by_length[int(length)]
        if bar["is_pivot"]:
            _draw_bar(
                out,
                row,
                center_col,
                int(length),
                int(bar["base_color"]),
                int(bar["inset_color"]),
            )
        else:
            _draw_bar(out, row, center_col, int(length), int(bar["base_color"]))

    return out


def _sample_odd(min_value: int, max_value: int) -> int:
    candidates = [value for value in range(min_value, max_value + 1) if value % 2 == 1]
    return random.choice(candidates)


def _sample_spaced_rows(height: int, count: int, blocked_rows: list[int]) -> list[int] | None:
    blocked = set()
    for row in blocked_rows:
        for rr in range(row - 1, row + 2):
            if 0 <= rr < height:
                blocked.add(rr)

    base_candidates = [row for row in range(height) if row not in blocked]
    for _ in range(200):
        candidates = base_candidates[:]
        random.shuffle(candidates)
        chosen: list[int] = []
        for row in candidates:
            if all(abs(row - other) >= 2 for other in chosen):
                chosen.append(row)
                if len(chosen) == count:
                    return chosen
    return None


def _sample_start_col(width: int, length: int, avoid: int) -> int:
    options = list(range(width - length + 1))
    if len(options) > 1 and avoid in options and random.random() < 0.9:
        options = [col for col in options if col != avoid]
    return random.choice(options)


def _has_row_order_inversion(index_to_row: dict[int, int]) -> bool:
    indices = sorted(index_to_row)
    for i, left in enumerate(indices):
        for right in indices[i + 1 :]:
            if index_to_row[left] < index_to_row[right]:
                return True
    return False


def _gen_attempt() -> np.ndarray | None:
    layer_count = random.randint(4, 8)
    longest_min = 2 * layer_count + 1
    longest_max = min(25, longest_min + 10)
    if longest_min > longest_max:
        return None

    longest_length = _sample_odd(longest_min, longest_max)
    full_lengths = [longest_length - 2 * i for i in range(layer_count)]
    missing_index = random.randint(1, layer_count - 2)
    observed_indices = [index for index in range(layer_count) if index != missing_index]
    pivot_index = random.choice(observed_indices)

    width_min = longest_length + 2
    width_max = min(30, longest_length + 10)
    height_min = max(layer_count + 2, 2 * (layer_count - 1) + 3)
    height_max = min(30, height_min + 6)
    if width_min > width_max or height_min > height_max:
        return None

    width = random.randint(width_min, width_max)
    height = random.randint(height_min, height_max)
    top_row = random.randint(0, height - layer_count)
    center_col = random.randint(longest_length // 2, width - 1 - longest_length // 2)

    palette = random.sample(NON_BG_COLORS, layer_count)
    inset_color = palette[0]
    bar_color_by_index = {
        index: color for index, color in zip(observed_indices, palette[1:], strict=True)
    }

    expected = np.full((height, width), BG, dtype=int)
    for index, length in enumerate(full_lengths):
        row = top_row + layer_count - 1 - index
        if index == missing_index:
            _draw_bar(expected, row, center_col, length, inset_color)
        elif index == pivot_index:
            _draw_bar(
                expected,
                row,
                center_col,
                length,
                bar_color_by_index[index],
                inset_color,
            )
        else:
            _draw_bar(expected, row, center_col, length, bar_color_by_index[index])

    pivot_row = top_row + layer_count - 1 - pivot_index
    inp = np.full((height, width), BG, dtype=int)
    _draw_bar(
        inp,
        pivot_row,
        center_col,
        full_lengths[pivot_index],
        bar_color_by_index[pivot_index],
        inset_color,
    )

    other_indices = [index for index in observed_indices if index != pivot_index]
    rows = _sample_spaced_rows(height, len(other_indices), [pivot_row])
    if rows is None:
        return None

    placements = {pivot_index: pivot_row}
    random.shuffle(other_indices)
    for index, row in zip(other_indices, rows, strict=True):
        length = full_lengths[index]
        target_c0 = center_col - length // 2
        c0 = _sample_start_col(width, length, target_c0)
        inp[row, c0 : c0 + length] = bar_color_by_index[index]
        placements[index] = row

    if not _has_row_order_inversion(placements):
        return None

    moved = False
    for index in other_indices:
        target_row = top_row + layer_count - 1 - index
        if placements[index] != target_row:
            moved = True
            break
    if not moved:
        return None

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pivot-bar pyramid input")
