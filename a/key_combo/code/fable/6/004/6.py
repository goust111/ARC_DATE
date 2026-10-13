from __future__ import annotations

# ARC task 6: 穿越交换的归边队列
#
# Design notes used before implementation:
# - The visible reference frame is row-local.  Every active row is solved
#   independently using the row's own left edge, right edge, blue bar, red bar,
#   and grey bars; no cross-row information is used.
# - To make the "pass-through swap" mechanism observable, the generator always
#   includes witness rows with multiple grey bars on at least one moving side,
#   because single-grey rows are a deliberate lure where swap and push coincide.
# - Central grey bars between blue and red must survive in place.  The generator
#   also includes at least one such row so a shortcut that compacts every grey
#   segment fails.
# - The validator rejects degenerate samples whose output matches a simpler
#   preserve-order compaction rule, ensuring the reversal step is necessary.
# - Multiple same-color grey bars must remain visually separable in the output.
#   The queue therefore uses the smallest visible separation (one background
#   cell) between consecutive grey bars on the same side; otherwise reversal
#   would collapse into an indistinguishable single grey block.

import random
from collections import Counter

import numpy as np


TEAL = 8
RED = 2
GREY = 5
ROW_RETRIES = 200
GRID_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _runs(row: np.ndarray) -> list[dict[str, int]]:
    vals = [int(v) for v in row]
    runs: list[dict[str, int]] = []
    start = 0
    for c in range(1, len(vals) + 1):
        if c == len(vals) or vals[c] != vals[start]:
            runs.append(
                {
                    "color": vals[start],
                    "start": start,
                    "end": c - 1,
                    "length": c - start,
                }
            )
            if c < len(vals):
                start = c
    return runs


def _row_info(row: np.ndarray, bg: int) -> dict[str, object] | None:
    fg_runs = [run for run in _runs(row) if int(run["color"]) != int(bg)]
    if not fg_runs:
        return None

    allowed = {TEAL, RED, GREY}
    assert all(int(run["color"]) in allowed for run in fg_runs), "unexpected foreground color"

    teal_idx = [i for i, run in enumerate(fg_runs) if int(run["color"]) == TEAL]
    red_idx = [i for i, run in enumerate(fg_runs) if int(run["color"]) == RED]
    assert len(teal_idx) == 1 and len(red_idx) == 1, "expected one teal bar and one red bar per active row"
    teal_i = int(teal_idx[0])
    red_i = int(red_idx[0])
    assert teal_i < red_i, "teal bar must be left of red bar"

    left = fg_runs[:teal_i]
    middle = fg_runs[teal_i + 1 : red_i]
    right = fg_runs[red_i + 1 :]
    assert all(int(run["color"]) == GREY for run in left + middle + right), "non-grey bars may only be teal/red"

    return {
        "teal": fg_runs[teal_i],
        "red": fg_runs[red_i],
        "left": left,
        "middle": middle,
        "right": right,
    }


def _paint(out: np.ndarray, start: int, length: int, color: int, bg: int) -> None:
    end = start + length
    assert 0 <= start <= end <= out.size
    assert np.all(out[start:end] == int(bg)), "segments must not overlap"
    out[start:end] = int(color)


def _transform_row(
    row: np.ndarray,
    bg: int,
    *,
    left_reversed: bool,
    right_reversed: bool,
) -> np.ndarray:
    info = _row_info(row, bg)
    if info is None:
        return np.array(row, dtype=int, copy=True)

    out = np.full_like(row, int(bg))
    width = int(row.size)
    teal = info["teal"]
    red = info["red"]
    left = list(info["left"])
    middle = list(info["middle"])
    right = list(info["right"])

    _paint(out, 0, int(teal["length"]), TEAL, bg)
    left_order = (
        sorted(left, key=lambda run: int(run["start"]), reverse=True)
        if left_reversed
        else sorted(left, key=lambda run: int(run["start"]))
    )
    cursor = int(teal["length"])
    for i, run in enumerate(left_order):
        _paint(out, cursor, int(run["length"]), GREY, bg)
        cursor += int(run["length"])
        if i < len(left_order) - 1:
            cursor += 1

    for run in middle:
        _paint(out, int(run["start"]), int(run["length"]), GREY, bg)

    red_start = width - int(red["length"])
    _paint(out, red_start, int(red["length"]), RED, bg)
    right_order = (
        sorted(right, key=lambda run: int(run["start"]))
        if right_reversed
        else sorted(right, key=lambda run: int(run["start"]), reverse=True)
    )
    cursor = red_start
    for i, run in enumerate(right_order):
        cursor -= int(run["length"])
        _paint(out, cursor, int(run["length"]), GREY, bg)
        if i < len(right_order) - 1:
            cursor -= 1

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the teal bar to the left edge and the red bar to the right edge while swapping through grey bars.

    Input structure:
    - The background is the dominant color.
    - Active rows contain exactly one horizontal teal bar and one horizontal red bar, with teal left of red.
    - Any other foreground segments on those rows are horizontal grey bars, placed either left of teal, between teal and red, or right of red.

    Rule:
    - Solve each active row independently.
    - Move the teal bar to the left edge and the red bar to the right edge.
    - Grey bars between teal and red stay fixed.
    - Grey bars on teal's left are crossed one by one, so they end up beside teal in the reverse of their original left-edge order.
    - Grey bars on red's right are crossed one by one, so they end up beside red in the reverse of their original right-edge order.
    - When several grey bars queue on the same side, keep them as distinct bars separated by a single background cell so their order remains visible.

    Key insight:
    - This is not order-preserving pushing: each encounter is a pass-through swap, so a side's final grey queue is determined by encounter order, which reverses the original distance-to-edge order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    out = g.copy()
    for r in range(g.shape[0]):
        if np.any(g[r] != int(bg)):
            out[r] = _transform_row(g[r], bg, left_reversed=True, right_reversed=True)
    return out


def _random_partition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    values: list[int] = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        values.append(cut - prev - 1)
        prev = cut
    return values


def _sample_counts(category: str) -> tuple[int, int, int]:
    if category == "left_reverse":
        return random.randint(2, 3), random.randint(0, 1), random.randint(0, 1)
    if category == "right_reverse":
        return random.randint(0, 1), random.randint(0, 1), random.randint(2, 3)
    if category == "center_bait":
        if random.random() < 0.5:
            return 1, random.randint(1, 2), 0
        return 0, random.randint(1, 2), 1
    if category == "mixed":
        left_n = random.randint(1, 2)
        mid_n = random.randint(0, 1)
        right_n = random.randint(1, 2)
        return left_n, mid_n, right_n
    raise ValueError(f"unknown category: {category}")


def _make_row(width: int, bg: int, category: str) -> np.ndarray | None:
    for _ in range(ROW_RETRIES):
        left_n, mid_n, right_n = _sample_counts(category)
        teal_len = random.randint(2, 5)
        red_len = random.randint(2, 5)

        grey_total = left_n + mid_n + right_n
        grey_lengths = random.sample(range(1, 7), grey_total)
        random.shuffle(grey_lengths)
        left_lengths = grey_lengths[:left_n]
        mid_lengths = grey_lengths[left_n : left_n + mid_n]
        right_lengths = grey_lengths[left_n + mid_n :]

        segments = (
            [(GREY, length) for length in left_lengths]
            + [(TEAL, teal_len)]
            + [(GREY, length) for length in mid_lengths]
            + [(RED, red_len)]
            + [(GREY, length) for length in right_lengths]
        )
        min_width = sum(length for _color, length in segments) + (len(segments) - 1)
        if min_width > width:
            continue

        extra = width - min_width
        slot_extra = _random_partition(extra, len(segments) + 1)
        leading = slot_extra[0]
        trailing = slot_extra[1]
        gap_extra = slot_extra[2:]

        row = np.full(width, int(bg), dtype=int)
        cursor = leading
        for i, (color, length) in enumerate(segments):
            row[cursor : cursor + length] = int(color)
            cursor += length
            if i < len(segments) - 1:
                cursor += 1 + gap_extra[i]
        cursor += trailing
        assert cursor == width

        info = _row_info(row, bg)
        assert info is not None
        if category == "left_reverse" and len(info["left"]) < 2:
            continue
        if category == "right_reverse" and len(info["right"]) < 2:
            continue
        if category == "center_bait" and len(info["middle"]) < 1:
            continue
        return row
    return None


def _preserve_order_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()
    for r in range(g.shape[0]):
        if np.any(g[r] != int(bg)):
            out[r] = _transform_row(g[r], bg, left_reversed=False, right_reversed=False)
    return out


def _middle_segments_preserved(inp: np.ndarray, out: np.ndarray, bg: int) -> bool:
    for r in range(inp.shape[0]):
        info = _row_info(inp[r], bg)
        if info is None:
            continue
        for run in info["middle"]:
            start = int(run["start"])
            end = int(run["end"]) + 1
            if not np.all(out[r, start:end] == GREY):
                return False
    return True


def _semantic_ok(grid: np.ndarray) -> bool:
    bg = _get_bg(grid)
    infos = []
    for r in range(grid.shape[0]):
        info = _row_info(grid[r], bg)
        if info is not None:
            infos.append(info)
    if len(infos) < 4:
        return False

    has_reverse_witness = any(len(info["left"]) >= 2 or len(info["right"]) >= 2 for info in infos)
    has_middle_witness = any(len(info["middle"]) >= 1 for info in infos)
    has_single_bait = any(max(len(info["left"]), len(info["right"])) == 1 for info in infos)
    has_empty_side = any(len(info["left"]) == 0 or len(info["right"]) == 0 for info in infos)
    if not (has_reverse_witness and has_middle_witness and has_single_bait and has_empty_side):
        return False

    out = main(grid)
    preserve = _preserve_order_output(grid)
    if np.array_equal(out, grid):
        return False
    if np.array_equal(out, preserve):
        return False
    if not _middle_segments_preserved(grid, out, bg):
        return False

    reverse_rows = 0
    for r in range(grid.shape[0]):
        if not np.array_equal(out[r], preserve[r]):
            reverse_rows += 1
    return reverse_rows >= 2


def _generate_attempt() -> np.ndarray | None:
    bg = random.choice([0, 1, 3, 4, 6, 7, 9])
    categories = ["left_reverse", "right_reverse", "center_bait", "mixed"]
    extra_count = random.randint(0, 2)
    extra_pool = ["left_reverse", "right_reverse", "center_bait", "mixed"]
    categories.extend(random.choice(extra_pool) for _ in range(extra_count))
    random.shuffle(categories)

    active_rows = len(categories)
    height = random.randint(max(2 * active_rows - 1, 8), 16)
    width = random.randint(20, 30)

    parity = random.randint(0, 1)
    available_rows = list(range(parity, height, 2))
    if len(available_rows) < active_rows:
        return None
    row_positions = sorted(random.sample(available_rows, active_rows))

    grid = np.full((height, width), int(bg), dtype=int)
    for row_idx, category in zip(row_positions, categories):
        row = _make_row(width, bg, category)
        if row is None:
            return None
        grid[row_idx] = row

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape:
        return None
    if not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GRID_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GRID_RETRIES} attempts")
