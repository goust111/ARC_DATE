from __future__ import annotations

import random
from typing import Any

import numpy as np


BG = 0
BAR = 1
MARK = 2
LOWER_LINE = 3
UPPER_LINE = 4

MAX_GENERATION_ATTEMPTS = 500
BAR_COUNT_CHOICES = (5, 6)
HEIGHT_MIN = 4
HEIGHT_MAX = 14
GRID_H_RANGE = (16, 21)
SEGMENT_LENGTH_CHOICES = (2, 3)
GAP_CHOICES = (2, 3)


def _segment_mid_row(seg_top: int, seg_len: int) -> int:
    return int(seg_top + (seg_len - 1) // 2)


def _classify_half(bar_top: int, bar_bottom: int, seg_top: int, seg_len: int) -> str:
    seg_bottom = seg_top + seg_len - 1
    if seg_top + seg_bottom < bar_top + bar_bottom:
        return "upper"
    return "lower"


def _extract_bars(grid: np.ndarray) -> list[dict[str, Any]]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.size == 0:
        raise AssertionError("expected a non-empty 2D grid")

    bars: list[dict[str, Any]] = []
    for c in range(g.shape[1]):
        rows = np.flatnonzero(g[:, c] != BG)
        if rows.size == 0:
            continue
        top = int(rows[0])
        bottom = int(rows[-1])
        expected = np.arange(top, bottom + 1)
        if not np.array_equal(rows, expected):
            raise AssertionError("each occupied column must be one contiguous vertical object")
        colors = set(map(int, np.unique(g[top : bottom + 1, c])))
        if not colors.issubset({BAR, MARK}):
            raise AssertionError("input bars may only contain blue and red")
        red_rows = np.flatnonzero(g[:, c] == MARK)
        if red_rows.size:
            red_top = int(red_rows[0])
            red_bottom = int(red_rows[-1])
            if not np.array_equal(red_rows, np.arange(red_top, red_bottom + 1)):
                raise AssertionError("the red mark must be one contiguous segment")
        else:
            red_top = None
            red_bottom = None

        bars.append(
            {
                "col": int(c),
                "top": top,
                "bottom": bottom,
                "height": int(bottom - top + 1),
                "red_top": red_top,
                "red_bottom": red_bottom,
            }
        )

    if not bars:
        raise AssertionError("expected at least one bar")
    return bars


def _transform_spec(grid: np.ndarray) -> dict[str, Any]:
    bars = _extract_bars(grid)
    bottoms = {bar["bottom"] for bar in bars}
    if len(bottoms) != 1:
        raise AssertionError("all bars must be bottom-aligned")

    heights = [bar["height"] for bar in bars]
    if len(set(heights)) != len(heights):
        raise AssertionError("bar heights must be distinct")

    marked = [bar for bar in bars if bar["red_top"] is not None]
    if len(marked) != 1:
        raise AssertionError("expected exactly one marked bar")

    source = marked[0]
    seg_top = int(source["red_top"])
    seg_bottom = int(source["red_bottom"])
    seg_len = int(seg_bottom - seg_top + 1)
    half = _classify_half(int(source["top"]), int(source["bottom"]), seg_top, seg_len)

    ordered = sorted(bars, key=lambda bar: bar["height"])
    source_rank = next(i for i, bar in enumerate(ordered) if bar["col"] == source["col"])
    if source_rank >= len(ordered) - 1:
        raise AssertionError("the marked bar cannot be the tallest bar")
    target = ordered[source_rank + 1]

    if half == "upper":
        offset = seg_top - int(source["top"])
        target_seg_top = int(target["top"]) + offset
    else:
        offset = int(source["bottom"]) - seg_bottom
        target_seg_top = int(target["bottom"]) - offset - seg_len + 1
    target_seg_bottom = target_seg_top + seg_len - 1

    if not (int(target["top"]) <= target_seg_top <= target_seg_bottom <= int(target["bottom"])):
        raise AssertionError("translated mark must stay within the target bar")
    if _classify_half(int(target["top"]), int(target["bottom"]), target_seg_top, seg_len) != half:
        raise AssertionError("translated mark must remain on the same side")

    return {
        "bars": bars,
        "ordered": ordered,
        "source": source,
        "target": target,
        "half": half,
        "seg_top": seg_top,
        "seg_bottom": seg_bottom,
        "seg_len": seg_len,
        "target_seg_top": target_seg_top,
        "target_seg_bottom": target_seg_bottom,
        "line_row": _segment_mid_row(seg_top, seg_len),
        "line_color": UPPER_LINE if half == "upper" else LOWER_LINE,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort the blue bars by height, move the red segment to the next taller bar on the same side, and connect the pair with a side-coded horizontal line.

    Input structure:
    - A black background contains several single-column blue vertical bars with distinct heights and a shared bottom row.
    - Exactly one bar contains one contiguous red segment that replaces part of that blue bar.
    - The red segment lies entirely in either the upper half or the lower half of its bar.

    Rule:
    - Measure each bar's full height, counting red cells as part of that bar, and sort the bars from shortest to tallest.
    - Find the marked bar in that sorted order, recolor its red segment back to blue, and move the segment to the next taller bar.
    - Preserve the segment's side by keeping the same top offset for an upper-half mark or the same bottom offset for a lower-half mark.
    - Draw a horizontal line on the marked segment's midpoint row between the source and target columns, using yellow for an upper-half mark and green for a lower-half mark, while never overwriting any bar pixels that the line crosses.

    Key insight:
    - The physical left-to-right layout is irrelevant; the true connection is between height-adjacent bars in the logical size ordering, and the segment side simultaneously determines both its destination placement and the line color.
    """
    g = np.asarray(grid, dtype=int)
    spec = _transform_spec(g)
    out = g.copy()

    source = spec["source"]
    target = spec["target"]
    seg_top = int(spec["seg_top"])
    seg_bottom = int(spec["seg_bottom"])
    target_seg_top = int(spec["target_seg_top"])
    target_seg_bottom = int(spec["target_seg_bottom"])
    line_row = int(spec["line_row"])
    line_color = int(spec["line_color"])

    out[seg_top : seg_bottom + 1, int(source["col"])] = BAR
    out[target_seg_top : target_seg_bottom + 1, int(target["col"])] = MARK

    c0 = min(int(source["col"]), int(target["col"]))
    c1 = max(int(source["col"]), int(target["col"]))
    for c in range(c0, c1 + 1):
        if out[line_row, c] == BG:
            out[line_row, c] = line_color

    return out


def _choose_heights(bar_count: int, grid_h: int) -> list[int] | None:
    values = list(range(HEIGHT_MIN, min(HEIGHT_MAX, grid_h - 2) + 1))
    choices = []

    def search(start: int, picked: list[int]) -> None:
        if len(picked) == bar_count:
            choices.append(picked.copy())
            return
        remaining = bar_count - len(picked)
        if len(values) - start < remaining:
            return
        for i in range(start, len(values)):
            value = values[i]
            if picked and value - picked[-1] < 2:
                continue
            picked.append(value)
            search(i + 1, picked)
            picked.pop()

    search(0, [])
    if not choices:
        return None
    return list(random.choice(choices))


def _choose_columns(bar_count: int, grid_h: int) -> tuple[int, list[int]] | None:
    gaps = [random.choice(GAP_CHOICES) for _ in range(bar_count - 1)]
    min_w = bar_count + sum(gaps) + 2
    width_lo = max(min_w, grid_h + 1)
    if width_lo > 30:
        return None
    grid_w = random.randint(width_lo, 30)
    slack = grid_w - (bar_count + sum(gaps))
    if slack < 2:
        return None
    left_margin = random.randint(1, slack - 1)
    cols = [left_margin]
    for gap in gaps:
        cols.append(cols[-1] + 1 + gap)
    return grid_w, cols


def _candidate_marks(
    grid_h: int,
    ordered_heights: list[int],
    rank_to_phys: list[int],
) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    for rank, src_height in enumerate(ordered_heights[:-1]):
        src_phys = int(rank_to_phys[rank])
        tgt_phys = int(rank_to_phys[rank + 1])
        if abs(src_phys - tgt_phys) < 2:
            continue
        src_top = grid_h - src_height
        src_bottom = grid_h - 1
        between = range(min(src_phys, tgt_phys) + 1, max(src_phys, tgt_phys))
        intervening_heights = []
        for phys_idx in between:
            rank_at_phys = next(i for i, p in enumerate(rank_to_phys) if p == phys_idx)
            intervening_heights.append(ordered_heights[rank_at_phys])

        for seg_len in SEGMENT_LENGTH_CHOICES:
            max_rel_top = src_height - seg_len
            for rel_top in range(max_rel_top + 1):
                seg_top = src_top + rel_top
                half = _classify_half(src_top, src_bottom, seg_top, seg_len)
                line_row = _segment_mid_row(seg_top, seg_len)
                crossing = any(line_row >= grid_h - h for h in intervening_heights)
                options.append(
                    {
                        "rank": rank,
                        "seg_len": seg_len,
                        "rel_top": rel_top,
                        "half": half,
                        "crossing": crossing,
                    }
                )
    return options


def _arrange_ranks(bar_count: int, source_rank: int, crossing: bool) -> list[int] | None:
    target_rank = source_rank + 1
    if crossing:
        between_pool = [rank for rank in range(bar_count) if rank > target_rank]
    else:
        between_pool = [rank for rank in range(source_rank)]
    if not between_pool:
        return None

    between_rank = random.choice(between_pool)
    trio_start = random.randint(0, bar_count - 3)
    trio_positions = [trio_start, trio_start + 1, trio_start + 2]
    rank_perm: list[int | None] = [None] * bar_count
    if random.random() < 0.5:
        left_rank, right_rank = source_rank, target_rank
    else:
        left_rank, right_rank = target_rank, source_rank
    rank_perm[trio_positions[0]] = left_rank
    rank_perm[trio_positions[1]] = between_rank
    rank_perm[trio_positions[2]] = right_rank

    remaining_ranks = [rank for rank in range(bar_count) if rank not in {source_rank, target_rank, between_rank}]
    remaining_positions = [idx for idx, rank in enumerate(rank_perm) if rank is None]
    random.shuffle(remaining_ranks)
    random.shuffle(remaining_positions)
    for idx, rank in zip(remaining_positions, remaining_ranks):
        rank_perm[idx] = rank
    return [int(rank) for rank in rank_perm]


def _build_input_from_spec(spec: dict[str, Any]) -> np.ndarray:
    grid = np.zeros((spec["grid_h"], spec["grid_w"]), dtype=int)
    for phys_idx, col in enumerate(spec["cols"]):
        height = spec["phys_heights"][phys_idx]
        top = spec["grid_h"] - height
        grid[top:, col] = BAR

    source_phys = spec["rank_to_phys"][spec["source_rank"]]
    source_col = spec["cols"][source_phys]
    source_height = spec["ordered_heights"][spec["source_rank"]]
    source_top = spec["grid_h"] - source_height
    seg_top = source_top + spec["rel_top"]
    grid[seg_top : seg_top + spec["seg_len"], source_col] = MARK
    return grid


def _semantic_check(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        spec = _transform_spec(inp)
    except AssertionError:
        return False

    if inp.shape != out.shape:
        return False

    source = spec["source"]
    target = spec["target"]
    source_col = int(source["col"])
    target_col = int(target["col"])
    line_row = int(spec["line_row"])
    line_color = int(spec["line_color"])
    other_line = LOWER_LINE if line_color == UPPER_LINE else UPPER_LINE

    bar_map = {int(bar["col"]): bar for bar in spec["bars"]}
    for r in range(inp.shape[0]):
        for c in range(inp.shape[1]):
            expected = BG
            if min(source_col, target_col) <= c <= max(source_col, target_col) and r == line_row:
                expected = line_color
            if c in bar_map:
                bar = bar_map[c]
                if int(bar["top"]) <= r <= int(bar["bottom"]):
                    expected = BAR
                    if c == target_col and int(spec["target_seg_top"]) <= r <= int(spec["target_seg_bottom"]):
                        expected = MARK
            if int(out[r, c]) != expected:
                return False

    if np.count_nonzero(out == MARK) != int(spec["seg_len"]):
        return False
    if np.any(out == other_line):
        return False
    if np.array_equal(inp, out):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    bar_count = random.choice(BAR_COUNT_CHOICES)
    ordered_heights = _choose_heights(bar_count, grid_h)
    if ordered_heights is None:
        return None

    width_cols = _choose_columns(bar_count, grid_h)
    if width_cols is None:
        return None
    grid_w, cols = width_cols

    desired_crossing = random.random() < 0.6
    if desired_crossing:
        source_rank = random.randint(0, bar_count - 3)
        desired_half = random.choice(("upper", "lower"))
    else:
        source_rank = random.randint(1, bar_count - 2)
        desired_half = "upper"
    rank_perm = _arrange_ranks(bar_count, source_rank, desired_crossing)
    if rank_perm is None:
        return None
    phys_heights = [ordered_heights[rank] for rank in rank_perm]
    if phys_heights == sorted(phys_heights):
        return None

    rank_to_phys = [0] * bar_count
    for phys_idx, rank in enumerate(rank_perm):
        rank_to_phys[rank] = phys_idx

    options = _candidate_marks(grid_h, ordered_heights, rank_to_phys)
    if not options:
        return None

    desired_len = random.choice(SEGMENT_LENGTH_CHOICES)
    filtered = [
        opt
        for opt in options
        if opt["rank"] == source_rank
        and opt["half"] == desired_half
        and opt["crossing"] == desired_crossing
        and opt["seg_len"] == desired_len
    ]
    if not filtered:
        return None

    chosen = random.choice(filtered)
    spec = {
        "grid_h": grid_h,
        "grid_w": grid_w,
        "cols": cols,
        "ordered_heights": ordered_heights,
        "phys_heights": phys_heights,
        "rank_to_phys": rank_to_phys,
        "source_rank": int(chosen["rank"]),
        "rel_top": int(chosen["rel_top"]),
        "seg_len": int(chosen["seg_len"]),
    }
    inp = _build_input_from_spec(spec)

    if max(inp.shape) > 30 or inp.shape[1] <= inp.shape[0]:
        return None
    try:
        parsed = _transform_spec(inp)
    except AssertionError:
        return None

    source_phys = rank_to_phys[chosen["rank"]]
    target_phys = rank_to_phys[chosen["rank"] + 1]
    if abs(source_phys - target_phys) < 2:
        return None
    if chosen["crossing"]:
        left = min(source_phys, target_phys)
        right = max(source_phys, target_phys)
        line_row = int(parsed["line_row"])
        if not any(line_row >= grid_h - phys_heights[i] for i in range(left + 1, right)):
            return None

    out = main(inp)
    if not _semantic_check(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
