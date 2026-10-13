from __future__ import annotations

import random
from typing import Any

import numpy as np


BACKGROUND = 0
NONZERO_COLORS = tuple(range(1, 10))
GENERATION_RETRIES = 500


def _runs(values: list[int]) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    start = 0
    color = int(values[0])
    for idx in range(1, len(values)):
        nxt = int(values[idx])
        if nxt != color:
            runs.append((start, idx - 1, color))
            start = idx
            color = nxt
    runs.append((start, len(values) - 1, color))
    return runs


def _compose_with_min(total: int, parts: int, minimum: int) -> list[int]:
    assert total >= parts * minimum
    pieces = [minimum] * parts
    slack = total - parts * minimum
    for _ in range(slack):
        pieces[random.randrange(parts)] += 1
    random.shuffle(pieces)
    return pieces


def _adjacent_marker_exists(markers: dict[tuple[int, int], int], r: int, c: int) -> bool:
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if (nr, nc) in markers:
            return True
    return False


def _place_marker(
    grid: np.ndarray,
    markers: dict[tuple[int, int], int],
    row: int,
    candidates: list[int],
    color: int,
) -> int:
    random.shuffle(candidates)
    for col in candidates:
        if grid[row, col] != BACKGROUND:
            continue
        if _adjacent_marker_exists(markers, row, col):
            continue
        grid[row, col] = int(color)
        markers[(row, col)] = int(color)
        return int(col)
    raise ValueError("no valid marker placement")


def _analyze_grid(grid: np.ndarray, *, strict: bool) -> dict[str, Any]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("grid must be 2D")
    h, w = g.shape
    if min(h, w) < 5:
        raise AssertionError("grid is too small")

    barrier_cols = [c for c in range(w) if np.all(g[:, c] != BACKGROUND)]
    if len(barrier_cols) != 1:
        raise AssertionError("expected exactly one full-height barrier column")
    barrier_col = int(barrier_cols[0])
    barrier_values = [int(v) for v in g[:, barrier_col]]
    segments = _runs(barrier_values)
    if len(segments) < 2:
        raise AssertionError("barrier must contain at least two color segments")
    if strict:
        for start, end, _color in segments:
            if end - start + 1 < 2:
                raise AssertionError("every barrier segment must span at least two rows")

    barrier_colors = {color for _start, _end, color in segments}
    row_infos: list[dict[str, Any]] = []
    marker_cells: set[tuple[int, int]] = set()

    for r in range(h):
        left = [(int(c), int(g[r, c])) for c in range(barrier_col) if int(g[r, c]) != BACKGROUND]
        right = [(int(c), int(g[r, c])) for c in range(barrier_col + 1, w) if int(g[r, c]) != BACKGROUND]
        if strict:
            if len(left) > 1 or len(right) > 1:
                raise AssertionError("generator should keep at most one marker per side per row")
            for _c, color in left + right:
                if color in barrier_colors:
                    raise AssertionError("marker colors must be disjoint from barrier colors")
        for c, _color in left + right:
            if (r, c) in marker_cells:
                raise AssertionError("duplicate marker cell")
            marker_cells.add((r, c))
        row_infos.append(
            {
                "row": int(r),
                "segment_color": int(g[r, barrier_col]),
                "left": left,
                "right": right,
                "paired": len(left) == 1 and len(right) == 1,
            }
        )

    if strict:
        for r, c in marker_cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in marker_cells:
                    raise AssertionError("markers should stay visually isolated from other markers")

    return {
        "shape": (h, w),
        "barrier_col": barrier_col,
        "segments": segments,
        "barrier_colors": barrier_colors,
        "rows": row_infos,
        "marker_cells": marker_cells,
    }


def _apply_rule(grid: np.ndarray, *, recolor_mode: str) -> np.ndarray:
    info = _analyze_grid(grid, strict=False)
    barrier_col = int(info["barrier_col"])
    first_segment_color = int(info["segments"][0][2])
    out = np.asarray(grid, dtype=int).copy()

    for row_info in info["rows"]:
        r = int(row_info["row"])
        left = row_info["left"]
        right = row_info["right"]
        if not (len(left) == 1 and len(right) == 1):
            continue

        left_col, left_color = left[0]
        right_col, right_color = right[0]
        out[r, left_col] = BACKGROUND
        out[r, right_col] = BACKGROUND

        if recolor_mode == "segment":
            new_color = int(row_info["segment_color"])
            out[r, barrier_col - 1] = new_color
            out[r, barrier_col + 1] = new_color
        elif recolor_mode == "preserve":
            out[r, barrier_col - 1] = int(right_color)
            out[r, barrier_col + 1] = int(left_color)
        elif recolor_mode == "fixed":
            out[r, barrier_col - 1] = first_segment_color
            out[r, barrier_col + 1] = first_segment_color
        else:
            raise ValueError(f"unknown recolor_mode: {recolor_mode}")

    return out


def _move_single_sided_too(grid: np.ndarray) -> np.ndarray:
    info = _analyze_grid(grid, strict=False)
    barrier_col = int(info["barrier_col"])
    out = np.asarray(grid, dtype=int).copy()

    for row_info in info["rows"]:
        r = int(row_info["row"])
        left = row_info["left"]
        right = row_info["right"]
        segment_color = int(row_info["segment_color"])

        if len(left) == 1:
            left_col, _left_color = left[0]
            out[r, left_col] = BACKGROUND
            out[r, barrier_col + 1] = segment_color
        if len(right) == 1:
            right_col, _right_color = right[0]
            out[r, right_col] = BACKGROUND
            out[r, barrier_col - 1] = segment_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap paired marker pixels across a segmented vertical barrier and recolor them by the crossed barrier segment.

    Input structure:
    - The grid has a black background and one vertical barrier column that runs from top to bottom.
    - The barrier is split into contiguous color segments, so each row crossing the barrier sees exactly one barrier color.
    - Single-pixel markers may appear on the left and/or right side of the barrier, with at most one marker per side on a row.

    Rule:
    - Find the unique full-height non-black column and read its row-wise color segments.
    - On any row with exactly one marker on each side of the barrier, erase both markers and place them in the cells immediately adjacent to the opposite side of the barrier.
    - After the swap, both moved markers take the color of the barrier pixel on that row; rows with only one-sided markers or no markers stay unchanged, and the barrier itself stays unchanged.

    Key insight:
    - The barrier is not just an obstacle: its local segment color is an embedded legend telling you how the swapped markers must be recolored on that row.
    """
    return _apply_rule(np.asarray(grid, dtype=int), recolor_mode="segment")


def _choose_pair_rows(segments: list[tuple[int, int, int]], pair_count: int) -> list[int]:
    segment_rows = [list(range(start, end + 1)) for start, end, _color in segments]
    eligible_segment_ids = [idx for idx, rows in enumerate(segment_rows) if rows]
    pair_segment_count = random.randint(2, min(3, len(eligible_segment_ids)))
    chosen_segments = random.sample(eligible_segment_ids, k=pair_segment_count)

    chosen_rows: list[int] = []
    for seg_id in chosen_segments:
        row = random.choice(segment_rows[seg_id])
        chosen_rows.append(int(row))

    remaining = [row for rows in segment_rows for row in rows if row not in chosen_rows]
    random.shuffle(remaining)
    while len(chosen_rows) < pair_count and remaining:
        chosen_rows.append(int(remaining.pop()))
    if len(chosen_rows) != pair_count:
        raise ValueError("not enough rows for requested pair count")
    return sorted(chosen_rows)


def _validate_case(inp: np.ndarray) -> None:
    info = _analyze_grid(inp, strict=True)
    row_infos = info["rows"]
    paired_rows = [row["row"] for row in row_infos if row["paired"]]
    one_sided_rows = [row["row"] for row in row_infos if (len(row["left"]) + len(row["right"])) == 1]

    if not (2 <= len(paired_rows) <= 6):
        raise AssertionError("generator must create 2-6 paired rows")
    if not one_sided_rows:
        raise AssertionError("generator must create at least one unchanged one-sided row")

    paired_segment_colors = {int(row["segment_color"]) for row in row_infos if row["paired"]}
    if len(paired_segment_colors) < 2:
        raise AssertionError("paired rows must cover at least two barrier segment colors")

    for row in row_infos:
        if row["paired"]:
            if row["left"][0][1] == row["right"][0][1]:
                raise AssertionError("paired markers on the same row must have different colors")

    out = main(inp)
    if np.array_equal(out, _apply_rule(inp, recolor_mode="preserve")):
        raise AssertionError("sample does not rule out color-preserving swap shortcut")
    if np.array_equal(out, _apply_rule(inp, recolor_mode="fixed")):
        raise AssertionError("sample does not rule out fixed recolor shortcut")
    if np.array_equal(out, _move_single_sided_too(inp)):
        raise AssertionError("sample does not rule out moving one-sided markers shortcut")


def generate_input() -> np.ndarray:
    for _attempt in range(GENERATION_RETRIES):
        height = random.randint(13, 18)
        width = random.randint(12, 18)
        if width < 9:
            continue

        barrier_col = random.randint(max(3, width // 2 - 2), min(width - 4, width // 2 + 2))
        segment_count = random.randint(2, min(4, height // 2))
        segment_lengths = _compose_with_min(height, segment_count, 2)

        segment_colors = random.sample(NONZERO_COLORS, k=segment_count)
        marker_palette = [color for color in NONZERO_COLORS if color not in segment_colors]
        if len(marker_palette) < 3:
            continue
        marker_color_count = random.randint(3, min(4, len(marker_palette)))
        marker_palette = random.sample(marker_palette, k=marker_color_count)

        grid = np.zeros((height, width), dtype=int)
        segments: list[tuple[int, int, int]] = []
        r0 = 0
        for length, color in zip(segment_lengths, segment_colors):
            r1 = r0 + length - 1
            grid[r0 : r1 + 1, barrier_col] = int(color)
            segments.append((r0, r1, int(color)))
            r0 = r1 + 1

        max_pairs = min(6, height - 1)
        pair_count = random.randint(2, max_pairs)
        try:
            pair_rows = _choose_pair_rows(segments, pair_count)
        except ValueError:
            continue

        remaining_rows = [r for r in range(height) if r not in pair_rows]
        if not remaining_rows:
            continue
        unpaired_count = random.randint(1, min(4, len(remaining_rows)))
        unpaired_rows = random.sample(remaining_rows, k=unpaired_count)

        markers: dict[tuple[int, int], int] = {}
        try:
            for row in pair_rows:
                left_color, right_color = random.sample(marker_palette, k=2)
                left_candidates = list(range(0, barrier_col - 1))
                right_candidates = list(range(barrier_col + 2, width))
                _place_marker(grid, markers, row, left_candidates, left_color)
                _place_marker(grid, markers, row, right_candidates, right_color)

            adjacent_unpaired_done = False
            random.shuffle(unpaired_rows)
            for row in unpaired_rows:
                side = random.choice(("left", "right"))
                color = random.choice(marker_palette)
                if side == "left":
                    near = [barrier_col - 1] if not adjacent_unpaired_done else []
                    far = list(range(0, barrier_col))
                    candidates = near + far
                else:
                    near = [barrier_col + 1] if not adjacent_unpaired_done else []
                    far = list(range(barrier_col + 1, width))
                    candidates = near + far
                placed_col = _place_marker(grid, markers, row, candidates, color)
                if placed_col in (barrier_col - 1, barrier_col + 1):
                    adjacent_unpaired_done = True
        except ValueError:
            continue

        try:
            _validate_case(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid obstacle-coded swap sample")
