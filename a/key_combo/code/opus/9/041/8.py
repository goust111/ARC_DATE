from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


_NEIGHBOR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
_PALETTE = list(range(1, 10))
_GEN_ATTEMPTS = 500
_GROUP_ATTEMPTS = 200


def _background_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _diag_col_bounds(shape: tuple[int, int], diag_sum: int) -> tuple[int, int]:
    h, w = shape
    c0 = max(0, diag_sum - (h - 1))
    c1 = min(w - 1, diag_sum)
    return c0, c1


def _segment_cells(diag_sum: int, start_col: int, length: int) -> list[tuple[int, int]]:
    return [(diag_sum - c, c) for c in range(start_col, start_col + length)]


def _extract_groups(grid: np.ndarray) -> dict[int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    by_diag: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c in np.argwhere(g != bg):
        by_diag[int(r + c)].append((int(c), int(g[r, c])))

    groups: dict[int, list[dict[str, object]]] = {}
    for diag_sum, cells in by_diag.items():
        cells.sort()
        segments: list[dict[str, object]] = []
        start_col = cells[0][0]
        prev_col = cells[0][0]
        color = cells[0][1]
        for col, new_color in cells[1:]:
            if col == prev_col + 1 and new_color == color:
                prev_col = col
                continue
            length = prev_col - start_col + 1
            segments.append(
                {
                    "diag_sum": diag_sum,
                    "start_col": start_col,
                    "end_col": prev_col,
                    "length": length,
                    "color": color,
                    "coords": _segment_cells(diag_sum, start_col, length),
                }
            )
            start_col = col
            prev_col = col
            color = new_color
        length = prev_col - start_col + 1
        segments.append(
            {
                "diag_sum": diag_sum,
                "start_col": start_col,
                "end_col": prev_col,
                "length": length,
                "color": color,
                "coords": _segment_cells(diag_sum, start_col, length),
            }
        )
        groups[int(diag_sum)] = segments
    return groups


def _classify_group(segments: list[dict[str, object]]) -> tuple[dict[str, object], tuple[int, int], tuple[int, int]]:
    assert 3 <= len(segments) <= 5
    sigs = [(
        int(segment["color"]),
        int(segment["length"]),
    ) for segment in segments]
    counts = Counter(sigs)
    unique = [sig for sig, count in counts.items() if count == 1]
    majority = [sig for sig, count in counts.items() if count > 1]
    assert len(unique) == 1
    assert len(majority) == 1
    assert counts[majority[0]] == len(segments) - 1
    outlier = next(segment for segment in segments if (int(segment["color"]), int(segment["length"])) == unique[0])
    return outlier, majority[0], unique[0]


def _apply_rule_from_groups(grid: np.ndarray, groups: dict[int, list[dict[str, object]]]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    bg = _background_color(out)
    h, w = out.shape
    for diag_sum, segments in groups.items():
        outlier, _majority_sig, _outlier_sig = _classify_group(segments)
        for r, c in outlier["coords"]:
            out[int(r), int(c)] = bg
        new_start = int(outlier["end_col"]) + 1
        new_len = int(outlier["length"]) + 1
        for r, c in _segment_cells(int(diag_sum), new_start, new_len):
            assert 0 <= r < h and 0 <= c < w
            out[int(r), int(c)] = int(outlier["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect the unique segment within each anti-diagonal group and advance only that segment one step farther along the same anti-diagonal while increasing its length by one.

    Input structure:
    - The background is black.
    - Foreground pixels form 3-4 groups of disconnected monochromatic segments.
    - Every group lies on a single anti-diagonal `r+c = D` and contains 3-5 separate segments on that anti-diagonal.
    - Within each group, all but one segment share the same `(color, length)` pair; the remaining segment is the unique outlier.

    Rule:
    - Group all segments by their anti-diagonal sum `r+c`.
    - In each group, find the segment whose `(color, length)` pair occurs exactly once.
    - Erase that outlier segment.
    - Redraw it on the same anti-diagonal starting immediately after its original upper-right end, preserving its color and increasing its length by one.
    - Leave every non-outlier segment unchanged.

    Key insight:
    - The comparison context is not row-wise or column-wise: the odd-one-out must be found inside each anti-diagonal group, and only that segment undergoes the move-and-grow transform.
    """
    groups = _extract_groups(np.asarray(grid, dtype=int))
    return _apply_rule_from_groups(grid, groups)


def _random_partition(total: int, parts: int) -> list[int]:
    bins = [0] * parts
    for _ in range(total):
        bins[random.randrange(parts)] += 1
    return bins


def _cells_clear(cells: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        for dr, dc in _NEIGHBOR8:
            if (r + dr, c + dc) in occupied:
                return False
    return True


def _make_group(
    shape: tuple[int, int],
    diag_sum: int,
    mode: str,
    majority_color: int,
    used_outlier_colors: set[int],
    require_interior_outlier: bool,
) -> dict[str, object] | None:
    c_min, c_max = _diag_col_bounds(shape, diag_sum)
    diag_len = c_max - c_min + 1
    if diag_len < 8:
        return None

    for _ in range(_GROUP_ATTEMPTS):
        seg_count = random.randint(3, 5)
        majority_len = random.randint(1, 3)
        if mode == "color":
            outlier_len = majority_len
            color_choices = [c for c in _PALETTE if c != majority_color and c not in used_outlier_colors]
            if not color_choices:
                continue
            outlier_color = random.choice(color_choices)
        else:
            length_choices = [x for x in (1, 2, 3, 4) if x != majority_len]
            if not length_choices:
                continue
            outlier_len = random.choice(length_choices)
            outlier_color = majority_color

        majority_count = seg_count - 1
        if require_interior_outlier and majority_count >= 2:
            left_count = random.randint(1, majority_count - 1)
        else:
            left_count = random.randint(0, majority_count)
        right_count = majority_count - left_count

        block_lengths = [majority_len] * left_count + [2 * outlier_len + 1] + [majority_len] * right_count
        min_required = sum(block_lengths) + (len(block_lengths) - 1)
        if min_required > diag_len:
            continue

        slack = diag_len - min_required
        extras = _random_partition(slack, len(block_lengths) + 1)
        block_starts: list[int] = []
        cursor = c_min + extras[0]
        for i, block_len in enumerate(block_lengths):
            block_starts.append(cursor)
            cursor += block_len
            if i < len(block_lengths) - 1:
                cursor += 1 + extras[i + 1]

        segments = []
        for i in range(left_count):
            start_col = block_starts[i]
            segments.append(
                {
                    "diag_sum": diag_sum,
                    "start_col": start_col,
                    "end_col": start_col + majority_len - 1,
                    "length": majority_len,
                    "color": majority_color,
                    "coords": _segment_cells(diag_sum, start_col, majority_len),
                    "role": "majority",
                }
            )

        outlier_block_start = block_starts[left_count]
        outlier_segment = {
            "diag_sum": diag_sum,
            "start_col": outlier_block_start,
            "end_col": outlier_block_start + outlier_len - 1,
            "length": outlier_len,
            "color": outlier_color,
            "coords": _segment_cells(diag_sum, outlier_block_start, outlier_len),
            "role": "outlier",
        }
        segments.append(outlier_segment)

        for i in range(right_count):
            start_col = block_starts[left_count + 1 + i]
            segments.append(
                {
                    "diag_sum": diag_sum,
                    "start_col": start_col,
                    "end_col": start_col + majority_len - 1,
                    "length": majority_len,
                    "color": majority_color,
                    "coords": _segment_cells(diag_sum, start_col, majority_len),
                    "role": "majority",
                }
            )

        future_cells = _segment_cells(diag_sum, outlier_segment["end_col"] + 1, outlier_len + 1)
        if any(not (0 <= r < shape[0] and 0 <= c < shape[1]) for r, c in future_cells):
            continue

        return {
            "diag_sum": diag_sum,
            "segments": segments,
            "mode": mode,
            "majority_sig": (majority_color, majority_len),
            "outlier_sig": (outlier_color, outlier_len),
            "input_cells": [cell for segment in segments for cell in segment["coords"]],
            "output_cells": [cell for segment in segments if segment["role"] == "majority" for cell in segment["coords"]]
            + future_cells,
            "outlier_index": left_count,
        }
    return None


def _group_spacing_ok(groups: list[dict[str, object]], input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
    in_groups = _extract_groups(input_grid)
    out_groups = _extract_groups(output_grid)
    if set(in_groups) != set(out_groups):
        return False

    by_diag = {int(group["diag_sum"]): group for group in groups}
    for diag_sum, segments in in_groups.items():
        ref = by_diag[int(diag_sum)]
        if len(segments) != len(ref["segments"]):
            return False
    return True


def _validate_generated(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30:
        return False
    if 0 not in set(map(int, grid.ravel())):
        return False

    groups = _extract_groups(grid)
    if not (3 <= len(groups) <= 4):
        return False

    saw_color_only = False
    saw_length_only = False
    saw_interior_outlier = False
    expected = np.asarray(grid, dtype=int).copy()
    bg = _background_color(expected)

    for diag_sum, segments in groups.items():
        try:
            outlier, majority_sig, outlier_sig = _classify_group(segments)
        except AssertionError:
            return False
        if len({int(seg["color"]) for seg in segments if seg is not outlier}) != 1:
            return False
        if int(outlier_sig[0]) != int(majority_sig[0]) and int(outlier_sig[1]) == int(majority_sig[1]):
            saw_color_only = True
        elif int(outlier_sig[0]) == int(majority_sig[0]) and int(outlier_sig[1]) != int(majority_sig[1]):
            saw_length_only = True
        else:
            return False

        outlier_index = next(i for i, seg in enumerate(segments) if seg == outlier)
        if 0 < outlier_index < len(segments) - 1:
            saw_interior_outlier = True

        for r, c in outlier["coords"]:
            expected[int(r), int(c)] = bg
        for r, c in _segment_cells(int(diag_sum), int(outlier["end_col"]) + 1, int(outlier["length"]) + 1):
            if not (0 <= r < expected.shape[0] and 0 <= c < expected.shape[1]):
                return False
            expected[int(r), int(c)] = int(outlier["color"])

    if not (saw_color_only and saw_length_only and saw_interior_outlier):
        return False

    out = main(grid)
    if not np.array_equal(out, expected):
        return False

    for state in (grid, out):
        bg_state = _background_color(state)
        occupied = [tuple(map(int, rc)) for rc in np.argwhere(state != bg_state)]
        occupied_set = set(occupied)
        if len(occupied_set) != len(occupied):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 19)
    w = random.randint(14, 19)
    group_count = random.randint(3, 4)

    majority_colors = random.sample(_PALETTE, group_count)
    modes = ["color", "length"] + [random.choice(["color", "length"]) for _ in range(group_count - 2)]
    random.shuffle(modes)

    diag_candidates = list(range(h + w - 1))
    random.shuffle(diag_candidates)

    groups: list[dict[str, object]] = []
    used_outlier_colors: set[int] = set()
    occupied_input: set[tuple[int, int]] = set()
    occupied_output: set[tuple[int, int]] = set()
    interior_needed = True

    for majority_color, mode in zip(majority_colors, modes):
        placed = False
        for diag_sum in diag_candidates:
            group = _make_group((h, w), diag_sum, mode, majority_color, used_outlier_colors, interior_needed)
            if group is None:
                continue
            input_cells = group["input_cells"]
            output_cells = group["output_cells"]
            if not _cells_clear(input_cells, occupied_input):
                continue
            if not _cells_clear(output_cells, occupied_output):
                continue
            occupied_input.update(input_cells)
            occupied_output.update(output_cells)
            groups.append(group)
            if mode == "color":
                used_outlier_colors.add(int(group["outlier_sig"][0]))
            if 0 < int(group["outlier_index"]) < len(group["segments"]) - 1:
                interior_needed = False
            placed = True
            break
        if not placed:
            return None

    if interior_needed:
        return None

    grid = np.zeros((h, w), dtype=int)
    for group in groups:
        for segment in group["segments"]:
            color = int(segment["color"])
            for r, c in segment["coords"]:
                grid[int(r), int(c)] = color

    if not _validate_generated(grid):
        return None

    out = main(grid)
    if not _group_spacing_ok(groups, grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid anti-diagonal outlier puzzle input")
