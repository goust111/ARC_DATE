from __future__ import annotations

import random
from functools import lru_cache

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GRAY = 5
ORANGE = 7

SCATTER_COLORS = (BLUE, RED, GREEN)
WIDTH_CHOICES = (2, 3, 4)
GEN_RETRIES = 500


def _find_separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GRAY)]
    assert len(rows) == 1, "expected exactly one full-width gray separator row"
    return int(rows[0])


def _wall_columns(grid: np.ndarray, separator_row: int) -> list[int]:
    cols = [
        c
        for c in range(grid.shape[1])
        if np.all(grid[: separator_row + 1, c] == GRAY)
    ]
    assert 3 <= len(cols) <= 5, "expected 3-5 vertical wall columns"
    return [int(c) for c in cols]


def _channel_spans(width: int, wall_cols: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for wall_col in wall_cols:
        if start <= wall_col - 1:
            spans.append((int(start), int(wall_col - 1)))
        start = int(wall_col + 1)
    if start <= width - 1:
        spans.append((int(start), int(width - 1)))
    assert 4 <= len(spans) <= 6, "expected 4-6 wall-bounded channels"
    return spans


def _marker_count(grid: np.ndarray, separator_row: int) -> int:
    marker_rows = np.argwhere(grid == ORANGE)
    assert marker_rows.size > 0, "expected an orange marker run"
    assert np.all(marker_rows[:, 0] == 0), "marker run must lie on the top row"
    assert np.all(marker_rows[:, 0] < separator_row), "marker must be above the separator"
    cols = sorted(int(c) for c in marker_rows[:, 1])
    assert 2 <= len(cols) <= 4, "marker run length must be 2-4"
    assert all(b == a + 1 for a, b in zip(cols, cols[1:])), "marker run must be contiguous"
    return len(cols)


def _channel_scatter(
    grid: np.ndarray,
    separator_row: int,
    span: tuple[int, int],
) -> tuple[int, int, list[int]]:
    c0, c1 = span
    band = grid[:separator_row, c0 : c1 + 1]
    scatter_mask = np.isin(band, SCATTER_COLORS)
    coords = np.argwhere(scatter_mask)
    assert coords.size > 0, "each channel needs at least one scattered pixel"

    colors = {int(band[r, c]) for r, c in coords}
    assert len(colors) == 1, "all scattered pixels in one channel must share one color"
    color = next(iter(colors))
    count = int(coords.shape[0])
    assert 1 <= count <= 4, "each channel must contain 1-4 scattered pixels"

    allowed = {BG, ORANGE, color}
    assert set(map(int, np.unique(band))) <= allowed, "unexpected object inside a channel"
    col_counts = [int(np.count_nonzero(scatter_mask[:, offset])) for offset in range(band.shape[1])]
    return int(color), count, col_counts


def _erase_scatter(grid: np.ndarray, separator_row: int) -> np.ndarray:
    out = grid.copy()
    band = out[:separator_row, :]
    band[np.isin(band, SCATTER_COLORS)] = BG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each channel's scattered pixels to the separator, then extend only the count-matching channels with centered bars.

    Input structure:
    - The grid has black background, one full-width gray separator row, and 3-5 gray vertical wall columns that partition the area above the separator into vertical channels.
    - Each channel contains 1-4 scattered pixels in a single non-gray color chosen from blue, red, and green; one top-row orange marker run has length 2-4.
    - The area below the separator is empty, and the marker run stays fixed above the wall-bounded channels.

    Rule:
    - Treat the gray columns as channel walls and drop every scattered pixel straight downward within its own column until it rests on the separator or on another settled pixel from the same channel column.
    - Count the orange marker-run length to get target count T.
    - For each channel, count its settled colored pixels after gravity.
    - If a channel count equals T, draw a 1-pixel-wide vertical bar of height T in that channel's center column, immediately above the settled pile's top row, using the channel color.
    - If a channel count does not equal T, keep only the gravity-settled pixels in that channel.
    - Preserve the walls, separator row, and orange marker run.

    Key insight:
    - Gravity creates the intermediate per-channel piles, but the visible extra structure is gated by a separate count comparison against the marker-run length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    separator_row = _find_separator_row(g)
    wall_cols = _wall_columns(g, separator_row)
    spans = _channel_spans(g.shape[1], wall_cols)
    target_count = _marker_count(g, separator_row)

    out = _erase_scatter(g, separator_row)
    for c0, c1 in spans:
        color, count, col_counts = _channel_scatter(g, separator_row, (c0, c1))
        for offset, pile_height in enumerate(col_counts):
            if pile_height == 0:
                continue
            col = c0 + offset
            out[separator_row - pile_height : separator_row, col] = color

        if count != target_count:
            continue

        tallest = max(col_counts)
        top_row = separator_row - tallest
        start_row = top_row - target_count
        center_col = (c0 + c1) // 2
        assert start_row >= 0, "gated bar must fit above the separator"
        assert np.all(out[start_row:top_row, center_col] == BG), "bar target must stay empty"
        out[start_row:top_row, center_col] = color

    return out


@lru_cache(maxsize=None)
def _compositions(total: int, slots: int) -> tuple[tuple[int, ...], ...]:
    if slots == 1:
        return ((total,),)
    results: list[tuple[int, ...]] = []
    for first in range(total + 1):
        for rest in _compositions(total - first, slots - 1):
            results.append((first,) + rest)
    return tuple(results)


def _sample_target_count() -> int:
    return random.choices([2, 3, 4], weights=[4, 4, 1], k=1)[0]


def _sample_widths(channel_count: int, target_count: int) -> list[int] | None:
    for _ in range(200):
        widths = [random.choice(WIDTH_CHOICES) for _ in range(channel_count)]
        if widths.count(3) < 3:
            continue
        if max(widths) < target_count:
            continue
        if sum(widths) + channel_count - 1 > 29:
            continue
        if random.random() < 0.65 and not any(width != 3 for width in widths):
            continue
        return widths
    return None


def _choose_match_indices(widths: list[int], target_count: int) -> tuple[list[int], int] | None:
    three_wide = [idx for idx, width in enumerate(widths) if width == 3]
    reserve_nonmatch = 2 if target_count < 4 else 1
    max_matches = min(3, len(three_wide) - 1, len(widths) - reserve_nonmatch)
    if max_matches < 2:
        return None
    match_total = random.randint(2, max_matches)
    return random.sample(three_wide, match_total), match_total


def _pick_marker_channel(
    widths: list[int],
    target_count: int,
    match_indices: list[int],
) -> int | None:
    candidates = [idx for idx, width in enumerate(widths) if width >= target_count]
    if not candidates:
        return None
    if target_count == 4:
        candidates = [idx for idx in candidates if idx not in match_indices]
        if not candidates:
            return None
        return random.choice(candidates)

    if random.random() < 0.5:
        preferred = [idx for idx in match_indices if widths[idx] >= target_count]
    else:
        preferred = [idx for idx in candidates if idx not in match_indices]
    if preferred:
        return random.choice(preferred)
    return random.choice(candidates)


def _assign_counts(
    widths: list[int],
    target_count: int,
    match_indices: list[int],
) -> tuple[list[int], int] | None:
    counts = [0] * len(widths)
    for idx in match_indices:
        counts[idx] = target_count

    nonmatch_indices = [idx for idx in range(len(widths)) if idx not in match_indices]
    extra_three_wide = [idx for idx in nonmatch_indices if widths[idx] == 3]
    if not extra_three_wide:
        return None
    special_three_wide = random.choice(extra_three_wide)

    random.shuffle(nonmatch_indices)
    if special_three_wide in nonmatch_indices:
        nonmatch_indices.remove(special_three_wide)
    ordered_nonmatches = [special_three_wide] + nonmatch_indices

    low_choices = list(range(1, target_count))
    high_choices = list(range(target_count + 1, 5))
    special_choices = [value for value in (target_count - 1, target_count + 1) if 1 <= value <= 4 and value != target_count]
    if not special_choices:
        return None
    counts[special_three_wide] = random.choice(special_choices)

    assigned_low = counts[special_three_wide] < target_count
    assigned_high = counts[special_three_wide] > target_count

    remaining = ordered_nonmatches[1:]
    if target_count < 4 and len(ordered_nonmatches) < 2:
        return None

    for idx in remaining:
        options = [1, 2, 3, 4]
        options.remove(target_count)
        if target_count < 4 and not assigned_low and low_choices:
            value = random.choice(low_choices)
            assigned_low = True
        elif target_count < 4 and not assigned_high and high_choices:
            value = random.choice(high_choices)
            assigned_high = True
        else:
            value = random.choice(options)
            assigned_low = assigned_low or value < target_count
            assigned_high = assigned_high or value > target_count
        counts[idx] = value

    if target_count < 4 and (not assigned_low or not assigned_high):
        return None
    return counts, special_three_wide


def _sample_distribution(width: int, total: int, center_like: bool) -> tuple[int, ...] | None:
    options: list[tuple[int, ...]] = []
    center = width // 2
    for parts in _compositions(total, width):
        if center_like:
            if width % 2 == 0:
                continue
            if parts[center] == 0 or parts[center] != max(parts):
                continue
        if total > 1 and sum(1 for value in parts if value > 0) > 1:
            options.append(parts)
        elif not center_like:
            options.append(parts)
        else:
            options.append(parts)
    if not options:
        return None
    return random.choice(options)


def _sample_rows_for_column(pile_height: int, separator_row: int) -> list[int] | None:
    final_rows = set(range(separator_row - pile_height, separator_row))
    pool = list(range(1, separator_row))
    if len(pool) < pile_height:
        return None
    for _ in range(100):
        rows = sorted(random.sample(pool, pile_height))
        if set(rows) != final_rows:
            return rows
    return None


def _layout_from_widths(widths: list[int]) -> tuple[int, list[int], list[tuple[int, int]]]:
    wall_cols: list[int] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for idx, width in enumerate(widths):
        spans.append((cursor, cursor + width - 1))
        cursor += width
        if idx != len(widths) - 1:
            wall_cols.append(cursor)
            cursor += 1
    return cursor, wall_cols, spans


def _render_input(case: dict) -> np.ndarray:
    height = int(case["height"])
    width = int(case["width"])
    separator_row = int(case["separator_row"])
    wall_cols = list(case["wall_cols"])
    marker_start = int(case["marker_start"])
    target_count = int(case["target_count"])

    grid = np.zeros((height, width), dtype=int)
    grid[separator_row, :] = GRAY
    for col in wall_cols:
        grid[: separator_row + 1, col] = GRAY
    grid[0, marker_start : marker_start + target_count] = ORANGE

    for channel in case["channels"]:
        color = int(channel["color"])
        for row, col in channel["positions"]:
            assert grid[row, col] == BG, "scatter pixels must not overlap existing structures"
            grid[row, col] = color
    return grid


def _render_expected(case: dict) -> np.ndarray:
    height = int(case["height"])
    width = int(case["width"])
    separator_row = int(case["separator_row"])
    wall_cols = list(case["wall_cols"])
    marker_start = int(case["marker_start"])
    target_count = int(case["target_count"])

    out = np.zeros((height, width), dtype=int)
    out[separator_row, :] = GRAY
    for col in wall_cols:
        out[: separator_row + 1, col] = GRAY
    out[0, marker_start : marker_start + target_count] = ORANGE

    for channel in case["channels"]:
        c0, c1 = channel["span"]
        color = int(channel["color"])
        distribution = tuple(int(value) for value in channel["distribution"])
        total = int(channel["count"])
        for offset, pile_height in enumerate(distribution):
            if pile_height == 0:
                continue
            out[separator_row - pile_height : separator_row, c0 + offset] = color

        if total == target_count:
            top_row = separator_row - max(distribution)
            center_col = (c0 + c1) // 2
            out[top_row - target_count : top_row, center_col] = color

    return out


def _render_settled_only(case: dict) -> np.ndarray:
    height = int(case["height"])
    width = int(case["width"])
    separator_row = int(case["separator_row"])
    wall_cols = list(case["wall_cols"])
    marker_start = int(case["marker_start"])
    target_count = int(case["target_count"])

    out = np.zeros((height, width), dtype=int)
    out[separator_row, :] = GRAY
    for col in wall_cols:
        out[: separator_row + 1, col] = GRAY
    out[0, marker_start : marker_start + target_count] = ORANGE

    for channel in case["channels"]:
        c0, _c1 = channel["span"]
        color = int(channel["color"])
        distribution = tuple(int(value) for value in channel["distribution"])
        for offset, pile_height in enumerate(distribution):
            if pile_height == 0:
                continue
            out[separator_row - pile_height : separator_row, c0 + offset] = color

    return out


def _validate_case(case: dict, grid: np.ndarray, expected: np.ndarray) -> None:
    counts = [int(channel["count"]) for channel in case["channels"]]
    widths = [int(channel["span"][1] - channel["span"][0] + 1) for channel in case["channels"]]
    colors = [int(channel["color"]) for channel in case["channels"]]
    target_count = int(case["target_count"])
    separator_row = int(case["separator_row"])
    match_indices = [idx for idx, count in enumerate(counts) if count == target_count]
    nonmatch_indices = [idx for idx, count in enumerate(counts) if count != target_count]
    settled_only = _render_settled_only(case)

    assert len(match_indices) >= 2, "need at least two matching channels"
    assert nonmatch_indices, "need at least one non-matching channel"
    if target_count < 4:
        assert any(count < target_count for count in counts), "need a below-target counterexample"
        assert any(count > target_count for count in counts), "need an above-target counterexample"

    assert any(width == 3 and count != target_count for width, count in zip(widths, counts)), "need a 3-wide non-match"
    assert any(colors[i] == colors[j] for i in match_indices for j in nonmatch_indices), "color must not predict the gate"
    assert np.all(grid[int(case["separator_row"]) + 1 :, :] == BG), "below the separator must stay empty in the input"
    assert set(map(int, np.unique(grid))) <= {BG, BLUE, RED, GREEN, GRAY, ORANGE}, "unexpected colors in input"
    assert set(map(int, np.unique(expected))) <= {BG, BLUE, RED, GREEN, GRAY, ORANGE}, "unexpected colors in output"

    for channel in case["channels"]:
        c0, c1 = channel["span"]
        center_col = (c0 + c1) // 2
        top_row = separator_row - max(channel["distribution"])
        if channel["count"] == target_count:
            diff_mask = expected[:, c0 : c1 + 1] != settled_only[:, c0 : c1 + 1]
            target_diff = np.zeros_like(diff_mask, dtype=bool)
            target_diff[top_row - target_count : top_row, center_col - c0] = True
            assert np.array_equal(diff_mask, target_diff), "matching channel must differ only by the gated bar"
            bar = expected[top_row - target_count : top_row, center_col]
            assert np.all(bar == int(channel["color"])), "matching channel must receive a full-height bar"
            assert np.array_equal(
                expected[: top_row - target_count, c0 : c1 + 1],
                settled_only[: top_row - target_count, c0 : c1 + 1],
            ), "bar must start immediately above the settled pile"
        else:
            assert np.array_equal(
                expected[:, c0 : c1 + 1],
                settled_only[:, c0 : c1 + 1],
            ), "non-matching channels must keep only the gravity result"


def _gen_attempt() -> np.ndarray | None:
    target_count = _sample_target_count()
    channel_count = random.randint(4, 6)
    widths = _sample_widths(channel_count, target_count)
    if widths is None:
        return None

    chosen = _choose_match_indices(widths, target_count)
    if chosen is None:
        return None
    match_indices, _ = chosen

    marker_channel = _pick_marker_channel(widths, target_count, match_indices)
    if marker_channel is None:
        return None

    count_result = _assign_counts(widths, target_count, match_indices)
    if count_result is None:
        return None
    counts, special_three_wide = count_result

    width, wall_cols, spans = _layout_from_widths(widths)
    separator_row = random.randint(max(8, 2 * target_count + 1), 11)
    height = separator_row + 1 + random.randint(2, 4)

    marker_c0, marker_c1 = spans[marker_channel]
    marker_start = random.randint(marker_c0, marker_c1 - target_count + 1)

    match_pick = random.choice(match_indices)
    nonmatch_pick = random.choice([idx for idx in range(channel_count) if idx not in match_indices])
    shared_color = random.choice(SCATTER_COLORS)
    colors = [random.choice(SCATTER_COLORS) for _ in range(channel_count)]
    colors[match_pick] = shared_color
    colors[nonmatch_pick] = shared_color
    if len(set(colors)) == 1:
        other = next(color for color in SCATTER_COLORS if color != shared_color)
        colors[random.randrange(channel_count)] = other

    channels = []
    for idx, span in enumerate(spans):
        c0, c1 = span
        width_i = c1 - c0 + 1
        center_like = idx in match_indices or idx == special_three_wide
        distribution = _sample_distribution(width_i, counts[idx], center_like=center_like)
        if distribution is None:
            return None

        positions: list[tuple[int, int]] = []
        for offset, pile_height in enumerate(distribution):
            if pile_height == 0:
                continue
            rows = _sample_rows_for_column(pile_height, separator_row)
            if rows is None:
                return None
            col = c0 + offset
            for row in rows:
                positions.append((int(row), int(col)))

        channel = {
            "span": (int(c0), int(c1)),
            "color": int(colors[idx]),
            "count": int(counts[idx]),
            "distribution": tuple(int(value) for value in distribution),
            "positions": positions,
        }
        channels.append(channel)

    case = {
        "height": int(height),
        "width": int(width),
        "separator_row": int(separator_row),
        "target_count": int(target_count),
        "wall_cols": [int(col) for col in wall_cols],
        "marker_start": int(marker_start),
        "channels": channels,
    }

    grid = _render_input(case)
    expected = _render_expected(case)
    _validate_case(case, grid, expected)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gravity-count-gate ARC input")
