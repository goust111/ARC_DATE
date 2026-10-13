import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color


BACKGROUND = Color.BLACK
UNIFORM_FILL = Color.GREEN
MIXED_FILL = Color.BLUE
INPUT_PALETTE = (Color.RED, Color.YELLOW, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON)
MAX_GENERATION_ATTEMPTS = 400


def _row_is_uniform(row: np.ndarray) -> bool:
    return bool(np.all(row == row[0]))


def _row_groups(grid: np.ndarray, bg: int = BACKGROUND) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    start = None
    for r in range(grid.shape[0]):
        is_foreground = bool(np.any(grid[r] != bg))
        if is_foreground and start is None:
            start = r
        elif not is_foreground and start is not None:
            groups.append((start, r))
            start = None
    if start is not None:
        groups.append((start, grid.shape[0]))
    return groups


def _pattern_for_group(grid: np.ndarray, group: tuple[int, int]) -> tuple[bool, ...]:
    r0, r1 = group
    return tuple(_row_is_uniform(grid[r]) for r in range(r0, r1))


def main(grid: np.ndarray) -> np.ndarray:
    """Complete lower row-groups by extrapolating the top group's row-uniformity pattern.

    Input structure
    A black-background grid contains 3-4 row-groups separated by at least two
    black rows. Each group is a stack of full-width non-black rows, and each row
    is either uniform (one visible color across the row) or mixed (2-3 visible
    colors in the same row).

    Rule
    Detect the row-groups, convert each row to a binary signal based on whether
    it is uniform, and use the top group's full signal as the reference pattern.
    Every lower group's signal is a strict prefix of that reference, so append the
    missing rows directly beneath that group: append a full green row for each
    missing uniform step and a full blue row for each missing mixed step.

    Key insight
    The visible colors inside the original groups are distractors; the reusable
    information is the row-level U/M sequence obtained by testing uniformity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    groups = _row_groups(g, BACKGROUND)
    assert 3 <= len(groups) <= 4, "expected 3-4 row-groups"

    lengths = [r1 - r0 for r0, r1 in groups]
    assert lengths[0] == max(lengths) and lengths.count(lengths[0]) == 1, "top group must be the unique longest reference"
    assert all(a > b for a, b in zip(lengths, lengths[1:])), "group lengths must strictly decrease"

    for idx, (r0, r1) in enumerate(groups):
        assert np.all(g[r0:r1] != BACKGROUND), "group rows must be full-width non-black"
        if idx > 0:
            prev_r1 = groups[idx - 1][1]
            assert r0 - prev_r1 >= 2, "groups must be separated by at least two black rows"

    reference = _pattern_for_group(g, groups[0])
    assert len(reference) >= 4, "reference pattern must have room for both row types"
    assert sum(reference) >= 2 and (len(reference) - sum(reference)) >= 2, "reference pattern must contain at least two uniform and two mixed rows"

    out_rows: list[np.ndarray] = []
    prev = 0
    for idx, group in enumerate(groups):
        r0, r1 = group
        out_rows.extend(g[prev:r1].copy())
        if idx > 0:
            prefix = _pattern_for_group(g, group)
            assert 2 <= len(prefix) < len(reference), "lower groups must be strict prefixes of length at least 2"
            assert prefix == reference[: len(prefix)], "lower group must match the reference prefix"
            for is_uniform in reference[len(prefix) :]:
                fill_color = UNIFORM_FILL if is_uniform else MIXED_FILL
                out_rows.append(np.full(g.shape[1], fill_color, dtype=int))
        prev = r1
    out_rows.extend(g[prev:].copy())

    return np.array(out_rows, dtype=int)


def _partition_total(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    bounds = [0, *cuts, total]
    return [bounds[i + 1] - bounds[i] for i in range(parts)]


def _make_uniform_row(width: int, palette: tuple[int, ...]) -> np.ndarray:
    color = random.choice(palette)
    return np.full(width, color, dtype=int)


def _make_mixed_row(width: int, palette: tuple[int, ...]) -> np.ndarray:
    color_count = random.choice((2, 3)) if width >= 3 else 2
    segment_lengths = _partition_total(width, color_count)
    colors = random.sample(list(palette), color_count)
    cells: list[int] = []
    for length, color in zip(segment_lengths, colors):
        cells.extend([color] * length)
    row = np.array(cells, dtype=int)
    assert not _row_is_uniform(row)
    return row


def _sample_reference_pattern(length: int) -> tuple[bool, ...]:
    while True:
        pattern = [random.random() < 0.5 for _ in range(length)]
        if sum(pattern) >= 2 and (length - sum(pattern)) >= 2:
            return tuple(pattern)


def _suffix_has_both_types(pattern: tuple[bool, ...], prefix_length: int) -> bool:
    suffix = pattern[prefix_length:]
    return any(suffix) and any(not value for value in suffix)


def _choose_prefix_lengths(reference_length: int, lower_count: int, pattern: tuple[bool, ...]) -> list[int] | None:
    candidates = list(range(2, reference_length))
    if len(candidates) < lower_count:
        return None
    for _ in range(80):
        chosen = sorted(random.sample(candidates, lower_count), reverse=True)
        if _suffix_has_both_types(pattern, chosen[-1]):
            return chosen
    return None


def _render_group_rows(prefix: tuple[bool, ...], width: int) -> list[np.ndarray]:
    palette = tuple(random.sample(INPUT_PALETTE, random.randint(4, min(6, len(INPUT_PALETTE)))))
    rows: list[np.ndarray] = []
    for is_uniform in prefix:
        row = _make_uniform_row(width, palette) if is_uniform else _make_mixed_row(width, palette)
        rows.append(row)
    return rows


def _build_input_grid(reference: tuple[bool, ...], prefixes: list[tuple[bool, ...]], width: int) -> np.ndarray:
    group_rows = [_render_group_rows(reference, width)]
    group_rows.extend(_render_group_rows(prefix, width) for prefix in prefixes)

    groups = len(group_rows)
    gaps = [2] * (groups - 1)
    if random.random() < 0.5:
        choices = [i for i in range(len(gaps))]
        random.shuffle(choices)
        for idx in choices:
            gaps[idx] += 1
            input_height = sum(len(rows) for rows in group_rows) + sum(gaps)
            extra_rows = sum(len(reference) - len(prefix) for prefix in prefixes)
            if input_height + extra_rows > 30:
                gaps[idx] -= 1
                break

    top_margin = 1 if random.random() < 0.35 else 0
    bottom_margin = 1 if random.random() < 0.35 else 0
    input_height = top_margin + bottom_margin + sum(len(rows) for rows in group_rows) + sum(gaps)
    extra_rows = sum(len(reference) - len(prefix) for prefix in prefixes)
    if input_height + extra_rows > 30:
        top_margin = 0
        bottom_margin = 0
        input_height = sum(len(rows) for rows in group_rows) + sum(gaps)
    if input_height + extra_rows > 30:
        raise ValueError("constructed grid exceeds ARC size limit")

    canvas = np.full((input_height, width), BACKGROUND, dtype=int)
    r = top_margin
    for idx, rows in enumerate(group_rows):
        for row in rows:
            canvas[r] = row
            r += 1
        if idx < len(gaps):
            r += gaps[idx]
    return canvas


def _validate_grid_semantics(grid: np.ndarray) -> None:
    groups = _row_groups(grid, BACKGROUND)
    assert 3 <= len(groups) <= 4

    lengths = [r1 - r0 for r0, r1 in groups]
    assert lengths[0] == max(lengths) and lengths.count(lengths[0]) == 1
    assert all(a > b for a, b in zip(lengths, lengths[1:]))

    reference = _pattern_for_group(grid, groups[0])
    assert sum(reference) >= 2 and (len(reference) - sum(reference)) >= 2

    saw_uniform_suffix = False
    saw_mixed_suffix = False
    for idx, group in enumerate(groups[1:], start=1):
        prefix = _pattern_for_group(grid, group)
        assert 2 <= len(prefix) < len(reference)
        assert prefix == reference[: len(prefix)]
        suffix = reference[len(prefix) :]
        saw_uniform_suffix |= any(suffix)
        saw_mixed_suffix |= any(not value for value in suffix)
        for r in range(group[0], group[1]):
            assert np.all(grid[r] != BACKGROUND)
        assert group[0] - groups[idx - 1][1] >= 2

    assert saw_uniform_suffix and saw_mixed_suffix, "examples should expose both completion colors"

    out = main(grid)
    expected_height = grid.shape[0] + sum(len(reference) - (r1 - r0) for r0, r1 in groups[1:])
    assert out.shape == (expected_height, grid.shape[1])

    out_groups = _row_groups(out, BACKGROUND)
    assert len(out_groups) == len(groups)
    for idx, ((in_r0, in_r1), (out_r0, out_r1)) in enumerate(zip(groups, out_groups)):
        prefix_len = in_r1 - in_r0
        assert np.array_equal(out[out_r0 : out_r0 + prefix_len], grid[in_r0:in_r1])
        if idx == 0:
            assert out_r1 - out_r0 == len(reference)
            continue
        suffix = reference[prefix_len:]
        appended = out[out_r0 + prefix_len : out_r1]
        assert appended.shape[0] == len(suffix)
        for row, is_uniform in zip(appended, suffix):
            expected_color = UNIFORM_FILL if is_uniform else MIXED_FILL
            assert np.all(row == expected_color)


def _gen_attempt() -> np.ndarray | None:
    group_count = random.choice((3, 4))
    reference_length = random.randint(5, 7 if group_count == 3 else 6)
    width = random.randint(8, 12)

    reference = _sample_reference_pattern(reference_length)
    prefix_lengths = _choose_prefix_lengths(reference_length, group_count - 1, reference)
    if prefix_lengths is None:
        return None

    prefixes = [reference[:length] for length in prefix_lengths]
    try:
        grid = _build_input_grid(reference, prefixes, width)
        _validate_grid_semantics(grid)
        return grid
    except (AssertionError, ValueError):
        return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
