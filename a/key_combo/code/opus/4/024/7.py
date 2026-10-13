from __future__ import annotations

from itertools import combinations

import numpy as np


BG = 0
FILL_COLOR = 1
HIGHLIGHT_COLOR = 6
SOURCE_COLORS = (2, 3, 4, 5, 7, 8, 9)
MAX_ATTEMPTS = 400


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = sorted(int(v) for v in np.unique(grid))
    if exclude is None:
        return colors
    return [color for color in colors if color not in exclude]


def _row_groups(grid: np.ndarray, bg: int = BG) -> list[tuple[int, int]]:
    occupied = np.any(grid != bg, axis=1)
    groups: list[tuple[int, int]] = []
    r = 0
    while r < grid.shape[0]:
        if not occupied[r]:
            r += 1
            continue
        r0 = r
        while r + 1 < grid.shape[0] and occupied[r + 1]:
            r += 1
        groups.append((r0, r))
        r += 1
    return groups


def _column_states(block: np.ndarray, bg: int = BG) -> list[tuple[int, ...]]:
    states: list[tuple[int, ...]] = []
    for c in range(block.shape[1]):
        states.append(tuple(int(r) for r in np.flatnonzero(block[:, c] != bg)))
    return states


def _paint_state(grid: np.ndarray, top: int, col: int, state: tuple[int, ...], color: int) -> None:
    for dr in state:
        grid[top + dr, col] = color


def _parse_instance(grid: np.ndarray) -> tuple[int, list[tuple[int, ...]], list[tuple[int, int, int]]]:
    groups = _row_groups(grid, BG)
    assert len(groups) >= 3, "expected one reference group and at least two lower groups"

    group_height = groups[0][1] - groups[0][0] + 1
    width = int(grid.shape[1])
    ref_r0, ref_r1 = groups[0]
    reference = grid[ref_r0 : ref_r1 + 1]
    ref_states = _column_states(reference, BG)
    assert all(ref_states), "reference group must occupy every column"

    lower: list[tuple[int, int, int]] = []
    for r0, r1 in groups[1:]:
        assert r1 - r0 + 1 == group_height, "all row groups must share one height"
        block = grid[r0 : r1 + 1]
        occupied_cols = np.any(block != BG, axis=0)
        assert occupied_cols[0], "each lower group must start at the left edge"
        prefix_len = 0
        while prefix_len < width and occupied_cols[prefix_len]:
            prefix_len += 1
        assert 0 < prefix_len < width, "lower groups must be strict prefixes"
        assert np.all(occupied_cols[:prefix_len]), "prefix columns must be contiguous"
        assert not np.any(occupied_cols[prefix_len:]), "suffix columns must be empty before completion"

        states = _column_states(block, BG)
        for c in range(prefix_len):
            assert states[c] == ref_states[c], "lower groups must match the reference prefix exactly"
        for c in range(prefix_len, width):
            assert states[c] == (), "suffix cells must be empty"
        lower.append((r0, r1, prefix_len))

    return group_height, ref_states, lower


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each lower row-group from the top full-width reference, then highlight every (N+1)-th column.

    Input structure:
    - The grid has black background and one foreground color forming several horizontal row groups.
    - The topmost row group spans the full width and defines one non-empty column-state for every column.
    - Each lower row group has the same height, starts at the left edge, and contains a strict prefix of that reference column-state sequence, with the rest of the columns empty.

    Rule:
    - Read the top row group's per-column occupied-row offsets as the reference sequence.
    - Count the number N of lower row groups.
    - For each lower group, keep its existing prefix and fill every empty suffix column with the corresponding reference state using blue (1).
    - After all completions, recolor every non-black pixel in columns c where c mod (N+1) == 0 to magenta (6).

    Key insight:
    - The completion pattern comes from the top row group, but the later recolor period is not fixed: it is derived from how many lower row groups the input contains.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    src_colors = _colors_present(g, exclude={BG})
    assert len(src_colors) == 1, "expected exactly one input foreground color"

    _group_height, ref_states, lower = _parse_instance(g)
    out = g.copy()

    for r0, _r1, prefix_len in lower:
        for c in range(prefix_len, g.shape[1]):
            _paint_state(out, r0, c, ref_states[c], FILL_COLOR)

    period = len(lower) + 1
    for c in range(g.shape[1]):
        if c % period == 0:
            mask = out[:, c] != BG
            out[mask, c] = HIGHLIGHT_COLOR
    return out


def _all_states(height: int) -> list[tuple[int, ...]]:
    states: list[tuple[int, ...]] = []
    rows = list(range(height))
    for size in range(1, height + 1):
        for combo in combinations(rows, size):
            states.append(tuple(int(v) for v in combo))
    return states


def _is_periodic(seq: list[tuple[int, ...]], period: int) -> bool:
    return all(seq[i] == seq[i % period] for i in range(len(seq)))


def _sequence_is_good(states: list[tuple[int, ...]], height: int, tail_start: int) -> bool:
    if len(set(states)) < 3:
        return False
    if sum(states[i] != states[i - 1] for i in range(1, len(states))) < 3:
        return False
    final_state = states[-1]
    if final_state in states[:tail_start]:
        return False
    if tail_start > 0 and states[tail_start - 1] == final_state:
        return False
    for period in range(1, min(4, len(states) - 1) + 1):
        if _is_periodic(states, period):
            return False
    for row in range(height):
        if not any(row in state for state in states):
            return False
    return True


def _make_reference_states(height: int, width: int, rng: np.random.Generator) -> tuple[list[tuple[int, ...]], int]:
    all_states = _all_states(height)
    max_tail = min(4, width - 3)
    for _ in range(MAX_ATTEMPTS):
        final_state = all_states[int(rng.integers(0, len(all_states)))]
        tail_len = int(rng.integers(2, max_tail + 1))
        tail_start = width - tail_len
        leading_choices = [state for state in all_states if state != final_state]
        leading: list[tuple[int, ...]] = []
        while len(leading) < tail_start:
            if not leading:
                candidates = leading_choices
            elif len(leading) >= 2 and leading[-1] == leading[-2]:
                candidates = [state for state in leading_choices if state != leading[-1]]
            elif rng.random() < 0.3 and len(leading) + 1 < tail_start:
                candidates = leading_choices
            else:
                candidates = [state for state in leading_choices if state != leading[-1]]
                if not candidates:
                    candidates = leading_choices
            leading.append(candidates[int(rng.integers(0, len(candidates)))])

        states = leading + [final_state] * tail_len
        if _sequence_is_good(states, height, tail_start):
            return states, tail_start
    raise RuntimeError("failed to build a non-degenerate reference sequence")


def _prefix_covers_all_rows(states: list[tuple[int, ...]], prefix_len: int, height: int) -> bool:
    return all(any(row in states[c] for c in range(prefix_len)) for row in range(height))


def _choose_prefix_lengths(
    width: int,
    n_lower: int,
    tail_start: int,
    height: int,
    states: list[tuple[int, ...]],
    rng: np.random.Generator,
) -> list[int]:
    valid_lengths = [k for k in range(1, width) if _prefix_covers_all_rows(states, k, height)]
    if len(valid_lengths) < n_lower:
        raise RuntimeError("not enough valid prefix lengths for the chosen reference")

    chosen: set[int] = set()
    early_candidates = [k for k in valid_lengths if k < tail_start]
    late_candidates = [k for k in valid_lengths if k >= tail_start]

    if early_candidates:
        chosen.add(int(rng.choice(early_candidates)))
    if late_candidates and len(chosen) < n_lower:
        chosen.add(int(rng.choice(late_candidates)))

    while len(chosen) < n_lower:
        chosen.add(int(rng.choice(valid_lengths)))

    lengths = list(chosen)
    rng.shuffle(lengths)
    return lengths


def _build_input(
    width: int,
    height: int,
    gaps: list[int],
    states: list[tuple[int, ...]],
    prefix_lengths: list[int],
    source_color: int,
    bottom_pad: int,
) -> np.ndarray:
    total_rows = height * (len(prefix_lengths) + 1) + sum(gaps) + bottom_pad
    grid = np.zeros((total_rows, width), dtype=int)

    tops = [0]
    for gap in gaps:
        tops.append(tops[-1] + height + gap)

    for c, state in enumerate(states):
        _paint_state(grid, tops[0], c, state, source_color)

    for idx, prefix_len in enumerate(prefix_lengths, start=1):
        for c in range(prefix_len):
            _paint_state(grid, tops[idx], c, states[c], source_color)
    return grid


def _apply_completion_then_highlight(
    width: int,
    height: int,
    gaps: list[int],
    states: list[tuple[int, ...]],
    prefix_lengths: list[int],
    source_color: int,
    bottom_pad: int,
    period: int,
    fill_suffix_with_final: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    grid = _build_input(width, height, gaps, states, prefix_lengths, source_color, bottom_pad)
    out = grid.copy()

    final_state = states[-1]
    tops = [0]
    for gap in gaps:
        tops.append(tops[-1] + height + gap)

    for idx, prefix_len in enumerate(prefix_lengths, start=1):
        top = tops[idx]
        for c in range(prefix_len, width):
            state = final_state if fill_suffix_with_final else states[c]
            _paint_state(out, top, c, state, FILL_COLOR)

    pre_highlight = out.copy()
    for c in range(width):
        if c % period == 0:
            mask = out[:, c] != BG
            out[mask, c] = HIGHLIGHT_COLOR
    return pre_highlight, out


def _validate_generated(
    grid: np.ndarray,
    states: list[tuple[int, ...]],
    prefix_lengths: list[int],
    source_color: int,
    height: int,
    gaps: list[int],
    bottom_pad: int,
) -> bool:
    width = int(grid.shape[1])
    n_lower = len(prefix_lengths)
    period = n_lower + 1

    parsed_height, parsed_states, lower = _parse_instance(grid)
    if parsed_height != height or parsed_states != states:
        return False
    if [prefix_len for _r0, _r1, prefix_len in lower] != prefix_lengths:
        return False

    pre_highlight, expected = _apply_completion_then_highlight(
        width, height, gaps, states, prefix_lengths, source_color, bottom_pad, period
    )
    if not np.array_equal(main(grid), expected):
        return False

    if not np.any(expected == FILL_COLOR):
        return False
    if not np.any((pre_highlight == FILL_COLOR) & (expected == HIGHLIGHT_COLOR)):
        return False
    if not np.any((pre_highlight == source_color) & (expected == HIGHLIGHT_COLOR)):
        return False

    wrong_completion = _apply_completion_then_highlight(
        width,
        height,
        gaps,
        states,
        prefix_lengths,
        source_color,
        bottom_pad,
        period,
        fill_suffix_with_final=True,
    )[1]
    if np.array_equal(wrong_completion, expected):
        return False

    for wrong_period in (2, 3, 4, 5):
        if wrong_period == period:
            continue
        wrong_output = _apply_completion_then_highlight(
            width, height, gaps, states, prefix_lengths, source_color, bottom_pad, wrong_period
        )[1]
        if np.array_equal(wrong_output, expected):
            return False

    return True


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    n_lower = int(rng.integers(2, 5))
    group_height = int(rng.integers(2, 4))
    width = int(rng.integers(10, 19))
    gaps = [int(rng.integers(1, 3)) for _ in range(n_lower)]
    bottom_pad = int(rng.integers(0, 3))

    states, tail_start = _make_reference_states(group_height, width, rng)
    prefix_lengths = _choose_prefix_lengths(width, n_lower, tail_start, group_height, states, rng)
    if len(set(prefix_lengths)) != len(prefix_lengths):
        return None

    source_color = int(rng.choice(SOURCE_COLORS))
    grid = _build_input(width, group_height, gaps, states, prefix_lengths, source_color, bottom_pad)
    if max(grid.shape) > 30:
        return None
    if not _validate_generated(grid, states, prefix_lengths, source_color, group_height, gaps, bottom_pad):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")
