import random
from typing import List, Sequence, Tuple

import numpy as np


BG = 0
BLUE = 1
RED = 2

MIN_GROUPS = 2
MAX_GROUPS = 4
MAX_ATTEMPTS = 500

State = Tuple[int, ...]


def _background_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel())
    return int(np.argmax(counts))


def _row_groups(grid: np.ndarray, bg: int) -> List[Tuple[int, int]]:
    rows = np.any(grid != bg, axis=1)
    groups: List[Tuple[int, int]] = []
    r = 0
    while r < grid.shape[0]:
        if not rows[r]:
            r += 1
            continue
        r0 = r
        while r + 1 < grid.shape[0] and rows[r + 1]:
            r += 1
        groups.append((r0, r))
        r += 1
    return groups


def _column_states(grid: np.ndarray, bg: int, r0: int, r1: int) -> List[State]:
    band = grid[r0 : r1 + 1, :]
    states: List[State] = []
    for c in range(grid.shape[1]):
        states.append(tuple(int(rr) for rr in np.flatnonzero(band[:, c] != bg)))
    return states


def _prefix_length(states: Sequence[State]) -> int:
    filled = [c for c, state in enumerate(states) if state]
    assert filled, "lower group must contain foreground"
    assert filled[-1] < len(states) - 1, "lower group must be a strict prefix"
    assert filled == list(range(filled[-1] + 1)), "lower group must be left-aligned"
    return filled[-1] + 1


def _paint_state_columns(height: int, states: Sequence[State], color: int) -> np.ndarray:
    patch = np.zeros((height, len(states)), dtype=int)
    for c, state in enumerate(states):
        for rr in state:
            patch[int(rr), c] = int(color)
    return patch


def _completed_patch(height: int, top_states: Sequence[State], prefix_len: int, fg: int) -> np.ndarray:
    width = len(top_states)
    patch = np.zeros((height, width), dtype=int)
    for c in range(prefix_len):
        for rr in top_states[c]:
            patch[int(rr), c] = int(fg)
    for c in range(prefix_len, width):
        for rr in top_states[c]:
            patch[int(rr), c] = BLUE
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """Complete each lower prefix from the top row-group, then shift it upward by its number of newly added columns.

    Input structure:
    A black-background grid contains several separated horizontal row-groups.
    The topmost widest group is a full-width red reference pattern whose columns
    are described by per-column occupied row-offset states. Every lower group
    has the same height, starts at the left edge, and shows only a strict prefix
    of that same concrete column-state sequence.

    Rule:
    Keep the top reference group fixed. For each lower group, determine its
    prefix length by matching its occupied columns against the top pattern,
    complete the missing suffix columns using the top states in blue, then move
    the whole completed group upward by exactly the number of added columns.
    Original lower-group positions are cleared, and lower groups are pasted in
    top-to-bottom processing order so later ones may overwrite earlier ones.

    Key insight:
    The completion amount is not just decoration: the missing suffix length is
    the vertical translation distance, so less-complete prefixes move farther.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)
    colors = sorted(set(map(int, np.unique(g))) - {bg})
    assert len(colors) == 1, "input must use one foreground color"
    fg = int(colors[0])

    groups = _row_groups(g, bg)
    assert len(groups) >= 3, "expected one reference group and multiple lower groups"

    coverages = [sum(bool(state) for state in _column_states(g, bg, r0, r1)) for r0, r1 in groups]
    top_index = min(i for i, cov in enumerate(coverages) if cov == max(coverages))
    top_r0, top_r1 = groups[top_index]
    top_height = top_r1 - top_r0 + 1
    top_states = _column_states(g, bg, top_r0, top_r1)
    assert all(top_states), "reference group must define every column"

    out = np.full_like(g, bg)
    out[top_r0 : top_r1 + 1, :] = g[top_r0 : top_r1 + 1, :]

    for idx, (r0, r1) in enumerate(groups):
        if idx == top_index:
            continue
        assert r1 - r0 + 1 == top_height, "all groups must share the reference height"
        states = _column_states(g, bg, r0, r1)
        prefix_len = _prefix_length(states)
        assert states[:prefix_len] == top_states[:prefix_len], "lower group must be a concrete prefix of the top pattern"
        shift = g.shape[1] - prefix_len
        assert shift > 0
        dst_r0 = r0 - shift
        assert dst_r0 >= 0, "shifted group must remain in-bounds"

        patch = _completed_patch(top_height, top_states, prefix_len, fg)
        sub = out[dst_r0 : dst_r0 + top_height, :]
        mask = patch != bg
        sub[mask] = patch[mask]
        out[dst_r0 : dst_r0 + top_height, :] = sub

    return out


def _random_partition(total: int, parts: int) -> List[int]:
    assert 1 <= parts <= total
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    bounds = [0] + cuts + [total]
    return [bounds[i + 1] - bounds[i] for i in range(parts)]


def _all_states(height: int) -> List[State]:
    states: List[State] = []
    for mask in range(1, 1 << height):
        states.append(tuple(r for r in range(height) if mask & (1 << r)))
    return states


def _run_boundaries(widths: Sequence[int]) -> List[int]:
    out: List[int] = []
    pos = 0
    for w in widths[:-1]:
        pos += int(w)
        out.append(pos)
    return out


def _generate_reference_pattern(width: int, height: int) -> Tuple[List[State], List[int]]:
    states_pool = _all_states(height)
    max_runs = min(8, width)
    for _ in range(200):
        run_count = random.randint(5, max_runs)
        widths = _random_partition(width, run_count)

        run_states = [random.choice(states_pool)]
        while len(run_states) < run_count:
            prev = run_states[-1]
            choices = [state for state in states_pool if state != prev]
            run_states.append(random.choice(choices))

        if len(set(run_states)) < min(3, len(states_pool)):
            continue
        if set().union(*(set(state) for state in run_states)) != set(range(height)):
            continue

        top_states: List[State] = []
        for state, run_width in zip(run_states, widths):
            top_states.extend([state] * int(run_width))
        if len(top_states) != width:
            continue
        return top_states, widths
    raise RuntimeError("failed to generate a reference pattern")


def _choose_prefix_lengths(width: int, widths: Sequence[int], height: int, lower_count: int) -> List[int]:
    boundary_positions = set(_run_boundaries(widths))
    max_shift = min(11, width - 4)
    min_prefix = max(4, width - max_shift)
    candidates = [p for p in range(min_prefix, width - 1) if width - p != height]
    assert len(candidates) >= lower_count

    boundary_candidates = [p for p in candidates if p in boundary_positions]
    interior_candidates = [p for p in candidates if p not in boundary_positions]
    picked: List[int] = []

    if boundary_candidates:
        picked.append(random.choice(boundary_candidates))
    if interior_candidates and len(picked) < lower_count:
        choice = random.choice(interior_candidates)
        if choice not in picked:
            picked.append(choice)

    remaining = [p for p in candidates if p not in picked]
    random.shuffle(remaining)
    picked.extend(remaining[: lower_count - len(picked)])
    random.shuffle(picked)

    assert len(set(picked)) == lower_count
    assert any(p in boundary_positions for p in picked)
    assert any(p not in boundary_positions for p in picked)
    return picked


def _plan_rows(height: int, shifts: Sequence[int], want_overlap: bool) -> Tuple[int, List[int], int]:
    for _ in range(300):
        top_r0 = random.randint(0, 1)
        top_r1 = top_r0 + height - 1
        rows: List[int] = []
        prev_bottom = top_r1
        for shift in shifts:
            gap = random.randint(1, 2) if want_overlap else random.randint(2, 4)
            base_r0 = max(prev_bottom + 1 + gap, top_r1 + 1 + int(shift))
            slack = random.randint(0, 1) if want_overlap else random.randint(0, 3)
            r0 = base_r0 + slack
            rows.append(r0)
            prev_bottom = r0 + height - 1

        total_height = prev_bottom + 1 + random.randint(0, 2)
        if total_height > 30:
            continue

        dest_rows = [r0 - int(shift) for r0, shift in zip(rows, shifts)]
        overlaps = []
        for a, b in zip(dest_rows, dest_rows[1:]):
            overlaps.append(max(0, min(a + height - 1, b + height - 1) - max(a, b) + 1))

        has_overlap = any(ov > 0 for ov in overlaps)
        if has_overlap != want_overlap:
            continue
        if overlaps and max(overlaps) > 1:
            continue

        total_height = max(total_height, random.randint(18, min(26, 30)))
        if total_height <= 30:
            return top_r0, rows, total_height

    raise RuntimeError("failed to place row groups")


def _render_input(height: int, width: int, top_r0: int, top_states: Sequence[State], row_starts: Sequence[int], prefix_lengths: Sequence[int]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    band_height = max(max(state) for state in top_states) + 1
    grid[top_r0 : top_r0 + band_height, :] = _paint_state_columns(band_height, top_states, RED)
    for r0, prefix_len in zip(row_starts, prefix_lengths):
        for c in range(prefix_len):
            for rr in top_states[c]:
                grid[r0 + int(rr), c] = RED
        assert np.all(grid[r0 : r0 + band_height, prefix_len:] == BG)
    return grid


def _render_expected(height: int, width: int, top_r0: int, band_height: int, top_states: Sequence[State], row_starts: Sequence[int], prefix_lengths: Sequence[int]) -> np.ndarray:
    out = np.zeros((height, width), dtype=int)
    out[top_r0 : top_r0 + band_height, :] = _paint_state_columns(band_height, top_states, RED)

    for r0, prefix_len in zip(row_starts, prefix_lengths):
        patch = _completed_patch(band_height, top_states, prefix_len, RED)
        dst_r0 = r0 - (width - prefix_len)
        sub = out[dst_r0 : dst_r0 + band_height, :]
        mask = patch != BG
        sub[mask] = patch[mask]
        out[dst_r0 : dst_r0 + band_height, :] = sub
    return out


def _validate_plan(inp: np.ndarray, expected: np.ndarray, top_states: Sequence[State], widths: Sequence[int], row_starts: Sequence[int], prefix_lengths: Sequence[int], want_overlap: bool) -> bool:
    band_height = max(max(state) for state in top_states) + 1
    transitions = sum(top_states[i] != top_states[i - 1] for i in range(1, len(top_states)))
    if transitions < 4:
        return False
    if len(prefix_lengths) < MIN_GROUPS or len(prefix_lengths) > MAX_GROUPS:
        return False
    if len(set(prefix_lengths)) != len(prefix_lengths):
        return False

    shifts = [inp.shape[1] - p for p in prefix_lengths]
    if len(set(shifts)) != len(shifts):
        return False
    if any(shift == band_height for shift in shifts):
        return False
    if min(shifts) < 2 or max(shifts) <= min(shifts):
        return False

    boundaries = set(_run_boundaries(widths))
    if not any(p in boundaries for p in prefix_lengths):
        return False
    if not any(p not in boundaries for p in prefix_lengths):
        return False

    moved_masks = []
    for r0, prefix_len in zip(row_starts, prefix_lengths):
        patch = _completed_patch(band_height, top_states, prefix_len, RED) != BG
        mask = np.zeros_like(inp, dtype=bool)
        dst_r0 = r0 - (inp.shape[1] - prefix_len)
        mask[dst_r0 : dst_r0 + band_height, :] = patch
        moved_masks.append(mask)

    if want_overlap:
        hidden = sum(int(mask.sum()) for mask in moved_masks) - int(np.any(np.stack(moved_masks), axis=0).sum())
        if hidden <= 0:
            return False
        if hidden > int(sum(int(mask.sum()) for mask in moved_masks) * 0.35):
            return False
    else:
        for a in range(len(moved_masks)):
            for b in range(a + 1, len(moved_masks)):
                if np.any(moved_masks[a] & moved_masks[b]):
                    return False

    if np.array_equal(inp, expected):
        return False
    if np.count_nonzero(expected == BLUE) == 0:
        return False
    if np.count_nonzero(inp == RED) == 0:
        return False
    return True


def _generate_case() -> Tuple[np.ndarray, np.ndarray]:
    width = random.randint(16, 24)
    band_height = random.randint(2, 3)
    lower_count = random.randint(MIN_GROUPS, MAX_GROUPS)

    top_states, widths = _generate_reference_pattern(width, band_height)
    prefix_lengths = _choose_prefix_lengths(width, widths, band_height, lower_count)

    want_overlap = random.random() < 0.35
    shifts = [width - p for p in prefix_lengths]
    ordered_pairs = list(zip(prefix_lengths, shifts))
    if want_overlap:
        ordered_pairs.sort(key=lambda item: item[1])
    else:
        ordered_pairs.sort(key=lambda item: item[1], reverse=True)
        if random.random() < 0.5:
            random.shuffle(ordered_pairs)
    prefix_lengths = [p for p, _shift in ordered_pairs]
    shifts = [shift for _p, shift in ordered_pairs]

    top_r0, row_starts, height = _plan_rows(band_height, shifts, want_overlap)
    inp = _render_input(height, width, top_r0, top_states, row_starts, prefix_lengths)
    expected = _render_expected(height, width, top_r0, band_height, top_states, row_starts, prefix_lengths)

    if not _validate_plan(inp, expected, top_states, widths, row_starts, prefix_lengths, want_overlap):
        raise RuntimeError("constructed plan failed validation")
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            inp, expected = _generate_case()
        except Exception:
            continue
        try:
            observed = main(inp)
        except Exception:
            continue
        if np.array_equal(observed, expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
