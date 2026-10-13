from __future__ import annotations

from collections import Counter
import itertools
import math

import numpy as np


def _dominant_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(grid).ravel()))
    return int(counts.most_common(1)[0][0])


def _is_periodic(bits: np.ndarray) -> bool:
    width = int(bits.shape[0])
    for period in range(1, width):
        if np.array_equal(bits[:-period], bits[period:]):
            return True
    return False


def _parse_fragments(grid: np.ndarray, bg: int) -> list[dict]:
    height, width = map(int, grid.shape)
    fragments: list[dict] = []

    for row_index in range(height):
        row = grid[row_index]
        cols = np.flatnonzero(row != bg)
        if cols.size == 0:
            continue

        left = int(cols[0])
        right = int(cols[-1])
        assert cols.size == right - left + 1, "each non-empty row must contain one contiguous segment"
        assert cols.size < width, "no segment may already span the full row"

        segment = row[left : right + 1]
        first_color = int(segment[0])
        second_positions = np.flatnonzero(segment != first_color)
        assert second_positions.size > 0, "each segment must use two colors"
        second_color = int(segment[int(second_positions[0])])
        allowed = {first_color, second_color}
        assert len(allowed) == 2, "segment colors must be distinct"
        assert all(int(value) in allowed for value in segment), "segment may only use two colors"

        local_bits = np.array([0 if int(value) == first_color else 1 for value in segment], dtype=int)
        assert 0 in local_bits and 1 in local_bits, "both hidden roles must appear in every segment"
        fragments.append(
            {
                "row": int(row_index),
                "left": left,
                "right": right,
                "colors": (first_color, second_color),
                "bits": local_bits,
            }
        )

    return fragments


def _overlap_parity(fragment_a: dict, fragment_b: dict) -> int | None:
    left = max(int(fragment_a["left"]), int(fragment_b["left"]))
    right = min(int(fragment_a["right"]), int(fragment_b["right"]))
    if left > right:
        return None

    bits_a = fragment_a["bits"][left - int(fragment_a["left"]) : right - int(fragment_a["left"]) + 1]
    bits_b = fragment_b["bits"][left - int(fragment_b["left"]) : right - int(fragment_b["left"]) + 1]
    parity = bits_a ^ bits_b
    assert np.all(parity == parity[0]), "overlapping fragments must agree up to one global flip"
    return int(parity[0])


def _infer_hidden_bits(width: int, fragments: list[dict]) -> tuple[np.ndarray, list[int]]:
    count = len(fragments)
    assert count >= 2, "expected multiple fragment rows"

    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(count)]
    edge_count = 0
    for i in range(count):
        for j in range(i + 1, count):
            parity = _overlap_parity(fragments[i], fragments[j])
            if parity is None:
                continue
            adjacency[i].append((j, parity))
            adjacency[j].append((i, parity))
            edge_count += 1
    assert edge_count > 0, "expected at least one overlapping pair"

    flips: list[int | None] = [None] * count
    flips[0] = 0
    stack = [0]
    while stack:
        current = stack.pop()
        current_flip = int(flips[current])
        for other, parity in adjacency[current]:
            wanted = current_flip ^ int(parity)
            if flips[other] is None:
                flips[other] = wanted
                stack.append(other)
            else:
                assert int(flips[other]) == wanted, "fragment overlaps must induce a consistent global stitching"

    assert all(value is not None for value in flips), "fragment overlap graph must be connected"
    resolved_flips = [int(value) for value in flips]

    hidden_bits = np.full(width, -1, dtype=int)
    for fragment, flip in zip(fragments, resolved_flips):
        left = int(fragment["left"])
        local_bits = fragment["bits"] ^ int(flip)
        for offset, bit in enumerate(local_bits):
            col = left + offset
            if hidden_bits[col] == -1:
                hidden_bits[col] = int(bit)
            else:
                assert int(hidden_bits[col]) == int(bit), "all rows covering one column must imply the same hidden role"

    assert np.all(hidden_bits != -1), "the stitched fragments must cover the full width"
    return hidden_bits, resolved_flips


def main(grid: np.ndarray) -> np.ndarray:
    """Complete every two-color horizontal fragment into a full-width row by stitching one hidden binary ribbon.

    Input structure:
    - A dominant background fills the grid, and every non-background row contains exactly one contiguous horizontal segment.
    - Each such segment is shorter than the full width and uses exactly two colors, with both colors present.
    - The segment intervals chain-overlap and together cover every column of the grid.

    Rule:
    - Treat each segment as a local view of one hidden full-width binary role sequence, where its two colors encode the two roles on that segment's interval.
    - Use overlapping columns to determine which rows use the same role orientation and which rows are flipped, then stitch all partial views into one consistent hidden sequence.
    - Replace every non-background row by the full-width realization of that hidden sequence using that row's own two colors; leave all-background rows unchanged.

    Key insight:
    - A row only reveals the hidden ribbon up to swapping its two local colors, so the solution comes from overlap consistency across rows rather than from extending any single row by itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert width >= 2, "grid must be wide enough for a two-role ribbon"

    bg = _dominant_color(g)
    fragments = _parse_fragments(g, bg)
    hidden_bits, flips = _infer_hidden_bits(width, fragments)

    out = g.copy()
    for fragment, flip in zip(fragments, flips):
        role_colors = fragment["colors"]
        full_row = np.array([role_colors[int(bit) ^ int(flip)] for bit in hidden_bits], dtype=int)
        out[int(fragment["row"]), :] = full_row

    return out


_MAX_ATTEMPTS = 400
_WIDTH_RANGE = (10, 22)
_NONEMPTY_ROWS_RANGE = (3, 5)
_BLANK_ROWS_RANGE = (1, 4)
_RUN_COUNT_RANGE = (4, 8)
_RUN_LEN_MAX = 5
_ADVANCE_MIN_FIRST = 4
_ADVANCE_MIN_OTHER = 2
_ADVANCE_MAX = 8
_OVERLAP_MIN = 2
_OVERLAP_MAX = 5


def _bounded_composition(total: int, minima: list[int], maxima: list[int], rng: np.random.Generator) -> list[int] | None:
    values = list(minima)
    capacities = [int(hi) - int(lo) for lo, hi in zip(minima, maxima)]
    remaining = int(total - sum(values))
    if remaining < 0 or any(cap < 0 for cap in capacities):
        return None

    while remaining > 0:
        choices = [index for index, cap in enumerate(capacities) if cap > 0]
        if not choices:
            return None
        index = int(rng.choice(choices))
        add = int(rng.integers(1, min(remaining, capacities[index]) + 1))
        values[index] += add
        capacities[index] -= add
        remaining -= add
    return values


def _generate_hidden_bits(width: int, rng: np.random.Generator) -> np.ndarray | None:
    max_runs = min(_RUN_COUNT_RANGE[1], width)
    run_count = int(rng.integers(_RUN_COUNT_RANGE[0], max_runs + 1))
    minima = [1] * run_count
    maxima = [min(_RUN_LEN_MAX, width - run_count + 1)] * run_count
    run_lengths = _bounded_composition(width, minima, maxima, rng)
    if run_lengths is None:
        return None

    start_bit = int(rng.integers(0, 2))
    pieces = [np.full(length, (start_bit + index) % 2, dtype=int) for index, length in enumerate(run_lengths)]
    bits = np.concatenate(pieces)
    if _is_periodic(bits):
        return None
    return bits


def _generate_intervals(width: int, row_count: int, rng: np.random.Generator) -> list[tuple[int, int]] | None:
    minima = [_ADVANCE_MIN_FIRST] + [_ADVANCE_MIN_OTHER] * (row_count - 1)
    maxima = [min(_ADVANCE_MAX, width - 2)] * row_count
    advances = _bounded_composition(width, minima, maxima, rng)
    if advances is None:
        return None

    boundaries = [0]
    for advance in advances:
        boundaries.append(boundaries[-1] + int(advance))

    intervals: list[tuple[int, int]] = []
    for index in range(row_count):
        right = int(boundaries[index + 1] - 1)
        if index == 0:
            left = 0
        else:
            max_overlap = min(_OVERLAP_MAX, boundaries[index] - 1)
            if max_overlap < _OVERLAP_MIN:
                return None
            overlap = int(rng.integers(_OVERLAP_MIN, max_overlap + 1))
            left = int(boundaries[index] - overlap)
        if left < 0 or right >= width or left >= right:
            return None
        if left == 0 and right == width - 1:
            return None
        intervals.append((left, right))

    return intervals


def _segment_values(bits: np.ndarray, interval: tuple[int, int], role_colors: tuple[int, int]) -> np.ndarray:
    left, right = interval
    window = bits[left : right + 1]
    return np.array([role_colors[int(bit)] for bit in window], dtype=int)


def _validate_generated_structure(
    grid: np.ndarray,
    bg: int,
    hidden_bits: np.ndarray,
    intervals: list[tuple[int, int]],
    row_positions: list[int],
) -> None:
    assert not _is_periodic(hidden_bits), "hidden ribbon must be aperiodic"
    assert set(map(int, hidden_bits)) == {0, 1}, "both hidden roles must appear globally"

    ordered = sorted(intervals)
    assert ordered[0][0] == 0 and ordered[-1][1] == grid.shape[1] - 1, "interval union must reach both width edges"
    frontier = -1
    for index, (left, right) in enumerate(ordered):
        assert 0 <= left <= right < grid.shape[1]
        if index == 0:
            frontier = right
        else:
            assert left <= frontier, "intervals must chain-overlap"
            frontier = max(frontier, right)
        window = hidden_bits[left : right + 1]
        assert 0 in window and 1 in window, "every fragment must show both roles"
    assert frontier == grid.shape[1] - 1, "interval union must cover the whole width"

    for earlier, later in zip(ordered, ordered[1:]):
        overlap_left = max(earlier[0], later[0])
        overlap_right = min(earlier[1], later[1])
        overlap = hidden_bits[overlap_left : overlap_right + 1]
        assert overlap.size >= _OVERLAP_MIN, "consecutive fragments must overlap non-trivially"
        assert 0 in overlap and 1 in overlap, "overlap windows must reveal both roles"

    non_bg_rows = [r for r in range(grid.shape[0]) if np.any(grid[r] != bg)]
    assert sorted(non_bg_rows) == sorted(row_positions), "exactly the chosen rows may contain fragments"


def _generate_example(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    width = int(rng.integers(_WIDTH_RANGE[0], _WIDTH_RANGE[1] + 1))
    max_rows = min(_NONEMPTY_ROWS_RANGE[1], width // 2)
    row_count = int(rng.integers(_NONEMPTY_ROWS_RANGE[0], max_rows + 1))
    blank_rows = int(rng.integers(_BLANK_ROWS_RANGE[0], _BLANK_ROWS_RANGE[1] + 1))
    height = row_count + blank_rows

    hidden_bits = _generate_hidden_bits(width, rng)
    if hidden_bits is None:
        return None
    intervals = _generate_intervals(width, row_count, rng)
    if intervals is None:
        return None

    bg = int(rng.integers(0, 10))
    palette = [color for color in range(10) if color != bg]
    pair_pool = list(itertools.combinations(palette, 2))
    chosen_indices = rng.choice(len(pair_pool), size=row_count, replace=False)
    role_colors: list[tuple[int, int]] = []
    for index in np.atleast_1d(chosen_indices):
        color_a, color_b = pair_pool[int(index)]
        if int(rng.integers(0, 2)) == 0:
            role_colors.append((int(color_a), int(color_b)))
        else:
            role_colors.append((int(color_b), int(color_a)))

    row_positions = sorted(map(int, rng.choice(height, size=row_count, replace=False).tolist()))
    shuffled_intervals = intervals[:]
    rng.shuffle(shuffled_intervals)

    inp = np.full((height, width), bg, dtype=int)
    out = np.full((height, width), bg, dtype=int)
    for row_index, interval, colors in zip(row_positions, shuffled_intervals, role_colors):
        segment = _segment_values(hidden_bits, interval, colors)
        left, right = interval
        inp[row_index, left : right + 1] = segment
        out[row_index, :] = np.array([colors[int(bit)] for bit in hidden_bits], dtype=int)

    try:
        _validate_generated_structure(inp, bg, hidden_bits, shuffled_intervals, row_positions)
        solved = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(solved, out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_MAX_ATTEMPTS):
        example = _generate_example(rng)
        if example is None:
            continue
        return example[0]
    raise RuntimeError("failed to generate a valid fragmented-ribbon completion input")
