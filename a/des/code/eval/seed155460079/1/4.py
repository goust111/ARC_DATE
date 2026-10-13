from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

PATH_COLORS = [BLUE, RED, YELLOW, ORANGE, AZURE]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_RETRIES = 300
PATH_RETRIES = 80

PATTERN_LIBRARY = [
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array(
        [[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 1, 1], [1, 1, 1, 1]],
        dtype=bool,
    ),
    np.array(
        [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]],
        dtype=bool,
    ),
    np.array(
        [[0, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 0]],
        dtype=bool,
    ),
    np.array(
        [[1, 1, 1, 0], [0, 1, 0, 0], [0, 1, 1, 1], [0, 0, 0, 1]],
        dtype=bool,
    ),
]


def _neighbors4(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    for dr, dc in DIRS4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _group_indices(indices: np.ndarray) -> list[tuple[int, int]]:
    if len(indices) == 0:
        return []
    groups: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
            continue
        groups.append((start, prev))
        start = prev = value
    groups.append((start, prev))
    return groups


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if cell else "0" for cell in row) for row in mask.astype(bool))


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].astype(bool)


def _find_separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert cols, "expected a full-height grey separator"
    scored = []
    for c in cols:
        left_nonblack = int(np.count_nonzero(grid[:, :c] != BLACK))
        right_nonblack = int(np.count_nonzero(grid[:, c + 1 :] != BLACK))
        scored.append((min(left_nonblack, right_nonblack), c))
    scored.sort()
    return int(scored[-1][1])


def _path_component(left: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = left.shape
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for rr, cc in _neighbors4(r, c, h, w):
            if left[rr, cc] == BLACK or (rr, cc) in seen:
                continue
            seen.add((rr, cc))
            stack.append((rr, cc))
    return seen


def _ordered_path(left: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    component = _path_component(left, start)
    assert end in component, "start and end must lie on the same path"
    path = [start]
    while path[-1] != end:
        r, c = path[-1]
        prev = path[-2] if len(path) >= 2 else None
        options = []
        for rr, cc in _neighbors4(r, c, left.shape[0], left.shape[1]):
            if (rr, cc) not in component or (rr, cc) == prev:
                continue
            options.append((rr, cc))
        assert len(options) == 1, "maze path must be a single non-branching line"
        path.append(options[0])
    return path


def _extract_path_sequence(left: np.ndarray) -> tuple[int, ...]:
    start_pts = np.argwhere(left == GREEN)
    end_pts = np.argwhere(left == MAROON)
    assert len(start_pts) == 1 and len(end_pts) == 1
    start = tuple(int(v) for v in start_pts[0])
    end = tuple(int(v) for v in end_pts[0])
    path = _ordered_path(left, start, end)
    colors = [int(left[r, c]) for r, c in path[1:-1]]
    assert colors, "path must contain colored segments between the markers"
    sequence: list[int] = []
    for color in colors:
        if not sequence or sequence[-1] != color:
            sequence.append(color)
    return tuple(sequence)


def _parse_table_rows(right: np.ndarray) -> list[tuple[tuple[int, ...], np.ndarray]]:
    active_rows = np.flatnonzero(np.any(right != BLACK, axis=1))
    bands = _group_indices(active_rows)
    rows: list[tuple[tuple[int, ...], np.ndarray]] = []
    for r0, r1 in bands:
        band = right[r0 : r1 + 1]
        magenta = np.argwhere(band == MAGENTA)
        assert len(magenta) > 0, "each legend row must contain a magenta pattern"
        (pr0, pc0), (pr1, pc1) = magenta.min(axis=0), magenta.max(axis=0)
        pattern_mask = band[int(pr0) : int(pr1) + 1, int(pc0) : int(pc1) + 1] == MAGENTA
        pattern = np.zeros(pattern_mask.shape, dtype=int)
        pattern[pattern_mask] = MAGENTA

        left_part = band[:, : int(pc0)]
        seq_cols = np.flatnonzero(np.any((left_part != BLACK) & (left_part != MAGENTA), axis=0))
        col_groups = _group_indices(seq_cols)
        sequence: list[int] = []
        for c0, c1 in col_groups:
            colors = {int(v) for v in left_part[:, c0 : c1 + 1].ravel() if int(v) not in {BLACK, MAGENTA}}
            assert len(colors) == 1, "each code block should be monochrome"
            sequence.append(colors.pop())
        assert sequence, "each legend row must expose a color sequence"
        rows.append((tuple(sequence), pattern))
    assert rows, "encoding table must contain at least one row"
    return rows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the start-to-end path, compress its segment colors into an ordered code,
    and use that code to select the magenta symbol from the right-hand table.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    sep = _find_separator_col(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]

    sequence = _extract_path_sequence(left)
    rows = _parse_table_rows(right)
    matches = [pattern for code, pattern in rows if code == sequence]
    assert len(matches) == 1, "expected exactly one matching legend row"
    return matches[0]


def _partition_sum(total: int, parts: int, rng: random.Random) -> list[int]:
    assert parts >= 1 and total >= parts
    if parts == 1:
        return [total]
    cuts = sorted(rng.sample(range(1, total), parts - 1))
    values = []
    prev = 0
    for cut in cuts + [total]:
        values.append(cut - prev)
        prev = cut
    return values


def _turn_count(path: list[tuple[int, int]]) -> int:
    turns = 0
    for i in range(1, len(path) - 1):
        dr1 = path[i][0] - path[i - 1][0]
        dc1 = path[i][1] - path[i - 1][1]
        dr2 = path[i + 1][0] - path[i][0]
        dc2 = path[i + 1][1] - path[i][1]
        if (dr1, dc1) != (dr2, dc2):
            turns += 1
    return turns


def _candidate_score(cell: tuple[int, int], occupied: set[tuple[int, int]], h: int, w: int) -> int:
    temp = occupied | {cell}
    score = 0
    for rr, cc in _neighbors4(cell[0], cell[1], h, w):
        if (rr, cc) in temp:
            continue
        occupied_neighbors = sum((nr, nc) in temp for nr, nc in _neighbors4(rr, cc, h, w))
        if occupied_neighbors == 1:
            score += 1
    return score


def _extend_induced_path(
    path: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    target_len: int,
    h: int,
    w: int,
    rng: random.Random,
) -> bool:
    if len(path) == target_len:
        return True
    r, c = path[-1]
    options = []
    for rr, cc in _neighbors4(r, c, h, w):
        if (rr, cc) in occupied:
            continue
        occupied_neighbors = sum((nr, nc) in occupied for nr, nc in _neighbors4(rr, cc, h, w))
        if occupied_neighbors != 1:
            continue
        options.append((rr, cc))
    rng.shuffle(options)
    options.sort(key=lambda cell: _candidate_score(cell, occupied, h, w), reverse=True)
    for cell in options:
        path.append(cell)
        occupied.add(cell)
        if _extend_induced_path(path, occupied, target_len, h, w, rng):
            return True
        occupied.remove(cell)
        path.pop()
    return False


def _make_induced_path(
    h: int,
    w: int,
    total_len: int,
    rng: random.Random,
    min_turns: int,
) -> list[tuple[int, int]] | None:
    for _ in range(PATH_RETRIES):
        start = (rng.randrange(h), rng.randrange(w))
        path = [start]
        occupied = {start}
        if not _extend_induced_path(path, occupied, total_len, h, w, rng):
            continue
        if _turn_count(path) < min_turns:
            continue
        return path
    return None


def _random_pattern(rng: random.Random, used: set[tuple[str, ...]]) -> np.ndarray | None:
    choices = list(range(len(PATTERN_LIBRARY)))
    rng.shuffle(choices)
    for idx in choices:
        base = PATTERN_LIBRARY[idx]
        variants = []
        for flip in (False, True):
            mask = np.fliplr(base) if flip else base
            for k in range(4):
                variants.append(_trim_mask(np.rot90(mask, k)))
        rng.shuffle(variants)
        for variant in variants:
            key = _mask_key(variant)
            if key in used:
                continue
            used.add(key)
            return variant
    return None


def _random_sequence(length: int, rng: random.Random) -> tuple[int, ...]:
    return tuple(rng.sample(PATH_COLORS, length))


def _make_row_band(
    sequence: tuple[int, ...],
    pattern: np.ndarray,
    block_size: int,
    left_pad: int,
    pattern_gap: int,
) -> np.ndarray:
    seq_width = len(sequence) * block_size + (len(sequence) - 1)
    height = max(block_size, pattern.shape[0])
    width = left_pad + seq_width + pattern_gap + pattern.shape[1]
    band = np.zeros((height, width), dtype=int)

    col = left_pad
    for idx, color in enumerate(sequence):
        band[:block_size, col : col + block_size] = int(color)
        col += block_size
        if idx != len(sequence) - 1:
            col += 1

    pattern_col = col + pattern_gap
    band[: pattern.shape[0], pattern_col : pattern_col + pattern.shape[1]][pattern] = MAGENTA
    return band


def _build_table(
    rows: list[tuple[tuple[int, ...], np.ndarray]],
    rng: random.Random,
) -> np.ndarray:
    block_size = rng.choice([1, 2])
    bands = []
    max_w = 0
    for sequence, pattern in rows:
        band = _make_row_band(
            sequence,
            pattern,
            block_size=block_size,
            left_pad=rng.randint(0, 2),
            pattern_gap=rng.randint(1, 2),
        )
        bands.append(band)
        max_w = max(max_w, band.shape[1])

    gap_sizes = [rng.randint(1, 2) for _ in range(max(0, len(bands) - 1))]
    height = sum(band.shape[0] for band in bands) + sum(gap_sizes)
    table = np.zeros((height, max_w), dtype=int)

    r = 0
    for idx, band in enumerate(bands):
        table[r : r + band.shape[0], : band.shape[1]] = band
        r += band.shape[0]
        if idx < len(gap_sizes):
            r += gap_sizes[idx]
    return table


def _expected_output(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=int)
    out[mask] = MAGENTA
    return out


def _gen_attempt(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    seq_len = rng.randint(2, 4)
    correct_sequence = _random_sequence(seq_len, rng)

    row_count = rng.randint(3, 5)
    sequences = {correct_sequence}
    ordered_sequences = [correct_sequence]
    if row_count >= 2:
        reversed_sequence = tuple(reversed(correct_sequence))
        if reversed_sequence != correct_sequence:
            sequences.add(reversed_sequence)
            ordered_sequences.append(reversed_sequence)

    while len(ordered_sequences) < row_count:
        distractor = _random_sequence(rng.randint(2, 4), rng)
        if distractor in sequences:
            continue
        sequences.add(distractor)
        ordered_sequences.append(distractor)

    used_patterns: set[tuple[str, ...]] = set()
    rows: list[tuple[tuple[int, ...], np.ndarray]] = []
    expected = None
    for sequence in ordered_sequences:
        pattern = _random_pattern(rng, used_patterns)
        if pattern is None:
            return None
        rows.append((sequence, pattern))
        if sequence == correct_sequence:
            expected = _expected_output(pattern)
    assert expected is not None
    rng.shuffle(rows)

    table = _build_table(rows, rng)
    table_h, table_w = table.shape

    min_h = max(12, table_h + 1)
    max_h = min(26, min_h + 6)
    if min_h > max_h:
        return None
    height = rng.randint(min_h, max_h)

    left_w = rng.randint(10, 18)
    path_h = height - 2
    path_w = left_w - 2

    internal_len_max = min(12, path_h + path_w - 4, (path_h * path_w) // 3)
    internal_len_min = max(seq_len + 1, 4)
    if internal_len_max < internal_len_min:
        return None
    internal_len = rng.randint(internal_len_min, internal_len_max)
    segment_lengths = _partition_sum(internal_len, seq_len, rng)

    total_path_len = internal_len + 2
    min_turns = 1 if total_path_len >= 7 and rng.random() < 0.8 else 0
    local_path = _make_induced_path(path_h, path_w, total_path_len, rng, min_turns)
    if local_path is None:
        return None

    right_margin = rng.randint(1, 4)
    right_w = table_w + right_margin
    width = left_w + 1 + right_w
    grid = np.zeros((height, width), dtype=int)
    grid[:, left_w] = GREY

    path = [(r + 1, c + 1) for r, c in local_path]
    grid[path[0]] = GREEN
    grid[path[-1]] = MAROON
    index = 1
    for color, run_len in zip(correct_sequence, segment_lengths):
        for _ in range(run_len):
            grid[path[index]] = int(color)
            index += 1

    table_top = rng.randint(0, height - table_h)
    table_left = left_w + 1 + rng.randint(0, right_w - table_w)
    grid[table_top : table_top + table_h, table_left : table_left + table_w] = table

    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        result = _gen_attempt(rng)
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid input")
