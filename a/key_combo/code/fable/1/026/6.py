from __future__ import annotations

from collections import Counter

import numpy as np


BG = 0
WALL = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_H_RANGE = (10, 20)
GRID_W_RANGE = (12, 24)
WALL_LEN_RANGE = (5, 9)
UNIQUE_ROW_MAX = 3
MAX_ATTEMPTS = 300
ASSIGN_RETRIES = 200
POSITION_RETRIES = 100


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _color_counts(grid: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(grid, return_counts=True)
    return {int(v): int(n) for v, n in zip(values, counts)}


def _find_wall(grid: np.ndarray) -> tuple[int, int, int]:
    coords = np.argwhere(grid == WALL)
    assert coords.size > 0, "expected a grey wall"
    cols = sorted({int(c) for _r, c in coords})
    assert len(cols) == 1, "wall must occupy exactly one column"
    wall_col = cols[0]
    rows = sorted(int(r) for r in coords[:, 0])
    assert rows == list(range(rows[0], rows[-1] + 1)), "wall rows must be contiguous"
    return rows[0], rows[-1], wall_col


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each row's pixels against the wall, except globally unique colors that permeate through it.

    Input structure:
    - A black background contains one vertical grey wall segment occupying a single column
      across consecutive rows.
    - Colored single pixels lie only on the wall's left side; the wall's right side is empty.
    - Some colors appear multiple times in the whole grid, while other colors appear exactly once,
      with at most one globally unique-color pixel per row.

    Rule:
    - Count each non-black, non-wall color over the entire grid.
    - For every wall row, keep only the pixels on that row's left side and preserve their original
      left-to-right order.
    - Pixels whose color occurs more than once slide rightward and stack contiguously against the
      wall's left side; a color occurring exactly once skips that stack and instead moves to the
      first cell immediately right of the wall.

    Key insight:
    - The permeating pixel is selected by a global frequency test, not by row-local position such
      as being first, last, farthest, or nearest to the wall.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    wall_r0, wall_r1, wall_col = _find_wall(g)
    assert wall_col + 1 < g.shape[1], "wall needs a cell on its right"
    assert np.all(g[wall_r0 : wall_r1 + 1, wall_col + 1 :] == BG), "wall right side must be empty"

    counts = _color_counts(g)
    unique_colors = {
        color
        for color, count in counts.items()
        if color not in (BG, WALL) and count == 1
    }

    out = g.copy()
    for r in range(wall_r0, wall_r1 + 1):
        out[r, :wall_col] = BG
        out[r, wall_col + 1 :] = BG

        row_colors = [
            int(g[r, c])
            for c in range(wall_col)
            if int(g[r, c]) not in (BG, WALL)
        ]
        row_uniques = [color for color in row_colors if color in unique_colors]
        assert len(row_uniques) <= 1, "at most one globally unique color per row"

        left_stack = [color for color in row_colors if color not in unique_colors]
        start_col = wall_col - len(left_stack)
        assert start_col >= 0, "not enough room to stack pixels left of the wall"

        for offset, color in enumerate(left_stack):
            out[r, start_col + offset] = color
        if row_uniques:
            out[r, wall_col + 1] = row_uniques[0]

    return out


def _build_patterns(
    rng: np.random.Generator, row_count: int
) -> tuple[list[tuple[str, ...]], int, int]:
    patterns: list[tuple[str, ...] | None] = [None] * row_count
    order = rng.permutation(row_count)

    witness_row = int(order[0])
    plain_row = int(order[1])
    patterns[witness_row] = ("R", "U", "R")
    patterns[plain_row] = ("R", "R")

    unique_rows = _randint(rng, 1, min(UNIQUE_ROW_MAX, row_count - 1))
    extra_unique_rows = unique_rows - 1

    unique_options = [
        ("U",),
        ("U", "R"),
        ("R", "U"),
    ]
    repeated_options = [
        ("R",),
        ("R",),
        ("R", "R"),
    ]

    cursor = 2
    for _ in range(extra_unique_rows):
        idx = int(order[cursor])
        cursor += 1
        patterns[idx] = unique_options[_randint(rng, 0, len(unique_options) - 1)]

    while cursor < row_count:
        idx = int(order[cursor])
        cursor += 1
        patterns[idx] = repeated_options[_randint(rng, 0, len(repeated_options) - 1)]

    return [tuple(p) for p in patterns], witness_row, plain_row


def _assign_repeated_colors(
    rng: np.random.Generator,
    patterns: list[tuple[str, ...]],
    repeated_palette: list[int],
) -> list[list[int]] | None:
    total_slots = sum(pattern.count("R") for pattern in patterns)
    row_needs = [pattern.count("R") for pattern in patterns]

    for _ in range(ASSIGN_RETRIES):
        counts = {color: 2 for color in repeated_palette}
        for _extra in range(total_slots - 2 * len(repeated_palette)):
            chosen = int(rng.choice(repeated_palette))
            counts[chosen] += 1

        assignments: list[list[int]] = [[] for _ in patterns]
        ok = True
        for row_idx in sorted(range(len(patterns)), key=lambda i: row_needs[i], reverse=True):
            need = row_needs[row_idx]
            if need == 0:
                continue

            row_colors: list[int] = []
            for _slot in range(need):
                choices = [
                    color
                    for color in repeated_palette
                    if counts[color] > 0 and color not in row_colors
                ]
                if not choices:
                    ok = False
                    break
                weights = np.array([counts[color] for color in choices], dtype=float)
                weights /= weights.sum()
                chosen = int(rng.choice(choices, p=weights))
                row_colors.append(chosen)
                counts[chosen] -= 1
            if not ok:
                break
            rng.shuffle(row_colors)
            assignments[row_idx] = row_colors

        if ok and all(count == 0 for count in counts.values()):
            return assignments

    return None


def _sample_positions(
    rng: np.random.Generator, wall_col: int, count: int
) -> list[int] | None:
    final_stack = list(range(wall_col - count, wall_col))
    for _ in range(POSITION_RETRIES):
        positions = sorted(int(v) for v in rng.choice(wall_col, size=count, replace=False))
        if positions != final_stack:
            return positions
    return None


def _expected_output(
    shape: tuple[int, int],
    wall_r0: int,
    wall_len: int,
    wall_col: int,
    patterns: list[tuple[str, ...]],
    sequences: list[list[int]],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    out[wall_r0 : wall_r0 + wall_len, wall_col] = WALL

    for local_r, (pattern, sequence) in enumerate(zip(patterns, sequences)):
        r = wall_r0 + local_r
        left_stack = [color for token, color in zip(pattern, sequence) if token == "R"]
        uniques = [color for token, color in zip(pattern, sequence) if token == "U"]

        start_col = wall_col - len(left_stack)
        for offset, color in enumerate(left_stack):
            out[r, start_col + offset] = color
        if uniques:
            out[r, wall_col + 1] = uniques[0]

    return out


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    wall_r0: int,
    wall_len: int,
    wall_col: int,
    patterns: list[tuple[str, ...]],
    sequences: list[list[int]],
    witness_row: int,
    plain_row: int,
) -> bool:
    try:
        found_r0, found_r1, found_col = _find_wall(inp)
    except AssertionError:
        return False
    if (found_r0, found_r1, found_col) != (wall_r0, wall_r0 + wall_len - 1, wall_col):
        return False
    if not np.all(inp[wall_r0 : wall_r0 + wall_len, wall_col + 1 :] == BG):
        return False

    counts = _color_counts(inp)
    unique_colors = {color for color, count in counts.items() if color not in (BG, WALL) and count == 1}
    repeated_colors = {
        color for color, count in counts.items() if color not in (BG, WALL) and count > 1
    }
    if not unique_colors or not repeated_colors:
        return False
    if any(count < 2 for color, count in counts.items() if color in repeated_colors):
        return False

    witness_pattern = patterns[witness_row]
    witness_sequence = sequences[witness_row]
    if witness_pattern != ("R", "U", "R"):
        return False
    if witness_sequence[1] not in unique_colors:
        return False
    if witness_sequence[0] in unique_colors or witness_sequence[2] in unique_colors:
        return False

    plain_sequence = sequences[plain_row]
    if len(plain_sequence) != 2 or any(color in unique_colors for color in plain_sequence):
        return False

    for pattern, sequence in zip(patterns, sequences):
        if sum(color in unique_colors for color in sequence) != pattern.count("U"):
            return False

    return bool(np.array_equal(main(inp), expected))


def _generate_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    height = _randint(rng, *GRID_H_RANGE)
    width = _randint(rng, *GRID_W_RANGE)

    wall_len = _randint(rng, WALL_LEN_RANGE[0], min(WALL_LEN_RANGE[1], height - 2))
    wall_r0 = _randint(rng, 1, height - wall_len - 1)
    wall_col = _randint(rng, 5, width - 3)

    patterns, witness_row, plain_row = _build_patterns(rng, wall_len)
    unique_count = sum(pattern.count("U") for pattern in patterns)
    total_repeated_slots = sum(pattern.count("R") for pattern in patterns)

    min_palette = 2
    max_palette = min(5, total_repeated_slots // 2, len(PALETTE) - unique_count)
    if min_palette > max_palette:
        return None
    repeated_size = _randint(rng, min_palette, max_palette)

    palette_order = list(rng.permutation(PALETTE))
    repeated_palette = [int(color) for color in palette_order[:repeated_size]]
    unique_palette = [int(color) for color in palette_order[repeated_size : repeated_size + unique_count]]
    if len(unique_palette) != unique_count:
        return None

    repeated_assignments = _assign_repeated_colors(rng, patterns, repeated_palette)
    if repeated_assignments is None:
        return None

    unique_rows = [idx for idx, pattern in enumerate(patterns) if "U" in pattern]
    unique_by_row = {row: unique_palette[i] for i, row in enumerate(unique_rows)}

    sequences: list[list[int]] = []
    for row_idx, pattern in enumerate(patterns):
        repeated_iter = iter(repeated_assignments[row_idx])
        sequence: list[int] = []
        for token in pattern:
            if token == "R":
                sequence.append(next(repeated_iter))
            else:
                sequence.append(unique_by_row[row_idx])
        sequences.append(sequence)

    inp = np.full((height, width), BG, dtype=int)
    inp[wall_r0 : wall_r0 + wall_len, wall_col] = WALL

    for local_r, sequence in enumerate(sequences):
        positions = _sample_positions(rng, wall_col, len(sequence))
        if positions is None:
            return None
        for c, color in zip(positions, sequence):
            inp[wall_r0 + local_r, c] = color

    expected = _expected_output(
        inp.shape,
        wall_r0,
        wall_len,
        wall_col,
        patterns,
        sequences,
    )
    if not _semantic_checks(
        inp,
        expected,
        wall_r0,
        wall_len,
        wall_col,
        patterns,
        sequences,
        witness_row,
        plain_row,
    ):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _generate_case(rng)
        if case is not None:
            inp, _expected = case
            return inp
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
