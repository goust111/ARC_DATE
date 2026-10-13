from __future__ import annotations

import numpy as np


SAME_FAMILY_PAIRS = (
    frozenset((1, 8)),
    frozenset((4, 7)),
)
SAME_FAMILY_SET = set(SAME_FAMILY_PAIRS)
FAMILY_COLORS = {1, 4, 7, 8}
FOREGROUND_COLORS = tuple(range(1, 10))

ALL_DIFFERENT_PAIRS = tuple(
    (a, b)
    for a in FOREGROUND_COLORS
    for b in FOREGROUND_COLORS
    if a < b and frozenset((a, b)) not in SAME_FAMILY_SET
)
CROSS_FAMILY_DIFFERENT_PAIRS = tuple(
    pair for pair in ALL_DIFFERENT_PAIRS if FAMILY_COLORS.intersection(pair)
)
UNRELATED_DIFFERENT_PAIRS = tuple(
    pair for pair in ALL_DIFFERENT_PAIRS if not FAMILY_COLORS.intersection(pair)
)

ROW_COUNT_RANGE = (7, 12)
COL_COUNT_RANGE = (12, 18)
EMPTY_ROW_RANGE = (0, 2)
MAX_GENERATION_ATTEMPTS = 400

_RNG = np.random.default_rng()
_NEXT_SAME_MAJORITY = False


def _is_same_family(left_color: int, right_color: int) -> bool:
    return frozenset((int(left_color), int(right_color))) in SAME_FAMILY_SET


def _ordered_pair(pair: tuple[int, int]) -> tuple[int, int]:
    if int(_RNG.integers(0, 2)) == 0:
        return int(pair[0]), int(pair[1])
    return int(pair[1]), int(pair[0])


def _collect_row_info(grid: np.ndarray) -> list[dict[str, int | bool]]:
    info: list[dict[str, int | bool]] = []
    for r in range(grid.shape[0]):
        cols = np.flatnonzero(grid[r] != 0)
        assert len(cols) in (0, 2), "each row must be empty or contain exactly two foreground pixels"
        if len(cols) == 0:
            continue

        c0, c1 = map(int, cols)
        assert c1 - c0 >= 3, "the two pixels in each nonempty row must have at least two black cells between them"
        left_color = int(grid[r, c0])
        right_color = int(grid[r, c1])
        assert left_color != right_color, "row pixels must have distinct colors"
        info.append(
            {
                "row": int(r),
                "left_col": c0,
                "right_col": c1,
                "left_color": left_color,
                "right_color": right_color,
                "same_family": _is_same_family(left_color, right_color),
            }
        )
    return info


def _slide_rows(grid: np.ndarray, *, move_same_family: bool | None) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for row in _collect_row_info(out):
        should_move = move_same_family is None or bool(row["same_family"]) == move_same_family
        if not should_move:
            continue

        r = int(row["row"])
        c0 = int(row["left_col"])
        c1 = int(row["right_col"])
        dst = c0 + 1
        assert dst < grid.shape[1]
        assert c1 > dst
        assert out[r, dst] == 0, "the destination cell must be empty before compaction"
        out[r, dst] = out[r, c1]
        out[r, c1] = 0
    return out


def _majority_is_same_family(grid: np.ndarray) -> bool:
    rows = _collect_row_info(grid)
    same_count = sum(bool(row["same_family"]) for row in rows)
    diff_count = len(rows) - same_count
    assert same_count >= 2 and diff_count >= 2, "both pair classes must appear at least twice"
    assert same_count != diff_count, "the generator must avoid ties"
    return same_count > diff_count


def main(grid: np.ndarray) -> np.ndarray:
    """Compact only the majority class of two-pixel rows after counting same-family versus different-family pairs.

    Input structure
    - The grid has a black background.
    - Each nonempty row contains exactly two distinct foreground pixels with at least two black cells between them.
    - A row is a same-family pair iff its two colors are blue/teal or yellow/orange; all other two-color rows are different-family pairs.

    Rule
    - Count same-family rows and different-family rows across the whole grid.
    - The majority class is the only class that compacts horizontally.
    - In each compacted row, keep the left pixel fixed and slide the right pixel left until it becomes immediately adjacent, clearing its original cell.

    Key insight
    - The color pairing alone is not enough: the global majority count decides which kind of row behaves like the sliding rows and which kind becomes the ignored distractor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"
    assert np.all(g >= 0), "colors must be non-negative"

    move_same_family = _majority_is_same_family(g)
    return _slide_rows(g, move_same_family=move_same_family)


def _count_options(target_same_majority: bool) -> list[tuple[int, int, int]]:
    options: list[tuple[int, int, int]] = []
    for rows in range(ROW_COUNT_RANGE[0], ROW_COUNT_RANGE[1] + 1):
        for empty_rows in range(EMPTY_ROW_RANGE[0], EMPTY_ROW_RANGE[1] + 1):
            nonempty_rows = rows - empty_rows
            for same_count in range(2, nonempty_rows - 1):
                diff_count = nonempty_rows - same_count
                if diff_count < 2 or same_count == diff_count:
                    continue
                if (same_count > diff_count) == target_same_majority:
                    options.append((rows, same_count, diff_count))
    return options


def _sample_separated_columns(cols: int) -> tuple[int, int]:
    left_col = int(_RNG.integers(0, cols - 3))
    right_col = int(_RNG.integers(left_col + 3, cols))
    return left_col, right_col


def _sample_expand_col(cols: int, left_col: int) -> int:
    return int(_RNG.integers(left_col + 3, cols))


def _same_family_pairs(count: int) -> list[tuple[int, int]]:
    pairs = [(1, 8), (4, 7)]
    while len(pairs) < count:
        pairs.append(pairs[int(_RNG.integers(0, 2))])
    order = _RNG.permutation(len(pairs))[:count]
    return [_ordered_pair(pairs[int(i)]) for i in order]


def _different_family_pairs(count: int) -> list[tuple[int, int]]:
    chosen: list[tuple[int, int]] = []
    if count >= 1:
        chosen.append(CROSS_FAMILY_DIFFERENT_PAIRS[int(_RNG.integers(0, len(CROSS_FAMILY_DIFFERENT_PAIRS)))])
    if count >= 2:
        chosen.append(UNRELATED_DIFFERENT_PAIRS[int(_RNG.integers(0, len(UNRELATED_DIFFERENT_PAIRS)))])
    while len(chosen) < count:
        chosen.append(ALL_DIFFERENT_PAIRS[int(_RNG.integers(0, len(ALL_DIFFERENT_PAIRS)))])
    order = _RNG.permutation(len(chosen))[:count]
    return [_ordered_pair(chosen[int(i)]) for i in order]


def _build_pair_lists(same_count: int, diff_count: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    same_pairs = _same_family_pairs(same_count)
    diff_pairs = _different_family_pairs(diff_count)
    return same_pairs, diff_pairs


def _render_output_grid(
    rows: int,
    cols: int,
    same_pairs: list[tuple[int, int]],
    diff_pairs: list[tuple[int, int]],
    *,
    same_majority: bool,
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    out = np.zeros((rows, cols), dtype=int)
    row_types = ["same"] * len(same_pairs) + ["diff"] * len(diff_pairs)
    row_types += ["empty"] * (rows - len(row_types))
    row_types = [row_types[int(i)] for i in _RNG.permutation(len(row_types))]

    moving_rows: list[tuple[int, int, int]] = []
    same_idx = 0
    diff_idx = 0
    for r, row_type in enumerate(row_types):
        if row_type == "empty":
            continue

        if row_type == "same":
            left_color, right_color = same_pairs[same_idx]
            same_idx += 1
            compact = same_majority
        else:
            left_color, right_color = diff_pairs[diff_idx]
            diff_idx += 1
            compact = not same_majority

        if compact:
            left_col = int(_RNG.integers(0, cols - 3))
            out[r, left_col] = left_color
            out[r, left_col + 1] = right_color
            moving_rows.append((r, left_col, right_color))
        else:
            left_col, right_col = _sample_separated_columns(cols)
            out[r, left_col] = left_color
            out[r, right_col] = right_color

    return out, moving_rows


def _expand_majority_rows(output_grid: np.ndarray, moving_rows: list[tuple[int, int, int]]) -> np.ndarray:
    inp = np.array(output_grid, copy=True)
    for r, left_col, right_color in moving_rows:
        expanded_col = _sample_expand_col(inp.shape[1], left_col)
        inp[r, left_col + 1] = 0
        inp[r, expanded_col] = right_color
    return inp


def _semantic_checks(grid: np.ndarray, expected: np.ndarray) -> bool:
    row_info = _collect_row_info(grid)
    same_count = sum(bool(row["same_family"]) for row in row_info)
    diff_count = len(row_info) - same_count
    if same_count < 2 or diff_count < 2 or same_count == diff_count:
        return False

    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    same_only = _slide_rows(grid, move_same_family=True)
    diff_only = _slide_rows(grid, move_same_family=False)
    all_rows = _slide_rows(grid, move_same_family=None)
    if np.array_equal(actual, grid) or np.array_equal(actual, all_rows):
        return False

    if same_count > diff_count:
        if not np.array_equal(actual, same_only):
            return False
        if np.array_equal(actual, diff_only):
            return False
    else:
        if not np.array_equal(actual, diff_only):
            return False
        if np.array_equal(actual, same_only):
            return False

    same_pairs_present = {
        frozenset((int(row["left_color"]), int(row["right_color"])))
        for row in row_info
        if bool(row["same_family"])
    }
    if same_pairs_present != SAME_FAMILY_SET:
        return False

    different_pairs = [
        frozenset((int(row["left_color"]), int(row["right_color"])))
        for row in row_info
        if not bool(row["same_family"])
    ]
    has_cross_family_counterexample = any(pair & FAMILY_COLORS for pair in different_pairs)
    has_unrelated_counterexample = any(not (pair & FAMILY_COLORS) for pair in different_pairs)
    return has_cross_family_counterexample and has_unrelated_counterexample


def _gen_attempt(target_same_majority: bool) -> np.ndarray | None:
    count_options = _count_options(target_same_majority)
    rows, same_count, diff_count = count_options[int(_RNG.integers(0, len(count_options)))]
    cols = int(_RNG.integers(COL_COUNT_RANGE[0], COL_COUNT_RANGE[1] + 1))

    same_pairs, diff_pairs = _build_pair_lists(same_count, diff_count)
    expected, moving_rows = _render_output_grid(
        rows,
        cols,
        same_pairs,
        diff_pairs,
        same_majority=target_same_majority,
    )
    grid = _expand_majority_rows(expected, moving_rows)

    if not _semantic_checks(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _NEXT_SAME_MAJORITY

    _NEXT_SAME_MAJORITY = not _NEXT_SAME_MAJORITY
    target_same_majority = _NEXT_SAME_MAJORITY
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(target_same_majority)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for conditional majority sliding")
