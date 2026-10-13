from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLUE = 1
CYAN = 8
BG = 0
CONTENT_COLORS = (2, 3, 4, 5, 6, 7, 9)

MIN_LENGTH = 3
MAX_LENGTH = 5
ROW_PAIR_RANGE = (2, 3)
COL_PAIR_RANGE = (1, 2)
INTERVAL_GAP = 1
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values = Counter(map(int, grid.ravel()))
    return int(values.most_common(1)[0][0])


def _sample_disjoint_intervals(total: int, count: int, size: int, gap: int) -> list[tuple[int, int]] | None:
    if count == 0:
        return []
    starts = list(range(total - size + 1))
    random.shuffle(starts)
    intervals: list[tuple[int, int]] = []
    for start in starts:
        end = start + size - 1
        if all(end + gap < other_start or other_end + gap < start for other_start, other_end in intervals):
            intervals.append((start, end))
            if len(intervals) == count:
                intervals.sort()
                return intervals
    return None


def _sample_sequence(length: int, used: set[tuple[int, ...]]) -> np.ndarray | None:
    for _ in range(200):
        seq = np.array([random.choice(CONTENT_COLORS) for _ in range(length)], dtype=int)
        tpl = tuple(map(int, seq))
        if len(set(tpl)) < 2:
            continue
        if tpl == tpl[::-1]:
            continue
        if tpl in used or tpl[::-1] in used:
            continue
        used.add(tpl)
        return seq
    return None


def _collect_pairs(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    pairs: list[dict] = []

    for r in range(g.shape[0]):
        blue_cols = np.flatnonzero(g[r] == BLUE)
        cyan_cols = np.flatnonzero(g[r] == CYAN)
        if len(blue_cols) == 1 and len(cyan_cols) == 1:
            cb = int(blue_cols[0])
            cc = int(cyan_cols[0])
            lo, hi = sorted((cb, cc))
            raw = g[r, lo + 1 : hi].copy()
            assert raw.size > 0, "row pair must enclose at least one cell"
            blue_first = cb < cc
            pairs.append(
                {
                    "axis": "row",
                    "blue": (int(r), cb),
                    "blue_first": blue_first,
                    "raw": raw,
                    "line": int(r),
                    "bounds": (lo, hi),
                }
            )

    for c in range(g.shape[1]):
        blue_rows = np.flatnonzero(g[:, c] == BLUE)
        cyan_rows = np.flatnonzero(g[:, c] == CYAN)
        if len(blue_rows) == 1 and len(cyan_rows) == 1:
            rb = int(blue_rows[0])
            rc = int(cyan_rows[0])
            lo, hi = sorted((rb, rc))
            raw = g[lo + 1 : hi, c].copy()
            assert raw.size > 0, "column pair must enclose at least one cell"
            blue_first = rb < rc
            pairs.append(
                {
                    "axis": "col",
                    "blue": (rb, int(c)),
                    "blue_first": blue_first,
                    "raw": raw,
                    "line": int(c),
                    "bounds": (lo, hi),
                }
            )

    assert pairs, "expected at least one valid marker pair"
    lengths = {int(pair["raw"].size) for pair in pairs}
    assert len(lengths) == 1, "all marker pairs must enclose the same number of cells"
    return pairs


def _pair_output_row(pair: dict, reverse_on_cyan: bool = True) -> np.ndarray:
    raw = np.asarray(pair["raw"], dtype=int)
    if reverse_on_cyan and not bool(pair["blue_first"]):
        return raw[::-1].copy()
    return raw.copy()


def _assemble_output(pairs: list[dict], *, reverse_on_cyan: bool = True, sort_mode: str = "blue") -> np.ndarray:
    if sort_mode == "blue":
        ordered = sorted(pairs, key=lambda pair: (int(pair["blue"][0]), int(pair["blue"][1])))
    elif sort_mode == "row_then_col":
        rows = sorted((pair for pair in pairs if pair["axis"] == "row"), key=lambda pair: (int(pair["line"]), int(pair["bounds"][0])))
        cols = sorted((pair for pair in pairs if pair["axis"] == "col"), key=lambda pair: (int(pair["line"]), int(pair["bounds"][0])))
        ordered = rows + cols
    else:
        raise ValueError(f"unknown sort mode: {sort_mode}")

    rows = [_pair_output_row(pair, reverse_on_cyan=reverse_on_cyan) for pair in ordered]
    return np.vstack(rows).astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Harvest the row and column segments bracketed by blue-cyan marker pairs and stack them into one output.

    Input structure:
    - A background grid contains ordinary content pixels plus blue and cyan control pixels.
    - Some rows contain exactly one blue-cyan pair, and some columns contain exactly one blue-cyan pair.
    - Every valid pair encloses the same number of interior cells, while stray single blue or cyan pixels may appear as distractors.

    Rule:
    - For every valid row pair, extract the cells strictly between the two markers.
    - For every valid column pair, extract the cells strictly between the two markers and read that column segment as a row.
    - If blue comes first on its line (left for rows, top for columns), keep the extracted order; if cyan comes first, reverse the segment before writing it.
    - Sort all extracted rows by the reading order of the blue marker pixels and stack them to form the output.

    Key insight:
    - The marker pair acts like brackets: it both defines the exact harvest span and encodes whether that harvested segment must be flipped before stacking.
    """
    pairs = _collect_pairs(np.asarray(grid, dtype=int))
    return _assemble_output(pairs, reverse_on_cyan=True, sort_mode="blue")


def _free_rows(height: int, blocked: set[int]) -> list[int]:
    return [r for r in range(height) if r not in blocked]


def _free_cols(width: int, blocked: set[int]) -> list[int]:
    return [c for c in range(width) if c not in blocked]


def _paint_extra_same_row(grid: np.ndarray, row: int, interval: tuple[int, int], occupied: set[tuple[int, int]]) -> int:
    lo, hi = interval
    candidates = [(row, c) for c in range(grid.shape[1]) if c < lo or c > hi]
    random.shuffle(candidates)
    painted = 0
    for r, c in candidates:
        if (r, c) in occupied:
            continue
        grid[r, c] = random.choice(CONTENT_COLORS)
        occupied.add((r, c))
        painted += 1
        if painted == random.randint(1, 2):
            break
    return painted


def _paint_extra_same_col(grid: np.ndarray, col: int, interval: tuple[int, int], occupied: set[tuple[int, int]]) -> int:
    lo, hi = interval
    candidates = [(r, col) for r in range(grid.shape[0]) if r < lo or r > hi]
    random.shuffle(candidates)
    painted = 0
    for r, c in candidates:
        if (r, c) in occupied:
            continue
        grid[r, c] = random.choice(CONTENT_COLORS)
        occupied.add((r, c))
        painted += 1
        if painted == random.randint(1, 2):
            break
    return painted


def _paint_general_clutter(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    forbidden_marker_rows: set[int],
    forbidden_marker_cols: set[int],
) -> int:
    target = random.randint(4, 8)
    painted = 0
    candidates = [(r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1])]
    random.shuffle(candidates)
    for r, c in candidates:
        if painted >= target:
            break
        if (r, c) in occupied:
            continue
        if r in forbidden_marker_rows and c in forbidden_marker_cols:
            continue
        if random.random() < 0.35:
            dr, dc = random.choice(((0, 1), (1, 0), (0, -1), (-1, 0)))
            cells = [(r, c), (r + dr, c + dc)]
        else:
            cells = [(r, c)]
        if any(not (0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]) or (rr, cc) in occupied for rr, cc in cells):
            continue
        color = random.choice(CONTENT_COLORS)
        for rr, cc in cells:
            grid[rr, cc] = color
            occupied.add((rr, cc))
            painted += 1
            if painted >= target:
                break
    return painted


def _place_distractors(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    row_pair_rows: set[int],
    col_pair_cols: set[int],
) -> list[tuple[int, int, int]] | None:
    row_blue = np.sum(grid == BLUE, axis=1).astype(int)
    row_cyan = np.sum(grid == CYAN, axis=1).astype(int)
    col_blue = np.sum(grid == BLUE, axis=0).astype(int)
    col_cyan = np.sum(grid == CYAN, axis=0).astype(int)

    count = random.randint(2, 5)
    colors = [BLUE, CYAN] + [random.choice((BLUE, CYAN)) for _ in range(count - 2)]
    random.shuffle(colors)

    placed: list[tuple[int, int, int]] = []
    for color in colors:
        success = False
        candidates = [(r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1])]
        random.shuffle(candidates)
        for r, c in candidates:
            if (r, c) in occupied:
                continue
            if r in row_pair_rows or c in col_pair_cols:
                continue

            next_row_blue = int(row_blue[r] + (color == BLUE))
            next_row_cyan = int(row_cyan[r] + (color == CYAN))
            next_col_blue = int(col_blue[c] + (color == BLUE))
            next_col_cyan = int(col_cyan[c] + (color == CYAN))
            if (next_row_blue, next_row_cyan) == (1, 1):
                continue
            if (next_col_blue, next_col_cyan) == (1, 1):
                continue

            grid[r, c] = color
            occupied.add((r, c))
            row_blue[r] = next_row_blue
            row_cyan[r] = next_row_cyan
            col_blue[c] = next_col_blue
            col_cyan[c] = next_col_cyan
            placed.append((r, c, color))
            success = True
            break
        if not success:
            return None
    return placed


def _validate_instance(grid: np.ndarray, expected_pairs: list[dict]) -> bool:
    try:
        observed_pairs = _collect_pairs(grid)
    except AssertionError:
        return False

    if len(observed_pairs) != len(expected_pairs):
        return False

    expected_signature = sorted(
        (
            pair["axis"],
            tuple(map(int, pair["blue"])),
            bool(pair["blue_first"]),
            tuple(map(int, pair["seq"])),
        )
        for pair in expected_pairs
    )
    observed_signature = sorted(
        (
            pair["axis"],
            tuple(map(int, pair["blue"])),
            bool(pair["blue_first"]),
            tuple(map(int, _pair_output_row(pair))),
        )
        for pair in observed_pairs
    )
    if observed_signature != expected_signature:
        return False

    if sum(pair["axis"] == "row" for pair in expected_pairs) < 2:
        return False
    if sum(pair["axis"] == "col" for pair in expected_pairs) < 1:
        return False
    if not any(pair["blue_first"] for pair in expected_pairs):
        return False
    if not any(not pair["blue_first"] for pair in expected_pairs):
        return False

    expected_out = np.vstack(
        [
            np.asarray(pair["seq"], dtype=int)
            for pair in sorted(expected_pairs, key=lambda pair: (int(pair["blue"][0]), int(pair["blue"][1])))
        ]
    )
    actual_out = main(grid)
    if not np.array_equal(actual_out, expected_out):
        return False

    no_reverse = _assemble_output(observed_pairs, reverse_on_cyan=False, sort_mode="blue")
    if np.array_equal(no_reverse, actual_out):
        return False

    grouped = _assemble_output(observed_pairs, reverse_on_cyan=True, sort_mode="row_then_col")
    if np.array_equal(grouped, actual_out):
        return False

    harvested = np.zeros_like(grid, dtype=bool)
    for pair in observed_pairs:
        if pair["axis"] == "row":
            r = int(pair["line"])
            lo, hi = pair["bounds"]
            harvested[r, lo + 1 : hi] = True
        else:
            c = int(pair["line"])
            lo, hi = pair["bounds"]
            harvested[lo + 1 : hi, c] = True
    non_marker = (grid != BG) & (grid != BLUE) & (grid != CYAN)
    ignored = non_marker & ~harvested
    if int(np.sum(ignored)) == 0:
        return False

    output_rows = [tuple(map(int, row)) for row in actual_out]
    if len(output_rows) != len(set(output_rows)):
        return False

    return True


def _attempt() -> np.ndarray | None:
    row_pairs = random.randint(*ROW_PAIR_RANGE)
    col_pairs = random.randint(*COL_PAIR_RANGE)
    length = random.randint(MIN_LENGTH, MAX_LENGTH)
    span = length + 2

    min_h = col_pairs * span + max(0, col_pairs - 1) * INTERVAL_GAP + row_pairs + 5
    min_w = row_pairs * span + max(0, row_pairs - 1) * INTERVAL_GAP + col_pairs + 6
    if min_h > 30 or min_w > 30:
        return None

    height = random.randint(min_h, min(30, min_h + 5))
    width = random.randint(min_w, min(30, min_w + 5))

    col_row_intervals = _sample_disjoint_intervals(height, col_pairs, span, INTERVAL_GAP)
    row_col_intervals = _sample_disjoint_intervals(width, row_pairs, span, INTERVAL_GAP)
    if col_row_intervals is None or row_col_intervals is None:
        return None

    col_orientations = [False] + [random.choice((True, False)) for _ in range(col_pairs - 1)]
    random.shuffle(col_orientations)
    row_orientations = [True] + [random.choice((True, False)) for _ in range(row_pairs - 1)]
    random.shuffle(row_orientations)

    reversed_col_index = next(i for i, flag in enumerate(col_orientations) if not flag)
    target_blue_row = col_row_intervals[reversed_col_index][1]

    blocked_rows = {r for start, end in col_row_intervals for r in range(start, end + 1)}
    free_rows = _free_rows(height, blocked_rows)
    above = [r for r in free_rows if r < target_blue_row]
    below = [r for r in free_rows if r > target_blue_row]
    if not above or not below:
        return None

    chosen_rows = [random.choice(above), random.choice(below)]
    remaining_rows = [r for r in free_rows if r not in chosen_rows]
    if len(remaining_rows) < row_pairs - 2:
        return None
    chosen_rows += random.sample(remaining_rows, row_pairs - 2)
    random.shuffle(chosen_rows)

    blocked_cols = {c for start, end in row_col_intervals for c in range(start, end + 1)}
    free_cols = _free_cols(width, blocked_cols)
    if len(free_cols) < col_pairs:
        return None
    chosen_cols = random.sample(free_cols, col_pairs)

    grid = np.full((height, width), BG, dtype=int)
    occupied: set[tuple[int, int]] = set()
    expected_pairs: list[dict] = []
    used_sequences: set[tuple[int, ...]] = set()

    for row, interval, blue_first in zip(chosen_rows, row_col_intervals, row_orientations):
        seq = _sample_sequence(length, used_sequences)
        if seq is None:
            return None
        c0, c1 = interval
        if blue_first:
            grid[row, c0] = BLUE
            grid[row, c1] = CYAN
            grid[row, c0 + 1 : c1] = seq
            blue = (row, c0)
        else:
            grid[row, c0] = CYAN
            grid[row, c1] = BLUE
            grid[row, c0 + 1 : c1] = seq[::-1]
            blue = (row, c1)
        for c in range(c0, c1 + 1):
            occupied.add((row, c))
        expected_pairs.append(
            {
                "axis": "row",
                "blue": blue,
                "blue_first": bool(blue_first),
                "seq": seq.copy(),
                "line": row,
                "bounds": interval,
            }
        )

    for col, interval, blue_first in zip(chosen_cols, col_row_intervals, col_orientations):
        seq = _sample_sequence(length, used_sequences)
        if seq is None:
            return None
        r0, r1 = interval
        if blue_first:
            grid[r0, col] = BLUE
            grid[r1, col] = CYAN
            grid[r0 + 1 : r1, col] = seq
            blue = (r0, col)
        else:
            grid[r0, col] = CYAN
            grid[r1, col] = BLUE
            grid[r0 + 1 : r1, col] = seq[::-1]
            blue = (r1, col)
        for r in range(r0, r1 + 1):
            occupied.add((r, col))
        expected_pairs.append(
            {
                "axis": "col",
                "blue": blue,
                "blue_first": bool(blue_first),
                "seq": seq.copy(),
                "line": col,
                "bounds": interval,
            }
        )

    extra_same_line = 0
    for pair in expected_pairs:
        if pair["axis"] == "row":
            extra_same_line += _paint_extra_same_row(grid, int(pair["line"]), pair["bounds"], occupied)
        else:
            extra_same_line += _paint_extra_same_col(grid, int(pair["line"]), pair["bounds"], occupied)
    if extra_same_line == 0:
        return None

    row_pair_rows = {int(pair["line"]) for pair in expected_pairs if pair["axis"] == "row"}
    col_pair_cols = {int(pair["line"]) for pair in expected_pairs if pair["axis"] == "col"}
    clutter = _paint_general_clutter(grid, occupied, row_pair_rows, col_pair_cols)
    if clutter < 3:
        return None

    distractors = _place_distractors(grid, occupied, row_pair_rows, col_pair_cols)
    if distractors is None or len(distractors) < 2:
        return None

    if _get_bg(grid) != BG:
        return None
    if not _validate_instance(grid, expected_pairs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bracket-harvest input")
