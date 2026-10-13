from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
SEPARATOR = 5
REFERENCE_COLORS = (1, 2, 3, 4)
DISTRACTOR_COLORS = (6, 7, 8, 9)
REF_COL = 2
SEPARATOR_COL = 4

GRID_HEIGHT_RANGE = (10, 17)
GRID_WIDTH_RANGE = (18, 24)
NONZERO_COUNT_RANGE = (3, 8)
GENERATION_ATTEMPTS = 300


def _separator_col(grid: np.ndarray) -> int:
    cols = np.flatnonzero(np.all(grid == SEPARATOR, axis=0))
    assert len(cols) == 1, "expected exactly one full-height grey separator"
    return int(cols[0])


def _row_main_color(row: np.ndarray) -> int | None:
    colors = [int(v) for v in row if int(v) != BACKGROUND]
    if not colors:
        return None
    counts = Counter(colors)
    best = max(counts.values())
    tied = {color for color, count in counts.items() if count == best}
    for value in row:
        color = int(value)
        if color in tied:
            return color
    raise AssertionError("unreachable")


def _invert_row(row: np.ndarray, main_color: int) -> np.ndarray:
    out = np.full_like(row, int(main_color))
    out[row == int(main_color)] = BACKGROUND
    return out


def _slide_row(row: np.ndarray) -> np.ndarray:
    idx = np.flatnonzero(row != BACKGROUND)
    if len(idx) == 0:
        return row.copy()
    out = np.zeros_like(row)
    shifted = idx - int(idx[0])
    out[shifted] = row[idx]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare each row's reference pixel to the row's operating-color leader, then invert or slide that row accordingly.

    Input structure:
    - Black (0) background fills the grid, and one full-height grey (5) column splits it into a left reference zone and a right operating zone.
    - In each row, the reference zone contains one colored pixel, while the operating zone contains a sparse 1D colored pattern separated from the divider by at least one black cell.
    - The operating pattern may mix one dominant non-black color with one or more distractor colors.

    Rule:
    - Process rows independently. In the operating zone, find the non-black color with the highest count; if counts tie, choose the tied color whose first occurrence is leftmost.
    - If that row-major color matches the row's reference pixel, invert the operating zone: major-color pixels become black, and every black or distractor pixel becomes the major color.
    - Otherwise rigidly slide all non-black operating pixels left until the leftmost non-black pixel sits immediately to the right of the separator, preserving their colors and relative offsets.

    Key insight:
    - The left reference pixel is not copied or recolored; it is a per-row branch selector that chooses between two different transformations based only on the operating row's dominant color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    sep = _separator_col(g)
    ref_col = max(0, sep - 2)
    op_start = sep + 1
    out = g.copy()

    for r in range(g.shape[0]):
        ref_color = int(g[r, ref_col])
        row = g[r, op_start:]
        main_color = _row_main_color(row)
        if main_color is None:
            continue
        if ref_color == main_color:
            out[r, op_start:] = _invert_row(row, main_color)
        else:
            out[r, op_start:] = _slide_row(row)
    return out


def _sample_other_color(exclude: set[int], pool: tuple[int, ...]) -> int:
    choices = [color for color in pool if color not in exclude]
    return int(random.choice(choices))


def _local_positions(nonzero_count: int, span_len: int, require_gap: bool) -> list[int]:
    assert span_len >= nonzero_count
    if not require_gap:
        return list(range(nonzero_count))

    assert span_len >= nonzero_count + 1
    while True:
        positions = sorted(random.sample(range(span_len), nonzero_count))
        if positions[0] != 0:
            positions[positions.index(min(positions))] = 0
            positions = sorted(set(positions))
            while len(positions) < nonzero_count:
                candidate = random.randrange(span_len)
                positions = sorted(set(positions + [candidate]))
        if positions[0] == 0 and any(b - a > 1 for a, b in zip(positions, positions[1:])):
            return positions


def _place_sequence(op_width: int, sequence: list[int], require_gap: bool) -> tuple[np.ndarray, list[int]]:
    nonzero_count = len(sequence)
    min_span = nonzero_count + 1 if require_gap else nonzero_count
    max_span = min(op_width - 1, nonzero_count + 4)
    span_len = random.randint(min_span, max_span)
    positions = _local_positions(nonzero_count, span_len, require_gap)
    start = random.randint(1, op_width - span_len)

    row = np.zeros(op_width, dtype=int)
    global_positions = [start + pos for pos in positions]
    for color, pos in zip(sequence, global_positions):
        row[pos] = int(color)
    return row, global_positions


def _shuffle_tail(sequence: list[int]) -> list[int]:
    head = sequence[:1]
    tail = sequence[1:]
    random.shuffle(tail)
    return head + tail


def _build_row(op_width: int, kind: str) -> tuple[np.ndarray, dict[str, object]]:
    require_gap = kind in {"match_tie", "slide_tie", "slide_ref_present"}

    if kind == "match_unique":
        nonzero_count = random.randint(4, NONZERO_COUNT_RANGE[1])
        main_color = int(random.choice(REFERENCE_COLORS))
        main_count = random.randint(max(2, nonzero_count // 2 + 1), nonzero_count - 1)
        other_total = nonzero_count - main_count
        sequence = [main_color] * main_count
        if other_total:
            other_palette = [color for color in REFERENCE_COLORS + DISTRACTOR_COLORS if color != main_color]
            first_other = int(random.choice(other_palette))
            sequence.extend([first_other] * random.randint(1, other_total))
            while len(sequence) < nonzero_count:
                sequence.append(int(random.choice(other_palette)))
        random.shuffle(sequence)
        ref_color = main_color
        tie = False
        ref_present_nonmain = False

    elif kind == "slide_ref_present":
        nonzero_count = random.randint(4, NONZERO_COUNT_RANGE[1])
        main_color = int(random.choice(REFERENCE_COLORS))
        ref_color = _sample_other_color({main_color}, REFERENCE_COLORS)
        main_count = random.randint(max(2, nonzero_count // 2 + 1), nonzero_count - 1)
        ref_count = random.randint(1, min(main_count - 1, nonzero_count - main_count))
        extra = nonzero_count - main_count - ref_count
        sequence = [ref_color] + [main_color] * main_count + [ref_color] * (ref_count - 1)
        if extra:
            extra_color = _sample_other_color({main_color, ref_color}, DISTRACTOR_COLORS + REFERENCE_COLORS)
            sequence.extend([extra_color] * extra)
        sequence = _shuffle_tail(sequence)
        tie = False
        ref_present_nonmain = True

    elif kind == "match_tie":
        k = random.randint(2, 3)
        main_color = int(random.choice(REFERENCE_COLORS))
        second_color = _sample_other_color({main_color}, REFERENCE_COLORS)
        extra = random.randint(0, 1)
        nonzero_count = 2 * k + extra
        sequence = [main_color] + [main_color] * (k - 1) + [second_color] * k
        if extra:
            sequence.append(_sample_other_color({main_color, second_color}, DISTRACTOR_COLORS))
        sequence = _shuffle_tail(sequence)
        ref_color = main_color
        tie = True
        ref_present_nonmain = False

    elif kind == "slide_tie":
        k = random.randint(2, 3)
        main_color = int(random.choice(REFERENCE_COLORS))
        second_color = _sample_other_color({main_color}, REFERENCE_COLORS)
        extra = random.randint(0, 1)
        nonzero_count = 2 * k + extra
        sequence = [main_color] + [main_color] * (k - 1) + [second_color] * k
        if extra:
            sequence.append(_sample_other_color({main_color, second_color}, DISTRACTOR_COLORS))
        sequence = _shuffle_tail(sequence)
        ref_color = second_color
        tie = True
        ref_present_nonmain = True

    else:
        raise ValueError(f"unknown row kind: {kind}")

    row, positions = _place_sequence(op_width, sequence, require_gap=require_gap)
    observed_main = _row_main_color(row)
    assert observed_main is not None
    metadata = {
        "kind": kind,
        "ref_color": int(ref_color),
        "main_color": int(observed_main),
        "positions": positions,
        "nonzero_count": int(np.count_nonzero(row)),
        "tie": bool(tie),
        "internal_gap": any(b - a > 1 for a, b in zip(positions, positions[1:])),
        "ref_present_nonmain": bool(ref_present_nonmain),
    }
    return row, metadata


def _analyze_grid(grid: np.ndarray) -> list[dict[str, object]]:
    sep = _separator_col(grid)
    ref_col = max(0, sep - 2)
    op_start = sep + 1
    rows = []
    for r in range(grid.shape[0]):
        row = grid[r, op_start:]
        positions = np.flatnonzero(row != BACKGROUND).tolist()
        colors = [int(v) for v in row if int(v) != BACKGROUND]
        counts = Counter(colors)
        max_count = max(counts.values())
        main_color = _row_main_color(row)
        rows.append(
            {
                "ref_color": int(grid[r, ref_col]),
                "main_color": int(main_color) if main_color is not None else None,
                "branch": "match" if main_color == int(grid[r, ref_col]) else "slide",
                "positions": positions,
                "internal_gap": any(b - a > 1 for a, b in zip(positions, positions[1:])),
                "tie": sum(count == max_count for count in counts.values()) > 1,
                "nonzero_count": len(positions),
                "ref_present_nonmain": any(
                    color == int(grid[r, ref_col]) and color != main_color for color in colors
                ),
            }
        )
    return rows


def _validate_generated_grid(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert max(grid.shape) <= 30
    assert SEPARATOR_COL < grid.shape[1] - 2
    assert np.all(grid[:, SEPARATOR_COL] == SEPARATOR)
    assert np.all(grid[:, :REF_COL] == BACKGROUND)
    assert np.all(grid[:, REF_COL + 1 : SEPARATOR_COL] == BACKGROUND)
    assert set(int(v) for v in np.unique(grid[:, REF_COL])).issubset(set(REFERENCE_COLORS))
    assert set(int(v) for v in np.unique(grid[:, SEPARATOR_COL + 1 :])).issubset(
        {BACKGROUND, *REFERENCE_COLORS, *DISTRACTOR_COLORS}
    )

    analysis = _analyze_grid(grid)
    assert any(row["branch"] == "match" for row in analysis)
    assert any(row["branch"] == "slide" for row in analysis)
    match_count = sum(row["branch"] == "match" for row in analysis)
    slide_count = len(analysis) - match_count
    assert abs(match_count - slide_count) <= 2
    assert any(bool(row["tie"]) for row in analysis)
    assert any(bool(row["internal_gap"]) for row in analysis)
    assert any(bool(row["ref_present_nonmain"]) and row["branch"] == "slide" for row in analysis)
    assert len({int(row["ref_color"]) for row in analysis}) >= 2
    assert len({int(row["main_color"]) for row in analysis}) >= 2

    out = main(grid)
    sep = _separator_col(grid)
    for r, row in enumerate(analysis):
        op_in = grid[r, sep + 1 :]
        op_out = out[r, sep + 1 :]
        assert row["nonzero_count"] >= NONZERO_COUNT_RANGE[0]
        assert row["nonzero_count"] <= NONZERO_COUNT_RANGE[1]
        assert row["positions"][0] >= 1
        if row["branch"] == "match":
            expected = _invert_row(op_in, int(row["main_color"]))
        else:
            expected = _slide_row(op_in)
        assert np.array_equal(op_out, expected)
        assert not np.array_equal(op_in, op_out), "every generated row should visibly change"
    assert np.array_equal(out[:, :sep], grid[:, :sep])
    assert np.array_equal(out[:, sep], grid[:, sep])


def _build_kind_plan(height: int) -> list[str]:
    kinds = ["match_unique", "match_tie", "slide_ref_present", "slide_tie"]
    target_match = random.randint(max(2, height // 2 - 1), min(height - 2, height // 2 + 1))
    match_count = 2
    slide_count = 2
    while len(kinds) < height:
        if match_count < target_match and slide_count < height - target_match:
            kind = random.choice(["match_unique", "match_tie", "slide_ref_present", "slide_tie"])
        elif match_count < target_match:
            kind = random.choice(["match_unique", "match_tie"])
        else:
            kind = random.choice(["slide_ref_present", "slide_tie"])
        kinds.append(kind)
        if kind.startswith("match"):
            match_count += 1
        else:
            slide_count += 1
    random.shuffle(kinds)
    return kinds


def _gen_attempt() -> np.ndarray:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    op_width = width - SEPARATOR_COL - 1

    grid = np.zeros((height, width), dtype=int)
    grid[:, SEPARATOR_COL] = SEPARATOR

    for r, kind in enumerate(_build_kind_plan(height)):
        row, meta = _build_row(op_width, kind)
        grid[r, REF_COL] = int(meta["ref_color"])
        grid[r, SEPARATOR_COL + 1 :] = row

    _validate_generated_grid(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        try:
            return _gen_attempt()
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid reference-pixel branch puzzle input")
