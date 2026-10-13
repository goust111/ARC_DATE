from __future__ import annotations

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2


GEN_MAX_ATTEMPTS = 400
HEIGHT_RANGE = (10, 16)
WIDTH_RANGE = (15, 25)
ACTIVE_ROW_RANGE = (4, 6)
EXTRA_PER_BLOCK_MAX = 2


def _fill_row_gaps(row: np.ndarray) -> tuple[np.ndarray, int]:
    filled = np.array(row, copy=True)
    count = 0
    for c in range(1, row.shape[0] - 1):
        if (
            int(row[c]) == Color.BLACK
            and int(row[c - 1]) == Color.BLUE
            and int(row[c + 1]) == Color.BLUE
        ):
            filled[c] = Color.RED
            count += 1
    return filled, count


def _analyze_row(row: np.ndarray) -> dict[str, object] | None:
    values = set(map(int, np.unique(row)))
    if not values - {Color.BLACK}:
        return None
    if not values.issubset({Color.BLACK, Color.BLUE}):
        return None

    blue_cols = np.flatnonzero(row == Color.BLUE)
    if blue_cols.size == 0:
        return None

    start = int(blue_cols[0])
    end = int(blue_cols[-1])
    segment = row[start : end + 1]

    block_lengths: list[int] = []
    gap_positions: list[int] = []
    idx = 0
    while idx < segment.shape[0]:
        if int(segment[idx]) != Color.BLUE:
            return None

        j = idx
        while j < segment.shape[0] and int(segment[j]) == Color.BLUE:
            j += 1
        block_lengths.append(int(j - idx))

        if j == segment.shape[0]:
            break

        k = j
        while k < segment.shape[0] and int(segment[k]) == Color.BLACK:
            k += 1
        if k - j != 1:
            return None
        gap_positions.append(start + j)
        idx = k

    if not gap_positions:
        return None

    filled = np.array(row, copy=True)
    for c in gap_positions:
        filled[c] = Color.RED

    return {
        "start": start,
        "end": end,
        "span": end - start + 1,
        "gap_count": len(gap_positions),
        "gap_positions": tuple(gap_positions),
        "block_lengths": tuple(block_lengths),
        "blue_total": int(np.sum(row == Color.BLUE)),
        "filled": filled,
    }


def _analyze_grid(grid: np.ndarray) -> list[dict[str, object]] | None:
    infos: list[dict[str, object]] = []
    for r in range(grid.shape[0]):
        info = _analyze_row(grid[r])
        if info is None:
            if np.any(grid[r] != Color.BLACK):
                return None
            continue
        info["row"] = r
        infos.append(info)
    return infos


def _analyze_output_row(row: np.ndarray) -> dict[str, int] | None:
    values = set(map(int, np.unique(row)))
    if not values - {Color.BLACK}:
        return None
    if not values.issubset({Color.BLACK, Color.BLUE, Color.RED}):
        return None

    red_count = int(np.sum(row == Color.RED))
    if red_count < 1:
        return None

    red_cols = np.flatnonzero(row == Color.RED)
    for c in red_cols:
        if c == 0 or c == row.shape[0] - 1:
            return None
        if int(row[c - 1]) != Color.BLUE or int(row[c + 1]) != Color.BLUE:
            return None

    return {"gap_count": red_count}


def _manual_expected_output(infos: list[dict[str, object]], width: int) -> np.ndarray:
    order = np.argsort([int(info["gap_count"]) for info in infos], kind="stable")
    rows = [np.asarray(infos[int(i)]["filled"], dtype=int) for i in order]
    return np.stack(rows, axis=0).reshape(len(rows), width)


def _distribute_extra(extra_total: int, slots: int, rng: np.random.Generator) -> np.ndarray:
    extra = np.zeros(slots, dtype=int)
    for _ in range(extra_total):
        candidates = np.flatnonzero(extra < EXTRA_PER_BLOCK_MAX)
        assert candidates.size > 0
        pick = int(rng.choice(candidates))
        extra[pick] += 1
    return extra


def _build_row(width: int, gap_count: int, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, int]]:
    min_span = 2 * gap_count + 1
    capacity = width - min_span
    slots = gap_count + 1
    extra_cap = min(capacity, EXTRA_PER_BLOCK_MAX * slots)
    extra_total = int(rng.integers(0, extra_cap + 1))
    block_lengths = 1 + _distribute_extra(extra_total, slots, rng)
    span = int(np.sum(block_lengths) + gap_count)
    start = int(rng.integers(0, width - span + 1))

    row = np.zeros(width, dtype=int)
    c = start
    for i, block_len in enumerate(block_lengths):
        row[c : c + int(block_len)] = Color.BLUE
        c += int(block_len)
        if i != len(block_lengths) - 1:
            c += 1

    return row, {
        "gap_count": gap_count,
        "blue_total": int(np.sum(block_lengths)),
        "span": span,
        "start": start,
    }


def _valid_example(grid: np.ndarray) -> bool:
    infos = _analyze_grid(grid)
    if infos is None:
        return False
    if not (ACTIVE_ROW_RANGE[0] <= len(infos) <= ACTIVE_ROW_RANGE[1]):
        return False

    gap_counts = [int(info["gap_count"]) for info in infos]
    if len(set(gap_counts)) != len(gap_counts):
        return False
    if tuple(np.argsort(gap_counts, kind="stable")) == tuple(range(len(infos))):
        return False

    blue_totals = [int(info["blue_total"]) for info in infos]
    spans = [int(info["span"]) for info in infos]
    gap_order = tuple(np.argsort(gap_counts, kind="stable"))
    if tuple(np.argsort(blue_totals, kind="stable")) == gap_order:
        return False
    if tuple(np.argsort(spans, kind="stable")) == gap_order:
        return False

    expected = _manual_expected_output(infos, grid.shape[1])
    actual = main(grid)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        return False

    out_infos = [info for info in (_analyze_output_row(row) for row in actual) if info is not None]
    if len(out_infos) != len(infos):
        return False
    out_counts = [int(info["gap_count"]) for info in out_infos]
    if out_counts != sorted(gap_counts):
        return False

    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill one-cell horizontal blue gaps in red, then reorder the non-empty rows by how many red cells were created.

    Input structure:
    - The grid has a black background with 4 to 6 active rows and other rows fully black.
    - Each active row contains one horizontal blue dashed segment made of blue blocks separated by single black gaps.
    - Different active rows have different numbers of fillable gaps.

    Rule:
    - For each row, recolor every black cell whose left and right neighbors are both blue to red.
    - Count the newly created red cells in each row.
    - Keep only the non-black rows and sort them from smallest red count to largest, preserving each row's internal pattern.
    - Remove the trailing empty rows after the reordered non-empty rows are packed together.

    Key insight:
    - The ranking key is not an existing row label; it is the number of red cells created by the gap-filling step.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    filled = np.array(g, copy=True)
    red_counts = np.zeros(g.shape[0], dtype=int)

    for r in range(g.shape[0]):
        filled[r], red_counts[r] = _fill_row_gaps(g[r])

    nonempty = np.any(filled != Color.BLACK, axis=1)
    rows = filled[nonempty]
    counts = red_counts[nonempty]
    if rows.size == 0:
        return np.zeros((0, g.shape[1]), dtype=int)

    order = np.argsort(counts, kind="stable")
    return rows[order]


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(GEN_MAX_ATTEMPTS):
        active_rows = int(rng.integers(ACTIVE_ROW_RANGE[0], ACTIVE_ROW_RANGE[1] + 1))
        height = int(rng.integers(max(HEIGHT_RANGE[0], active_rows + 2), HEIGHT_RANGE[1] + 1))
        width = int(rng.integers(WIDTH_RANGE[0], WIDTH_RANGE[1] + 1))

        max_gap = min(7, (width - 1) // 2)
        if max_gap < active_rows:
            continue

        gap_counts = list(map(int, rng.choice(np.arange(1, max_gap + 1), size=active_rows, replace=False)))
        rng.shuffle(gap_counts)

        row_indices = list(map(int, rng.choice(np.arange(height), size=active_rows, replace=False)))
        row_indices.sort()

        grid = np.zeros((height, width), dtype=int)
        for r, gap_count in zip(row_indices, gap_counts):
            row, _meta = _build_row(width, gap_count, rng)
            grid[r] = row

        if _valid_example(grid):
            return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
