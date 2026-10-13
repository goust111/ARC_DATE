from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
CONTROL_MARKER = 4
AXIS_COLOR = 8
MARKER_PALETTE = (1, 2, 3, 6, 7, 9)
MAX_ATTEMPTS = 500


def _quadrant_of(r: int, c: int, axis_r: int, axis_c: int) -> int | None:
    if r == axis_r or c == axis_c:
        return None
    if r < axis_r and c < axis_c:
        return 0
    if r < axis_r and c > axis_c:
        return 1
    if r > axis_r and c < axis_c:
        return 2
    return 3


def _quadrant_cells(size: int, axis_r: int, axis_c: int, quadrant: int) -> list[tuple[int, int]]:
    rows = range(0, axis_r) if quadrant < 2 else range(axis_r + 1, size)
    cols = range(0, axis_c) if quadrant % 2 == 0 else range(axis_c + 1, size)
    return [(r, c) for r in rows for c in cols]


def _decode_shifts(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    assert grid.ndim == 2 and min(grid.shape) >= 3, "expected a 2D grid with control borders"
    top = grid[0, 1:]
    left = grid[1:, 0]
    assert int(grid[0, 0]) == BG, "top-left control corner must be background"
    assert np.all(top[(top != CONTROL_MARKER)] == BG), "top control row must be background plus one marker"
    assert np.all(left[(left != CONTROL_MARKER)] == BG), "left control column must be background plus one marker"

    row_hits = np.flatnonzero(left == CONTROL_MARKER)
    col_hits = np.flatnonzero(top == CONTROL_MARKER)
    assert row_hits.size == 1 and col_hits.size == 1, "expected exactly one control marker per band"
    return int(row_hits[0]), int(col_hits[0]), np.asarray(grid[1:, 1:], dtype=int)


def _find_cross_axes(payload: np.ndarray) -> tuple[int, int]:
    rows = np.flatnonzero(np.all(payload == AXIS_COLOR, axis=1))
    cols = np.flatnonzero(np.all(payload == AXIS_COLOR, axis=0))
    assert rows.size == 1 and cols.size == 1, "expected exactly one full teal row and column"
    return int(rows[0]), int(cols[0])


def _mirror_fill(payload: np.ndarray, axis_r: int, axis_c: int) -> np.ndarray:
    out = np.asarray(payload, dtype=int).copy()
    h, w = out.shape
    markers = np.argwhere((out != BG) & (out != AXIS_COLOR))

    for r_raw, c_raw in markers:
        r = int(r_raw)
        c = int(c_raw)
        color = int(out[r, c])
        for rr in (r, 2 * axis_r - r):
            for cc in (c, 2 * axis_c - c):
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if int(out[rr, cc]) in (BG, AXIS_COLOR):
                    out[rr, cc] = color
    return out


def _assign_colors(count: int) -> list[int]:
    max_distinct = min(4, count)
    min_distinct = max(1, (count + 2) // 3)
    distinct = random.randint(min_distinct, max_distinct)
    palette = random.sample(MARKER_PALETTE, distinct)

    counts = [1] * distinct
    remaining = count - distinct
    while remaining > 0:
        expandable = [i for i, used in enumerate(counts) if used < 3]
        pick = random.choice(expandable)
        counts[pick] += 1
        remaining -= 1

    colors: list[int] = []
    for color, used in zip(palette, counts):
        colors.extend([color] * used)
    random.shuffle(colors)
    return colors


def _edgeish_indices(size: int) -> list[int]:
    return sorted({idx for idx in (1, 2, size - 3, size - 2) if 1 <= idx <= size - 2})


def _has_clipped_reflection(cells: list[tuple[int, int]], axis_r: int, axis_c: int, size: int) -> bool:
    for r, c in cells:
        reflected = (
            (2 * axis_r - r, c),
            (r, 2 * axis_c - c),
            (2 * axis_r - r, 2 * axis_c - c),
        )
        if any(not (0 <= rr < size and 0 <= cc < size) for rr, cc in reflected):
            return True
    return False


def _roll_then_fill(payload: np.ndarray, row_shift: int, col_shift: int) -> np.ndarray:
    shifted = np.roll(payload, shift=(row_shift, col_shift), axis=(0, 1))
    axis_r, axis_c = _find_cross_axes(shifted)
    return _mirror_fill(shifted, axis_r, axis_c)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop away the control borders, cyclically move the payload, then complete the sparse pattern by reflecting it across the shifted teal cross.

    Input structure
    - The top row and left column are black control bands with one yellow marker each, and their shared corner is black.
    - The lower-right payload contains one full teal row and one full teal column forming a cross.
    - All other non-black payload pixels are sparse colored markers placed off the teal axes.

    Rule
    - Read the yellow marker offsets from the left control column and the top control row.
    - Remove the control row and column, then cyclically roll the payload down and right by those offsets.
    - In the shifted payload, use the full teal row and column as mirror axes and copy every sparse marker to its row-reflection, column-reflection, and double-reflection positions, clipping out-of-bounds copies and never overwriting an existing non-black non-teal marker.

    Key insight
    - The yellow controls do not describe colors to preserve; they reposition the teal cross, and that shifted cross is the only valid reference frame for the final symmetry completion.
    """
    row_shift, col_shift, payload = _decode_shifts(np.asarray(grid, dtype=int))
    shifted = np.roll(payload, shift=(row_shift, col_shift), axis=(0, 1))
    axis_r, axis_c = _find_cross_axes(shifted)
    return _mirror_fill(shifted, axis_r, axis_c)


def _build_payload(
    size: int,
    axis_r: int,
    axis_c: int,
    cells: list[tuple[int, int]],
    colors: list[int],
) -> np.ndarray:
    payload = np.full((size, size), BG, dtype=int)
    payload[axis_r, :] = AXIS_COLOR
    payload[:, axis_c] = AXIS_COLOR
    for (r, c), color in zip(cells, colors):
        payload[r, c] = color
    return payload


def _candidate_groups(
    size: int,
    axis0_r: int,
    axis0_c: int,
    axis1_r: int,
    axis1_c: int,
    row_shift: int,
    col_shift: int,
) -> list[tuple[int, int, list[tuple[tuple[int, int], tuple[int, int]]]]]:
    groups: list[tuple[int, int, list[tuple[tuple[int, int], tuple[int, int]]]]] = []
    for post_q in range(4):
        buckets: dict[int, list[tuple[tuple[int, int], tuple[int, int]]]] = defaultdict(list)
        for post_cell in _quadrant_cells(size, axis1_r, axis1_c, post_q):
            pre_r = (post_cell[0] - row_shift) % size
            pre_c = (post_cell[1] - col_shift) % size
            pre_q = _quadrant_of(pre_r, pre_c, axis0_r, axis0_c)
            if pre_q is None:
                continue
            buckets[pre_q].append((post_cell, (pre_r, pre_c)))
        for pre_q, items in buckets.items():
            if len(items) >= 2:
                groups.append((pre_q, post_q, items))
    return groups


def _choose_axes(size: int) -> tuple[int, int, int, int]:
    axis0_r = random.randint(2, size - 3)
    axis0_c = random.randint(2, size - 3)
    edgeish = _edgeish_indices(size)
    branch = random.choices(("both", "row", "col"), weights=(6, 2, 2), k=1)[0]

    if branch == "row":
        axis1_r = random.choice([v for v in edgeish if v != axis0_r] or edgeish)
        axis1_c = axis0_c
    elif branch == "col":
        axis1_r = axis0_r
        axis1_c = random.choice([v for v in edgeish if v != axis0_c] or edgeish)
    else:
        axis1_r = random.choice([v for v in edgeish if v != axis0_r] or edgeish)
        axis1_c = random.choice([v for v in edgeish if v != axis0_c] or edgeish)

    if axis1_r == axis0_r and axis1_c == axis0_c:
        axis1_r = random.choice([v for v in edgeish if v != axis0_r] or edgeish)
    return axis0_r, axis0_c, axis1_r, axis1_c


def _semantic_ok(
    inp: np.ndarray,
    payload: np.ndarray,
    row_shift: int,
    col_shift: int,
    shifted: np.ndarray,
    out: np.ndarray,
    axis0_r: int,
    axis0_c: int,
    axis1_r: int,
    axis1_c: int,
    pre_cells: list[tuple[int, int]],
    post_cells: list[tuple[int, int]],
) -> bool:
    if _find_cross_axes(payload) != (axis0_r, axis0_c):
        return False
    if _find_cross_axes(shifted) != (axis1_r, axis1_c):
        return False
    if len({_quadrant_of(r, c, axis0_r, axis0_c) for r, c in pre_cells}) != 1:
        return False
    if len({_quadrant_of(r, c, axis1_r, axis1_c) for r, c in post_cells}) != 1:
        return False
    if not _has_clipped_reflection(post_cells, axis1_r, axis1_c, payload.shape[0]):
        return False

    pre_completed = _mirror_fill(payload, axis0_r, axis0_c)
    if np.array_equal(out, shifted):
        return False
    if np.array_equal(out, pre_completed):
        return False
    if np.array_equal(out, np.roll(pre_completed, shift=(row_shift, col_shift), axis=(0, 1))):
        return False
    if row_shift != 0 and col_shift != 0:
        row_only = _roll_then_fill(payload, row_shift, 0)
        col_only = _roll_then_fill(payload, 0, col_shift)
        if np.array_equal(out, row_only) or np.array_equal(out, col_only):
            return False

    try:
        solved = main(inp)
    except AssertionError:
        return False
    return np.array_equal(solved, out)


def _gen_attempt() -> np.ndarray | None:
    size = random.randint(8, 13)
    axis0_r, axis0_c, axis1_r, axis1_c = _choose_axes(size)
    row_shift = (axis1_r - axis0_r) % size
    col_shift = (axis1_c - axis0_c) % size

    groups = _candidate_groups(size, axis0_r, axis0_c, axis1_r, axis1_c, row_shift, col_shift)
    if not groups:
        return None

    diff_groups = [g for g in groups if g[0] != g[1]]
    if diff_groups and random.random() < 0.7:
        pre_q, post_q, items = random.choice(diff_groups)
    else:
        pre_q, post_q, items = random.choice(groups)

    random.shuffle(items)
    max_count = min(5, len(items))
    if max_count < 2:
        return None

    for _ in range(24):
        marker_count = random.randint(2, max_count)
        chosen = items[:marker_count]
        post_cells = [post for post, _ in chosen]
        pre_cells = [pre for _, pre in chosen]
        if not _has_clipped_reflection(post_cells, axis1_r, axis1_c, size):
            random.shuffle(items)
            continue

        colors = _assign_colors(marker_count)
        payload = _build_payload(size, axis0_r, axis0_c, pre_cells, colors)
        shifted = np.roll(payload, shift=(row_shift, col_shift), axis=(0, 1))
        out = _mirror_fill(shifted, axis1_r, axis1_c)

        inp = np.zeros((size + 1, size + 1), dtype=int)
        inp[0, 1 + col_shift] = CONTROL_MARKER
        inp[1 + row_shift, 0] = CONTROL_MARKER
        inp[1:, 1:] = payload

        if _semantic_ok(
            inp,
            payload,
            row_shift,
            col_shift,
            shifted,
            out,
            axis0_r,
            axis0_c,
            axis1_r,
            axis1_c,
            pre_cells,
            post_cells,
        ):
            return inp

        random.shuffle(items)
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shifted-cross symmetry input")
