from __future__ import annotations

import random

import numpy as np


BLACK = 0
PINK = 6
MAROON = 9

GEN_MAX_ATTEMPTS = 500
GEN_PATTERN_COLORS = [2, 3, 4, 7, 8]
GEN_BLOCK_SEQS = {
    3: [(0, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 0)],
    4: [(0, 0, 1, 1), (1, 1, 0, 0), (0, 1, 1, 0), (1, 0, 0, 1)],
}


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _overlap_slices(shape: tuple[int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    h, w = shape
    r0 = max(0, -dr)
    r1 = min(h, h - dr)
    c0 = max(0, -dc)
    c1 = min(w, w - dc)
    return r0, r1, c0, c1


def _axis_supports(region: np.ndarray, axis: str) -> dict[int, int]:
    h, w = region.shape
    limit = h if axis == "v" else w
    supports: dict[int, int] = {}
    for step in range(1, limit):
        dr, dc = (step, 0) if axis == "v" else (0, step)
        r0, r1, c0, c1 = _overlap_slices(region.shape, dr, dc)
        if r0 >= r1 or c0 >= c1:
            continue
        a = region[r0:r1, c0:c1]
        b = region[r0 + dr : r1 + dr, c0 + dc : c1 + dc]
        visible = (a != BLACK) & (b != BLACK)
        support = int(np.count_nonzero(visible))
        if support == 0:
            continue
        if np.any(a[visible] != b[visible]):
            continue
        supports[step] = support
    return supports


def _best_axis_shift(region: np.ndarray, axis: str) -> int:
    supports = _axis_supports(region, axis)
    assert supports, f"no valid {axis}-axis translation found"
    return min(step for step, score in supports.items() if score == max(supports.values()))


def _fill_region_axis(region: np.ndarray, axis: str, step: int) -> np.ndarray:
    out = np.array(region, dtype=int, copy=True)
    h, w = out.shape

    if axis == "h":
        for r in range(h):
            for rem in range(step):
                vals = [int(out[r, c]) for c in range(rem, w, step) if int(out[r, c]) != BLACK]
                assert vals, "every horizontal class must retain a visible witness"
                color = vals[0]
                assert all(v == color for v in vals)
                for c in range(rem, w, step):
                    if int(out[r, c]) == BLACK:
                        out[r, c] = color
    else:
        for c in range(w):
            for rem in range(step):
                vals = [int(out[r, c]) for r in range(rem, h, step) if int(out[r, c]) != BLACK]
                assert vals, "every vertical class must retain a visible witness"
                color = vals[0]
                assert all(v == color for v in vals)
                for r in range(rem, h, step):
                    if int(out[r, c]) == BLACK:
                        out[r, c] = color

    assert not np.any(out == BLACK), "chosen-axis restoration must fill every hole"
    return out


def _control_color(
    filled_region: np.ndarray,
    axis: str,
    step: int,
    control: tuple[int, int],
    bbox_rc: tuple[int, int],
) -> int:
    pr, pc = control
    r0, c0 = bbox_rc
    h, w = filled_region.shape

    if axis == "h":
        rr = pr - r0
        if not (0 <= rr < h):
            return MAROON
        rem = (pc - c0) % step
        vals = {int(filled_region[rr, c]) for c in range(rem, w, step)}
        assert len(vals) == 1
        return next(iter(vals))

    cc = pc - c0
    if not (0 <= cc < w):
        return MAROON
    rem = (pr - r0) % step
    vals = {int(filled_region[r, cc]) for r in range(rem, h, step)}
    assert len(vals) == 1
    return next(iter(vals))


def main(grid: np.ndarray) -> np.ndarray:
    """Use the pink control pixel's side to choose the recovery axis of a rectangular periodic field.

    Input structure:
    - A mostly rectangular colored pattern region sits inside a black background; some cells inside that rectangle are black holes.
    - The visible pattern cells use ordinary non-black colors, while exactly one pink control pixel lies outside the pattern rectangle.
    - The visible pattern is compatible with both a horizontal and a vertical one-dimensional translation, but those two restorations can disagree on hole colors.

    Rule:
    - Find the bounding rectangle of the non-black, non-pink pattern cells and locate the single pink control pixel.
    - If the control is above or below the rectangle, choose a vertical translation; otherwise choose a horizontal translation.
    - Along that chosen axis, find the nonzero shift with maximum conflict-free visible overlap, then use its equivalence classes to fill every black hole in the rectangle.
    - Recolor the pink control pixel to the color implied by the recovered pattern if it lies on the chosen translational extension; otherwise recolor it maroon.

    Key insight:
    - The control pixel does not point to a cell inside the pattern; it disambiguates which of two valid periodic restorations should be trusted.
    """
    g = np.asarray(grid, dtype=int)

    controls = np.argwhere(g == PINK)
    assert controls.shape == (1, 2), f"expected one pink control pixel, got {len(controls)}"
    pr, pc = map(int, controls[0])

    pattern_mask = (g != BLACK) & (g != PINK)
    pattern_bbox = _bbox(pattern_mask)
    assert pattern_bbox is not None, "missing pattern region"
    r0, c0, r1, c1 = pattern_bbox
    assert not (r0 <= pr <= r1 and c0 <= pc <= c1), "control pixel must be outside the pattern rectangle"

    axis = "v" if pr < r0 or pr > r1 else "h"
    region = g[r0 : r1 + 1, c0 : c1 + 1]
    step = _best_axis_shift(region, axis)
    filled_region = _fill_region_axis(region, axis, step)

    out = np.array(g, copy=True)
    out[r0 : r1 + 1, c0 : c1 + 1] = filled_region
    out[pr, pc] = _control_color(filled_region, axis, step, (pr, pc), (r0, c0))
    return out


def _strict_best_step(region: np.ndarray, axis: str, target: int) -> bool:
    supports = _axis_supports(region, axis)
    if target not in supports:
        return False
    best = supports[target]
    return all(best > score for step, score in supports.items() if step != target)


def _sample_color_pairs(py: int, px: int) -> tuple[list[list[int]], list[list[int]]]:
    palette = random.sample(GEN_PATTERN_COLORS, 3)
    a_table = [[0] * px for _ in range(py)]
    b_table = [[0] * px for _ in range(py)]
    for t in range(py):
        for k in range(px):
            a, b = random.sample(palette, 2)
            a_table[t][k] = a
            b_table[t][k] = b
    return a_table, b_table


def _build_region(
    py: int,
    px: int,
    row_blocks: int,
    col_blocks: int,
    row_seq: tuple[int, ...],
    col_seq: tuple[int, ...],
    a_table: list[list[int]],
    b_table: list[list[int]],
) -> np.ndarray:
    h = py * row_blocks
    w = px * col_blocks
    region = np.zeros((h, w), dtype=int)
    for r in range(h):
        t = r % py
        row_type = row_seq[r // py]
        for c in range(w):
            k = c % px
            col_type = col_seq[c // px]
            if row_type == col_type:
                region[r, c] = a_table[t][k] if row_type == 0 else b_table[t][k]
    return region


def _expected_output(
    grid: np.ndarray,
    pattern_bbox: tuple[int, int, int, int],
    axis: str,
) -> np.ndarray:
    r0, c0, r1, c1 = pattern_bbox
    region = grid[r0 : r1 + 1, c0 : c1 + 1]
    step = _best_axis_shift(region, axis)
    filled = _fill_region_axis(region, axis, step)
    out = np.array(grid, copy=True)
    out[r0 : r1 + 1, c0 : c1 + 1] = filled
    pr, pc = map(int, np.argwhere(grid == PINK)[0])
    out[pr, pc] = _control_color(filled, axis, step, (pr, pc), (r0, c0))
    return out


def _choose_control_position(
    side: str,
    aligned: bool,
    top: int,
    left: int,
    rh: int,
    rw: int,
    gh: int,
    gw: int,
) -> tuple[int, int] | None:
    rows_inside = list(range(top, top + rh))
    cols_inside = list(range(left, left + rw))
    rows_outside = [r for r in range(gh) if r not in rows_inside]
    cols_outside = [c for c in range(gw) if c not in cols_inside]

    if side == "top":
        if top == 0:
            return None
        pr = random.randrange(0, top)
        pc = random.choice(cols_inside if aligned else cols_outside)
        return pr, pc
    if side == "bottom":
        if top + rh >= gh:
            return None
        pr = random.randrange(top + rh, gh)
        pc = random.choice(cols_inside if aligned else cols_outside)
        return pr, pc
    if side == "left":
        if left == 0:
            return None
        pc = random.randrange(0, left)
        pr = random.choice(rows_inside if aligned else rows_outside)
        return pr, pc
    if left + rw >= gw:
        return None
    pc = random.randrange(left + rw, gw)
    pr = random.choice(rows_inside if aligned else rows_outside)
    return pr, pc


def _gen_attempt() -> np.ndarray | None:
    py = random.randint(3, 4)
    px = random.randint(3, 4)
    row_blocks = random.randint(3, 4)
    col_blocks = random.randint(3, 4)
    row_seq = random.choice(GEN_BLOCK_SEQS[row_blocks])
    col_seq = random.choice(GEN_BLOCK_SEQS[col_blocks])

    a_table, b_table = _sample_color_pairs(py, px)
    region = _build_region(py, px, row_blocks, col_blocks, row_seq, col_seq, a_table, b_table)
    rh, rw = region.shape

    top = random.randint(1, 3)
    bottom = random.randint(1, 3)
    left = random.randint(1, 3)
    right = random.randint(1, 3)
    gh = top + rh + bottom
    gw = left + rw + right
    if gh > 30 or gw > 30:
        return None

    grid = np.zeros((gh, gw), dtype=int)
    grid[top : top + rh, left : left + rw] = region
    pattern_bbox = (top, left, top + rh - 1, left + rw - 1)

    side = random.choice(["top", "bottom", "left", "right"])
    axis = "v" if side in ("top", "bottom") else "h"
    aligned = random.choice([False, True])
    control = _choose_control_position(side, aligned, top, left, rh, rw, gh, gw)
    if control is None:
        return None
    grid[control] = PINK

    region_h = region.copy()
    region_v = region.copy()
    if not _strict_best_step(region_h, "h", px):
        return None
    if not _strict_best_step(region_v, "v", py):
        return None

    try:
        filled_h = _fill_region_axis(region_h, "h", px)
        filled_v = _fill_region_axis(region_v, "v", py)
    except AssertionError:
        return None

    if np.array_equal(filled_h, filled_v):
        return None
    if not np.any(region == BLACK):
        return None

    expected = _expected_output(grid, pattern_bbox, axis)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    control_color = int(expected[control])
    if aligned and control_color == MAROON:
        return None
    if not aligned and control_color != MAROON:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
