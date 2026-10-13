from __future__ import annotations

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

GENERATION_RETRIES = 400


def _detect_axes(grid: np.ndarray) -> tuple[int, int, int, int, int, int]:
    g = np.asarray(grid, dtype=int)
    red = g == RED
    assert red.any(), "expected red axis segments"

    row_counts = red.sum(axis=1)
    col_counts = red.sum(axis=0)
    h_rows = np.flatnonzero(row_counts >= 2)
    v_cols = np.flatnonzero(col_counts >= 2)
    assert len(h_rows) == 1 and len(v_cols) == 1, "expected one horizontal and one vertical red segment"

    h_row = int(h_rows[0])
    v_col = int(v_cols[0])
    h_cols = np.flatnonzero(red[h_row])
    v_rows = np.flatnonzero(red[:, v_col])
    assert h_cols.size >= 2 and v_rows.size >= 2, "axis segments must have length at least 2"

    hc0, hc1 = int(h_cols[0]), int(h_cols[-1])
    vr0, vr1 = int(v_rows[0]), int(v_rows[-1])

    expected = np.zeros_like(red, dtype=bool)
    expected[h_row, hc0 : hc1 + 1] = True
    expected[vr0 : vr1 + 1, v_col] = True
    assert np.array_equal(red, expected), "red pixels must form exactly one horizontal and one vertical segment"
    return h_row, hc0, hc1, v_col, vr0, vr1


def _classify_grey(
    grey: np.ndarray,
    h_row: int,
    hc0: int,
    hc1: int,
    v_col: int,
    vr0: int,
    vr1: int,
) -> tuple[dict[str, set[tuple[int, int]]], dict[tuple[int, int], str]]:
    h, w = grey.shape
    groups = {
        "both": set(),
        "vertical_only": set(),
        "horizontal_only": set(),
        "none": set(),
    }
    cell_to_group: dict[tuple[int, int], str] = {}

    for r, c in np.argwhere(grey):
        rr = int(r)
        cc = int(c)
        mirror_c = 2 * v_col - cc
        mirror_r = 2 * h_row - rr
        vertical_ok = vr0 <= rr <= vr1 and 0 <= mirror_c < w and bool(grey[rr, mirror_c])
        horizontal_ok = hc0 <= cc <= hc1 and 0 <= mirror_r < h and bool(grey[mirror_r, cc])
        cell = (rr, cc)
        if vertical_ok and horizontal_ok:
            label = "both"
        elif vertical_ok:
            label = "vertical_only"
        elif horizontal_ok:
            label = "horizontal_only"
        else:
            label = "none"
        groups[label].add(cell)
        cell_to_group[cell] = label
    return groups, cell_to_group


def _expected_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h_row, hc0, hc1, v_col, vr0, vr1 = _detect_axes(g)
    grey = g == GREY
    groups, _ = _classify_grey(grey, h_row, hc0, hc1, v_col, vr0, vr1)

    out = g.copy()
    for r, c in groups["vertical_only"]:
        out[r, c] = BLUE
    for r, c in groups["horizontal_only"]:
        out[r, c] = GREEN
    for r, c in groups["both"]:
        out[r, c] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey pixels by whether they pass vertical-segment and horizontal-segment mirror tests.

    Input structure:
    - Black background with one vertical red segment, one horizontal red segment, and several grey singleton pixels.
    - The red segments define a vertical mirror axis at their column and a horizontal mirror axis at their row.
    - Grey pixels never lie on the red row or red column.

    Rule:
    - A grey pixel passes the vertical test iff its row lies inside the vertical segment's row span and the cell mirrored across the red column is also grey.
    - A grey pixel passes the horizontal test iff its column lies inside the horizontal segment's column span and the cell mirrored across the red row is also grey.
    - Recolor vertical-only pixels blue, horizontal-only pixels green, pixels passing both tests yellow, and leave failing pixels grey.
    - Keep the red segments unchanged.

    Key insight:
    - The mirror axes are finite in-scene segments rather than whole-canvas midlines, so symmetry alone is insufficient unless the pixel also lies inside the corresponding segment span.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _expected_output(g)


def _choose_single(pool: list[int], used: set[int], rng: np.random.Generator) -> int | None:
    options = [int(v) for v in pool if int(v) not in used]
    if not options:
        return None
    value = int(options[int(rng.integers(len(options)))])
    used.add(value)
    return value


def _choose_pair(
    offsets: list[int],
    used: set[int],
    center: int,
    rng: np.random.Generator,
) -> int | None:
    options = [int(d) for d in offsets if (center - int(d)) not in used and (center + int(d)) not in used]
    if not options:
        return None
    delta = int(options[int(rng.integers(len(options)))])
    used.add(center - delta)
    used.add(center + delta)
    return delta


def _paint_role(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    role: str,
    roles: dict[tuple[int, int], str],
) -> None:
    for r, c in cells:
        assert grid[r, c] == BG, "grey pixels must not overlap"
        grid[r, c] = GREY
        roles[(int(r), int(c))] = role


def _validate_semantics(
    grid: np.ndarray,
    roles: dict[tuple[int, int], str],
    motifs: dict[str, list[tuple[int, int]]],
) -> bool:
    h_row, hc0, hc1, v_col, vr0, vr1 = _detect_axes(grid)
    grey = grid == GREY
    groups, cell_to_group = _classify_grey(grey, h_row, hc0, hc1, v_col, vr0, vr1)

    if not all(groups[key] for key in groups):
        return False
    if set(np.unique(grid)) - {BG, RED, GREY}:
        return False
    if grey[h_row, :].any() or grey[:, v_col].any():
        return False

    for cell, role in roles.items():
        if cell_to_group.get(cell) != role:
            return False

    for r, c in motifs["both"]:
        if not (vr0 <= r <= vr1 and hc0 <= c <= hc1):
            return False
        if cell_to_group[(r, c)] != "both":
            return False
        if not grey[r, 2 * v_col - c] or not grey[2 * h_row - r, c]:
            return False

    for r, c in motifs["vertical_only"]:
        if not (vr0 <= r <= vr1) or hc0 <= c <= hc1:
            return False
        if cell_to_group[(r, c)] != "vertical_only":
            return False
        if not grey[r, 2 * v_col - c]:
            return False

    for r, c in motifs["horizontal_only"]:
        if vr0 <= r <= vr1 or not (hc0 <= c <= hc1):
            return False
        if cell_to_group[(r, c)] != "horizontal_only":
            return False
        if not grey[2 * h_row - r, c]:
            return False

    for r, c in motifs["vertical_scope_decoy"]:
        if vr0 <= r <= vr1:
            return False
        if not (hc0 <= c <= hc1):
            return False
        if cell_to_group[(r, c)] != "none":
            return False
        if not grey[r, 2 * v_col - c]:
            return False

    for r, c in motifs["horizontal_scope_decoy"]:
        if not (vr0 <= r <= vr1):
            return False
        if hc0 <= c <= hc1:
            return False
        if cell_to_group[(r, c)] != "none":
            return False
        if not grey[2 * h_row - r, c]:
            return False

    for r, c in motifs["overlap_singleton"]:
        if not (vr0 <= r <= vr1 and hc0 <= c <= hc1):
            return False
        if cell_to_group[(r, c)] != "none":
            return False
        if grey[r, 2 * v_col - c] or grey[2 * h_row - r, c]:
            return False

    oracle = _expected_output(grid)
    if np.array_equal(oracle, grid):
        return False
    return np.array_equal(main(grid), oracle)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    h = int(rng.integers(19, 28))
    w = int(rng.integers(19, 28))
    axis_row = int(rng.integers(7, h - 7))
    axis_col = int(rng.integers(7, w - 7))
    row_limit = min(axis_row, h - 1 - axis_row)
    col_limit = min(axis_col, w - 1 - axis_col)
    if row_limit < 5 or col_limit < 5:
        return None

    up = int(rng.integers(3, row_limit - 1))
    down = int(rng.integers(3, row_limit - 1))
    left = int(rng.integers(3, col_limit - 1))
    right = int(rng.integers(3, col_limit - 1))

    vr0, vr1 = axis_row - up, axis_row + down
    hc0, hc1 = axis_col - left, axis_col + right

    grid = np.full((h, w), BG, dtype=int)
    grid[vr0 : vr1 + 1, axis_col] = RED
    grid[axis_row, hc0 : hc1 + 1] = RED

    inside_rows = [r for r in range(vr0, vr1 + 1) if r != axis_row]
    outside_rows = [r for r in range(h) if r != axis_row and not (vr0 <= r <= vr1)]
    inside_cols = [c for c in range(hc0, hc1 + 1) if c != axis_col]
    outside_cols = [c for c in range(w) if c != axis_col and not (hc0 <= c <= hc1)]
    inside_row_offsets = list(range(1, min(up, down) + 1))
    outside_row_offsets = list(range(max(up, down) + 1, row_limit + 1))
    inside_col_offsets = list(range(1, min(left, right) + 1))
    outside_col_offsets = list(range(max(left, right) + 1, col_limit + 1))

    used_rows: set[int] = set()
    used_cols: set[int] = set()
    roles: dict[tuple[int, int], str] = {}
    motifs: dict[str, list[tuple[int, int]]] = {}

    both_dr = _choose_pair(inside_row_offsets, used_rows, axis_row, rng)
    both_dc = _choose_pair(inside_col_offsets, used_cols, axis_col, rng)
    if both_dr is None or both_dc is None:
        return None
    motifs["both"] = [
        (axis_row - both_dr, axis_col - both_dc),
        (axis_row - both_dr, axis_col + both_dc),
        (axis_row + both_dr, axis_col - both_dc),
        (axis_row + both_dr, axis_col + both_dc),
    ]
    _paint_role(grid, motifs["both"], "both", roles)

    vonly_row = _choose_single(inside_rows, used_rows, rng)
    vonly_dc = _choose_pair(outside_col_offsets, used_cols, axis_col, rng)
    if vonly_row is None or vonly_dc is None:
        return None
    motifs["vertical_only"] = [
        (vonly_row, axis_col - vonly_dc),
        (vonly_row, axis_col + vonly_dc),
    ]
    _paint_role(grid, motifs["vertical_only"], "vertical_only", roles)

    honly_dr = _choose_pair(outside_row_offsets, used_rows, axis_row, rng)
    honly_col = _choose_single(inside_cols, used_cols, rng)
    if honly_dr is None or honly_col is None:
        return None
    motifs["horizontal_only"] = [
        (axis_row - honly_dr, honly_col),
        (axis_row + honly_dr, honly_col),
    ]
    _paint_role(grid, motifs["horizontal_only"], "horizontal_only", roles)

    vdecoy_row = _choose_single(outside_rows, used_rows, rng)
    vdecoy_dc = _choose_pair(inside_col_offsets, used_cols, axis_col, rng)
    if vdecoy_row is None or vdecoy_dc is None:
        return None
    motifs["vertical_scope_decoy"] = [
        (vdecoy_row, axis_col - vdecoy_dc),
        (vdecoy_row, axis_col + vdecoy_dc),
    ]
    _paint_role(grid, motifs["vertical_scope_decoy"], "none", roles)

    hdecoy_dr = _choose_pair(inside_row_offsets, used_rows, axis_row, rng)
    hdecoy_col = _choose_single(outside_cols, used_cols, rng)
    if hdecoy_dr is None or hdecoy_col is None:
        return None
    motifs["horizontal_scope_decoy"] = [
        (axis_row - hdecoy_dr, hdecoy_col),
        (axis_row + hdecoy_dr, hdecoy_col),
    ]
    _paint_role(grid, motifs["horizontal_scope_decoy"], "none", roles)

    singleton_row = _choose_single(inside_rows, used_rows, rng)
    singleton_col = _choose_single(inside_cols, used_cols, rng)
    if singleton_row is None or singleton_col is None:
        return None
    motifs["overlap_singleton"] = [(singleton_row, singleton_col)]
    _paint_role(grid, motifs["overlap_singleton"], "none", roles)

    if not _validate_semantics(grid, roles, motifs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
