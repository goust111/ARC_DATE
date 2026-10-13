from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
YELLOW = 4
GREY = 5
PINK = 6

MAX_GENERATION_ATTEMPTS = 400
GRID_MIN = 10
GRID_MAX = 16
SEPARATOR_COUNT_RANGE = (2, 3)
MARKER_COUNT_RANGE = (3, 5)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _find_separator_axis(grid: np.ndarray) -> tuple[int, list[int]]:
    row_separators = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    col_separators = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    has_rows = bool(row_separators)
    has_cols = bool(col_separators)
    assert has_rows != has_cols, "expected grey separators along exactly one axis"
    return (0, row_separators) if has_rows else (1, col_separators)


def _compress_grid(grid: np.ndarray) -> tuple[np.ndarray, int]:
    separator_axis, separators = _find_separator_axis(grid)
    if separator_axis == 0:
        kept_rows = [r for r in range(grid.shape[0]) if r not in set(separators)]
        compressed = grid[kept_rows, :]
    else:
        kept_cols = [c for c in range(grid.shape[1]) if c not in set(separators)]
        compressed = grid[:, kept_cols]
    return compressed, separator_axis


def _find_cross_axes(grid: np.ndarray) -> tuple[int, int]:
    row_candidates = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == PINK)]
    col_candidates = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == PINK)]
    assert len(row_candidates) == 1 and len(col_candidates) == 1, "expected one full pink row and one full pink column after compression"
    return row_candidates[0], col_candidates[0]


def _hourglass_hit(r: int, c: int, axis_r: int, axis_c: int, separator_axis: int) -> bool:
    dr = abs(r - axis_r)
    dc = abs(c - axis_c)
    dn, do = (dr, dc) if separator_axis == 0 else (dc, dr)
    return dn > 0 and do > 0 and dn <= do


def _mirror_positions(r: int, c: int, axis_r: int, axis_c: int) -> set[tuple[int, int]]:
    return {
        (2 * axis_r - r, c),
        (r, 2 * axis_c - c),
        (2 * axis_r - r, 2 * axis_c - c),
    }


def _solve_compressed(grid: np.ndarray, separator_axis: int) -> np.ndarray:
    axis_r, axis_c = _find_cross_axes(grid)
    originals = [tuple(map(int, rc)) for rc in np.argwhere(grid == BLUE)]

    out = grid.copy()
    original_set = set(originals)
    generated: set[tuple[int, int]] = set()

    for r, c in originals:
        assert r != axis_r and c != axis_c, "input markers must stay off the cross axes"
        for rr, cc in _mirror_positions(r, c, axis_r, axis_c):
            if not _in_bounds(grid.shape, rr, cc):
                continue
            assert out[rr, cc] != PINK, "mirrored markers must not overwrite the cross"
            out[rr, cc] = BLUE
            if (rr, cc) not in original_set:
                generated.add((rr, cc))

    for r, c in generated:
        if _hourglass_hit(r, c, axis_r, axis_c, separator_axis):
            out[r, c] = YELLOW

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress away grey separators, mirror sparse blue markers across a pink cross, and recolor only the mirrored copies that fall inside an hourglass.

    Input structure
    The grid has a black background, grey full-line separators along exactly one
    axis, and a pink cross encoded by one full pink row plus one pink column
    that becomes complete after the separators are removed. Blue single-pixel
    markers lie off the cross in one quadrant.

    Rule
    Remove every full grey separator row or column to obtain compressed
    coordinates. In that compressed grid, mirror each blue marker across the
    pink row, across the pink column, and across both axes. Keep the original
    blue markers blue. Among only the newly created mirrored pixels, recolor a
    pixel yellow when its distance from the cross along the separator axis is
    nonzero, its distance along the perpendicular axis is nonzero, and the
    separator-axis distance is no greater than the perpendicular distance.

    Key insight
    The grey lines define a coordinate-compression step and also choose the
    hourglass narrow axis, while the pink cross is the mirror reference frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    compressed, separator_axis = _compress_grid(g)
    return _solve_compressed(compressed, separator_axis)


def _quadrant_cells(h: int, w: int, axis_r: int, axis_c: int, quadrant: int) -> list[tuple[int, int]]:
    if quadrant == 0:
        rows = range(0, axis_r)
        cols = range(0, axis_c)
    elif quadrant == 1:
        rows = range(0, axis_r)
        cols = range(axis_c + 1, w)
    elif quadrant == 2:
        rows = range(axis_r + 1, h)
        cols = range(0, axis_c)
    else:
        rows = range(axis_r + 1, h)
        cols = range(axis_c + 1, w)
    return [(r, c) for r in rows for c in cols]


def _mirror_family_in_bounds_count(shape: tuple[int, int], r: int, c: int, axis_r: int, axis_c: int) -> int:
    return sum(_in_bounds(shape, rr, cc) for rr, cc in _mirror_positions(r, c, axis_r, axis_c))


def _well_spaced(cells: Iterable[tuple[int, int]]) -> bool:
    pts = list(cells)
    for i, (r0, c0) in enumerate(pts):
        for r1, c1 in pts[i + 1 :]:
            if max(abs(r0 - r1), abs(c0 - c1)) <= 1:
                return False
    return True


def _families_disjoint(cells: list[tuple[int, int]], axis_r: int, axis_c: int) -> bool:
    occupied: set[tuple[int, int]] = set()
    for r, c in cells:
        family = {(r, c), *_mirror_positions(r, c, axis_r, axis_c)}
        if occupied & family:
            return False
        occupied |= family
    return True


def _build_compressed_grid(
    h: int,
    w: int,
    axis_r: int,
    axis_c: int,
    markers: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    grid[axis_r, :] = PINK
    grid[:, axis_c] = PINK
    for r, c in markers:
        grid[r, c] = BLUE
    return grid


def _expand_with_separators(compressed: np.ndarray, separator_axis: int, separator_boundaries: list[int]) -> np.ndarray:
    h, w = compressed.shape
    if separator_axis == 0:
        raw_h = h + len(separator_boundaries)
        raw = np.full((raw_h, w), BLACK, dtype=int)
        kept_rows: list[int] = []
        separator_rows: set[int] = set()
        raw_r = 0
        for r in range(h):
            kept_rows.append(raw_r)
            raw_r += 1
            if (r + 1) in separator_boundaries:
                separator_rows.add(raw_r)
                raw_r += 1
        for src_r, dst_r in enumerate(kept_rows):
            raw[dst_r, :] = compressed[src_r, :]
        for sep_r in separator_rows:
            raw[sep_r, :] = GREY
        return raw

    raw_w = w + len(separator_boundaries)
    raw = np.full((h, raw_w), BLACK, dtype=int)
    kept_cols: list[int] = []
    separator_cols: set[int] = set()
    raw_c = 0
    for c in range(w):
        kept_cols.append(raw_c)
        raw_c += 1
        if (c + 1) in separator_boundaries:
            separator_cols.add(raw_c)
            raw_c += 1
    for src_c, dst_c in enumerate(kept_cols):
        raw[:, dst_c] = compressed[:, src_c]
    for sep_c in separator_cols:
        raw[:, sep_c] = GREY
    return raw


def _sample_marker_set(
    h: int,
    w: int,
    axis_r: int,
    axis_c: int,
    separator_axis: int,
    quadrant: int,
    count: int,
    want_clipping: bool,
) -> list[tuple[int, int]] | None:
    candidates = _quadrant_cells(h, w, axis_r, axis_c, quadrant)
    if len(candidates) < count:
        return None

    inside = []
    outside = []
    clipped = []
    for r, c in candidates:
        in_bounds_count = _mirror_family_in_bounds_count((h, w), r, c, axis_r, axis_c)
        if in_bounds_count == 0:
            continue
        bucket = inside if _hourglass_hit(r, c, axis_r, axis_c, separator_axis) else outside
        bucket.append((r, c))
        if in_bounds_count < 3:
            clipped.append((r, c))

    if not inside or not outside:
        return None

    random.shuffle(inside)
    random.shuffle(outside)
    base = [inside[0], outside[0]]
    if want_clipping:
        clip_choices = [cell for cell in clipped if cell not in base]
        if not clip_choices:
            return None
        base.append(clip_choices[0])

    pool = [cell for cell in candidates if cell not in base]
    random.shuffle(pool)
    chosen = base[:]
    for cell in pool:
        if len(chosen) >= count:
            break
        chosen.append(cell)

    if len(chosen) != count:
        return None
    if not _well_spaced(chosen):
        return None
    if not _families_disjoint(chosen, axis_r, axis_c):
        return None
    if not any(_hourglass_hit(r, c, axis_r, axis_c, separator_axis) for r, c in chosen):
        return None
    if not any(not _hourglass_hit(r, c, axis_r, axis_c, separator_axis) for r, c in chosen):
        return None
    return chosen


def _semantic_checks(raw: np.ndarray, compressed: np.ndarray, markers: list[tuple[int, int]], separator_axis: int) -> bool:
    expected = _solve_compressed(compressed, separator_axis)
    actual = main(raw)
    if not np.array_equal(actual, expected):
        return False
    detected_axis, separators = _find_separator_axis(raw)
    if detected_axis != separator_axis or not (SEPARATOR_COUNT_RANGE[0] <= len(separators) <= SEPARATOR_COUNT_RANGE[1]):
        return False

    axis_r, axis_c = _find_cross_axes(compressed)
    original_set = set(markers)
    generated = set()
    for r, c in markers:
        for rr, cc in _mirror_positions(r, c, axis_r, axis_c):
            if _in_bounds(compressed.shape, rr, cc) and (rr, cc) not in original_set:
                generated.add((rr, cc))

    if not generated:
        return False
    recolored = {cell for cell in generated if actual[cell] == YELLOW}
    stayed_blue = {cell for cell in generated if actual[cell] == BLUE}
    if not recolored or not stayed_blue:
        return False

    inside_originals = [cell for cell in markers if _hourglass_hit(cell[0], cell[1], axis_r, axis_c, separator_axis)]
    if not inside_originals:
        return False
    if any(actual[cell] != BLUE for cell in inside_originals):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    separator_axis = random.randint(0, 1)
    count = random.randint(*MARKER_COUNT_RANGE)

    for _ in range(80):
        h = random.randint(GRID_MIN, GRID_MAX)
        w = random.randint(GRID_MIN, GRID_MAX)
        axis_r = random.randint(2, h - 3)
        axis_c = random.randint(2, w - 3)
        quadrant = random.randint(0, 3)
        separator_count = random.randint(*SEPARATOR_COUNT_RANGE)
        want_clipping = random.random() < 0.35

        if separator_axis == 0:
            boundaries = list(range(1, h))
        else:
            boundaries = list(range(1, w))
        if len(boundaries) < separator_count:
            continue
        separator_boundaries = sorted(random.sample(boundaries, separator_count))

        markers = _sample_marker_set(h, w, axis_r, axis_c, separator_axis, quadrant, count, want_clipping)
        if markers is None:
            continue

        compressed = _build_compressed_grid(h, w, axis_r, axis_c, markers)
        raw = _expand_with_separators(compressed, separator_axis, separator_boundaries)
        if max(raw.shape) > 30:
            continue
        if _semantic_checks(raw, compressed, markers, separator_axis):
            return raw

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
