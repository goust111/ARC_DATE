import random

import numpy as np


BG = 0
GREY = 5
SHAPE_COLORS = (1, 2, 3, 4, 6)
GRID_H_RANGE = (14, 22)
GRID_W_RANGE = (14, 22)
SHAPE_COUNT_RANGE = (2, 4)
SHAPE_SIDE_RANGE = (3, 6)
LINE_COUNT_RANGE = (1, 2)
GENERATION_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _line_rows_and_cols(grid: np.ndarray) -> tuple[list[int], list[int]]:
    line_rows = [int(r) for r in range(grid.shape[0]) if np.all(grid[r] == GREY)]
    line_cols = [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    return line_rows, line_cols


def _reconstruct_shape(
    visible: np.ndarray,
    line_rows: list[int],
    line_cols: list[int],
) -> np.ndarray:
    if not np.any(visible):
        return np.zeros_like(visible, dtype=bool)

    r0, c0, r1, c1 = _bbox(visible)
    work = visible.copy()
    active_rows = [r for r in line_rows if r0 <= r <= r1]
    active_cols = [c for c in line_cols if c0 <= c <= c1]

    changed = True
    while changed:
        changed = False

        for r in active_rows:
            for c in range(c0, c1 + 1):
                if work[r, c]:
                    continue
                if np.any(work[r0:r, c]) and np.any(work[r + 1 : r1 + 1, c]):
                    work[r, c] = True
                    changed = True

        for c in active_cols:
            for r in range(r0, r1 + 1):
                if work[r, c]:
                    continue
                if np.any(work[r, c0:c]) and np.any(work[r, c + 1 : c1 + 1]):
                    work[r, c] = True
                    changed = True

        for r in active_rows:
            cols = np.flatnonzero(work[r, c0 : c1 + 1])
            if cols.size >= 2:
                lo = int(cols[0]) + c0
                hi = int(cols[-1]) + c0
                before = work[r, lo : hi + 1].copy()
                work[r, lo : hi + 1] = True
                if not np.array_equal(before, work[r, lo : hi + 1]):
                    changed = True

        for c in active_cols:
            rows = np.flatnonzero(work[r0 : r1 + 1, c])
            if rows.size >= 2:
                lo = int(rows[0]) + r0
                hi = int(rows[-1]) + r0
                before = work[lo : hi + 1, c].copy()
                work[lo : hi + 1, c] = True
                if not np.array_equal(before, work[lo : hi + 1, c]):
                    changed = True

    return work


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the exposed grey guide lines and restore the colored shapes they cut through.

    Input structure:
    - Black background.
    - One-cell grey rows and/or columns spanning the whole grid.
    - Solid colored shapes that may be sliced by those lines.

    Rule:
    - Grey pixels outside shapes disappear.
    - Grey pixels inside a shape become that shape's color.
    - Shapes otherwise stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    line_rows, line_cols = _line_rows_and_cols(g)
    out = np.full_like(g, BG)

    colors = [int(color) for color in np.unique(g) if color not in (BG, GREY)]
    for color in colors:
        visible = g == color
        if not np.any(visible):
            continue
        restored = _reconstruct_shape(visible, line_rows, line_cols)
        out[restored] = color

    return out


def _ranges_overlap(a0: int, a1: int, b0: int, b1: int, gap: int = 0) -> bool:
    return not (a1 + gap < b0 or b1 + gap < a0)


def _place_shapes(
    height: int,
    width: int,
    n_shapes: int,
) -> list[tuple[int, int, int, int, int]] | None:
    shapes: list[tuple[int, int, int, int, int]] = []
    palette = random.sample(SHAPE_COLORS, n_shapes)

    for color in palette:
        placed = False
        for _ in range(80):
            sh = random.randint(*SHAPE_SIDE_RANGE)
            sw = random.randint(*SHAPE_SIDE_RANGE)
            r0 = random.randint(1, height - sh - 2)
            c0 = random.randint(1, width - sw - 2)
            r1 = r0 + sh - 1
            c1 = c0 + sw - 1

            if any(
                _ranges_overlap(r0, r1, or0, or1, gap=1)
                and _ranges_overlap(c0, c1, oc0, oc1, gap=1)
                for or0, oc0, or1, oc1, _ in shapes
            ):
                continue

            shapes.append((r0, c0, r1, c1, color))
            placed = True
            break

        if not placed:
            return None

    return shapes


def _spaced_choices(candidates: list[int], count: int) -> list[int] | None:
    items = candidates[:]
    random.shuffle(items)
    chosen: list[int] = []
    for value in items:
        if all(abs(value - prev) >= 2 for prev in chosen):
            chosen.append(int(value))
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _rows_crossing_shapes(
    shapes: list[tuple[int, int, int, int, int]]
) -> tuple[dict[int, list[int]], list[int]]:
    by_shape: dict[int, list[int]] = {}
    union: set[int] = set()
    for idx, (r0, _c0, r1, _c1, _color) in enumerate(shapes):
        rows = list(range(r0 + 1, r1))
        by_shape[idx] = rows
        union.update(rows)
    return by_shape, sorted(union)


def _cols_crossing_shapes(
    shapes: list[tuple[int, int, int, int, int]]
) -> tuple[dict[int, list[int]], list[int]]:
    by_shape: dict[int, list[int]] = {}
    union: set[int] = set()
    for idx, (_r0, c0, _r1, c1, _color) in enumerate(shapes):
        cols = list(range(c0 + 1, c1))
        by_shape[idx] = cols
        union.update(cols)
    return by_shape, sorted(union)


def _unused_rows(
    height: int,
    shapes: list[tuple[int, int, int, int, int]],
    blocked: set[int],
) -> list[int]:
    free: list[int] = []
    for r in range(1, height - 1):
        if r in blocked:
            continue
        if all(not (r0 <= r <= r1) for r0, _c0, r1, _c1, _color in shapes):
            free.append(r)
    return free


def _unused_cols(
    width: int,
    shapes: list[tuple[int, int, int, int, int]],
    blocked: set[int],
) -> list[int]:
    free: list[int] = []
    for c in range(1, width - 1):
        if c in blocked:
            continue
        if all(not (c0 <= c <= c1) for _r0, c0, _r1, c1, _color in shapes):
            free.append(c)
    return free


def _choose_lines(
    height: int,
    width: int,
    shapes: list[tuple[int, int, int, int, int]],
    mode: str,
) -> tuple[list[int], list[int]] | None:
    rows_by_shape, row_union = _rows_crossing_shapes(shapes)
    cols_by_shape, col_union = _cols_crossing_shapes(shapes)
    anchor = random.randrange(len(shapes))

    used_rows: list[int] = []
    used_cols: list[int] = []

    if mode in ("horizontal", "both"):
        need = random.randint(*LINE_COUNT_RANGE)
        if not rows_by_shape[anchor]:
            return None
        used_rows = [random.choice(rows_by_shape[anchor])]
        if need > 1:
            extras = [r for r in row_union if r not in used_rows and abs(r - used_rows[0]) >= 2]
            extra_rows = _spaced_choices(extras, need - 1)
            if extra_rows is None:
                return None
            used_rows.extend(extra_rows)
            used_rows = sorted(used_rows)

    if mode in ("vertical", "both"):
        need = random.randint(*LINE_COUNT_RANGE)
        if not cols_by_shape[anchor]:
            return None
        used_cols = [random.choice(cols_by_shape[anchor])]
        if need > 1:
            extras = [c for c in col_union if c not in used_cols and abs(c - used_cols[0]) >= 2]
            extra_cols = _spaced_choices(extras, need - 1)
            if extra_cols is None:
                return None
            used_cols.extend(extra_cols)
            used_cols = sorted(used_cols)

    orphan_rows: list[int] = []
    orphan_cols: list[int] = []

    if mode in ("horizontal", "both") and random.random() < 0.5:
        free_rows = _unused_rows(height, shapes, set(used_rows))
        choice = _spaced_choices(free_rows, 1)
        if choice is not None:
            orphan_rows = choice

    if mode in ("vertical", "both") and random.random() < 0.5:
        free_cols = _unused_cols(width, shapes, set(used_cols))
        choice = _spaced_choices(free_cols, 1)
        if choice is not None:
            orphan_cols = choice

    rows = sorted(set(used_rows + orphan_rows))
    cols = sorted(set(used_cols + orphan_cols))

    if not rows and not cols:
        return None
    return rows, cols


def _paint_shapes(
    height: int,
    width: int,
    shapes: list[tuple[int, int, int, int, int]],
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for r0, c0, r1, c1, color in shapes:
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
    return grid


def _paint_input(
    base: np.ndarray,
    line_rows: list[int],
    line_cols: list[int],
) -> np.ndarray:
    grid = base.copy()
    for r in line_rows:
        grid[r, :] = GREY
    for c in line_cols:
        grid[:, c] = GREY
    return grid


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    n_shapes = random.randint(*SHAPE_COUNT_RANGE)
    mode = random.choice(("horizontal", "vertical", "both"))

    shapes = _place_shapes(height, width, n_shapes)
    if shapes is None:
        return None

    lines = _choose_lines(height, width, shapes, mode)
    if lines is None:
        return None
    line_rows, line_cols = lines

    base = _paint_shapes(height, width, shapes)
    grid = _paint_input(base, line_rows, line_cols)

    if not line_rows and not line_cols:
        return None
    if np.array_equal(grid, base):
        return None
    if not np.array_equal(main(grid), base):
        return None

    return grid, base


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        case = _gen_attempt()
        if case is not None:
            grid, _expected = case
            return grid
    raise RuntimeError("failed to generate a valid input")
