from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

GEN_RETRY_LIMIT = 400


def _extract_controls(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    assert g.shape[0] > 2 and g.shape[1] > 2, "input must be at least 3x3"
    assert np.all((g == BLACK) | (g == BLUE)), "input may only contain black background and blue controls"

    controls = [tuple(map(int, rc)) for rc in np.argwhere(g == BLUE)]
    assert 2 <= len(controls) <= 3, "expected 2 or 3 blue control pixels"

    rows = [r for r, _c in controls]
    cols = [c for _r, c in controls]
    assert len(set(rows)) == len(rows), "control pixels must occupy distinct rows"
    assert len(set(cols)) == len(cols), "control pixels must occupy distinct columns"
    return controls


def _solve_from_controls(shape: tuple[int, int], controls: list[tuple[int, int]]) -> np.ndarray:
    h, w = map(int, shape)
    out = np.zeros((2 * h, 2 * w), dtype=int)

    row_source: dict[int, int] = {}
    col_source: dict[int, int] = {}
    red_cells: list[tuple[int, int]] = []

    for source_id, (r, c) in enumerate(controls):
        rows = (int(r), int(2 * h - 1 - r))
        cols = (int(c), int(2 * w - 1 - c))

        for rr in rows:
            assert rr not in row_source, "distinct input rows should stay distinct after mirroring"
            row_source[rr] = source_id
        for cc in cols:
            assert cc not in col_source, "distinct input cols should stay distinct after mirroring"
            col_source[cc] = source_id

        for rr in rows:
            for cc in cols:
                red_cells.append((rr, cc))

    marked_rows = sorted(row_source)
    marked_cols = sorted(col_source)

    out[marked_rows, :] = BLUE
    out[:, marked_cols] = BLUE

    for rr in marked_rows:
        for cc in marked_cols:
            if row_source[rr] != col_source[cc]:
                out[rr, cc] = GREEN

    for rr, cc in red_cells:
        out[rr, cc] = RED

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the sparse controls into four quadrants, extend full row/column rays, and recolor cross-source intersections.

    Input structure:
    - The input is a small black grid containing 2 or 3 blue control pixels.
    - Control pixels occupy distinct rows and distinct columns.
    - No other foreground objects are present.

    Rule:
    - Build a 2H x 2W four-quadrant mirror tile whose upper-left quadrant is the input.
    - Treat every mirrored blue copy as a ray source and paint its full row and full column blue.
    - Recolor the original mirrored source pixels red.
    - Recolor a blue cell green exactly when it is a row/column crossing whose row ray and column ray come from different original control pixels.

    Key insight:
    - The visible row and column sets are not enough: after mirroring, each original control contributes a paired row-set and col-set, and green versus red depends on keeping that source pairing intact.
    """
    controls = _extract_controls(np.asarray(grid, dtype=int))
    return _solve_from_controls(grid.shape, controls)


def _sample_axis_positions(size: int, count: int) -> list[int]:
    assert size > count

    chosen: list[int] = []

    if count >= 2 and np.random.random() < 0.45:
        a = int(np.random.randint(0, size - 1))
        b = int(size - 1 - a)
        if a != b:
            chosen.extend([a, b])

    if len(chosen) < count and np.random.random() < 0.35:
        edge = int(np.random.choice([0, size - 1]))
        if edge not in chosen:
            chosen.append(edge)

    remaining = [idx for idx in range(size) if idx not in chosen]
    np.random.shuffle(remaining)
    chosen.extend(remaining[: count - len(chosen)])
    np.random.shuffle(chosen)
    return [int(x) for x in chosen[:count]]


def _sample_controls(h: int, w: int, count: int) -> list[tuple[int, int]]:
    rows = _sample_axis_positions(h, count)
    cols = _sample_axis_positions(w, count)

    base_cols = cols[:]
    for _ in range(12):
        np.random.shuffle(cols)
        if any(col != base for col, base in zip(cols, sorted(base_cols), strict=False)):
            break

    controls = list(zip(rows, cols, strict=False))
    return [(int(r), int(c)) for r, c in controls]


def _alternative_pairing(grid: np.ndarray) -> np.ndarray:
    controls = sorted(_extract_controls(grid))
    rows = [r for r, _c in controls]
    cols = [c for _r, c in controls]
    rotated_cols = cols[1:] + cols[:1]

    alt = np.zeros_like(grid)
    for r, c in zip(rows, rotated_cols, strict=False):
        alt[r, c] = BLUE
    return alt


def _validate_semantics(grid: np.ndarray) -> bool:
    controls = _extract_controls(grid)
    out = main(grid)

    h, w = grid.shape
    if out.shape != (2 * h, 2 * w):
        return False
    if max(out.shape) > 30:
        return False

    unique, counts = np.unique(out, return_counts=True)
    color_counts = {int(color): int(count) for color, count in zip(unique, counts, strict=False)}
    if set(color_counts) != {BLACK, BLUE, RED, GREEN}:
        return False

    count = len(controls)
    if color_counts[RED] != 4 * count:
        return False
    if color_counts[GREEN] != 4 * count * (count - 1):
        return False
    if color_counts[BLUE] <= 0 or color_counts[BLACK] <= 0:
        return False

    if np.array_equal(grid, np.fliplr(grid)) or np.array_equal(grid, np.flipud(grid)):
        return False

    alt = _alternative_pairing(grid)
    if np.array_equal(alt, grid):
        return False
    if np.array_equal(main(alt), out):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    count = int(np.random.choice([2, 3]))
    h = int(np.random.randint(count + 1, 8))
    w = int(np.random.randint(count + 1, 9))

    grid = np.zeros((h, w), dtype=int)
    for r, c in _sample_controls(h, w, count):
        grid[r, c] = BLUE

    if not _validate_semantics(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
