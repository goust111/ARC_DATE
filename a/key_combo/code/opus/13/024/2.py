from __future__ import annotations

import itertools

import numpy as np


BG = 0
LINE = 1
FG = 4

ORTHO_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

GEN_MIN_SIDE = 16
GEN_MAX_SIDE = 22
GEN_MIN_CENTERS = 4
GEN_MAX_CENTERS = 5
GEN_MIN_CENTER_GAP = 5
GEN_MIN_YELLOW = 40
GEN_MAX_YELLOW = 60
GEN_RETRY_LIMIT = 500


def _find_cross_centers(grid: np.ndarray) -> list[tuple[int, int]]:
    """Return all yellow cells whose four orthogonal neighbors are also yellow."""
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, height - 1):
        for c in range(1, width - 1):
            if int(g[r, c]) != FG:
                continue
            if all(int(g[r + dr, c + dc]) == FG for dr, dc in ORTHO_DIRS):
                centers.append((r, c))
    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect local yellow '+' centers, treat them as virtual anchors, and draw blue spans between anchors that share rows or columns.

    Input structure:
    - The grid is mostly black background.
    - Yellow pixels include several local '+' crosses mixed with unrelated yellow noise.
    - A valid anchor is not specially marked: it is any yellow cell whose up, down, left, and right neighbors are also yellow.

    Rule:
    - Scan the yellow pixels and collect every cross center.
    - For each pair of centers on the same row, fill the cells strictly between them with blue.
    - For each pair of centers on the same column, fill the cells strictly between them with blue.
    - Keep the yellow anchor cells themselves yellow, but allow blue spans to overwrite any other yellow cells they pass through.

    Key insight:
    - The meaningful endpoints are not given explicitly; they emerge only after local cross-pattern detection, and those detected centers drive the later span-filling step.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    centers = _find_cross_centers(g)
    out = g.copy()

    centers_by_row: dict[int, list[int]] = {}
    centers_by_col: dict[int, list[int]] = {}
    for r, c in centers:
        centers_by_row.setdefault(r, []).append(c)
        centers_by_col.setdefault(c, []).append(r)

    for r, cols in centers_by_row.items():
        cols = sorted(cols)
        for i, c0 in enumerate(cols):
            for c1 in cols[i + 1 :]:
                if c1 > c0 + 1:
                    out[r, c0 + 1 : c1] = LINE

    for c, rows in centers_by_col.items():
        rows = sorted(rows)
        for i, r0 in enumerate(rows):
            for r1 in rows[i + 1 :]:
                if r1 > r0 + 1:
                    out[r0 + 1 : r1, c] = LINE

    return out


def _paint_cross(grid: np.ndarray, center: tuple[int, int]) -> None:
    r, c = center
    grid[r, c] = FG
    for dr, dc in ORTHO_DIRS:
        grid[r + dr, c + dc] = FG


def _center_pairs(centers: list[tuple[int, int]]) -> tuple[int, int]:
    row_pairs = 0
    col_pairs = 0
    for (r0, c0), (r1, c1) in itertools.combinations(centers, 2):
        if r0 == r1:
            row_pairs += 1
        if c0 == c1:
            col_pairs += 1
    return row_pairs, col_pairs


def _corridor_cells(centers: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for (r0, c0), (r1, c1) in itertools.combinations(centers, 2):
        if r0 == r1:
            for c in range(min(c0, c1) + 1, max(c0, c1)):
                cells.add((r0, c))
        if c0 == c1:
            for r in range(min(r0, r1) + 1, max(r0, r1)):
                cells.add((r, c0))
    return sorted(cells)


def _protected_mask(shape: tuple[int, int], centers: list[tuple[int, int]]) -> np.ndarray:
    height, width = shape
    protected = np.zeros((height, width), dtype=bool)
    for r, c in centers:
        for rr in range(max(0, r - 2), min(height, r + 3)):
            for cc in range(max(0, c - 2), min(width, c + 3)):
                if abs(rr - r) + abs(cc - c) <= 2:
                    protected[rr, cc] = True
    return protected


def _valid_centers(centers: list[tuple[int, int]]) -> bool:
    rows: dict[int, int] = {}
    cols: dict[int, int] = {}
    for r, c in centers:
        rows[r] = rows.get(r, 0) + 1
        cols[c] = cols.get(c, 0) + 1
    if any(count > 2 for count in rows.values()):
        return False
    if any(count > 2 for count in cols.values()):
        return False
    return all(abs(r0 - r1) + abs(c0 - c1) >= GEN_MIN_CENTER_GAP for (r0, c0), (r1, c1) in itertools.combinations(centers, 2))


def _choose_centers(rng: np.random.Generator, height: int, width: int) -> list[tuple[int, int]] | None:
    max_row_gap = min(9, height - 3)
    max_col_gap = min(9, width - 3)
    row_gap_choices = [gap for gap in range(GEN_MIN_CENTER_GAP, max_row_gap + 1)]
    col_gap_choices = [gap for gap in range(GEN_MIN_CENTER_GAP, max_col_gap + 1)]
    if not row_gap_choices or not col_gap_choices:
        return None

    row_gap = int(rng.choice(row_gap_choices))
    col_gap = int(rng.choice(col_gap_choices))
    top = int(rng.integers(1, height - row_gap - 1))
    left = int(rng.integers(1, width - col_gap - 1))

    centers = [
        (top, left),
        (top, left + col_gap),
        (top + row_gap, left),
        (top + row_gap, left + col_gap),
    ]

    if rng.random() < 0.55:
        candidates = []
        used_rows = {r for r, _ in centers}
        used_cols = {c for _, c in centers}
        for r in range(1, height - 1):
            if r in used_rows:
                continue
            for c in range(1, width - 1):
                if c in used_cols:
                    continue
                candidate = (r, c)
                if all(abs(r - rr) + abs(c - cc) >= GEN_MIN_CENTER_GAP for rr, cc in centers):
                    candidates.append(candidate)
        if candidates:
            centers.append(candidates[int(rng.integers(0, len(candidates)))])

    if not _valid_centers(centers):
        return None
    row_pairs, col_pairs = _center_pairs(centers)
    if row_pairs < 2 or col_pairs < 1:
        return None
    return centers


def _try_add_shape(
    grid: np.ndarray,
    centers: list[tuple[int, int]],
    protected: np.ndarray,
    offsets: list[tuple[int, int]],
    rng: np.random.Generator,
    require_corridor: bool = False,
) -> bool:
    height, width = grid.shape
    corridor = set(_corridor_cells(centers))

    anchors = [(r, c) for r in range(height) for c in range(width)]
    rng.shuffle(anchors)
    for base_r, base_c in anchors:
        cells = []
        touches_corridor = False
        ok = True
        for dr, dc in offsets:
            r = base_r + dr
            c = base_c + dc
            if not (0 <= r < height and 0 <= c < width):
                ok = False
                break
            if protected[r, c] or int(grid[r, c]) != BG:
                ok = False
                break
            if (r, c) in corridor:
                touches_corridor = True
            cells.append((r, c))
        if not ok or (require_corridor and not touches_corridor):
            continue

        grid[[r for r, _ in cells], [c for _, c in cells]] = FG
        if sorted(_find_cross_centers(grid)) == sorted(centers):
            return True
        grid[[r for r, _ in cells], [c for _, c in cells]] = BG
    return False


def _add_noise(grid: np.ndarray, centers: list[tuple[int, int]], rng: np.random.Generator) -> bool:
    target_total = int(rng.integers(GEN_MIN_YELLOW, GEN_MAX_YELLOW + 1))
    protected = _protected_mask(grid.shape, centers)

    shapes: list[list[tuple[int, int]]] = [
        [(0, 0)],
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0)],
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 1)],
        [(0, 0), (1, 0), (1, -1)],
    ]

    extra_corridor = int(rng.integers(1, 4))
    for _ in range(extra_corridor):
        if int(np.count_nonzero(grid == FG)) >= target_total:
            break
        if not _try_add_shape(grid, centers, protected, [(0, 0)], rng, require_corridor=True):
            break

    attempts = 0
    while int(np.count_nonzero(grid == FG)) < target_total and attempts < 800:
        attempts += 1
        remaining = target_total - int(np.count_nonzero(grid == FG))
        valid_shapes = [shape for shape in shapes if len(shape) <= remaining]
        if not valid_shapes:
            break
        shape = valid_shapes[int(rng.integers(0, len(valid_shapes)))]
        if not _try_add_shape(grid, centers, protected, shape, rng):
            if remaining <= 3:
                if not _try_add_shape(grid, centers, protected, [(0, 0)], rng):
                    break
            continue

    return int(np.count_nonzero(grid == FG)) == target_total


def _is_valid_instance(inp: np.ndarray, centers: list[tuple[int, int]]) -> bool:
    found = sorted(_find_cross_centers(inp))
    if found != sorted(centers):
        return False

    count_yellow = int(np.count_nonzero(inp == FG))
    if not (GEN_MIN_YELLOW <= count_yellow <= GEN_MAX_YELLOW):
        return False

    row_pairs, col_pairs = _center_pairs(found)
    if row_pairs < 2 or col_pairs < 1:
        return False

    out = main(inp)
    if out.shape != inp.shape:
        return False
    if np.array_equal(out, inp):
        return False
    if int(np.count_nonzero(out == LINE)) < 6:
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    height = int(rng.integers(GEN_MIN_SIDE, GEN_MAX_SIDE + 1))
    width = int(rng.integers(GEN_MIN_SIDE, GEN_MAX_SIDE + 1))

    centers = _choose_centers(rng, height, width)
    if centers is None or not (GEN_MIN_CENTERS <= len(centers) <= GEN_MAX_CENTERS):
        return None

    grid = np.full((height, width), BG, dtype=int)
    for center in centers:
        _paint_cross(grid, center)

    if not _add_noise(grid, centers, rng):
        return None
    if not _is_valid_instance(grid, centers):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cross-anchor ARC input")
