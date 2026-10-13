from __future__ import annotations

from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
ORANGE = 7
TEAL = 8

GEN_MAX_ATTEMPTS = 400
GEN_REGION_COUNT_RANGE = (2, 4)
GEN_REGION_SIZE_RANGE = (1, 5)
GEN_PADDING_RANGE = (1, 3)

_RNG = np.random.default_rng()


def _get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values, minlength=10).argmax())


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _has_adjacent_pair(values: np.ndarray, color: int) -> bool:
    vals = np.asarray(values, dtype=int)
    if vals.size < 2:
        return False
    return bool(np.any((vals[:-1] == int(color)) & (vals[1:] == int(color))))


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _detect_line_rows_cols(
    grid: np.ndarray,
    grid_bbox: tuple[int, int, int, int],
    *,
    line_color: int = BLUE,
) -> tuple[list[int], list[int]]:
    r0, c0, r1, c1 = grid_bbox
    line_rows = [
        r
        for r in range(r0, r1 + 1)
        if _has_adjacent_pair(grid[r, c0 : c1 + 1], line_color)
    ]
    line_cols = [
        c
        for c in range(c0, c1 + 1)
        if _has_adjacent_pair(grid[r0 : r1 + 1, c], line_color)
    ]
    return line_rows, line_cols


def _complete_lines(
    grid: np.ndarray,
    grid_bbox: tuple[int, int, int, int],
    line_rows: Iterable[int],
    line_cols: Iterable[int],
    *,
    bg: int,
) -> np.ndarray:
    r0, c0, r1, c1 = grid_bbox
    out = np.asarray(grid, dtype=int).copy()
    for r in line_rows:
        span = out[int(r), c0 : c1 + 1]
        span[span == bg] = BLUE
    for c in line_cols:
        span = out[r0 : r1 + 1, int(c)]
        span[span == bg] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the real blue grid lines, then recolor the unique smallest and largest enclosed black regions.

    Input structure:
    - A black-background grid contains one broken blue rectilinear grid, and every blue pixel lies inside one foreground bbox.
    - A row is a true horizontal grid line iff that row contains a horizontally adjacent blue pair.
    - A column is a true vertical grid line iff that column contains a vertically adjacent blue pair; isolated blue points may be mere intersections.

    Rule:
    - Find the foreground bbox of all blue pixels, detect every true horizontal row and vertical column, and fill their missing background cells blue across the whole bbox.
    - Treat the completed blue lines as walls and split the remaining black cells inside the bbox into 4-connected regions.
    - Recolor the unique smallest region orange and the unique largest region teal, leaving all other black cells and all blue line pixels unchanged.

    Key insight:
    - The broken input cannot be partitioned directly: only adjacent blue pairs prove which rows and columns are genuine grid lines, and the region areas become meaningful only after those lines are completed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    grid_bbox = _bbox(g == BLUE)
    if grid_bbox is None:
        return g.copy()

    line_rows, line_cols = _detect_line_rows_cols(g, grid_bbox, line_color=BLUE)
    assert line_rows and line_cols, "expected at least one proved horizontal and vertical line"

    out = _complete_lines(g, grid_bbox, line_rows, line_cols, bg=bg)

    r0, c0, r1, c1 = grid_bbox
    box = out[r0 : r1 + 1, c0 : c1 + 1]
    regions = _components4(box == bg)
    areas = np.array([coords.shape[0] for coords in regions], dtype=int)
    assert areas.size >= 2, "expected at least two enclosed regions"

    min_area = int(areas.min())
    max_area = int(areas.max())
    assert int(np.sum(areas == min_area)) == 1, "smallest region must be unique"
    assert int(np.sum(areas == max_area)) == 1, "largest region must be unique"

    smallest = regions[int(np.argmin(areas))]
    largest = regions[int(np.argmax(areas))]
    for rr, cc in smallest:
        box[int(rr), int(cc)] = ORANGE
    for rr, cc in largest:
        box[int(rr), int(cc)] = TEAL

    return out


def _make_line_layout(lengths: list[int]) -> tuple[list[int], list[list[int]], int]:
    line_positions = [0]
    cell_spans: list[list[int]] = []
    cursor = 0
    for length in lengths:
        start = cursor + 1
        stop = start + int(length)
        cell_spans.append(list(range(start, stop)))
        cursor = stop
        line_positions.append(cursor)
    return line_positions, cell_spans, cursor + 1


def _areas_have_unique_extremes(row_sizes: list[int], col_sizes: list[int]) -> bool:
    areas = np.outer(np.asarray(row_sizes, dtype=int), np.asarray(col_sizes, dtype=int))
    flat = areas.ravel()
    return (
        flat.size >= 4
        and int(flat.min()) != int(flat.max())
        and int(np.sum(flat == flat.min())) == 1
        and int(np.sum(flat == flat.max())) == 1
    )


def _valid_proof_state(
    grid: np.ndarray,
    full_bbox: tuple[int, int, int, int],
    line_rows: list[int],
    line_cols: list[int],
) -> bool:
    if _bbox(grid == BLUE) != full_bbox:
        return False
    r0, c0, r1, c1 = full_bbox
    for r in line_rows:
        if not _has_adjacent_pair(grid[int(r), c0 : c1 + 1], BLUE):
            return False
    for c in line_cols:
        if not _has_adjacent_pair(grid[r0 : r1 + 1, int(c)], BLUE):
            return False
    return True


def _count_black_regions(grid: np.ndarray, grid_bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = grid_bbox
    return len(_components4(grid[r0 : r1 + 1, c0 : c1 + 1] == BLACK))


def _is_isolated_blue(grid: np.ndarray, r: int, c: int) -> bool:
    if int(grid[r, c]) != BLUE:
        return False
    h, w = map(int, grid.shape)
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == BLUE:
            return False
    return True


def _try_remove_cells(
    grid: np.ndarray,
    coords: Iterable[tuple[int, int]],
    full_bbox: tuple[int, int, int, int],
    line_rows: list[int],
    line_cols: list[int],
) -> int:
    removed = 0
    for r, c in coords:
        if int(grid[r, c]) != BLUE:
            continue
        grid[r, c] = BLACK
        if _valid_proof_state(grid, full_bbox, line_rows, line_cols):
            removed += 1
        else:
            grid[r, c] = BLUE
    return removed


def _gen_attempt() -> np.ndarray | None:
    row_count = int(_RNG.integers(GEN_REGION_COUNT_RANGE[0], GEN_REGION_COUNT_RANGE[1] + 1))
    col_count = int(_RNG.integers(GEN_REGION_COUNT_RANGE[0], GEN_REGION_COUNT_RANGE[1] + 1))

    row_sizes = _RNG.integers(GEN_REGION_SIZE_RANGE[0], GEN_REGION_SIZE_RANGE[1] + 1, size=row_count).tolist()
    col_sizes = _RNG.integers(GEN_REGION_SIZE_RANGE[0], GEN_REGION_SIZE_RANGE[1] + 1, size=col_count).tolist()
    if not _areas_have_unique_extremes(row_sizes, col_sizes):
        return None

    local_line_rows, local_row_spans, box_h = _make_line_layout(row_sizes)
    local_line_cols, local_col_spans, box_w = _make_line_layout(col_sizes)

    pad_top = int(_RNG.integers(GEN_PADDING_RANGE[0], GEN_PADDING_RANGE[1] + 1))
    pad_bottom = int(_RNG.integers(GEN_PADDING_RANGE[0], GEN_PADDING_RANGE[1] + 1))
    pad_left = int(_RNG.integers(GEN_PADDING_RANGE[0], GEN_PADDING_RANGE[1] + 1))
    pad_right = int(_RNG.integers(GEN_PADDING_RANGE[0], GEN_PADDING_RANGE[1] + 1))

    height = pad_top + box_h + pad_bottom
    width = pad_left + box_w + pad_right
    if height > 30 or width > 30:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    r0 = pad_top
    c0 = pad_left
    r1 = r0 + box_h - 1
    c1 = c0 + box_w - 1

    abs_line_rows = [r0 + r for r in local_line_rows]
    abs_line_cols = [c0 + c for c in local_line_cols]

    grid[r0 : r1 + 1, abs_line_cols] = BLUE
    grid[abs_line_rows, c0 : c1 + 1] = BLUE

    full_bbox = (r0, c0, r1, c1)
    expected_region_count = row_count * col_count
    if _count_black_regions(grid, full_bbox) != expected_region_count:
        return None

    broken = grid.copy()
    interior_rows = [r0 + rr for span in local_row_spans for rr in span]
    interior_cols = [c0 + cc for span in local_col_spans for cc in span]
    internal_line_rows = abs_line_rows[1:-1]
    internal_line_cols = abs_line_cols[1:-1]

    mode = _RNG.choice(["h", "v", "both"], p=[0.25, 0.25, 0.50])
    forced_gap = False
    if mode in {"h", "both"}:
        chosen_rows = _RNG.choice(
            np.asarray(internal_line_rows, dtype=int),
            size=int(_RNG.integers(1, len(internal_line_rows) + 1)),
            replace=False,
        )
        for row in np.asarray(chosen_rows, dtype=int).tolist():
            gap_count = int(_RNG.integers(1, min(3, len(interior_cols)) + 1))
            gap_cols = _RNG.choice(np.asarray(interior_cols, dtype=int), size=gap_count, replace=False)
            broken[int(row), np.asarray(gap_cols, dtype=int)] = BLACK
            forced_gap = True
    if mode in {"v", "both"}:
        chosen_cols = _RNG.choice(
            np.asarray(internal_line_cols, dtype=int),
            size=int(_RNG.integers(1, len(internal_line_cols) + 1)),
            replace=False,
        )
        for col in np.asarray(chosen_cols, dtype=int).tolist():
            gap_count = int(_RNG.integers(1, min(3, len(interior_rows)) + 1))
            gap_rows = _RNG.choice(np.asarray(interior_rows, dtype=int), size=gap_count, replace=False)
            broken[np.asarray(gap_rows, dtype=int), int(col)] = BLACK
            forced_gap = True
    if not forced_gap:
        return None

    if not _valid_proof_state(broken, full_bbox, abs_line_rows, abs_line_cols):
        return None

    blue_coords = [tuple(map(int, rc)) for rc in np.argwhere(broken == BLUE)]
    _RNG.shuffle(blue_coords)
    extra_target = int(_RNG.integers(max(1, len(blue_coords) // 10), max(2, len(blue_coords) // 3) + 1))
    _try_remove_cells(broken, blue_coords[: extra_target * 3], full_bbox, abs_line_rows, abs_line_cols)

    if float(_RNG.random()) < 0.7:
        intersections = [(r, c) for r in abs_line_rows for c in abs_line_cols]
        _RNG.shuffle(intersections)
        target_isolated = int(_RNG.integers(1, min(4, len(intersections)) + 1))
        isolated = 0
        for r, c in intersections:
            if isolated >= target_isolated or int(broken[r, c]) != BLUE:
                continue
            neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
            removed_neighbors: list[tuple[int, int]] = []
            for nr, nc in neighbors:
                if r0 <= nr <= r1 and c0 <= nc <= c1 and int(broken[nr, nc]) == BLUE:
                    broken[nr, nc] = BLACK
                    removed_neighbors.append((nr, nc))
            if removed_neighbors and _valid_proof_state(broken, full_bbox, abs_line_rows, abs_line_cols) and _is_isolated_blue(broken, r, c):
                isolated += 1
            else:
                for nr, nc in removed_neighbors:
                    broken[nr, nc] = BLUE

    if not _valid_proof_state(broken, full_bbox, abs_line_rows, abs_line_cols):
        return None
    if np.array_equal(broken, grid):
        return None
    if _count_black_regions(broken, full_bbox) >= expected_region_count:
        return None

    line_rows, line_cols = _detect_line_rows_cols(broken, full_bbox, line_color=BLUE)
    if line_rows != abs_line_rows or line_cols != abs_line_cols:
        return None

    out = main(broken)
    if out.shape != broken.shape:
        return None
    if not np.any(out == ORANGE) or not np.any(out == TEAL):
        return None

    completed = _complete_lines(broken, full_bbox, abs_line_rows, abs_line_cols, bg=BLACK)
    if _count_black_regions(completed, full_bbox) != expected_region_count:
        return None

    return broken


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
