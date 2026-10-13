from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4

GEN_RETRY_LIMIT = 400


def _connect_red_spans(grid: np.ndarray) -> np.ndarray:
    """Fill red row/column spans using only the original red markers as endpoints."""
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    markers = g == RED

    for r in range(g.shape[0]):
        cols = np.flatnonzero(markers[r])
        if cols.size >= 2:
            out[r, cols[0] : cols[-1] + 1] = RED

    for c in range(g.shape[1]):
        rows = np.flatnonzero(markers[:, c])
        if rows.size >= 2:
            out[rows[0] : rows[-1] + 1, c] = RED

    return out


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)

    return comps


def _extract_enclosed_regions(filled: np.ndarray) -> list[tuple[int, int, int, int]]:
    red_box = _bbox(filled == RED)
    assert red_box is not None, "expected at least one red marker"
    r0, c0, r1, c1 = red_box
    sub = filled[r0 : r1 + 1, c0 : c1 + 1]
    open_mask = sub != RED
    regions: list[tuple[int, int, int, int]] = []

    for comp in _components(open_mask):
        rows = [rc[0] for rc in comp]
        cols = [rc[1] for rc in comp]
        rr0, rr1 = min(rows), max(rows)
        cc0, cc1 = min(cols), max(cols)
        if rr0 == 0 or rr1 == sub.shape[0] - 1 or cc0 == 0 or cc1 == sub.shape[1] - 1:
            continue

        area = (rr1 - rr0 + 1) * (cc1 - cc0 + 1)
        assert len(comp) == area, "each enclosed region must be a solid rectangle"
        for rr in range(rr0, rr1 + 1):
            for cc in range(cc0, cc1 + 1):
                assert open_mask[rr, cc], "enclosed region bbox must be fully open"

        regions.append((r0 + rr0, c0 + cc0, r0 + rr1, c0 + cc1))

    assert regions, "expected at least one enclosed region"
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect red marker spans to create an implicit partition, then compress each enclosed region by blue-marker presence.

    Input structure:
    - A black background contains sparse red marker pixels, sparse blue marker pixels, and optional yellow distractor pixels.
    - Red markers align so that connecting their same-row and same-column spans creates a grid-like separator structure.
    - Blue markers lie inside the enclosed rectangular regions defined by that red structure; yellow pixels are irrelevant noise.

    Rule:
    - Using the original red markers as endpoints, fill every row span between the leftmost and rightmost red markers on that row.
    - Using the same original red markers, fill every column span between the topmost and bottommost red markers in that column.
    - Treat the resulting red lines as separators, ignore any open area touching the outer red bounding box, and keep only the fully enclosed rectangular regions.
    - Output one cell per enclosed region, ordered by the regions' spatial rows and columns: blue if the region contains a blue marker, otherwise black.

    Key insight:
    - The red pixels are not output objects; they define the hidden coordinate system whose enclosed cells determine how blue-marker occupancy is compressed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    filled = _connect_red_spans(g)
    regions = _extract_enclosed_regions(filled)

    row_starts = sorted({box[0] for box in regions})
    col_starts = sorted({box[1] for box in regions})
    assert len(regions) == len(row_starts) * len(col_starts), "regions must form a full rectangular grid"

    row_index = {r: i for i, r in enumerate(row_starts)}
    col_index = {c: j for j, c in enumerate(col_starts)}
    out = np.zeros((len(row_starts), len(col_starts)), dtype=int)

    for r0, c0, r1, c1 in regions:
        rr = row_index[r0]
        cc = col_index[c0]
        region = g[r0 : r1 + 1, c0 : c1 + 1]
        out[rr, cc] = BLUE if np.any(region == BLUE) else BLACK

    return out


def _sample_axis(
    rng: np.random.Generator,
    line_count: int,
    endpoint_count: int,
) -> tuple[list[int], list[int], list[int], list[int], int]:
    for _ in range(200):
        start_pad = int(rng.integers(0, 2))
        end_pad = int(rng.integers(0, 2))
        start_gap = int(rng.integers(1, 3))
        end_gap = int(rng.integers(1, 3))
        region_sizes = [int(rng.integers(1, 4)) for _ in range(line_count - 1)]

        endpoints_before = [start_pad + i for i in range(endpoint_count)]
        cursor = start_pad + endpoint_count + start_gap
        lines = [cursor]
        for size in region_sizes:
            cursor += 1 + size
            lines.append(cursor)

        after_start = cursor + 1 + end_gap
        endpoints_after = [after_start + i for i in range(endpoint_count)]
        total = endpoints_after[-1] + 1 + end_pad
        if 12 <= total <= 30:
            return lines, endpoints_before, endpoints_after, region_sizes, total

    raise RuntimeError("failed to sample a valid axis layout")


def _region_boxes(
    row_lines: list[int],
    col_lines: list[int],
) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for i in range(len(row_lines) - 1):
        for j in range(len(col_lines) - 1):
            boxes.append((row_lines[i] + 1, col_lines[j] + 1, row_lines[i + 1] - 1, col_lines[j + 1] - 1))
    return boxes


def _box_lookup(
    row_lines: list[int],
    col_lines: list[int],
) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    out: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for i in range(len(row_lines) - 1):
        for j in range(len(col_lines) - 1):
            out[(i, j)] = (row_lines[i] + 1, col_lines[j] + 1, row_lines[i + 1] - 1, col_lines[j + 1] - 1)
    return out


def _place_free_cells(
    rng: np.random.Generator,
    grid: np.ndarray,
    color: int,
    count: int,
) -> None:
    if count == 0:
        return
    free = np.argwhere(grid == BLACK)
    assert free.shape[0] >= count, "not enough free cells for distractors"
    picks = rng.choice(free.shape[0], size=count, replace=False)
    for idx in np.atleast_1d(picks):
        r, c = free[int(idx)]
        grid[int(r), int(c)] = color


def _validate_instance(
    grid: np.ndarray,
    expected: np.ndarray,
    row_lines: list[int],
    col_lines: list[int],
) -> None:
    red_mask = grid == RED
    row_counts = np.count_nonzero(red_mask, axis=1)
    col_counts = np.count_nonzero(red_mask, axis=0)
    assert row_counts.max(initial=0) <= 2, "original red markers must not exceed two per row"
    assert col_counts.max(initial=0) <= 2, "original red markers must not exceed two per col"
    assert np.count_nonzero(row_counts == 2) == len(row_lines), "horizontal separators must be uniquely encoded"
    assert np.count_nonzero(col_counts == 2) == len(col_lines), "vertical separators must be uniquely encoded"

    filled = _connect_red_spans(grid)
    actual_regions = sorted(_extract_enclosed_regions(filled))
    expected_regions = sorted(_region_boxes(row_lines, col_lines))
    assert actual_regions == expected_regions, "red spans must induce the intended region grid"

    actual = main(grid)
    assert np.array_equal(actual, expected), "solver output must match designed occupancy"
    assert np.any(actual == BLUE) and np.any(actual == BLACK), "need both occupied and empty regions"

    blue_cells = np.argwhere(grid == BLUE)
    assert 3 <= blue_cells.shape[0] <= 6, "blue marker count must stay in range"
    seen_regions: set[tuple[int, int]] = set()
    for r, c in blue_cells:
        matches = [
            (i, j)
            for i in range(expected.shape[0])
            for j in range(expected.shape[1])
            if row_lines[i] < int(r) < row_lines[i + 1] and col_lines[j] < int(c) < col_lines[j + 1]
        ]
        assert len(matches) == 1, "each blue marker must belong to exactly one enclosed region"
        assert matches[0] not in seen_regions, "each enclosed region may contain at most one blue marker"
        seen_regions.add(matches[0])

    stripped = grid.copy()
    stripped[stripped == YELLOW] = BLACK
    assert np.array_equal(main(grid), main(stripped)), "yellow distractors must not affect the output"


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    region_rows = int(rng.integers(2, 5))
    region_cols = int(rng.integers(2, 5))
    horizontal_lines = region_rows + 1
    vertical_lines = region_cols + 1

    row_lines, top_rows, bottom_rows, _row_sizes, height = _sample_axis(rng, horizontal_lines, vertical_lines)
    col_lines, left_cols, right_cols, _col_sizes, width = _sample_axis(rng, vertical_lines, horizontal_lines)

    grid = np.zeros((height, width), dtype=int)
    left_perm = list(rng.permutation(left_cols))
    right_perm = list(rng.permutation(right_cols))
    top_perm = list(rng.permutation(top_rows))
    bottom_perm = list(rng.permutation(bottom_rows))

    for i, row in enumerate(row_lines):
        grid[row, left_perm[i]] = RED
        grid[row, right_perm[i]] = RED

    for j, col in enumerate(col_lines):
        grid[top_perm[j], col] = RED
        grid[bottom_perm[j], col] = RED

    total_regions = region_rows * region_cols
    blue_count = int(rng.integers(3, min(6, total_regions - 1) + 1))
    chosen = rng.choice(total_regions, size=blue_count, replace=False)
    chosen_set = {int(idx) for idx in np.atleast_1d(chosen)}

    expected = np.zeros((region_rows, region_cols), dtype=int)
    boxes = _box_lookup(row_lines, col_lines)
    for flat in range(total_regions):
        if flat not in chosen_set:
            continue
        rr = flat // region_cols
        cc = flat % region_cols
        r0, c0, r1, c1 = boxes[(rr, cc)]
        br = int(rng.integers(r0, r1 + 1))
        bc = int(rng.integers(c0, c1 + 1))
        grid[br, bc] = BLUE
        expected[rr, cc] = BLUE

    yellow_count = int(rng.integers(0, 3))
    _place_free_cells(rng, grid, YELLOW, yellow_count)

    try:
        _validate_instance(grid, expected, row_lines, col_lines)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        rng = np.random.default_rng()
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRY_LIMIT} attempts")
