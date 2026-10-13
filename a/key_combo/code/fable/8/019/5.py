from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

MARKER_COLORS = [BLUE, RED, GREEN, YELLOW, PINK, ORANGE, TEAL, MAROON]

BLOCK_SIZE = 2
GUTTER_SIZE = 1
STEP = BLOCK_SIZE + GUTTER_SIZE
MIN_BANDS = 2
MAX_BANDS = 6
MAX_MARGIN = 6
GEN_RETRIES = 400


def _true_intervals(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(mask_1d.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            intervals.append((start, idx - 1))
            start = None
    if start is not None:
        intervals.append((start, len(mask_1d) - 1))
    return intervals


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert len(coords) > 0, "expected a nonempty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _lattice_structure(grid: np.ndarray) -> dict:
    grey_mask = grid == GREY
    assert grey_mask.any(), "expected grey lattice blocks"

    row_bands = _true_intervals(grey_mask.any(axis=1))
    col_bands = _true_intervals(grey_mask.any(axis=0))
    assert len(row_bands) >= 2 and len(col_bands) >= 2, "expected a block array with gutters in both axes"
    assert all(end - start + 1 == BLOCK_SIZE for start, end in row_bands), "row bands must be 2 pixels tall"
    assert all(end - start + 1 == BLOCK_SIZE for start, end in col_bands), "column bands must be 2 pixels wide"

    row_gutters = np.zeros(grid.shape[0], dtype=bool)
    col_gutters = np.zeros(grid.shape[1], dtype=bool)
    for (_prev_start, prev_end), (next_start, _next_end) in zip(row_bands, row_bands[1:]):
        assert next_start - prev_end - 1 == GUTTER_SIZE, "row gutters must be 1 pixel wide"
        row_gutters[prev_end + 1 : next_start] = True
    for (_prev_start, prev_end), (next_start, _next_end) in zip(col_bands, col_bands[1:]):
        assert next_start - prev_end - 1 == GUTTER_SIZE, "column gutters must be 1 pixel wide"
        col_gutters[prev_end + 1 : next_start] = True

    row_band_mask = np.zeros(grid.shape[0], dtype=bool)
    col_band_mask = np.zeros(grid.shape[1], dtype=bool)
    for start, end in row_bands:
        row_band_mask[start : end + 1] = True
    for start, end in col_bands:
        col_band_mask[start : end + 1] = True

    expected_blocks = row_band_mask[:, None] & col_band_mask[None, :]
    assert np.array_equal(grey_mask, expected_blocks), "grey pixels must form a complete rectangular 2x2 block lattice"

    r0, c0, r1, c1 = _bbox(grey_mask)
    return {
        "bbox": (r0, c0, r1, c1),
        "row_gutters": row_gutters,
        "col_gutters": col_gutters,
    }


def _collect_markers(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    row_gutters: np.ndarray,
    col_gutters: np.ndarray,
) -> tuple[dict[int, tuple[int, int]], dict[int, tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    row_markers: dict[int, tuple[int, int]] = {}
    col_markers: dict[int, tuple[int, int]] = {}

    marker_coords = np.argwhere((grid != BLACK) & (grid != GREY))
    assert len(marker_coords) > 0, "expected at least one colored edge marker"

    for r, c in marker_coords:
        rr = int(r)
        cc = int(c)
        color = int(grid[rr, cc])
        on_row_extension = bool(row_gutters[rr]) and (cc < c0 or cc > c1)
        on_col_extension = bool(col_gutters[cc]) and (rr < r0 or rr > r1)
        assert on_row_extension ^ on_col_extension, "each marker must align to exactly one gutter extension"
        if on_row_extension:
            assert rr not in row_markers, "each gutter row may have at most one marker"
            row_markers[rr] = (color, cc)
        else:
            assert cc not in col_markers, "each gutter column may have at most one marker"
            col_markers[cc] = (color, rr)
    return row_markers, col_markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint lattice gutters from edge-aligned markers, with each colored intersection won by the closer marker along its gutter.

    Input structure:
    - A black background contains a complete rectangular array of grey 2x2 blocks.
    - Adjacent block bands are separated by 1-pixel-wide empty gutter rows and columns.
    - Outside the lattice bounding box, some gutter-row or gutter-column extensions carry one colored single-pixel marker.

    Rule:
    - Keep the grey blocks and all marker pixels unchanged.
    - Inside the lattice bounding box, paint every marked gutter row or marked gutter column with that marker's color.
    - At a pixel belonging to both a marked row gutter and a marked column gutter, compare the axis-aligned distances to the two markers and keep the closer color.
    - If only one of the two crossing gutters has a marker, use that color; unmarked gutters stay black.

    Key insight:
    - The markers do not point to blocks; they color the internal gutter lines aligned with them, and crossings are resolved locally by nearest-marker distance rather than by a fixed paint priority.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    structure = _lattice_structure(g)
    r0, c0, r1, c1 = structure["bbox"]
    row_gutters = structure["row_gutters"]
    col_gutters = structure["col_gutters"]
    row_markers, col_markers = _collect_markers(g, (r0, c0, r1, c1), row_gutters, col_gutters)

    out = g.copy()
    for r in range(r0, r1 + 1):
        row_info = row_markers.get(r) if row_gutters[r] else None
        for c in range(c0, c1 + 1):
            if g[r, c] != BLACK:
                continue
            col_info = col_markers.get(c) if col_gutters[c] else None
            if row_info is not None and col_info is not None:
                row_color, row_marker_c = row_info
                col_color, col_marker_r = col_info
                row_dist = abs(c - row_marker_c)
                col_dist = abs(r - col_marker_r)
                assert row_dist != col_dist, "equal-distance gutter conflicts are intentionally excluded"
                out[r, c] = row_color if row_dist < col_dist else col_color
            elif row_info is not None:
                out[r, c] = row_info[0]
            elif col_info is not None:
                out[r, c] = col_info[0]
    return out


def _render_lattice(
    n_block_rows: int,
    n_block_cols: int,
    top: int,
    left: int,
    bottom: int,
    right: int,
) -> tuple[np.ndarray, list[int], list[int], tuple[int, int, int, int]]:
    bbox_h = n_block_rows * BLOCK_SIZE + (n_block_rows - 1) * GUTTER_SIZE
    bbox_w = n_block_cols * BLOCK_SIZE + (n_block_cols - 1) * GUTTER_SIZE
    grid = np.full((top + bbox_h + bottom, left + bbox_w + right), BLACK, dtype=int)

    row_starts = [top + STEP * idx for idx in range(n_block_rows)]
    col_starts = [left + STEP * idx for idx in range(n_block_cols)]
    for rs in row_starts:
        for cs in col_starts:
            grid[rs : rs + BLOCK_SIZE, cs : cs + BLOCK_SIZE] = GREY

    row_gutters = [row_starts[idx] + BLOCK_SIZE for idx in range(n_block_rows - 1)]
    col_gutters = [col_starts[idx] + BLOCK_SIZE for idx in range(n_block_cols - 1)]
    bbox = (top, left, top + bbox_h - 1, left + bbox_w - 1)
    return grid, row_gutters, col_gutters, bbox


def _marker_counts(n_rows: int, n_cols: int) -> tuple[int, int]:
    mode = random.random()
    if mode < 0.72:
        return random.randint(1, n_rows), random.randint(1, n_cols)
    if mode < 0.86:
        return random.randint(1, n_rows), 0
    return 0, random.randint(1, n_cols)


def _assign_marker_colors(total: int) -> list[int]:
    if total <= len(MARKER_COLORS):
        return random.sample(MARKER_COLORS, total)
    return [random.choice(MARKER_COLORS) for _ in range(total)]


def _place_markers(
    grid: np.ndarray,
    row_gutters: list[int],
    col_gutters: list[int],
) -> tuple[dict[int, tuple[int, int]], dict[int, tuple[int, int]]]:
    row_count, col_count = _marker_counts(len(row_gutters), len(col_gutters))
    chosen_rows = random.sample(row_gutters, row_count)
    chosen_cols = random.sample(col_gutters, col_count)
    colors = _assign_marker_colors(row_count + col_count)
    random.shuffle(colors)

    row_markers: dict[int, tuple[int, int]] = {}
    col_markers: dict[int, tuple[int, int]] = {}
    color_idx = 0

    for r in chosen_rows:
        color = colors[color_idx]
        color_idx += 1
        c = 0 if random.random() < 0.5 else grid.shape[1] - 1
        grid[r, c] = color
        row_markers[r] = (color, c)

    for c in chosen_cols:
        color = colors[color_idx]
        color_idx += 1
        r = 0 if random.random() < 0.5 else grid.shape[0] - 1
        grid[r, c] = color
        col_markers[c] = (color, r)

    return row_markers, col_markers


def _has_equal_distance_tie(
    row_markers: dict[int, tuple[int, int]],
    col_markers: dict[int, tuple[int, int]],
) -> bool:
    for r, (_row_color, row_marker_c) in row_markers.items():
        for c, (_col_color, col_marker_r) in col_markers.items():
            if abs(c - row_marker_c) == abs(r - col_marker_r):
                return True
    return False


def _gen_attempt() -> np.ndarray | None:
    n_block_rows = random.randint(MIN_BANDS, MAX_BANDS)
    n_block_cols = random.randint(MIN_BANDS, MAX_BANDS)
    bbox_h = n_block_rows * BLOCK_SIZE + (n_block_rows - 1) * GUTTER_SIZE
    bbox_w = n_block_cols * BLOCK_SIZE + (n_block_cols - 1) * GUTTER_SIZE

    if bbox_h + 2 > 30 or bbox_w + 2 > 30:
        return None

    top = random.randint(1, MAX_MARGIN)
    bottom = random.randint(1, MAX_MARGIN)
    left = random.randint(1, MAX_MARGIN)
    right = random.randint(1, MAX_MARGIN)
    if bbox_h + top + bottom > 30 or bbox_w + left + right > 30:
        return None

    grid, row_gutters, col_gutters, bbox = _render_lattice(n_block_rows, n_block_cols, top, left, bottom, right)
    row_markers, col_markers = _place_markers(grid, row_gutters, col_gutters)
    if not row_markers and not col_markers:
        return None
    if row_markers and col_markers and _has_equal_distance_tie(row_markers, col_markers):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None

    r0, c0, r1, c1 = bbox
    painted = out[r0 : r1 + 1, c0 : c1 + 1]
    if not np.any((painted != BLACK) & (painted != GREY)):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gutter-marker lattice")
