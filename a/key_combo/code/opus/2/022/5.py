from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask


BLACK = Color.BLACK
BLUE = Color.BLUE
RED = Color.RED
GREEN = Color.GREEN
YELLOW = Color.YELLOW
GREY = Color.GREY
TEAL = Color.TEAL
PINK = Color.PINK

FILL_BY_COUNT = {
    0: BLACK,
    1: YELLOW,
    2: GREEN,
    3: TEAL,
    4: PINK,
}

MARKER_ARMS = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_RETRIES = 300
GEN_LAYOUT_RETRIES = 4000


def _marker_centers(grid: np.ndarray) -> list[tuple[int, int]]:
    centers: list[tuple[int, int]] = []
    for coords in components_mask(grid == RED, connectivity=8):
        r0, c0, r1, c1 = bbox_coords(coords)
        assert coords.shape[0] == 4, "each marker must contain exactly four red pixels"
        assert (r1 - r0, c1 - c0) == (2, 2), "each marker must fit in a 3x3 box"

        cr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        expected = {(cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)}
        assert set(map(tuple, coords)) == expected, "red object must be a hollow plus marker"
        assert int(grid[cr, cc]) == BLACK, "marker center must be background"
        centers.append((int(cr), int(cc)))

    assert 4 <= len(centers) <= 8, "expected four to eight markers"
    return sorted(centers)


def _line_indices_from_grey(grid: np.ndarray, axis: int) -> list[int]:
    lattice_counts = np.count_nonzero(grid != BLACK, axis=axis)
    positive = lattice_counts[lattice_counts > 0]
    assert positive.size > 0, "expected a grey lattice skeleton"

    threshold = float(np.median(positive))
    lines = [int(i) for i, count in enumerate(lattice_counts) if float(count) > threshold]
    assert len(lines) >= 4, "expected multiple lattice lines"

    steps = np.diff(lines)
    assert steps.size > 0 and np.all(steps == steps[0]), "lattice lines must be regularly spaced"
    assert int(steps[0]) >= 5, "chambers must leave visible interior"
    return lines


def _paint_connector_segment(
    out: np.ndarray,
    original: np.ndarray,
    fixed_axis: int,
    start: int,
    end: int,
    *,
    horizontal: bool,
) -> int:
    painted = 0
    if horizontal:
        for c in range(start + 2, end - 1):
            if int(original[fixed_axis, c]) == BLACK:
                out[fixed_axis, c] = BLUE
                painted += 1
    else:
        for r in range(start + 2, end - 1):
            if int(original[r, fixed_axis]) == BLACK:
                out[r, fixed_axis] = BLUE
                painted += 1
    return painted


def _connect_markers(grid: np.ndarray, centers: list[tuple[int, int]]) -> np.ndarray:
    out = grid.copy()
    rows: dict[int, list[int]] = defaultdict(list)
    cols: dict[int, list[int]] = defaultdict(list)
    for r, c in centers:
        rows[r].append(c)
        cols[c].append(r)

    painted = 0
    for r, row_cols in rows.items():
        ordered = sorted(row_cols)
        for c0, c1 in zip(ordered, ordered[1:]):
            painted += _paint_connector_segment(out, grid, r, int(c0), int(c1), horizontal=True)

    for c, col_rows in cols.items():
        ordered = sorted(col_rows)
        for r0, r1 in zip(ordered, ordered[1:]):
            painted += _paint_connector_segment(out, grid, c, int(r0), int(r1), horizontal=False)

    assert painted > 0, "expected at least one visible blue connector pixel"
    return out


def _fill_chambers(grid: np.ndarray, row_lines: list[int], col_lines: list[int]) -> np.ndarray:
    out = grid.copy()
    for r0, r1 in zip(row_lines, row_lines[1:]):
        for c0, c1 in zip(col_lines, col_lines[1:]):
            top = np.any(out[r0, c0 + 1 : c1] == BLUE)
            bottom = np.any(out[r1, c0 + 1 : c1] == BLUE)
            left = np.any(out[r0 + 1 : r1, c0] == BLUE)
            right = np.any(out[r0 + 1 : r1, c1] == BLUE)
            count = int(top) + int(bottom) + int(left) + int(right)
            fill = FILL_BY_COUNT[count]
            if fill != BLACK:
                interior = out[r0 + 1 : r1, c0 + 1 : c1]
                interior[interior == BLACK] = fill
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect aligned hollow-plus anchors on the lattice and fill each chamber by its number of covered sides.

    Input structure:
    - The grid has a black background and a regular grey lattice skeleton that partitions the canvas into rectangular chambers.
    - Four to eight red hollow-plus markers sit on lattice intersections; each marker is a 3x3 cross with only the four arm pixels present.
    - Chamber interiors are black, and no other foreground objects are present.

    Rule:
    - Detect every red hollow-plus marker and use its empty center as an anchor.
    - For each row and each column, connect adjacent anchors by painting blue only onto background gaps along the lattice segment between their opposing arms.
    - For every chamber, count how many of its top, bottom, left, and right lattice edges contain at least one blue connector pixel.
    - Fill the chamber interior by that edge-count code: 1->yellow, 2->green, 3->teal, 4->pink, while 0 stays black.

    Key insight:
    - The output color of a chamber is determined by connector topology on its four boundary edges, not by the marker count alone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    centers = _marker_centers(g)
    row_lines = _line_indices_from_grey(g, axis=1)
    col_lines = _line_indices_from_grey(g, axis=0)

    center_rows = {r for r, _c in centers}
    center_cols = {c for _r, c in centers}
    assert center_rows.issubset(set(row_lines)), "marker centers must lie on lattice rows"
    assert center_cols.issubset(set(col_lines)), "marker centers must lie on lattice columns"

    connected = _connect_markers(g, centers)
    return _fill_chambers(connected, row_lines, col_lines)


def _coverage_from_anchor_mask(anchor_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, int, int, int]]]:
    m = anchor_mask.shape[0] - 1
    n = anchor_mask.shape[1] - 1
    horizontal = np.zeros((m + 1, n), dtype=int)
    vertical = np.zeros((m, n + 1), dtype=int)
    intervals: list[tuple[str, int, int, int]] = []

    for r in range(m + 1):
        cols = np.flatnonzero(anchor_mask[r])
        for c0, c1 in zip(cols, cols[1:]):
            horizontal[r, int(c0) : int(c1)] = 1
            intervals.append(("H", int(r), int(c0), int(c1)))

    for c in range(n + 1):
        rows = np.flatnonzero(anchor_mask[:, c])
        for r0, r1 in zip(rows, rows[1:]):
            vertical[int(r0) : int(r1), c] = 1
            intervals.append(("V", int(c), int(r0), int(r1)))

    counts = horizontal[:-1, :] + horizontal[1:, :] + vertical[:, :-1] + vertical[:, 1:]
    return horizontal, vertical, counts, intervals


def _corner_marker_counts(anchor_mask: np.ndarray) -> np.ndarray:
    return (
        anchor_mask[:-1, :-1].astype(int)
        + anchor_mask[1:, :-1].astype(int)
        + anchor_mask[:-1, 1:].astype(int)
        + anchor_mask[1:, 1:].astype(int)
    )


def _endpoint_pair_counts(anchor_mask: np.ndarray) -> np.ndarray:
    top = anchor_mask[:-1, :-1] & anchor_mask[:-1, 1:]
    bottom = anchor_mask[1:, :-1] & anchor_mask[1:, 1:]
    left = anchor_mask[:-1, :-1] & anchor_mask[1:, :-1]
    right = anchor_mask[:-1, 1:] & anchor_mask[1:, 1:]
    return top.astype(int) + bottom.astype(int) + left.astype(int) + right.astype(int)


def _line_presence_counts(anchor_mask: np.ndarray) -> np.ndarray:
    row_has = anchor_mask.any(axis=1).astype(int)
    col_has = anchor_mask.any(axis=0).astype(int)
    return row_has[:-1, None] + row_has[1:, None] + col_has[None, :-1] + col_has[None, 1:]


def _layout_is_semantically_good(anchor_mask: np.ndarray) -> bool:
    horizontal, vertical, counts, intervals = _coverage_from_anchor_mask(anchor_mask)
    flat = counts.ravel()
    if set(map(int, flat)) != {0, 1, 2, 3, 4}:
        return False
    if not any(kind == "H" for kind, _idx, _a, _b in intervals):
        return False
    if not any(kind == "V" for kind, _idx, _a, _b in intervals):
        return False
    if not any((end - start) > 1 for _kind, _idx, start, end in intervals):
        return False

    if not np.any(flat != _corner_marker_counts(anchor_mask).ravel()):
        return False
    if not np.any(flat != _endpoint_pair_counts(anchor_mask).ravel()):
        return False
    if not np.any(flat != _line_presence_counts(anchor_mask).ravel()):
        return False

    # Require at least one chamber whose true count needs both horizontal and vertical evidence.
    if not np.any((horizontal[:-1, :] + horizontal[1:, :]) > 0):
        return False
    if not np.any((vertical[:, :-1] + vertical[:, 1:]) > 0):
        return False
    if not np.any(
        ((horizontal[:-1, :] + horizontal[1:, :]) > 0)
        & ((vertical[:, :-1] + vertical[:, 1:]) > 0)
    ):
        return False
    return True


def _random_anchor_mask(m: int, n: int) -> np.ndarray:
    nodes = [(r, c) for r in range(m + 1) for c in range(n + 1)]
    max_markers = min(8, len(nodes))
    for _ in range(GEN_LAYOUT_RETRIES):
        count = random.randint(4, max_markers)
        chosen = random.sample(nodes, count)
        mask = np.zeros((m + 1, n + 1), dtype=bool)
        for r, c in chosen:
            mask[r, c] = True
        if _layout_is_semantically_good(mask):
            return mask
    raise RuntimeError("failed to find a non-degenerate anchor layout")


def _gap_positions(start: int, end: int, offset: int) -> list[int]:
    gaps = list(range(start + offset, end - 1, 3))
    if gaps:
        return gaps

    fallback = start + (end - start) // 2
    fallback = min(max(start + 2, fallback), end - 2)
    return [int(fallback)]


def _render_input(
    row_lines: list[int],
    col_lines: list[int],
    anchor_mask: np.ndarray,
    *,
    height: int,
    width: int,
    row_gap_offset: int,
    col_gap_offset: int,
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)

    for r in row_lines:
        for c0, c1 in zip(col_lines, col_lines[1:]):
            grid[r, c0 : c1 + 1] = GREY
            for gap in _gap_positions(int(c0), int(c1), row_gap_offset):
                grid[r, gap] = BLACK

    for c in col_lines:
        for r0, r1 in zip(row_lines, row_lines[1:]):
            grid[r0 : r1 + 1, c] = GREY
            for gap in _gap_positions(int(r0), int(r1), col_gap_offset):
                grid[gap, c] = BLACK

    for r_idx, c_idx in np.argwhere(anchor_mask):
        r = row_lines[int(r_idx)]
        c = col_lines[int(c_idx)]
        grid[r, c] = BLACK
        for dr, dc in MARKER_ARMS:
            grid[r + dr, c + dc] = RED
    return grid


def _render_expected_from_plan(
    inp: np.ndarray,
    row_lines: list[int],
    col_lines: list[int],
    anchor_mask: np.ndarray,
) -> np.ndarray:
    out = inp.copy()
    centers = [(row_lines[r], col_lines[c]) for r, c in np.argwhere(anchor_mask)]
    out = _connect_markers(out, centers)

    _horizontal, _vertical, counts, _intervals = _coverage_from_anchor_mask(anchor_mask)
    for i, (r0, r1) in enumerate(zip(row_lines, row_lines[1:])):
        for j, (c0, c1) in enumerate(zip(col_lines, col_lines[1:])):
            fill = FILL_BY_COUNT[int(counts[i, j])]
            if fill == BLACK:
                continue
            interior = out[r0 + 1 : r1, c0 + 1 : c1]
            interior[interior == BLACK] = fill
    return out


def _validate_rendered_output(
    out: np.ndarray,
    row_lines: list[int],
    col_lines: list[int],
    anchor_mask: np.ndarray,
) -> bool:
    horizontal, vertical, counts, _intervals = _coverage_from_anchor_mask(anchor_mask)

    for i, (r0, r1) in enumerate(zip(row_lines, row_lines[1:])):
        for j, (c0, c1) in enumerate(zip(col_lines, col_lines[1:])):
            top = np.any(out[r0, c0 + 1 : c1] == BLUE)
            bottom = np.any(out[r1, c0 + 1 : c1] == BLUE)
            left = np.any(out[r0 + 1 : r1, c0] == BLUE)
            right = np.any(out[r0 + 1 : r1, c1] == BLUE)
            if int(top) != int(horizontal[i, j]):
                return False
            if int(bottom) != int(horizontal[i + 1, j]):
                return False
            if int(left) != int(vertical[i, j]):
                return False
            if int(right) != int(vertical[i, j + 1]):
                return False

            interior = out[r0 + 1 : r1, c0 + 1 : c1]
            expected_color = FILL_BY_COUNT[int(counts[i, j])]
            visible = {int(v) for v in np.unique(interior)}
            if expected_color == BLACK:
                if visible != {BLACK}:
                    return False
            else:
                if visible != {expected_color}:
                    return False
    return True


def _generate_case() -> np.ndarray | None:
    m = random.choice([3, 4])
    n = random.choice([3, 4])
    cell_h = random.choice([4, 5])
    cell_w = random.choice([4, 5])
    top = random.choice([1, 2])
    left = random.choice([1, 2])
    bottom = random.choice([1, 2])
    right = random.choice([1, 2])

    row_step = cell_h + 1
    col_step = cell_w + 1
    row_lines = [top + i * row_step for i in range(m + 1)]
    col_lines = [left + j * col_step for j in range(n + 1)]
    height = row_lines[-1] + bottom + 1
    width = col_lines[-1] + right + 1
    if max(height, width) > 30:
        return None

    anchor_mask = _random_anchor_mask(m, n)
    row_gap_offset = random.choice([2, 3])
    col_gap_offset = random.choice([2, 3])
    inp = _render_input(
        row_lines,
        col_lines,
        anchor_mask,
        height=height,
        width=width,
        row_gap_offset=row_gap_offset,
        col_gap_offset=col_gap_offset,
    )
    expected = _render_expected_from_plan(inp, row_lines, col_lines, anchor_mask)

    if not _validate_rendered_output(expected, row_lines, col_lines, anchor_mask):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid anchor-lattice puzzle")
