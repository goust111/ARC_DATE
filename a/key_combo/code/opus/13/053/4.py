from __future__ import annotations

import numpy as np


BG = 0
BLUE = 1
GREY = 5
MARKER_COLORS = (2, 3, 4, 6, 7, 8, 9)

GEN_HEIGHT_RANGE = (9, 14)
GEN_PANEL_WIDTH_RANGE = (6, 10)
GEN_MARKER_COUNT_RANGE = (3, 5)
GEN_TOTAL_BLUE_RANGE = (8, 15)
GEN_ATTEMPTS = 400


def _find_separator(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(cols) == 1, "expected exactly one full-height gray separator"
    return int(cols[0])


def _read_diagonal_palette(left_panel: np.ndarray) -> list[int]:
    h, w = left_panel.shape
    limit = min(h, w)
    colors: list[int] = []
    for i in range(limit):
        value = int(left_panel[i, i])
        if value in (BG, BLUE, GREY):
            break
        colors.append(value)
    assert colors, "expected a non-empty diagonal marker palette"
    assert len(colors) == len(set(colors)), "expected distinct diagonal marker colors"
    return colors


def _overlap_positions(left_panel: np.ndarray, right_panel: np.ndarray) -> list[tuple[int, int]]:
    coords = np.argwhere((left_panel == BLUE) & (right_panel == BLUE))
    positions = [tuple(map(int, rc)) for rc in coords]
    positions.sort()
    return positions


def _render_overlap_output(
    shape: tuple[int, int],
    positions: list[tuple[int, int]],
    palette: list[int],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for i, (r, c) in enumerate(positions):
        out[r, c] = int(palette[i % len(palette)])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color the logical overlap of two separated blue masks by cycling through the top-left diagonal palette.

    Input structure:
    - The grid has a black background and one full-height gray separator column.
    - On the left and right of the separator are equal-sized panels containing blue/black bitmasks.
    - The left panel also contains a contiguous down-right diagonal of distinct non-blue marker colors starting at (0,0).

    Rule:
    - Split the grid at the gray separator and align the two panels cell by cell.
    - Keep only positions where both panels are blue.
    - Read the diagonal marker colors from upper-left to lower-right.
    - Order the overlap positions in reading order and recolor them by cycling through the marker sequence.
    - Output only the left-panel-sized grid; every non-overlap cell is black.

    Key insight:
    - The separator defines a pixelwise logical AND, while the diagonal markers provide an independent ordered palette that must be paired with the overlap cells in reading order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    sep = _find_separator(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.shape == right.shape, "expected equal-sized left and right mask panels"

    palette = _read_diagonal_palette(left)
    positions = _overlap_positions(left, right)
    assert positions, "expected at least one overlap position"
    return _render_overlap_output(left.shape, positions, palette)


def _build_overlap_positions(
    rng: np.random.Generator,
    rows: list[int],
    panel_w: int,
    count: int,
) -> list[tuple[int, int]] | None:
    if len(rows) < 2 or panel_w < 2 or count < 4:
        return None

    all_cells = [(r, c) for r in rows for c in range(panel_w)]
    for _ in range(80):
        r0, r1 = sorted(rng.choice(rows, size=2, replace=False).tolist())
        c0, c1 = sorted(rng.choice(panel_w, size=2, replace=False).tolist())
        anchors = {(r0, c0), (r0, c1), (r1, c0)}
        remaining = [cell for cell in all_cells if cell not in anchors]
        need = count - len(anchors)
        if need < 0 or len(remaining) < need:
            continue
        extras = rng.choice(len(remaining), size=need, replace=False)
        positions = sorted(list(anchors) + [remaining[int(i)] for i in np.atleast_1d(extras)])
        if len({r for r, _ in positions}) < 2 or len({c for _, c in positions}) < 2:
            continue
        return positions
    return None


def _match_shortcut_sequences(
    positions: list[tuple[int, int]],
    palette: list[int],
    mask_top: int,
) -> bool:
    actual = [int(palette[i % len(palette)]) for i in range(len(positions))]
    unique_rows = sorted({r for r, _ in positions})
    unique_cols = sorted({c for _, c in positions})

    row_rank = {r: i for i, r in enumerate(unique_rows)}
    col_rank = {c: i for i, c in enumerate(unique_cols)}
    candidate_sequences = [
        [int(palette[(r - mask_top) % len(palette)]) for r, _ in positions],
        [int(palette[c % len(palette)]) for _, c in positions],
        [int(palette[(r + c) % len(palette)]) for r, c in positions],
        [int(palette[row_rank[r] % len(palette)]) for r, _ in positions],
        [int(palette[col_rank[c] % len(palette)]) for _, c in positions],
    ]
    return any(seq == actual for seq in candidate_sequences)


def _semantic_ok(
    grid: np.ndarray,
    expected: np.ndarray,
    palette: list[int],
    overlap: list[tuple[int, int]],
    left_only: list[tuple[int, int]],
    right_only: list[tuple[int, int]],
    mask_top: int,
) -> bool:
    if len(overlap) <= len(palette):
        return False
    if len(left_only) < 2 or len(right_only) < 2:
        return False

    sep = _find_separator(grid)
    left = grid[:, :sep]
    right = grid[:, sep + 1 :]
    if left.shape != right.shape:
        return False
    if _read_diagonal_palette(left) != palette:
        return False
    if _overlap_positions(left, right) != overlap:
        return False

    cycle = [int(palette[i % len(palette)]) for i in range(len(overlap))]
    if len(set(cycle)) != len(palette):
        return False

    by_row: dict[int, list[int]] = {}
    by_col: dict[int, list[int]] = {}
    for (r, c), color in zip(overlap, cycle):
        by_row.setdefault(r, []).append(color)
        by_col.setdefault(c, []).append(color)
    if not any(len(colors) >= 2 and len(set(colors)) >= 2 for colors in by_row.values()):
        return False
    if not any(len(colors) >= 2 and len(set(colors)) >= 2 for colors in by_col.values()):
        return False

    repeated = {}
    for (r, c), color in zip(overlap, cycle):
        repeated.setdefault(color, []).append((r, c))
    if not any(len(cells) >= 2 and len({r for r, _ in cells}) >= 2 and len({c for _, c in cells}) >= 2 for cells in repeated.values()):
        return False

    if _match_shortcut_sequences(overlap, palette, mask_top):
        return False

    if np.count_nonzero(expected) != len(overlap):
        return False
    expected_palette = [int(expected[r, c]) for r, c in overlap]
    if expected_palette != cycle:
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    rng = np.random.default_rng()

    h = int(rng.integers(GEN_HEIGHT_RANGE[0], GEN_HEIGHT_RANGE[1] + 1))
    panel_w = int(rng.integers(GEN_PANEL_WIDTH_RANGE[0], GEN_PANEL_WIDTH_RANGE[1] + 1))
    max_markers = min(GEN_MARKER_COUNT_RANGE[1], panel_w, h - 4)
    if max_markers < GEN_MARKER_COUNT_RANGE[0]:
        return None

    marker_count = int(rng.integers(GEN_MARKER_COUNT_RANGE[0], max_markers + 1))
    palette = rng.choice(MARKER_COLORS, size=marker_count, replace=False).tolist()

    extra_gap = max(0, h - marker_count - 4)
    mask_top = marker_count + int(rng.integers(0, min(2, extra_gap) + 1))
    active_rows = list(range(mask_top, h))
    active_cells = len(active_rows) * panel_w
    if active_cells < marker_count + 6:
        return None

    min_overlap = max(4, marker_count + 1)
    max_overlap = min(10, active_cells - 4)
    if min_overlap > max_overlap:
        return None
    overlap_count = int(rng.integers(min_overlap, max_overlap + 1))
    overlap = _build_overlap_positions(rng, active_rows, panel_w, overlap_count)
    if overlap is None:
        return None

    min_total = max(overlap_count + 2, GEN_TOTAL_BLUE_RANGE[0])
    max_total = min(active_cells, overlap_count + 6, GEN_TOTAL_BLUE_RANGE[1])
    if min_total > max_total:
        return None

    left_total = int(rng.integers(min_total, max_total + 1))
    right_total = int(rng.integers(min_total, max_total + 1))
    left_only_n = left_total - overlap_count
    right_only_n = right_total - overlap_count
    if left_only_n < 2 or right_only_n < 2:
        return None

    remaining = [(r, c) for r in active_rows for c in range(panel_w) if (r, c) not in set(overlap)]
    if len(remaining) < left_only_n + right_only_n:
        return None
    left_pick = rng.choice(len(remaining), size=left_only_n, replace=False)
    left_only = [remaining[int(i)] for i in np.atleast_1d(left_pick)]
    remaining_after_left = [cell for cell in remaining if cell not in set(left_only)]
    right_pick = rng.choice(len(remaining_after_left), size=right_only_n, replace=False)
    right_only = [remaining_after_left[int(i)] for i in np.atleast_1d(right_pick)]

    grid = np.zeros((h, 2 * panel_w + 1), dtype=int)
    grid[:, panel_w] = GREY
    for i, color in enumerate(palette):
        grid[i, i] = int(color)
    for r, c in overlap + left_only:
        grid[r, c] = BLUE
    for r, c in overlap + right_only:
        grid[r, panel_w + 1 + c] = BLUE

    expected = _render_overlap_output((h, panel_w), overlap, palette)
    if not _semantic_ok(grid, expected, palette, overlap, left_only, right_only, mask_top):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a non-degenerate dual-mask cyclic-color sample")
