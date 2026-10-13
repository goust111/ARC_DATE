import numpy as np


BACKGROUND = 0
MARKER = 5
PAYLOAD_COLORS = (1, 2, 3, 4, 6, 7, 8)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

PANEL_SIDE_RANGE = (5, 9)  # high-exclusive
PANEL_ROWS_RANGE = (2, 5)  # high-exclusive
PANEL_COLS_RANGE = (2, 5)  # high-exclusive
SEPARATOR_WIDTH_RANGE = (1, 3)  # high-exclusive
MAX_SHAPE_ATTEMPTS = 120
MAX_GENERATION_ATTEMPTS = 300

_RNG = np.random.default_rng()


def _group_consecutive(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    groups: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        groups.append((start, prev + 1))
        start = prev = idx
    groups.append((start, prev + 1))
    return groups


def _intervals_from_bands(bands: list[tuple[int, int]], limit: int) -> list[tuple[int, int]]:
    prev = 0
    intervals: list[tuple[int, int]] = []
    for start, end in bands:
        assert prev < start, "separator bands cannot touch the border or each other"
        intervals.append((prev, start))
        prev = end
    assert prev < limit, "separator bands cannot touch the border"
    intervals.append((prev, limit))
    return intervals


def _find_separator_color(grid: np.ndarray) -> int:
    candidates: list[tuple[int, list[tuple[int, int]], list[tuple[int, int]]]] = []
    for color in map(int, np.unique(grid)):
        row_bands = _group_consecutive([r for r in range(grid.shape[0]) if np.all(grid[r, :] == color)])
        col_bands = _group_consecutive([c for c in range(grid.shape[1]) if np.all(grid[:, c] == color)])
        if not row_bands or not col_bands:
            continue
        try:
            row_intervals = _intervals_from_bands(row_bands, grid.shape[0])
            col_intervals = _intervals_from_bands(col_bands, grid.shape[1])
        except AssertionError:
            continue
        row_sizes = {r1 - r0 for r0, r1 in row_intervals}
        col_sizes = {c1 - c0 for c0, c1 in col_intervals}
        if len(row_intervals) < 2 or len(col_intervals) < 2:
            continue
        if len(row_sizes) != 1 or len(col_sizes) != 1:
            continue
        if next(iter(row_sizes)) != next(iter(col_sizes)):
            continue
        candidates.append((color, row_bands, col_bands))
    assert len(candidates) == 1, "expected exactly one separator color"
    return candidates[0][0]


def _panel_layout(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    sep = _find_separator_color(grid)
    row_bands = _group_consecutive([r for r in range(grid.shape[0]) if np.all(grid[r, :] == sep)])
    col_bands = _group_consecutive([c for c in range(grid.shape[1]) if np.all(grid[:, c] == sep)])
    row_intervals = _intervals_from_bands(row_bands, grid.shape[0])
    col_intervals = _intervals_from_bands(col_bands, grid.shape[1])
    panel_side = row_intervals[0][1] - row_intervals[0][0]
    assert all(r1 - r0 == panel_side for r0, r1 in row_intervals)
    assert all(c1 - c0 == panel_side for c0, c1 in col_intervals)
    return sep, row_intervals, col_intervals


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each panel clockwise according to how many grey corner markers it contains.

    The input is partitioned by uniform separator rows and columns into equal square
    panels. Within each panel, grey pixels appear only at the four corners and act
    as a counter from 0 to 3. Remove those markers, rotate the entire panel content
    clockwise by 0/90/180/270 degrees, and preserve the separator structure.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    _, row_intervals, col_intervals = _panel_layout(g)
    panel_side = row_intervals[0][1] - row_intervals[0][0]
    corners = {
        (0, 0),
        (0, panel_side - 1),
        (panel_side - 1, 0),
        (panel_side - 1, panel_side - 1),
    }

    out = g.copy()
    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            panel = g[r0:r1, c0:c1].copy()
            marker_positions = {tuple(map(int, rc)) for rc in np.argwhere(panel == MARKER)}
            assert marker_positions <= corners, "grey markers must sit at panel corners"
            assert len(marker_positions) <= 3, "expected 0-3 grey markers per panel"
            assert np.any(panel != BACKGROUND), "each panel must contain a visible pattern"

            for mr, mc in marker_positions:
                panel[mr, mc] = BACKGROUND
            out[r0:r1, c0:c1] = np.rot90(panel, -len(marker_positions))

    return out


def _marker_counts(num_panels: int) -> list[int]:
    counts = list(range(4)) * (num_panels // 4)
    remainder = list(_RNG.permutation(4)[: num_panels % 4])
    counts.extend(int(v) for v in remainder)
    _RNG.shuffle(counts)
    return counts


def _corner_positions(side: int) -> list[tuple[int, int]]:
    return [(0, 0), (0, side - 1), (side - 1, 0), (side - 1, side - 1)]


def _random_shape(side: int) -> np.ndarray | None:
    corners = set(_corner_positions(side))
    allowed = [(r, c) for r in range(side) for c in range(side) if (r, c) not in corners]
    max_area = min(len(allowed), 8)

    for _ in range(MAX_SHAPE_ATTEMPTS):
        target_area = int(_RNG.integers(3, max_area + 1))
        start = allowed[int(_RNG.integers(len(allowed)))]
        cells = {start}

        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < side and 0 <= cc < side and (rr, cc) not in corners and (rr, cc) not in cells:
                        frontier.append((rr, cc))
            if not frontier:
                break
            cells.add(frontier[int(_RNG.integers(len(frontier)))])

        if len(cells) != target_area:
            continue

        mask = np.zeros((side, side), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        rows, cols = np.nonzero(mask)
        if rows.ptp() == 0 or cols.ptp() == 0:
            continue
        if any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
            continue
        return mask

    return None


def _build_panel(side: int, color: int, marker_count: int) -> np.ndarray | None:
    mask = _random_shape(side)
    if mask is None:
        return None

    panel = np.full((side, side), BACKGROUND, dtype=int)
    panel[mask] = color
    panel = np.rot90(panel, -int(_RNG.integers(4)))

    if marker_count:
        marker_indices = _RNG.choice(4, size=marker_count, replace=False)
        for idx in marker_indices:
            r, c = _corner_positions(side)[int(idx)]
            panel[r, c] = MARKER
    return panel


def _gen_attempt() -> np.ndarray | None:
    panel_rows = int(_RNG.integers(*PANEL_ROWS_RANGE))
    panel_cols = int(_RNG.integers(*PANEL_COLS_RANGE))
    if panel_rows * panel_cols < 4:
        return None

    side = int(_RNG.integers(*PANEL_SIDE_RANGE))
    sep_width = int(_RNG.integers(*SEPARATOR_WIDTH_RANGE))
    height = panel_rows * side + (panel_rows - 1) * sep_width
    width = panel_cols * side + (panel_cols - 1) * sep_width

    grid = np.full((height, width), BACKGROUND, dtype=int)
    row_intervals: list[tuple[int, int]] = []
    col_intervals: list[tuple[int, int]] = []

    cursor = 0
    for idx in range(panel_rows):
        row_intervals.append((cursor, cursor + side))
        cursor += side
        if idx != panel_rows - 1:
            grid[cursor : cursor + sep_width, :] = 9
            cursor += sep_width

    cursor = 0
    for idx in range(panel_cols):
        col_intervals.append((cursor, cursor + side))
        cursor += side
        if idx != panel_cols - 1:
            grid[:, cursor : cursor + sep_width] = 9
            cursor += sep_width

    palette_size = int(_RNG.integers(2, min(5, len(PAYLOAD_COLORS) + 1)))
    palette = list(_RNG.choice(PAYLOAD_COLORS, size=palette_size, replace=False))
    counts = _marker_counts(panel_rows * panel_cols)

    panel_index = 0
    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            color = int(palette[int(_RNG.integers(len(palette)))])
            panel = _build_panel(side, color, counts[panel_index])
            if panel is None:
                return None
            grid[r0:r1, c0:c1] = panel
            panel_index += 1

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is None:
            continue
        try:
            main(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid sample")
