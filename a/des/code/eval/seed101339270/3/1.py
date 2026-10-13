from __future__ import annotations

import random

import numpy as np


BG = 0
SEP = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for sep in separators:
        assert start < sep, "separator lines must leave nonempty regions"
        spans.append((start, sep - 1))
        start = sep + 1
    assert start < length, "separator lines must leave a final region"
    spans.append((start, length - 1))
    return spans


def _find_span_index(index: int, spans: list[tuple[int, int]]) -> int:
    for i, (lo, hi) in enumerate(spans):
        if lo <= index <= hi:
            return i
    raise AssertionError("coordinate must belong to a region span")


def _separator_rows(g: np.ndarray) -> list[int]:
    width = int(g.shape[1])
    return [
        int(r)
        for r in range(g.shape[0])
        if np.count_nonzero(g[r, :] == BG) == 0 and np.count_nonzero(g[r, :] == SEP) >= width - 1
    ]


def _separator_cols(g: np.ndarray) -> list[int]:
    height = int(g.shape[0])
    return [
        int(c)
        for c in range(g.shape[1])
        if np.count_nonzero(g[:, c] == BG) == 0 and np.count_nonzero(g[:, c] == SEP) >= height - 1
    ]


def _extract_seeds(
    g: np.ndarray,
    sep_rows: list[int],
    sep_cols: list[int],
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    horizontal: list[tuple[int, int, int]] = []
    vertical: list[tuple[int, int, int]] = []

    sep_col_set = set(sep_cols)
    sep_row_set = set(sep_rows)

    for r in sep_rows:
        cols = np.flatnonzero(g[r, :] != SEP)
        assert len(cols) <= 1, "each horizontal separator line may carry at most one seed"
        if len(cols) == 1:
            c = int(cols[0])
            color = int(g[r, c])
            assert c not in sep_col_set, "seed pixels cannot sit on separator intersections"
            assert color != BG, "separator seeds must be colored"
            horizontal.append((int(r), c, color))

    for c in sep_cols:
        rows = np.flatnonzero(g[:, c] != SEP)
        assert len(rows) <= 1, "each vertical separator line may carry at most one seed"
        if len(rows) == 1:
            r = int(rows[0])
            color = int(g[r, c])
            assert r not in sep_row_set, "seed pixels cannot sit on separator intersections"
            assert color != BG, "separator seeds must be colored"
            vertical.append((r, int(c), color))

    assert horizontal or vertical, "expected at least one separator seed"
    return horizontal, vertical


def _paint_reflection(
    out: np.ndarray,
    g: np.ndarray,
    *,
    color: int,
    src_bounds: tuple[int, int, int, int],
    dst_bounds: tuple[int, int, int, int],
    axis_kind: str,
    axis_index: int,
) -> None:
    sr0, sr1, sc0, sc1 = map(int, src_bounds)
    dr0, dr1, dc0, dc1 = map(int, dst_bounds)
    src_coords = np.argwhere(g[sr0 : sr1 + 1, sc0 : sc1 + 1] == int(color))
    assert len(src_coords) > 0, "source region must contain seed-colored pixels"

    for rr, cc in src_coords:
        sr = sr0 + int(rr)
        sc = sc0 + int(cc)
        if axis_kind == "horizontal":
            tr = 2 * int(axis_index) - sr
            tc = sc
        else:
            tr = sr
            tc = 2 * int(axis_index) - sc

        assert dr0 <= tr <= dr1 and dc0 <= tc <= dc1, "reflected pattern must stay inside the target region"
        out[tr, tc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror-copy a seed-colored pattern from one partition across the touched separator line.

    The grid uses black interiors and grey partition lines. A colored seed on a
    separator line identifies the adjacent source/target partition pair and the
    reflection axis. The pattern of the matching color is copied into the
    opposite partition, reflected across that separator, while the seed itself
    becomes grey again.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    sep_rows = _separator_rows(g)
    sep_cols = _separator_cols(g)
    assert sep_rows and sep_cols, "expected grey rows and columns partitioning the grid"

    row_spans = _spans(g.shape[0], sep_rows)
    col_spans = _spans(g.shape[1], sep_cols)
    sep_row_to_index = {sep: i for i, sep in enumerate(sep_rows)}
    sep_col_to_index = {sep: i for i, sep in enumerate(sep_cols)}
    horizontal_seeds, vertical_seeds = _extract_seeds(g, sep_rows, sep_cols)

    out = g.copy()

    for r, c, color in horizontal_seeds:
        row_idx = sep_row_to_index[int(r)]
        col_idx = _find_span_index(int(c), col_spans)
        top_r0, top_r1 = row_spans[row_idx]
        bot_r0, bot_r1 = row_spans[row_idx + 1]
        c0, c1 = col_spans[col_idx]

        top_region = g[top_r0 : top_r1 + 1, c0 : c1 + 1]
        bottom_region = g[bot_r0 : bot_r1 + 1, c0 : c1 + 1]
        top_has = bool(np.any(top_region == int(color)))
        bottom_has = bool(np.any(bottom_region == int(color)))
        assert top_has != bottom_has, "exactly one adjacent region must contain the seed color"

        if top_has:
            src_bounds = (top_r0, top_r1, c0, c1)
            dst_bounds = (bot_r0, bot_r1, c0, c1)
        else:
            src_bounds = (bot_r0, bot_r1, c0, c1)
            dst_bounds = (top_r0, top_r1, c0, c1)

        _paint_reflection(
            out,
            g,
            color=int(color),
            src_bounds=src_bounds,
            dst_bounds=dst_bounds,
            axis_kind="horizontal",
            axis_index=int(r),
        )
        out[int(r), int(c)] = SEP

    for r, c, color in vertical_seeds:
        row_idx = _find_span_index(int(r), row_spans)
        col_idx = sep_col_to_index[int(c)]
        r0, r1 = row_spans[row_idx]
        left_c0, left_c1 = col_spans[col_idx]
        right_c0, right_c1 = col_spans[col_idx + 1]

        left_region = g[r0 : r1 + 1, left_c0 : left_c1 + 1]
        right_region = g[r0 : r1 + 1, right_c0 : right_c1 + 1]
        left_has = bool(np.any(left_region == int(color)))
        right_has = bool(np.any(right_region == int(color)))
        assert left_has != right_has, "exactly one adjacent region must contain the seed color"

        if left_has:
            src_bounds = (r0, r1, left_c0, left_c1)
            dst_bounds = (r0, r1, right_c0, right_c1)
        else:
            src_bounds = (r0, r1, right_c0, right_c1)
            dst_bounds = (r0, r1, left_c0, left_c1)

        _paint_reflection(
            out,
            g,
            color=int(color),
            src_bounds=src_bounds,
            dst_bounds=dst_bounds,
            axis_kind="vertical",
            axis_index=int(c),
        )
        out[int(r), int(c)] = SEP

    return out


# --------------- generator ---------------

ROW_COUNT_RANGE = (2, 4)
COL_COUNT_RANGE = (2, 4)
CELL_HEIGHT_RANGE = (3, 6)
CELL_WIDTH_RANGE = (3, 6)

LINE_SEED_PROB = 0.55
TARGET_STATES = ("empty", "disjoint", "overlap")
DISTRACTOR_PROB = 0.4

MIN_SOURCE_CELLS = 3
MAX_SOURCE_FILL_RATIO = 0.45
MAX_OVERLAY_FILL_RATIO = 0.35
MAX_DISTRACTOR_FILL_RATIO = 0.35

SHAPE_ATTEMPTS = 100
GENERATION_ATTEMPTS = 400


def _cell_start(index: int, size: int) -> int:
    return int(index) * (int(size) + 1)


def _cell_view(grid: np.ndarray, row: int, col: int, cell_h: int, cell_w: int) -> np.ndarray:
    r0 = _cell_start(row, cell_h)
    c0 = _cell_start(col, cell_w)
    return grid[r0 : r0 + cell_h, c0 : c0 + cell_w]


def _paint_mask(grid: np.ndarray, row: int, col: int, cell_h: int, cell_w: int, mask: np.ndarray, color: int) -> None:
    cell = _cell_view(grid, row, col, cell_h, cell_w)
    cell[mask] = int(color)


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _grow_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    frontier = set(_neighbors4(start[0], start[1], h, w))

    while len(cells) < area:
        assert frontier, "connected growth frontier unexpectedly empty"
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nxt in _neighbors4(cell[0], cell[1], h, w):
            if nxt not in cells:
                frontier.add(nxt)

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    return mask


def _mask_spread_ok(mask: np.ndarray) -> bool:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return int(rows.sum()) >= 2 and int(cols.sum()) >= 2


def _make_connected_mask(
    h: int,
    w: int,
    *,
    axis: str | None = None,
    max_fill_ratio: float = MAX_SOURCE_FILL_RATIO,
) -> np.ndarray | None:
    max_cells = min(h * w - 1, max(MIN_SOURCE_CELLS, int(round(h * w * max_fill_ratio))))
    if max_cells < MIN_SOURCE_CELLS:
        return None

    for _ in range(SHAPE_ATTEMPTS):
        area = random.randint(MIN_SOURCE_CELLS, max_cells)
        mask = _grow_connected_mask(h, w, area)
        if not _mask_spread_ok(mask):
            continue
        if axis == "horizontal" and np.array_equal(mask, np.flipud(mask)):
            continue
        if axis == "vertical" and np.array_equal(mask, np.fliplr(mask)):
            continue
        return mask
    return None


def _make_overlay_mask(h: int, w: int, reflected: np.ndarray, mode: str) -> np.ndarray | None:
    if mode == "empty":
        return None

    for _ in range(SHAPE_ATTEMPTS):
        mask = _make_connected_mask(h, w, axis=None, max_fill_ratio=MAX_OVERLAY_FILL_RATIO)
        if mask is None:
            continue
        overlaps = bool(np.any(mask & reflected))
        has_outside = bool(np.any(mask & (~reflected)))
        if mode == "disjoint" and not overlaps:
            return mask
        if mode == "overlap" and overlaps and has_outside:
            return mask
    return None


def _choose_seed_edges(rows: int, cols: int) -> list[tuple[str, int, int]]:
    selected: list[tuple[str, int, int]] = []
    used_cells: set[tuple[int, int]] = set()

    line_specs = [("h", i) for i in range(rows - 1)] + [("v", j) for j in range(cols - 1)]
    random.shuffle(line_specs)

    for kind, line_idx in line_specs:
        if random.random() >= LINE_SEED_PROB:
            continue
        if kind == "h":
            choices = list(range(cols))
            random.shuffle(choices)
            for col in choices:
                cells = {(line_idx, col), (line_idx + 1, col)}
                if used_cells.isdisjoint(cells):
                    selected.append(("h", int(line_idx), int(col)))
                    used_cells |= cells
                    break
        else:
            choices = list(range(rows))
            random.shuffle(choices)
            for row in choices:
                cells = {(row, line_idx), (row, line_idx + 1)}
                if used_cells.isdisjoint(cells):
                    selected.append(("v", int(row), int(line_idx)))
                    used_cells |= cells
                    break

    if selected:
        return selected

    forced: list[tuple[str, int, int]] = [
        ("h", i, j) for i in range(rows - 1) for j in range(cols)
    ] + [
        ("v", i, j) for i in range(rows) for j in range(cols - 1)
    ]
    return [random.choice(forced)]


def _line_indices(rows: int, cols: int, cell_h: int, cell_w: int) -> tuple[list[int], list[int]]:
    sep_rows = [_cell_start(r, cell_h) + cell_h for r in range(rows - 1)]
    sep_cols = [_cell_start(c, cell_w) + cell_w for c in range(cols - 1)]
    return sep_rows, sep_cols


def _clear_structure_ok(grid: np.ndarray, sep_rows: list[int], sep_cols: list[int]) -> bool:
    sep_row_set = set(sep_rows)
    sep_col_set = set(sep_cols)

    for r in range(grid.shape[0]):
        row = grid[r, :]
        if r in sep_row_set:
            if np.count_nonzero(row == BG) != 0:
                return False
            if np.count_nonzero(row != SEP) > 1:
                return False
        elif np.count_nonzero(row == BG) == 0:
            return False

    for c in range(grid.shape[1]):
        col = grid[:, c]
        if c in sep_col_set:
            if np.count_nonzero(col == BG) != 0:
                return False
            if np.count_nonzero(col != SEP) > 1:
                return False
        elif np.count_nonzero(col == BG) == 0:
            return False

    for r in sep_rows:
        for c in sep_cols:
            if int(grid[r, c]) != SEP:
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*ROW_COUNT_RANGE)
    cols = random.randint(*COL_COUNT_RANGE)
    cell_h = random.randint(*CELL_HEIGHT_RANGE)
    cell_w = random.randint(*CELL_WIDTH_RANGE)

    height = rows * cell_h + (rows - 1)
    width = cols * cell_w + (cols - 1)
    grid = np.full((height, width), BG, dtype=int)

    sep_rows, sep_cols = _line_indices(rows, cols, cell_h, cell_w)
    for r in sep_rows:
        grid[int(r), :] = SEP
    for c in sep_cols:
        grid[:, int(c)] = SEP

    seed_edges = _choose_seed_edges(rows, cols)
    if len(seed_edges) > len(COLOR_POOL):
        return None
    seed_colors = random.sample(COLOR_POOL, len(seed_edges))
    spare_colors = [c for c in COLOR_POOL if c not in seed_colors]
    if not spare_colors:
        return None

    involved_cells: set[tuple[int, int]] = set()
    expected = grid.copy()

    for color, (kind, a, b) in zip(seed_colors, seed_edges, strict=False):
        axis = "horizontal" if kind == "h" else "vertical"
        source_mask = _make_connected_mask(cell_h, cell_w, axis=axis, max_fill_ratio=MAX_SOURCE_FILL_RATIO)
        if source_mask is None:
            return None

        if kind == "h":
            cell_a = (a, b)
            cell_b = (a + 1, b)
            seed_pos = (sep_rows[a], random.randint(_cell_start(b, cell_w), _cell_start(b, cell_w) + cell_w - 1))
            reflected = np.flipud(source_mask)
        else:
            cell_a = (a, b)
            cell_b = (a, b + 1)
            seed_pos = (random.randint(_cell_start(a, cell_h), _cell_start(a, cell_h) + cell_h - 1), sep_cols[b])
            reflected = np.fliplr(source_mask)

        source_cell, target_cell = (cell_a, cell_b) if random.random() < 0.5 else (cell_b, cell_a)
        involved_cells.add(source_cell)
        involved_cells.add(target_cell)

        _paint_mask(grid, source_cell[0], source_cell[1], cell_h, cell_w, source_mask, int(color))
        _paint_mask(expected, source_cell[0], source_cell[1], cell_h, cell_w, source_mask, int(color))

        target_state = random.choice(TARGET_STATES)
        overlay_mask = _make_overlay_mask(cell_h, cell_w, reflected, target_state)
        if target_state != "empty" and overlay_mask is None:
            return None
        if overlay_mask is not None:
            overlay_color = random.choice(spare_colors)
            _paint_mask(grid, target_cell[0], target_cell[1], cell_h, cell_w, overlay_mask, overlay_color)
            _paint_mask(expected, target_cell[0], target_cell[1], cell_h, cell_w, overlay_mask, overlay_color)

        grid[int(seed_pos[0]), int(seed_pos[1])] = int(color)
        expected[int(seed_pos[0]), int(seed_pos[1])] = SEP
        _paint_mask(expected, target_cell[0], target_cell[1], cell_h, cell_w, reflected, int(color))

    free_cells = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in involved_cells]
    random.shuffle(free_cells)
    for row, col in free_cells:
        if random.random() >= DISTRACTOR_PROB:
            continue
        mask = _make_connected_mask(cell_h, cell_w, axis=None, max_fill_ratio=MAX_DISTRACTOR_FILL_RATIO)
        if mask is None:
            return None
        color = random.choice(spare_colors)
        _paint_mask(grid, row, col, cell_h, cell_w, mask, color)
        _paint_mask(expected, row, col, cell_h, cell_w, mask, color)

    if not _clear_structure_ok(grid, sep_rows, sep_cols):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
