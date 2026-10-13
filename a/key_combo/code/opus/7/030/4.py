from __future__ import annotations

import numpy as np


BACKGROUND = 0
CONTROL_COLORS = (1, 2, 3, 4)
SEPARATOR = 5
SEED = 7

GEN_ATTEMPTS = 400
GEN_CELL_COUNTS = (3, 4)
GEN_CELL_SIZES = (4,)
GEN_SHAPE_LIBRARY = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
)


def _open_runs(length: int, separators: set[int], start: int = 0) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for idx in range(start, length):
        if idx in separators:
            if run_start is not None:
                runs.append((run_start, idx - 1))
                run_start = None
            continue
        if run_start is None:
            run_start = idx
    if run_start is not None:
        runs.append((run_start, length - 1))
    return runs


def _locate_run(runs: list[tuple[int, int]], value: int) -> int:
    hits = [idx for idx, (lo, hi) in enumerate(runs) if lo <= value <= hi]
    assert len(hits) == 1, "coordinate must lie in exactly one open run"
    return int(hits[0])


def _receiver_indices(count: int, source_index: int) -> list[int]:
    parity = int(source_index) % 2
    return [idx for idx in range(int(count)) if idx % 2 == parity]


def _stamp_recolored_copies(
    out: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    source_row: int,
    source_col: int,
    offsets: list[tuple[int, int]],
    row_controls: list[int],
) -> None:
    for ri, (r0, r1) in enumerate(row_bands):
        if ri % 2 != source_row % 2:
            continue
        stamp_color = int(row_controls[ri])
        for ci, (c0, c1) in enumerate(col_bands):
            if ci % 2 != source_col % 2:
                continue
            for dr, dc in offsets:
                r = r0 + dr
                c = c0 + dc
                if r <= r1 and c <= c1:
                    out[r, c] = stamp_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the seed shape into same-parity scaffold cells, recoloring each copy by its row-band key.

    Input structure:
    - Full grey separator rows and columns divide the grid into open scaffold cells to the right of a dedicated control column.
    - The first non-separator row of each horizontal band carries one colored control pixel in column 0.
    - Exactly one open cell contains a small orange seed pattern, while the other open cells are black.

    Rule:
    - Split the grid into row bands using full grey rows and into scaffold columns using full grey columns, ignoring column 0 as the control strip.
    - Find the cell containing the orange seed and record its pixel offsets from that cell's upper-left corner.
    - For every cell whose row index parity and column index parity both match the seed cell, stamp those offsets into the cell.
    - Recolor every stamped copy, including the source cell, with the control color from that cell's row band; keep separators, control pixels, and untouched background unchanged.

    Key insight:
    - The seed supplies only the shape; placement comes from scaffold parity, and color comes independently from the left-edge row key.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert width >= 3, "expected a control column plus scaffold area"

    separator_rows = set(map(int, np.flatnonzero(np.all(g == SEPARATOR, axis=1))))
    separator_cols = set(map(int, np.flatnonzero(np.all(g == SEPARATOR, axis=0))))
    assert separator_rows and separator_cols, "expected full separator lines"

    row_bands = _open_runs(height, separator_rows, start=0)
    col_bands = _open_runs(width, separator_cols, start=1)
    assert row_bands and col_bands, "expected open scaffold cells"

    row_controls: list[int] = []
    for r0, r1 in row_bands:
        control_rows = [r for r in range(r0, r1 + 1) if int(g[r, 0]) in CONTROL_COLORS]
        assert len(control_rows) == 1 and control_rows[0] == r0, "expected one band control at the band start"
        row_controls.append(int(g[r0, 0]))

    seed_coords = np.argwhere(g == SEED)
    assert 2 <= len(seed_coords) <= 3, "expected a 2-3 pixel orange seed"
    source_rows = {_locate_run(row_bands, int(r)) for r, _c in seed_coords}
    source_cols = {_locate_run(col_bands, int(c)) for _r, c in seed_coords}
    assert len(source_rows) == 1 and len(source_cols) == 1, "seed must stay inside one open cell"

    source_row = int(next(iter(source_rows)))
    source_col = int(next(iter(source_cols)))
    source_r0 = row_bands[source_row][0]
    source_c0 = col_bands[source_col][0]
    offsets = sorted((int(r - source_r0), int(c - source_c0)) for r, c in seed_coords)

    out = g.copy()
    out[out == SEED] = BACKGROUND
    _stamp_recolored_copies(out, row_bands, col_bands, source_row, source_col, offsets, row_controls)
    return out


def _build_layout(
    cell_count: int,
    cell_h: int,
    cell_w: int,
    crop_h: int,
    crop_w: int,
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]], np.ndarray, np.ndarray]:
    height = cell_count * cell_h + (cell_count - 1) - crop_h
    width = 1 + cell_count * cell_w + (cell_count - 1) - crop_w
    row_separators = cell_h + np.arange(cell_count - 1) * (cell_h + 1)
    col_separators = 1 + cell_w + np.arange(cell_count - 1) * (cell_w + 1)
    row_bands = _open_runs(height, set(map(int, row_separators)), start=0)
    col_bands = _open_runs(width, set(map(int, col_separators)), start=1)
    return height, width, row_bands, col_bands, row_separators, col_separators


def _source_candidates(cell_count: int, cropped: bool) -> list[int]:
    if cropped:
        last = cell_count - 1
        return [idx for idx in range(cell_count - 1) if idx % 2 == last % 2]
    return [idx for idx in range(cell_count) if len(_receiver_indices(cell_count, idx)) >= 2]


def _sample_controls(cell_count: int, source_row: int) -> list[int]:
    controls = [int(np.random.choice(CONTROL_COLORS)) for _ in range(cell_count)]
    receiver_rows = _receiver_indices(cell_count, source_row)
    chosen = list(np.random.permutation(CONTROL_COLORS)[: len(receiver_rows)])
    for idx, row in enumerate(receiver_rows):
        controls[row] = int(chosen[idx])
    return controls


def _sample_offsets(
    cell_h: int,
    cell_w: int,
    require_bottom: bool,
    require_right: bool,
) -> list[tuple[int, int]]:
    fitting_shapes = []
    for shape in GEN_SHAPE_LIBRARY:
        max_r = max(r for r, _c in shape)
        max_c = max(c for _r, c in shape)
        if max_r < cell_h and max_c < cell_w:
            fitting_shapes.append(shape)
    assert fitting_shapes, "expected at least one shape to fit"

    choices = []
    for shape in fitting_shapes:
        max_r = max(r for r, _c in shape)
        max_c = max(c for _r, c in shape)
        anchor_r = cell_h - 1 - max_r if require_bottom else int(np.random.randint(0, cell_h - max_r))
        anchor_c = cell_w - 1 - max_c if require_right else int(np.random.randint(0, cell_w - max_c))
        offsets = sorted((anchor_r + r, anchor_c + c) for r, c in shape)
        if require_bottom and max(r for r, _c in offsets) != cell_h - 1:
            continue
        if require_right and max(c for _r, c in offsets) != cell_w - 1:
            continue
        choices.append(offsets)
    assert choices, "expected at least one seed shape to satisfy clipping constraints"
    return list(choices[int(np.random.randint(len(choices)))])


def _render_base(
    height: int,
    width: int,
    row_bands: list[tuple[int, int]],
    row_separators: np.ndarray,
    col_separators: np.ndarray,
    controls: list[int],
) -> np.ndarray:
    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[row_separators, :] = SEPARATOR
    grid[:, col_separators] = SEPARATOR
    for ri, (r0, _r1) in enumerate(row_bands):
        grid[r0, 0] = int(controls[ri])
    return grid


def _render_input(
    base: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    source_row: int,
    source_col: int,
    offsets: list[tuple[int, int]],
) -> np.ndarray:
    inp = base.copy()
    r0 = row_bands[source_row][0]
    c0 = col_bands[source_col][0]
    for dr, dc in offsets:
        inp[r0 + dr, c0 + dc] = SEED
    return inp


def _render_output(
    base: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    source_row: int,
    source_col: int,
    offsets: list[tuple[int, int]],
    controls: list[int],
) -> np.ndarray:
    out = base.copy()
    _stamp_recolored_copies(out, row_bands, col_bands, source_row, source_col, offsets, controls)
    return out


def _generate_one() -> np.ndarray | None:
    cell_count = int(np.random.choice(GEN_CELL_COUNTS))
    cell_h = int(np.random.choice(GEN_CELL_SIZES))
    cell_w = int(np.random.choice(GEN_CELL_SIZES))

    crop_h = 0
    crop_w = 0
    if bool(np.random.randint(0, 2)):
        crop_h = int(np.random.randint(0, min(3, cell_h)))
        crop_w = int(np.random.randint(0, min(3, cell_w)))
        if crop_h == 0 and crop_w == 0:
            if bool(np.random.randint(0, 2)):
                crop_h = 1
            else:
                crop_w = 1

    height, width, row_bands, col_bands, row_separators, col_separators = _build_layout(
        cell_count, cell_h, cell_w, crop_h, crop_w
    )
    if max(height, width) > 30:
        return None

    row_candidates = _source_candidates(cell_count, crop_h > 0)
    col_candidates = _source_candidates(cell_count, crop_w > 0)
    source_row = int(row_candidates[int(np.random.randint(len(row_candidates)))])
    source_col = int(col_candidates[int(np.random.randint(len(col_candidates)))])

    controls = _sample_controls(cell_count, source_row)
    receiver_rows = _receiver_indices(cell_count, source_row)
    if len({controls[idx] for idx in receiver_rows}) < 2:
        return None

    require_bottom = crop_h > 0 and (cell_count - 1) in receiver_rows
    require_right = crop_w > 0 and (cell_count - 1) in _receiver_indices(cell_count, source_col)
    offsets = _sample_offsets(cell_h, cell_w, require_bottom=require_bottom, require_right=require_right)

    base = _render_base(height, width, row_bands, row_separators, col_separators, controls)
    inp = _render_input(base, row_bands, col_bands, source_row, source_col, offsets)
    expected = _render_output(base, row_bands, col_bands, source_row, source_col, offsets, controls)

    if np.count_nonzero(inp == SEED) not in (2, 3):
        return None
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
