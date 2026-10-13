from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_shared_for_opus_4_003_2",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()
Color = _ARC_UTILS.Color
paste = _ARC_UTILS.paste


BG = Color.BLACK
STAMP = Color.RED
SCAFFOLD = Color.GREEN
MARKER = Color.YELLOW

CELL_SIZE = 3
BASE_SIDE = 17
BASE_SEPARATORS = np.array([0, 4, 8, 12, 16], dtype=int)
MAX_GENERATION_ATTEMPTS = 400

COMPLETE_MASKS = (
    np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
)


def _open_runs(length: int, separators: np.ndarray) -> list[tuple[int, int]]:
    separator_set = set(map(int, separators))
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx in range(int(length)):
        if idx in separator_set:
            if start is not None:
                runs.append((start, idx - 1))
                start = None
        elif start is None:
            start = idx
    if start is not None:
        runs.append((start, int(length) - 1))
    return runs


def _run_index(runs: list[tuple[int, int]], value: int) -> int:
    hits = [idx for idx, (lo, hi) in enumerate(runs) if lo <= int(value) <= hi]
    assert len(hits) == 1, "coordinate must lie in exactly one open cell"
    return int(hits[0])


def _cell_size(run: tuple[int, int]) -> int:
    return int(run[1] - run[0] + 1)


def _cell_parity(cell: tuple[int, int]) -> tuple[int, int]:
    return int(cell[0] % 2), int(cell[1] % 2)


def _locate_unique_cell(
    row_cells: list[tuple[int, int]],
    col_cells: list[tuple[int, int]],
    coords: np.ndarray,
) -> tuple[int, int]:
    row_hits = {_run_index(row_cells, int(r)) for r, _c in coords}
    col_hits = {_run_index(col_cells, int(c)) for _r, c in coords}
    assert len(row_hits) == 1, "object must stay inside one row-cell"
    assert len(col_hits) == 1, "object must stay inside one col-cell"
    return int(next(iter(row_hits))), int(next(iter(col_hits)))


def _layout_from_grid(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_lines = np.flatnonzero(np.all(grid == SCAFFOLD, axis=1))
    col_lines = np.flatnonzero(np.all(grid == SCAFFOLD, axis=0))
    row_cells = _open_runs(grid.shape[0], row_lines)
    col_cells = _open_runs(grid.shape[1], col_lines)
    assert len(row_cells) == 4 and len(col_cells) == 4, "expected a 4x4 open-cell array"

    open_mask = np.ones_like(grid, dtype=bool)
    open_mask[row_lines, :] = False
    open_mask[:, col_lines] = False
    assert not np.any(grid[open_mask] == SCAFFOLD), "green should only appear on separators"
    return row_cells, col_cells


def _rotational_completion(source_patch: np.ndarray) -> np.ndarray:
    assert source_patch.shape == (CELL_SIZE, CELL_SIZE), "source stamp cell must be full 3x3"
    assert np.all((source_patch == BG) | (source_patch == STAMP))

    center_r = CELL_SIZE // 2
    center_c = CELL_SIZE // 2
    out = np.full_like(source_patch, BG)
    for r_raw, c_raw in np.argwhere(source_patch == STAMP):
        dr = int(r_raw) - center_r
        dc = int(c_raw) - center_c
        for rr, cc in ((dr, dc), (-dc, dr), (-dr, -dc), (dc, -dr)):
            nr = center_r + rr
            nc = center_c + cc
            assert 0 <= nr < CELL_SIZE and 0 <= nc < CELL_SIZE
            out[nr, nc] = STAMP
    return out


def _stamp_same_parity(
    out: np.ndarray,
    row_cells: list[tuple[int, int]],
    col_cells: list[tuple[int, int]],
    parity: tuple[int, int],
    stamp_patch: np.ndarray,
) -> None:
    for row_idx, (r0, _r1) in enumerate(row_cells):
        if row_idx % 2 != int(parity[0]):
            continue
        for col_idx, (c0, _c1) in enumerate(col_cells):
            if col_idx % 2 != int(parity[1]):
                continue
            paste(out, stamp_patch, r0, c0, transparent=BG, clip=True)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the broken red motif by 90-degree rotation and stamp it into every scaffold cell selected by an independent parity marker.

    Input structure:
    - The grid uses black background with green full-row and full-column separators that carve the canvas into a 4x4 array of open cells.
    - Exactly one full 3x3 open cell contains a broken red motif, and exactly one different open cell contains a single yellow marker pixel.
    - The red motif is centered on the source cell's middle pixel and becomes a 90-degree rotationally symmetric stamp after filling its missing rotated copies.

    Rule:
    - Split the grid into open cells between the green separator lines.
    - In the red source cell, rotate every red pixel around the cell center by 90, 180, and 270 degrees; fill the rotated positions with red to recover the full stamp.
    - Read the yellow marker cell's open-cell row and column indices, keep only their parities, erase the yellow pixel, and stamp the completed red motif into every open cell with the same row-parity and column-parity, clipping at truncated outer cells.
    - Keep the completed red motif in its original source cell even when that cell is not in the marker-selected parity class.

    Key insight:
    - The final stamp is not fully visible in the input: it must first be reconstructed by rotational closure, and only then can a separate yellow cell tell which checkerboard parity class receives copies.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    row_cells, col_cells = _layout_from_grid(g)

    red_coords = np.argwhere(g == STAMP)
    yellow_coords = np.argwhere(g == MARKER)
    assert 2 <= len(red_coords) <= 4, "expected a broken 2-4 pixel red motif"
    assert len(yellow_coords) == 1, "expected one yellow marker pixel"

    source_cell = _locate_unique_cell(row_cells, col_cells, red_coords)
    marker_cell = _locate_unique_cell(row_cells, col_cells, yellow_coords)
    assert source_cell != marker_cell, "red source and yellow marker must be in different cells"

    source_r0, source_r1 = row_cells[source_cell[0]]
    source_c0, source_c1 = col_cells[source_cell[1]]
    assert _cell_size(row_cells[source_cell[0]]) == CELL_SIZE
    assert _cell_size(col_cells[source_cell[1]]) == CELL_SIZE

    source_patch = np.where(
        g[source_r0 : source_r1 + 1, source_c0 : source_c1 + 1] == STAMP,
        STAMP,
        BG,
    )
    stamp_patch = _rotational_completion(source_patch)
    stamp_area = int(np.count_nonzero(stamp_patch == STAMP))
    assert stamp_area in (4, 5), "completed stamp should occupy 4 or 5 pixels"
    assert stamp_area > len(red_coords), "input motif must be incomplete"

    out = g.copy()
    out[out == MARKER] = BG
    paste(out, stamp_patch, source_r0, source_c0, transparent=BG)
    _stamp_same_parity(out, row_cells, col_cells, _cell_parity(marker_cell), stamp_patch)
    return out


def _choose(seq):
    return seq[int(np.random.randint(len(seq)))]


def _build_scaffold(crop_r: int, crop_c: int) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    height = int(BASE_SIDE - crop_r)
    width = int(BASE_SIDE - crop_c)
    grid = np.full((height, width), BG, dtype=int)
    row_lines = BASE_SEPARATORS[BASE_SEPARATORS < height]
    col_lines = BASE_SEPARATORS[BASE_SEPARATORS < width]
    grid[row_lines, :] = SCAFFOLD
    grid[:, col_lines] = SCAFFOLD
    row_cells = _open_runs(height, row_lines)
    col_cells = _open_runs(width, col_lines)
    assert len(row_cells) == 4 and len(col_cells) == 4
    return grid, row_cells, col_cells


def _source_candidates(
    row_cells: list[tuple[int, int]],
    col_cells: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    full_rows = [idx for idx, run in enumerate(row_cells) if _cell_size(run) == CELL_SIZE]
    full_cols = [idx for idx, run in enumerate(col_cells) if _cell_size(run) == CELL_SIZE]
    return [(row_idx, col_idx) for row_idx in full_rows for col_idx in full_cols]


def _make_incomplete_mask(full_mask: np.ndarray) -> np.ndarray:
    incomplete = full_mask.copy()
    removable = [
        (int(r), int(c))
        for r, c in np.argwhere(full_mask == 1)
        if (int(r), int(c)) != (CELL_SIZE // 2, CELL_SIZE // 2)
    ]
    max_remove = min(2, len(removable) - 1)
    remove_count = int(np.random.randint(1, max_remove + 1))
    for idx in np.random.permutation(len(removable))[:remove_count]:
        r, c = removable[int(idx)]
        incomplete[r, c] = 0
    assert 2 <= int(incomplete.sum()) <= 4
    assert int(incomplete.sum()) < int(full_mask.sum())
    if full_mask[CELL_SIZE // 2, CELL_SIZE // 2]:
        assert incomplete[CELL_SIZE // 2, CELL_SIZE // 2] == 1
    return incomplete


def _canonical_marker_cell(
    source_cell: tuple[int, int],
    parity: tuple[int, int],
) -> tuple[int, int]:
    candidates = [
        (row_idx, col_idx)
        for row_idx in range(4)
        for col_idx in range(4)
        if (row_idx, col_idx) != source_cell and _cell_parity((row_idx, col_idx)) == parity
    ]
    assert candidates, "chosen parity must have a non-source marker cell"
    return candidates[0]


def _stamp_coords(stamp_patch: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(stamp_patch == STAMP)]


def _parity_has_visible_clip(
    row_cells: list[tuple[int, int]],
    col_cells: list[tuple[int, int]],
    parity: tuple[int, int],
    stamp_patch: np.ndarray,
) -> bool:
    full_count = int(np.count_nonzero(stamp_patch == STAMP))
    coords = _stamp_coords(stamp_patch)
    for row_idx, row_run in enumerate(row_cells):
        if row_idx % 2 != int(parity[0]):
            continue
        height = _cell_size(row_run)
        for col_idx, col_run in enumerate(col_cells):
            if col_idx % 2 != int(parity[1]):
                continue
            width = _cell_size(col_run)
            if height == CELL_SIZE and width == CELL_SIZE:
                continue
            kept = sum(1 for r, c in coords if r < height and c < width)
            if 0 < kept < full_count:
                return True
    return False


def _center_of_cell(row_run: tuple[int, int], col_run: tuple[int, int]) -> tuple[int, int]:
    return int((row_run[0] + row_run[1]) // 2), int((col_run[0] + col_run[1]) // 2)


def _build_expected(
    scaffold: np.ndarray,
    row_cells: list[tuple[int, int]],
    col_cells: list[tuple[int, int]],
    source_cell: tuple[int, int],
    parity: tuple[int, int],
    stamp_patch: np.ndarray,
) -> np.ndarray:
    out = scaffold.copy()
    source_r0 = row_cells[source_cell[0]][0]
    source_c0 = col_cells[source_cell[1]][0]
    paste(out, stamp_patch, source_r0, source_c0, transparent=BG)
    _stamp_same_parity(out, row_cells, col_cells, parity, stamp_patch)
    return out


def _red_bearing_cells(
    grid: np.ndarray,
    row_cells: list[tuple[int, int]],
    col_cells: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    hits: list[tuple[int, int]] = []
    for row_idx, (r0, r1) in enumerate(row_cells):
        for col_idx, (c0, c1) in enumerate(col_cells):
            if np.any(grid[r0 : r1 + 1, c0 : c1 + 1] == STAMP):
                hits.append((row_idx, col_idx))
    return hits


def _gen_attempt() -> np.ndarray | None:
    crop_r = int(_choose((0, 0, 1, 1, 2, 2, 2)))
    crop_c = int(_choose((0, 0, 1, 1, 2, 2, 2)))
    scaffold, row_cells, col_cells = _build_scaffold(crop_r, crop_c)

    full_mask = _choose(COMPLETE_MASKS)
    incomplete_mask = _make_incomplete_mask(full_mask)
    stamp_patch = np.where(full_mask == 1, STAMP, BG).astype(int)
    incomplete_patch = np.where(incomplete_mask == 1, STAMP, BG).astype(int)

    source_cell = _choose(_source_candidates(row_cells, col_cells))
    source_parity = _cell_parity(source_cell)
    want_source_targeted = bool(np.random.randint(2))
    want_visible_clip = (
        (crop_r == 2 or crop_c == 2) and bool(np.random.randint(100) < 70)
    )

    parity_candidates = []
    for parity in ((0, 0), (0, 1), (1, 0), (1, 1)):
        if want_source_targeted != (parity == source_parity):
            continue
        if want_visible_clip and not _parity_has_visible_clip(row_cells, col_cells, parity, stamp_patch):
            continue
        parity_candidates.append(parity)
    if not parity_candidates:
        parity_candidates = [
            parity
            for parity in ((0, 0), (0, 1), (1, 0), (1, 1))
            if want_source_targeted == (parity == source_parity)
        ]
    marker_parity = _choose(parity_candidates)
    marker_cell = _canonical_marker_cell(source_cell, marker_parity)

    inp = scaffold.copy()
    source_r0 = row_cells[source_cell[0]][0]
    source_c0 = col_cells[source_cell[1]][0]
    paste(inp, incomplete_patch, source_r0, source_c0, transparent=BG)
    marker_r, marker_c = _center_of_cell(row_cells[marker_cell[0]], col_cells[marker_cell[1]])
    inp[marker_r, marker_c] = MARKER

    expected = _build_expected(scaffold, row_cells, col_cells, source_cell, marker_parity, stamp_patch)

    if int(np.count_nonzero(inp == STAMP)) not in (2, 3, 4):
        return None
    if int(np.count_nonzero(expected == STAMP)) <= int(np.count_nonzero(inp == STAMP)):
        return None
    if np.count_nonzero(inp == MARKER) != 1 or np.any(expected == MARKER):
        return None

    red_cells = _red_bearing_cells(expected, row_cells, col_cells)
    if len(red_cells) < 4:
        return None
    if source_cell not in red_cells:
        return None
    if want_visible_clip and not _parity_has_visible_clip(row_cells, col_cells, marker_parity, stamp_patch):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotational parity stamp task")
