from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg


def _open_runs(length: int, separators: np.ndarray) -> list[tuple[int, int]]:
    separator_set = set(map(int, separators))
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx in range(length):
        if idx in separator_set:
            if start is not None:
                runs.append((start, idx - 1))
                start = None
        elif start is None:
            start = idx
    if start is not None:
        runs.append((start, length - 1))
    return runs


def _run_index(runs: list[tuple[int, int]], value: int) -> int:
    hits = [idx for idx, (lo, hi) in enumerate(runs) if lo <= value <= hi]
    assert len(hits) == 1, "each motif pixel must lie inside one scaffold cell"
    return hits[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the rare-colored motif into every same-parity cell of the scaffold grid.

    Input structure
    - The background surrounds one scaffold color that forms full horizontal and vertical separator lines.
    - A second foreground color appears only as a small motif inside one open scaffold cell.

    Rule
    - Split rows and columns into open cells between the full scaffold separator lines.
    - Find the open cell containing all motif pixels and record those pixels as offsets from that cell's upper-left corner.
    - Stamp those offsets into every open cell whose row-cell index and column-cell index have the same parity as the source cell, clipping at truncated edge cells.

    Key insight
    - The scaffold turns the grid into a checkerboard of cells; the seed motif marks which parity class should receive copies.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    bg = int(get_bg(g))
    colors = colors_present(g, exclude={bg})
    assert len(colors) == 2, "expected exactly one scaffold color and one motif color"

    scaffold_candidates: list[tuple[int, np.ndarray, np.ndarray]] = []
    for color in colors:
        color = int(color)
        row_lines = np.flatnonzero(np.all(g == color, axis=1))
        col_lines = np.flatnonzero(np.all(g == color, axis=0))
        if len(row_lines) > 0 and len(col_lines) > 0:
            scaffold_candidates.append((color, row_lines, col_lines))

    assert len(scaffold_candidates) == 1, "expected one color to define full scaffold lines"
    scaffold_color, row_lines, col_lines = scaffold_candidates[0]
    motif_colors = [color for color in colors if int(color) != scaffold_color]
    assert len(motif_colors) == 1, "expected one non-scaffold motif color"
    motif_color = int(motif_colors[0])

    row_cells = _open_runs(height, row_lines)
    col_cells = _open_runs(width, col_lines)
    assert row_cells and col_cells, "scaffold must leave open cells"

    motif_coords = np.argwhere(g == motif_color)
    assert len(motif_coords) > 0, "expected a nonempty motif"
    source_row_indices = {_run_index(row_cells, int(r)) for r, _c in motif_coords}
    source_col_indices = {_run_index(col_cells, int(c)) for _r, c in motif_coords}
    assert len(source_row_indices) == 1, "motif must occupy one row cell"
    assert len(source_col_indices) == 1, "motif must occupy one column cell"

    source_row = source_row_indices.pop()
    source_col = source_col_indices.pop()
    source_r0 = row_cells[source_row][0]
    source_c0 = col_cells[source_col][0]
    offsets = [(int(r - source_r0), int(c - source_c0)) for r, c in motif_coords]

    out = g.copy()
    for row_idx, (r0, r1) in enumerate(row_cells):
        if row_idx % 2 != source_row % 2:
            continue
        for col_idx, (c0, c1) in enumerate(col_cells):
            if col_idx % 2 != source_col % 2:
                continue
            for dr, dc in offsets:
                r = r0 + dr
                c = c0 + dc
                if r <= r1 and c <= c1:
                    assert out[r, c] in (bg, motif_color), "motif copies must land in open cells"
                    out[r, c] = motif_color

    return out

# --------------- generator (o5x) ---------------
# Canvas/scaffold geometry.
BACKGROUND_COLOR = 0
CELL_SIZE, CELL_STRIDE, TRUNCATED_CROP = 3, 4, 2
SEPARATOR_COUNT_RANGE, TRUNCATION_COIN_RANGE = (5, 8), (0, 2)
FORCE_FULL_SEP_COUNT, FORCE_TRUNCATED_SEP_COUNT = 5, 7
# Palette.
FOREGROUND_COLORS, FOREGROUND_PAIR_SIZE = np.arange(1, 10), 2
# Motif shape.
MOTIF_PIXEL_RANGE, MOTIF_COORD_RANGE = (4, 7), (0, CELL_SIZE)
MOTIF_EDGE_COORDS = (0, CELL_SIZE - 1)
# Retry limit.
MAX_GENERATION_ATTEMPTS = 300

def _gen_open_cells(side: int, separators: np.ndarray) -> list[tuple[int, int]]:
    starts = [0] + [int(sep) + 1 for sep in separators]
    ends = [int(sep) - 1 for sep in separators] + [side - 1]
    return [(lo, hi) for lo, hi in zip(starts, ends) if lo <= hi]


def _sample_scaffold_layout() -> tuple[int, np.ndarray, list[tuple[int, int]]]:
    """Sample side length, separator coordinates, and square open-cell runs."""
    n_seps = int(np.random.randint(*SEPARATOR_COUNT_RANGE))
    if n_seps == FORCE_FULL_SEP_COUNT:
        truncated = False
    elif n_seps == FORCE_TRUNCATED_SEP_COUNT:
        truncated = True
    else:
        truncated = bool(np.random.randint(*TRUNCATION_COIN_RANGE))
    side = CELL_SIZE + n_seps * CELL_STRIDE - (TRUNCATED_CROP if truncated else 0)
    separators = CELL_SIZE + CELL_STRIDE * np.arange(n_seps)
    return side, separators, _gen_open_cells(side, separators)


def _sample_full_cell_index(cells: list[tuple[int, int]]) -> int:
    full_indices = [idx for idx, (lo, hi) in enumerate(cells) if hi - lo + 1 == CELL_SIZE]
    return int(full_indices[int(np.random.randint(len(full_indices)))])


def _gen_motif_offsets(pixel_count: int) -> list[tuple[int, int]]:
    """Return offsets that span the full cell bbox with the requested pixel count."""
    edge_lo, edge_hi = MOTIF_EDGE_COORDS
    offsets = {
        (edge_lo, int(np.random.randint(*MOTIF_COORD_RANGE))),
        (edge_hi, int(np.random.randint(*MOTIF_COORD_RANGE))),
        (int(np.random.randint(*MOTIF_COORD_RANGE)), edge_lo),
        (int(np.random.randint(*MOTIF_COORD_RANGE)), edge_hi),
    }
    remaining = [(r, c) for r in range(CELL_SIZE) for c in range(CELL_SIZE) if (r, c) not in offsets]
    needed = pixel_count - len(offsets)
    if needed > 0:
        for idx in np.random.permutation(len(remaining))[:needed]:
            offsets.add(remaining[int(idx)])
    return sorted(offsets)


def _stamp_same_parity_motif(grid, cells, source_row, source_col, offsets, motif_color) -> None:
    """Stamp motif offsets into same-parity cells, clipping truncated edge cells."""
    for row_idx, (r0, r1) in enumerate(cells):
        if row_idx % 2 != source_row % 2:
            continue
        for col_idx, (c0, c1) in enumerate(cells):
            if col_idx % 2 != source_col % 2:
                continue
            for dr, dc in offsets:
                r = r0 + dr
                c = c0 + dc
                if r <= r1 and c <= c1:
                    grid[r, c] = motif_color


def _gen_attempt() -> np.ndarray | None:
    side, separators, cells = _sample_scaffold_layout()
    scaffold_color, motif_color = map(int, np.random.permutation(FOREGROUND_COLORS)[:FOREGROUND_PAIR_SIZE])
    out = np.full((side, side), BACKGROUND_COLOR, dtype=int)
    out[separators, :] = scaffold_color
    out[:, separators] = scaffold_color

    source_row = _sample_full_cell_index(cells)
    source_col = _sample_full_cell_index(cells)
    offsets = _gen_motif_offsets(int(np.random.randint(*MOTIF_PIXEL_RANGE)))
    _stamp_same_parity_motif(out, cells, source_row, source_col, offsets, motif_color)

    inp = out.copy()
    inp[inp == motif_color] = BACKGROUND_COLOR
    source_r0 = cells[source_row][0]
    source_c0 = cells[source_col][0]
    for dr, dc in offsets:
        inp[source_r0 + dr, source_c0 + dc] = motif_color

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 92e50de0 input")
