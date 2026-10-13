"""ARC task 4: cyclic coloring of intact cells in a broken lattice."""

from __future__ import annotations

from collections import deque

import numpy as np


_NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_MAX_GENERATION_ATTEMPTS = 500


def _mode(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _label_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label True pixels into 4-connected components using (row, column) coordinates."""
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int16)
    next_label = 0
    for start_r, start_c in np.argwhere(mask):
        start_r, start_c = int(start_r), int(start_c)
        if labels[start_r, start_c] != 0:
            continue
        next_label += 1
        labels[start_r, start_c] = next_label
        queue = deque([(start_r, start_c)])
        while queue:
            r, c = queue.popleft()
            for dr, dc in _NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < height
                    and 0 <= nc < width
                    and mask[nr, nc]
                    and labels[nr, nc] == 0
                ):
                    labels[nr, nc] = next_label
                    queue.append((nr, nc))
    return labels, next_label


def _lattice_structure(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    """Recover the lattice color and its regularly spaced boundary rows/columns."""
    height, width = grid.shape
    assert height >= 7 and width >= 7, "expected a top key and a multi-cell lattice"

    colors, counts = np.unique(grid[1:], return_counts=True)
    candidates = [
        (int(count), int(color))
        for color, count in zip(colors, counts)
        if int(color) != bg
    ]
    assert candidates, "expected a non-background lattice below the key"
    candidates.sort(reverse=True)
    lattice_color = candidates[0][1]

    # A boundary row spans the lattice width; an ordinary interior row only
    # meets the comparatively sparse vertical boundaries. Gaps are short, so
    # the 60% projection threshold keeps broken walls and rejects interiors.
    row_counts = np.count_nonzero(grid == lattice_color, axis=1)
    line_rows = [r for r in range(1, height) if row_counts[r] * 5 >= width * 3]
    assert len(line_rows) >= 3, "expected at least two lattice rows"
    assert line_rows == list(range(line_rows[0], line_rows[-1] + 1, line_rows[1] - line_rows[0]))

    top, bottom = line_rows[0], line_rows[-1]
    lattice_height = bottom - top + 1
    col_counts = np.count_nonzero(grid[top : bottom + 1] == lattice_color, axis=0)
    line_cols = [c for c in range(width) if col_counts[c] * 5 >= lattice_height * 3]
    assert len(line_cols) >= 3, "expected at least two lattice columns"
    assert line_cols == list(range(line_cols[0], line_cols[-1] + 1, line_cols[1] - line_cols[0]))
    assert line_cols[0] == 0 and line_cols[-1] == width - 1
    assert line_rows[-1] == height - 1
    return lattice_color, line_rows, line_cols


def _intact_cells(
    grid: np.ndarray, bg: int, line_rows: list[int], line_cols: list[int]
) -> np.ndarray:
    """Classify each slot from the bbox and extent of its background component."""
    labels, _ = _label_mask(grid == bg)
    intact = np.zeros((len(line_rows) - 1, len(line_cols) - 1), dtype=bool)
    for cell_r, (wall_top, wall_bottom) in enumerate(zip(line_rows, line_rows[1:])):
        for cell_c, (wall_left, wall_right) in enumerate(zip(line_cols, line_cols[1:])):
            r0, r1 = wall_top + 1, wall_bottom - 1
            c0, c1 = wall_left + 1, wall_right - 1
            assert r0 <= r1 and c0 <= c1
            component_label = int(labels[r0, c0])
            assert component_label > 0, "cell interiors must be background"
            coords = np.argwhere(labels == component_label)
            cr0, cc0 = coords.min(axis=0)
            cr1, cc1 = coords.max(axis=0)
            bbox_area = (int(cr1) - int(cr0) + 1) * (int(cc1) - int(cc0) + 1)
            inside_this_cell = (
                int(cr0) >= r0
                and int(cr1) <= r1
                and int(cc0) >= c0
                and int(cc1) <= c1
            )
            intact[cell_r, cell_c] = inside_this_cell and len(coords) == bbox_area
    return intact


def main(grid: np.ndarray) -> np.ndarray:
    """Color intact lattice cells with the top-row key in one row-major cycle.

    Input structure:
    - The top row contains a contiguous non-background color sequence, and below it a
      one-color regular lattice encloses background cell interiors.
    - Short missing wall segments connect some neighboring interiors, while the outer
      lattice frame and all wall intersections remain present.

    Rule:
    - Enumerate every lattice slot from left to right and then top to bottom, assigning
      successive top-row colors cyclically without restarting at a new lattice row.
    - Fill a slot only when its background component is a solid rectangle wholly inside
      that slot; a broken slot remains background but still consumes its assigned color.
      Preserve the key and every remaining lattice pixel.

    Key insight:
    - The invisible broken slots advance the color cycle, so colors are indexed by all
      geometric slots rather than only by the intact cells that become visible.
    """
    g = np.asarray(grid)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    bg = _mode(g)

    key_positions = np.flatnonzero(g[0] != bg)
    assert len(key_positions) >= 2, "expected a top-row color sequence"
    assert np.array_equal(key_positions, np.arange(key_positions[0], key_positions[-1] + 1))
    key = [int(g[0, c]) for c in key_positions]

    _lattice_color, line_rows, line_cols = _lattice_structure(g, bg)
    intact = _intact_cells(g, bg, line_rows, line_cols)

    out = g.copy()
    slot = 0
    for cell_r, (wall_top, wall_bottom) in enumerate(zip(line_rows, line_rows[1:])):
        for cell_c, (wall_left, wall_right) in enumerate(zip(line_cols, line_cols[1:])):
            if intact[cell_r, cell_c]:
                out[wall_top + 1 : wall_bottom, wall_left + 1 : wall_right] = key[slot % len(key)]
            slot += 1
    return out


def _render_case(
    rng: np.random.Generator,
    n_rows: int,
    n_cols: int,
    cell_h: int,
    cell_w: int,
    bg: int,
    lattice_color: int,
    key: list[int],
) -> tuple[np.ndarray, np.ndarray, set[tuple[int, int]]]:
    lattice_top = 2
    height = lattice_top + n_rows * (cell_h + 1) + 1
    width = n_cols * (cell_w + 1) + 1
    grid = np.full((height, width), bg, dtype=np.int8)

    key_left = int(rng.integers(0, width - len(key) + 1))
    grid[0, key_left : key_left + len(key)] = key

    line_rows = [lattice_top + i * (cell_h + 1) for i in range(n_rows + 1)]
    line_cols = [i * (cell_w + 1) for i in range(n_cols + 1)]
    grid[line_rows, :] = lattice_color
    grid[:, line_cols] = lattice_color
    # Restore the spacer and top key after drawing the vertical walls: the
    # lattice is explicitly the framed object below the blank separator row.
    grid[:lattice_top, :] = bg
    grid[0, key_left : key_left + len(key)] = key

    vertical_edges = [("v", r, c) for r in range(n_rows) for c in range(n_cols - 1)]
    horizontal_edges = [("h", r, c) for r in range(n_rows - 1) for c in range(n_cols)]
    chosen = [
        vertical_edges[int(rng.integers(len(vertical_edges)))],
        horizontal_edges[int(rng.integers(len(horizontal_edges)))],
    ]
    remaining = list(set(vertical_edges + horizontal_edges) - set(chosen))
    rng.shuffle(remaining)
    extra_count = int(rng.integers(0, min(4, len(remaining)) + 1))
    chosen.extend(remaining[:extra_count])

    broken: set[tuple[int, int]] = set()
    for orientation, r, c in chosen:
        if orientation == "v":
            wall_c = line_cols[c + 1]
            gap_r = int(rng.integers(line_rows[r] + 1, line_rows[r + 1]))
            grid[gap_r, wall_c] = bg
            broken.update({(r, c), (r, c + 1)})
        else:
            wall_r = line_rows[r + 1]
            gap_c = int(rng.integers(line_cols[c] + 1, line_cols[c + 1]))
            grid[wall_r, gap_c] = bg
            broken.update({(r, c), (r + 1, c)})

    expected = grid.copy()
    for r in range(n_rows):
        for c in range(n_cols):
            if (r, c) not in broken:
                slot = r * n_cols + c
                expected[
                    line_rows[r] + 1 : line_rows[r + 1],
                    line_cols[c] + 1 : line_cols[c + 1],
                ] = key[slot % len(key)]
    return grid, expected, broken


def _semantic_checks(
    grid: np.ndarray,
    expected: np.ndarray,
    broken: set[tuple[int, int]],
    n_rows: int,
    n_cols: int,
    key: list[int],
    bg: int,
) -> bool:
    """Reject cases that fail the intended rule or allow the main shortcuts."""
    try:
        actual = main(grid)
        _line_color, line_rows, line_cols = _lattice_structure(grid, bg)
        observed_intact = _intact_cells(grid, bg, line_rows, line_cols)
    except (AssertionError, IndexError, ValueError):
        return False

    declared_intact = np.ones((n_rows, n_cols), dtype=bool)
    for r, c in broken:
        declared_intact[r, c] = False
    if not np.array_equal(observed_intact, declared_intact):
        return False
    if not np.array_equal(actual, expected) or np.array_equal(actual, grid):
        return False
    if not broken or int(declared_intact.sum()) < 3:
        return False

    intact_slots = [i for i in range(n_rows * n_cols) if (i // n_cols, i % n_cols) not in broken]
    if not any(i > min(r * n_cols + c for r, c in broken) for i in intact_slots):
        return False

    # At least one visible cell must disprove counting only painted cells.
    consumes_slot_witness = any(
        key[slot % len(key)] != key[paint_rank % len(key)]
        for paint_rank, slot in enumerate(intact_slots)
    )
    # At least one intact cell below row zero must disprove restarting the key each row.
    cross_row_witness = any(
        slot >= n_cols and key[slot % len(key)] != key[(slot % n_cols) % len(key)]
        for slot in intact_slots
    )
    filled_colors = {key[slot % len(key)] for slot in intact_slots}
    return consumes_slot_witness and cross_row_witness and len(filled_colors) >= 2


def generate_input() -> np.ndarray:
    """Generate a varied, semantically checked broken-lattice input of at most 30x30."""
    rng = np.random.default_rng()
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        n_rows = int(rng.integers(3, 6))
        n_cols = int(rng.integers(3, 7))
        cell_h = int(rng.integers(3, 5))
        cell_w = int(rng.integers(3, 5))
        height = 3 + n_rows * (cell_h + 1)
        width = 1 + n_cols * (cell_w + 1)
        if height > 30 or width > 30:
            continue

        possible_key_lengths = [length for length in range(3, 6) if n_cols % length != 0]
        key_length = int(rng.choice(possible_key_lengths))
        palette = list(map(int, rng.permutation(10)))
        bg, lattice_color = palette[:2]
        key = palette[2 : 2 + key_length]

        grid, expected, broken = _render_case(
            rng, n_rows, n_cols, cell_h, cell_w, bg, lattice_color, key
        )
        if _mode(grid) != bg:
            continue
        if _semantic_checks(grid, expected, broken, n_rows, n_cols, key, bg):
            return grid
    raise RuntimeError("failed to generate a non-degenerate broken-lattice input")
