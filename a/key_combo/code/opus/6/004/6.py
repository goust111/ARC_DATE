from __future__ import annotations

import random

import numpy as np


BG = 0
MARKER = 3
BLOCK_COLORS = (1, 2, 4, 5, 6, 7, 8, 9)
GEN_RETRIES = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _marker_rows(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    height, width = grid.shape
    rows = []
    for r in range(height):
        cols = np.flatnonzero(grid[r] == MARKER)
        if cols.size == 0:
            continue
        assert cols.size == 2, "each marker row must contain exactly two green pixels"
        c0, c1 = map(int, cols)
        assert c0 + c1 == width - 1, "marker pair must be symmetric about the grid center axis"
        rows.append((int(r), c0, c1, c1 - c0 - 1))
    assert rows, "expected at least one marker row"
    return rows


def _foreground_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    mask = (grid != BG) & (grid != MARKER)
    box = _bbox(mask)
    assert box is not None, "expected one non-black, non-green foreground block"
    r0, c0, r1, c1 = box
    assert np.all(mask[r0 : r1 + 1, c0 : c1 + 1]), "foreground block must be a filled rectangle"
    return r0, c0, r1, c1


def _shear_rows(block_rows: np.ndarray, block_left: int, hinge_row: int, out: np.ndarray, block_top: int) -> None:
    height, width = out.shape
    block_width = block_rows.shape[1]
    for local_r in range(block_rows.shape[0]):
        r = block_top + local_r
        shift = r - hinge_row
        start = block_left + shift
        src0 = max(0, -start)
        dst0 = max(0, start)
        length = min(block_width - src0, width - dst0)
        if length > 0:
            out[r, dst0 : dst0 + length] = block_rows[local_r, src0 : src0 + length]


def main(grid: np.ndarray) -> np.ndarray:
    """Shear the filled foreground rectangle using the row of the tightest green marker pair as the hinge.

    Input structure:
    - The grid uses black as background.
    - One filled rectangular foreground block of non-black, non-green colors sits near the center.
    - Above and below the block, each marker row contains exactly two green pixels mirrored across the grid's vertical center axis.

    Rule:
    - Compute each marker row's gap as the number of cells strictly between its two green pixels.
    - Find the unique marker row with the smallest gap; its row index is the hinge row.
    - Erase all green markers.
    - For every row r of the foreground rectangle, shift that entire row horizontally by r - hinge_row cells.
    - Pixels shifted beyond the left or right edge are discarded; vacated cells become black.

    Key insight:
    - The green pairs do not become output objects; their relative spacing encodes a geometric parameter, and that parameter controls the rectangle's shear axis.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    markers = _marker_rows(g)
    gaps = [gap for _r, _c0, _c1, gap in markers]
    min_gap = min(gaps)
    hinge_rows = [r for r, _c0, _c1, gap in markers if gap == min_gap]
    assert len(hinge_rows) == 1, "expected a unique minimum marker gap"
    hinge_row = hinge_rows[0]

    r0, c0, r1, c1 = _foreground_bbox(g)
    block = g[r0 : r1 + 1, c0 : c1 + 1]

    out = np.zeros_like(g)
    _shear_rows(block, c0, hinge_row, out, r0)
    return out


def _random_palette() -> list[int]:
    n_colors = random.randint(2, 4)
    return random.sample(BLOCK_COLORS, n_colors)


def _build_block(height: int, width: int) -> np.ndarray:
    palette = _random_palette()
    block = np.zeros((height, width), dtype=int)
    used_rows: set[tuple[int, ...]] = set()

    for r in range(height):
        for _ in range(80):
            n_cuts = random.randint(1, min(3, width - 1))
            cuts = sorted(random.sample(range(1, width), n_cuts))
            spans = [0] + cuts + [width]
            row = np.empty(width, dtype=int)
            last_color = None
            for c0, c1 in zip(spans[:-1], spans[1:]):
                choices = [color for color in palette if color != last_color] or palette
                color = random.choice(choices)
                row[c0:c1] = color
                last_color = color
            key = tuple(map(int, row))
            if len(set(key)) >= 2 and key not in used_rows:
                block[r] = row
                used_rows.add(key)
                break
        else:
            row = np.asarray([(palette[(c + r) % len(palette)]) for c in range(width)], dtype=int)
            block[r] = row
            used_rows.add(tuple(map(int, row)))

    if len(np.unique(block)) < 2:
        block[0, 0] = palette[0]
        block[-1, -1] = palette[-1]
    return block


def _choose_hinge_row(rows: list[int], *, near_block_row: int, prefer_non_nearest: bool) -> int:
    if len(rows) == 1:
        return rows[0]
    if not prefer_non_nearest:
        return random.choice(rows)
    alternatives = [r for r in rows if r != near_block_row]
    return random.choice(alternatives or rows)


def _candidate_positions(width: int, block_width: int, max_left_shift: int, max_right_shift: int) -> list[int]:
    axis_left = (width - 1) // 2
    axis_right = width // 2
    positions = []
    for c0 in range(width - block_width + 1):
        c1 = c0 + block_width - 1
        if c0 - max_left_shift < 0 or c1 + max_right_shift >= width:
            continue
        if not (c0 <= axis_left and c1 >= axis_right):
            continue
        positions.append(c0)
    return positions


def _render_input(
    height: int,
    width: int,
    block: np.ndarray,
    block_top: int,
    block_left: int,
    marker_cols: dict[int, int],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    block_height, block_width = block.shape
    grid[block_top : block_top + block_height, block_left : block_left + block_width] = block
    for r, left in marker_cols.items():
        grid[r, left] = MARKER
        grid[r, width - 1 - left] = MARKER
    return grid


def _render_expected(
    height: int,
    width: int,
    block: np.ndarray,
    block_top: int,
    block_left: int,
    hinge_row: int,
) -> np.ndarray:
    out = np.zeros((height, width), dtype=int)
    _shear_rows(block, block_left, hinge_row, out, block_top)
    return out


def _validate_input_structure(
    grid: np.ndarray,
    expected_hinge: int,
    expected_bbox: tuple[int, int, int, int],
) -> bool:
    try:
        markers = _marker_rows(grid)
        bbox = _foreground_bbox(grid)
    except AssertionError:
        return False

    if bbox != expected_bbox:
        return False

    r0, _c0, r1, _c1 = bbox
    top_rows = [r for r, _l, _rr, _gap in markers if r < r0]
    bottom_rows = [r for r, _l, _rr, _gap in markers if r > r1]
    if len(top_rows) not in (2, 3) or len(bottom_rows) not in (2, 3):
        return False
    if any(r0 <= r <= r1 for r, _l, _rr, _gap in markers):
        return False

    gaps = [gap for _r, _l, _rr, gap in markers]
    if len(set(gaps)) != len(gaps):
        return False
    hinge_candidates = [r for r, _l, _rr, gap in markers if gap == min(gaps)]
    if hinge_candidates != [expected_hinge]:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    block_height = random.randint(5, 8)
    block_width = random.randint(6, 8)
    n_top = random.randint(2, 3)
    n_bottom = random.randint(2, 3)
    top_space = random.randint(n_top, n_top + 2)
    bottom_space = random.randint(n_bottom, n_bottom + 2)
    height = top_space + block_height + bottom_space
    if height > 22:
        return None

    block_top = top_space
    block_bottom = block_top + block_height - 1
    top_region = list(range(block_top))
    bottom_region = list(range(block_bottom + 1, height))
    if len(top_region) < n_top or len(bottom_region) < n_bottom:
        return None

    top_rows = sorted(random.sample(top_region, n_top))
    bottom_rows = sorted(random.sample(bottom_region, n_bottom))

    hinge_side = random.choice(("top", "bottom"))
    if hinge_side == "top":
        hinge_row = _choose_hinge_row(
            top_rows,
            near_block_row=max(top_rows),
            prefer_non_nearest=len(top_rows) >= 2 and random.random() < 0.7,
        )
    else:
        hinge_row = _choose_hinge_row(
            bottom_rows,
            near_block_row=min(bottom_rows),
            prefer_non_nearest=len(bottom_rows) >= 2 and random.random() < 0.7,
        )

    max_left_shift = max(0, hinge_row - block_top)
    max_right_shift = max(0, block_bottom - hinge_row)

    total_marker_rows = n_top + n_bottom
    min_width = max(12, 2 * total_marker_rows, block_width + max_left_shift + max_right_shift + 1)
    viable_widths = []
    position_catalog: dict[int, list[int]] = {}
    for width in range(min_width, 21):
        if width // 2 < total_marker_rows:
            continue
        positions = _candidate_positions(width, block_width, max_left_shift, max_right_shift)
        if positions:
            viable_widths.append(width)
            position_catalog[width] = positions
    if not viable_widths:
        return None

    width = random.choice(viable_widths)
    block_left = random.choice(position_catalog[width])

    all_rows = top_rows + bottom_rows
    left_choices = random.sample(range(width // 2), total_marker_rows)
    hinge_left = max(left_choices)
    other_lefts = [left for left in left_choices if left != hinge_left]
    random.shuffle(other_lefts)

    marker_cols: dict[int, int] = {}
    for r in all_rows:
        if r == hinge_row:
            marker_cols[r] = hinge_left
        else:
            marker_cols[r] = other_lefts.pop()

    block = _build_block(block_height, block_width)
    inp = _render_input(height, width, block, block_top, block_left, marker_cols)
    expected = _render_expected(height, width, block, block_top, block_left, hinge_row)

    bbox = (block_top, block_left, block_bottom, block_left + block_width - 1)
    if not _validate_input_structure(inp, hinge_row, bbox):
        return None
    try:
        produced = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(produced, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
