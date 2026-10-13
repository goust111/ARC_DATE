from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
MARKER = 2
SHAPE_COLORS = [1, 3, 4, 5, 6, 7, 8, 9]
MAX_RETRIES = 400
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid_key(grid: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(map(int, row)) for row in grid.tolist())


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in NEIGHBORS4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    seen = {next(iter(cells))}
    stack = list(seen)
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nbr = (r + dr, c + dc)
            if nbr in cells and nbr not in seen:
                seen.add(nbr)
                stack.append(nbr)
    return seen == cells


def _find_separator_pair(grid: np.ndarray) -> int:
    h, w = grid.shape
    candidates = []
    for c in range(w - 1):
        if np.all(grid[:, c] == BG) and np.all(grid[:, c + 1] == BG):
            left = grid[:, :c]
            right = grid[:, c + 2 :]
            if left.size == 0 or right.size == 0:
                continue
            if left.shape[0] % 2 != 1 or left.shape[1] % 2 != 1:
                continue
            mid_r = left.shape[0] // 2
            mid_c = left.shape[1] // 2
            if np.all(left[mid_r, :] == BG) and np.all(left[:, mid_c] == BG):
                candidates.append(c)
    assert len(candidates) == 1, "expected exactly one 2-column separator"
    return candidates[0]


def _quadrants(left: np.ndarray) -> tuple[list[np.ndarray], int, int]:
    h, w = left.shape
    assert h % 2 == 1 and w % 2 == 1
    qh, qw = h // 2, w // 2
    mid_r, mid_c = qh, qw
    assert np.all(left[mid_r, :] == BG)
    assert np.all(left[:, mid_c] == BG)
    quads = [
        left[:qh, :qw],
        left[:qh, qw + 1 :],
        left[qh + 1 :, :qw],
        left[qh + 1 :, qw + 1 :],
    ]
    return quads, qh, qw


def _unique_quadrant(quadrants: list[np.ndarray]) -> np.ndarray:
    keys = [_grid_key(q) for q in quadrants]
    counts = Counter(keys)
    unique = [idx for idx, key in enumerate(keys) if counts[key] == 1]
    assert len(unique) == 1, "expected one odd quadrant"
    return quadrants[unique[0]]


def _template_cells(template_grid: np.ndarray) -> list[tuple[int, int, int]]:
    cells = []
    for r, c in np.argwhere(template_grid != BG):
        cells.append((int(r), int(c), int(template_grid[r, c])))
    assert cells, "template must be non-empty"
    return cells


def _active_marker_height(marker_region: np.ndarray, block_h: int) -> int:
    nonzero_rows = np.where(np.any(marker_region != BG, axis=1))[0]
    assert len(nonzero_rows) > 0, "marker region must contain red markers"
    active_h = ((int(nonzero_rows[-1]) + 1 + block_h - 1) // block_h) * block_h
    assert active_h <= marker_region.shape[0]
    return active_h


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the odd quadrant, turn its colored shape into a stamp template, and render it in marked blocks on the right.

    Input structure:
    - A black background grid is split by two black separator columns into a left quadrant area and a right marker area.
    - The left area has a black separator cross: one full black row and one full black column divide it into four equal quadrants.
    - Three quadrants contain the same non-black pattern; the remaining quadrant contains one different colored pattern.
    - The right area is partitioned into blocks the same size as a quadrant, and some blocks contain red marker pixels.

    Rule:
    - Compare the four quadrant arrays and locate the unique quadrant.
    - Extract every non-black pixel of that unique quadrant as a template, keeping each pixel's offset and color.
    - Crop the right side to the active marker region, partition it into quadrant-sized blocks, and stamp the template into every block that contains at least one red pixel.
    - Leave blocks without any red marker pixel black, and output only the rendered right-side region.

    Key insight:
    - The odd quadrant is not itself the answer; it is a hidden template that must be discovered before the marker blocks on the right can be interpreted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    sep_c = _find_separator_pair(g)
    left = g[:, :sep_c]
    right = g[:, sep_c + 2 :]
    quadrants, qh, qw = _quadrants(left)
    odd = _unique_quadrant(quadrants)
    template = _template_cells(odd)

    active_h = _active_marker_height(right, qh)
    active = right[:active_h, :]
    assert active.shape[1] % qw == 0, "right side width must be a multiple of the quadrant width"

    out = np.zeros_like(active)
    block_rows = active.shape[0] // qh
    block_cols = active.shape[1] // qw
    for br in range(block_rows):
        for bc in range(block_cols):
            block = active[br * qh : (br + 1) * qh, bc * qw : (bc + 1) * qw]
            if not np.any(block == MARKER):
                continue
            for dr, dc, color in template:
                out[br * qh + dr, bc * qw + dc] = color
    return out


def _random_connected_shape(h: int, w: int, *, min_cells: int, max_cells: int) -> set[tuple[int, int]] | None:
    cells = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    if not cells:
        return None
    max_cells = min(max_cells, len(cells))
    if max_cells < min_cells:
        return None
    target = random.randint(min_cells, max_cells)
    shape = {random.choice(cells)}
    while len(shape) < target:
        frontier = sorted(
            {
                nbr
                for r, c in shape
                for nbr in _neighbors4(r, c, h, w)
                if nbr in cells and nbr not in shape
            }
        )
        if not frontier:
            return None
        shape.add(random.choice(frontier))
    rows = {r for r, _ in shape}
    cols = {c for _, c in shape}
    if len(rows) < 2 or len(cols) < 2:
        return None
    return shape


def _paint_shape(shape: set[tuple[int, int]], h: int, w: int, color: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for r, c in shape:
        grid[r, c] = color
    return grid


def _make_quadrant_pair(qh: int, qw: int) -> tuple[np.ndarray, np.ndarray] | None:
    majority_color, odd_color = random.sample(SHAPE_COLORS, 2)
    majority_shape = _random_connected_shape(qh, qw, min_cells=3, max_cells=7)
    odd_shape = _random_connected_shape(qh, qw, min_cells=3, max_cells=8)
    if majority_shape is None or odd_shape is None:
        return None
    if majority_shape == odd_shape:
        return None
    majority = _paint_shape(majority_shape, qh, qw, majority_color)
    odd = _paint_shape(odd_shape, qh, qw, odd_color)
    return majority, odd


def _render_left(majority: np.ndarray, odd: np.ndarray, odd_slot: int) -> np.ndarray:
    qh, qw = majority.shape
    left = np.zeros((2 * qh + 1, 2 * qw + 1), dtype=int)
    slots = [
        (0, 0),
        (0, qw + 1),
        (qh + 1, 0),
        (qh + 1, qw + 1),
    ]
    for idx, (r0, c0) in enumerate(slots):
        left[r0 : r0 + qh, c0 : c0 + qw] = odd if idx == odd_slot else majority
    return left


def _template_offsets(template: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.argwhere(template != BG)}


def _choose_marked_blocks(block_rows: int, block_cols: int) -> list[tuple[int, int]] | None:
    all_blocks = [(br, bc) for br in range(block_rows) for bc in range(block_cols)]
    if len(all_blocks) < 3:
        return None
    marked = set()
    for br in range(block_rows):
        choices = [(br, bc) for bc in range(block_cols)]
        marked.add(random.choice(choices))
    extra_count = random.randint(0, len(all_blocks) - len(marked) - 1)
    extras = [block for block in all_blocks if block not in marked]
    random.shuffle(extras)
    for block in extras[:extra_count]:
        marked.add(block)
    if len(marked) < 2 or len(marked) >= len(all_blocks):
        return None
    return sorted(marked)


def _build_right(template: np.ndarray, qh: int, qw: int) -> tuple[np.ndarray, np.ndarray] | None:
    block_rows = random.choice((1, 2))
    block_cols = random.choice((2, 3))
    marked_blocks = _choose_marked_blocks(block_rows, block_cols)
    if marked_blocks is None:
        return None

    right_h = block_rows * qh
    right_w = block_cols * qw
    markers = np.zeros((right_h, right_w), dtype=int)
    out = np.zeros((right_h, right_w), dtype=int)
    template_cells = _template_cells(template)
    template_offsets = {(r, c) for r, c, _ in template_cells}
    all_offsets = [(r, c) for r in range(qh) for c in range(qw)]
    outside_offsets = [pos for pos in all_offsets if pos not in template_offsets]
    if not outside_offsets:
        return None

    overlap_block = random.choice(marked_blocks)
    multi_block = random.choice(marked_blocks)
    for br, bc in marked_blocks:
        base_r = br * qh
        base_c = bc * qw
        local_marks = set()
        count = random.randint(1, 3)
        if (br, bc) == multi_block:
            count = max(2, count)
        if (br, bc) == overlap_block:
            local_marks.add(random.choice(sorted(template_offsets)))
        if (br, bc) == multi_block:
            local_marks.add(random.choice(outside_offsets))
        while len(local_marks) < count:
            local_marks.add(random.choice(all_offsets))
        for lr, lc in local_marks:
            markers[base_r + lr, base_c + lc] = MARKER
        for dr, dc, color in template_cells:
            out[base_r + dr, base_c + dc] = color

    for br in range(block_rows):
        row_blocks = [(r, c) for r, c in marked_blocks if r == br]
        anchor_br, anchor_bc = random.choice(row_blocks)
        anchor_c = random.randrange(qw)
        markers[(anchor_br + 1) * qh - 1, anchor_bc * qw + anchor_c] = MARKER

    block_has_multi = any(
        np.count_nonzero(markers[br * qh : (br + 1) * qh, bc * qw : (bc + 1) * qw] == MARKER) >= 2
        for br, bc in marked_blocks
    )
    if not block_has_multi:
        return None
    if not any(
        (markers[br * qh : (br + 1) * qh, bc * qw : (bc + 1) * qw] == MARKER).any()
        and any(
            markers[br * qh + r, bc * qw + c] == MARKER and (r, c) not in template_offsets
            for r in range(qh)
            for c in range(qw)
        )
        for br, bc in marked_blocks
    ):
        return None
    if not any(
        any(markers[br * qh + r, bc * qw + c] == MARKER for r, c in template_offsets)
        for br, bc in marked_blocks
    ):
        return None
    return markers, out


def _assemble_input(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    h = left.shape[0]
    grid = np.zeros((h, left.shape[1] + 2 + right.shape[1]), dtype=int)
    grid[:, : left.shape[1]] = left
    grid[: right.shape[0], left.shape[1] + 2 :] = right
    return grid


def _semantic_checks(
    inp: np.ndarray,
    expected_out: np.ndarray,
    *,
    qh: int,
    qw: int,
    right_h: int,
) -> bool:
    try:
        sep_c = _find_separator_pair(inp)
        left = inp[:, :sep_c]
        right = inp[:, sep_c + 2 :]
        quadrants, got_qh, got_qw = _quadrants(left)
        if (got_qh, got_qw) != (qh, qw):
            return False
        odd = _unique_quadrant(quadrants)
        if np.count_nonzero(odd) < 3:
            return False
        active_h = _active_marker_height(right, qh)
        if active_h != right_h:
            return False
        if active_h % qh != 0 or right.shape[1] % qw != 0:
            return False
        block_rows = active_h // qh
        block_cols = right.shape[1] // qw
        marked_blocks = 0
        blank_blocks = 0
        for br in range(block_rows):
            row_has_bottom = False
            for bc in range(block_cols):
                block = right[br * qh : (br + 1) * qh, bc * qw : (bc + 1) * qw]
                has_marker = np.any(block == MARKER)
                if has_marker:
                    marked_blocks += 1
                else:
                    blank_blocks += 1
                if np.any(block[-1, :] == MARKER):
                    row_has_bottom = True
            if not row_has_bottom:
                return False
        if marked_blocks < 2 or blank_blocks < 1:
            return False
        return np.array_equal(main(inp), expected_out)
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    qh = random.randint(4, 5)
    qw = random.randint(4, 5)
    pair = _make_quadrant_pair(qh, qw)
    if pair is None:
        return None
    majority, odd = pair
    odd_slot = random.randrange(4)
    left = _render_left(majority, odd, odd_slot)
    built = _build_right(odd, qh, qw)
    if built is None:
        return None
    right_markers, expected_out = built
    inp = _assemble_input(left, right_markers)
    if not _semantic_checks(inp, expected_out, qh=qh, qw=qw, right_h=right_markers.shape[0]):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
