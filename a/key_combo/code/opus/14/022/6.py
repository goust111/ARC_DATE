from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
GREY = 5
RED = 2
GEN_ATTEMPTS = 800
MOTIF_COLOR_POOL = (1, 3, 4, 6, 7, 8, 9)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_width(mask: np.ndarray) -> int:
    cols = np.flatnonzero(np.any(mask, axis=0))
    assert cols.size > 0
    return int(cols[-1] - cols[0] + 1)


def _parse_input(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int], int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = map(int, g.shape)
    assert 1 <= h <= 30 and 1 <= w <= 30

    bg = _get_bg(g)
    assert bg == BACKGROUND

    marker_rows = np.flatnonzero(g[:, 0] == GREY)
    assert 2 <= marker_rows.size <= 4
    assert np.array_equal(marker_rows, np.arange(marker_rows.size))
    assert int(np.count_nonzero(g == GREY)) == int(marker_rows.size)

    motif_raw = g[: marker_rows.size].copy()
    assert not np.any(motif_raw[:, 1:] == GREY)
    assert not np.any(motif_raw[:, 1:] == RED)
    motif = motif_raw.copy()
    motif[:, 0] = bg
    assert np.all(np.any(motif != bg, axis=1))
    assert _bbox_width(motif != bg) <= w // 2

    lower = g[marker_rows.size :]
    assert not np.any((lower != bg) & (lower != RED))

    red_coords = np.argwhere(g == RED)
    assert 2 <= red_coords.shape[0] <= 4
    red_coords = red_coords[np.argsort(red_coords[:, 0])]
    red_rows = [int(r) for r, _ in red_coords]
    red_cols = [int(c) for _, c in red_coords]
    assert all(r >= int(marker_rows.size) for r in red_rows)
    assert len(set(red_rows)) == len(red_rows)
    assert len(set(red_cols)) == len(red_cols)
    assert all(b - a >= 2 for a, b in zip(red_rows, red_rows[1:]))

    suffix_start = red_rows[-1] + 1
    assert suffix_start < h
    assert np.all(g[suffix_start:] == bg)
    assert np.count_nonzero(g[marker_rows.size : suffix_start] == RED) == len(red_rows)
    assert suffix_start - red_rows[0] >= 2
    assert h - suffix_start >= 2 * motif.shape[0]
    return g, motif_raw, red_cols, suffix_start, bg


def _render_shifted_block(motif: np.ndarray, shift: int, bg: int) -> np.ndarray:
    h, w = motif.shape
    block = np.full((h, w), bg, dtype=int)
    rows, cols = np.where(motif != bg)
    for r, c in zip(rows, cols):
        block[int(r), (int(c) + int(shift)) % w] = int(motif[int(r), int(c)])
    return block


def _transform(grid: np.ndarray, motif_rows: np.ndarray, offsets: list[int], suffix_start: int, bg: int) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    motif_h, width = motif_rows.shape
    out[:motif_h, 0] = bg
    out[out == RED] = bg

    shifted_blocks = [_render_shifted_block(motif_rows, shift, bg) for shift in offsets]
    for r in range(suffix_start, out.shape[0]):
        copy_idx = (r - suffix_start) // motif_h
        local_r = (r - suffix_start) % motif_h
        row = shifted_blocks[copy_idx % len(shifted_blocks)][local_r]
        paint = row != bg
        out[r, paint] = row[paint]

    assert out.shape[1] == width
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Fill the blank suffix with cyclically repeated motif copies whose horizontal shift is encoded by red anchor columns.

    Input structure:
    - The grid has a black background.
    - A contiguous grey run starts at the top of column 0; those rows contain the motif, with grey used only as a left-edge marker.
    - Below the motif are 2-4 isolated red pixels on distinct rows, one per row, whose column positions encode offsets; everything below the lowest red pixel is an empty suffix.

    Rule:
    - Extract the motif from the grey-marked rows, replacing the grey marker cells with background.
    - Read the red pixels from top to bottom and collect their column indices as a cyclic offset sequence.
    - Starting at the first empty suffix row, write motif copies back-to-back; copy i is shifted right by the corresponding offset, wrapping horizontally modulo the grid width.
    - Erase all grey and red marker pixels from the output while keeping the original motif colors.

    Key insight:
    - Grey marks which rows define the repeating motif, while red does not mark geometry to keep: its row order and column coordinates encode the repeating shift instructions.
    """
    g, motif_raw, offsets, suffix_start, bg = _parse_input(grid)
    motif = motif_raw.copy()
    motif[:, 0] = bg
    return _transform(g, motif, offsets, suffix_start, bg)


def _make_local_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for r in range(height):
        cols = random.sample(range(width), random.randint(1, width))
        take = random.randint(1, min(width, max(2, width - 1)))
        mask[r, cols[:take]] = True
    for c in range(width):
        mask[random.randrange(height), c] = True
    for r in range(height):
        for c in range(width):
            if not mask[r, c] and random.random() < 0.25:
                mask[r, c] = True
    assert np.all(mask.any(axis=1))
    assert np.all(mask.any(axis=0))
    return mask


def _make_motif(width: int, motif_h: int) -> np.ndarray:
    motif_w = random.randint(3, min(6, width // 2))
    max_left = min(width - motif_w, max(3, width // 3))
    left = random.randint(1, max_left)
    local_mask = _make_local_mask(motif_h, motif_w)

    row_signatures = {tuple(int(v) for v in row) for row in local_mask.astype(int)}
    if len(row_signatures) == 1:
        rr = random.randrange(motif_h)
        cc = random.randrange(motif_w)
        local_mask[rr, cc] = not local_mask[rr, cc]
        if not local_mask[rr].any():
            local_mask[rr, cc] = True

    motif = np.full((motif_h, width), BACKGROUND, dtype=int)
    color_count = random.randint(2, 3)
    colors = random.sample(list(MOTIF_COLOR_POOL), color_count)
    filled = list(zip(*np.where(local_mask)))
    random.shuffle(filled)
    assignments = colors + [random.choice(colors) for _ in range(len(filled) - color_count)]
    random.shuffle(assignments)
    for (r, c), color in zip(filled, assignments):
        motif[int(r), int(left + c)] = int(color)
    return motif


def _choose_offsets(width: int, motif: np.ndarray, count: int) -> tuple[list[int], bool]:
    cols = np.flatnonzero(np.any(motif != BACKGROUND, axis=0))
    max_col = int(cols[-1])
    no_wrap_limit = width - 1 - max_col
    want_wrap = random.random() < 0.5

    no_wrap = [c for c in range(1, width) if c <= no_wrap_limit]
    wrap = [c for c in range(1, width) if c > no_wrap_limit]
    if want_wrap and wrap:
        chosen = [random.choice(wrap)]
        pool = [c for c in range(1, width) if c not in chosen]
        chosen.extend(random.sample(pool, count - 1))
    else:
        source = no_wrap if len(no_wrap) >= count else list(range(1, width))
        chosen = random.sample(source, count)
        want_wrap = False

    random.shuffle(chosen)
    if chosen == list(reversed(chosen)):
        chosen[0], chosen[-1] = chosen[-1], chosen[0]
    assert len(set(chosen)) == count
    if want_wrap:
        assert any(shift + max_col >= width for shift in chosen)
    else:
        assert all(shift + max_col < width for shift in chosen)
    return [int(x) for x in chosen], want_wrap


def _build_input() -> np.ndarray | None:
    width = random.randint(14, 22)
    motif_h = random.randint(2, 4)
    red_count = random.randint(2, 4)
    pre_red_blank = random.randint(1, 2)
    fixed_prefix = motif_h + pre_red_blank + 2 * (red_count - 1) + 1
    max_blocks = min(5, (30 - fixed_prefix + motif_h - 1) // motif_h)
    if max_blocks < 3:
        return None

    block_min = 3
    if red_count == 4 and max_blocks >= 5:
        block_count = 5
    elif max_blocks >= red_count + 1 and random.random() < 0.55:
        block_count = random.randint(red_count + 1, max_blocks)
    else:
        block_count = random.randint(block_min, max_blocks)
    truncation = random.randint(0, motif_h - 1)
    suffix_h = block_count * motif_h - truncation
    if suffix_h < 2 * motif_h:
        return None
    height = fixed_prefix + suffix_h
    if height > 30:
        return None

    motif = _make_motif(width, motif_h)
    offsets, _ = _choose_offsets(width, motif, red_count)

    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[:motif_h] = motif
    grid[:motif_h, 0] = GREY

    first_red_row = motif_h + pre_red_blank
    red_rows = [first_red_row + 2 * i for i in range(red_count)]
    for row, col in zip(red_rows, offsets):
        grid[int(row), int(col)] = RED
    return grid


def _validate_example(grid: np.ndarray) -> bool:
    try:
        g, motif_raw, offsets, suffix_start, bg = _parse_input(grid)
        motif = motif_raw.copy()
        motif[:, 0] = bg
        out = main(g)
        expected = _transform(g, motif, offsets, suffix_start, bg)
        if not np.array_equal(out, expected):
            return False

        no_shift = _transform(g, motif, [0] * len(offsets), suffix_start, bg)
        if np.array_equal(out, no_shift):
            return False

        same_shift = _transform(g, motif, [offsets[0]] * len(offsets), suffix_start, bg)
        if np.array_equal(out, same_shift):
            return False

        reversed_offsets = list(reversed(offsets))
        if reversed_offsets == offsets:
            return False
        reversed_out = _transform(g, motif, reversed_offsets, suffix_start, bg)
        if np.array_equal(out, reversed_out):
            return False

        raw_repeat = _transform(g, motif_raw, offsets, suffix_start, bg)
        if np.array_equal(out, raw_repeat):
            return False

        distinct_blocks = {
            _render_shifted_block(motif, shift, bg).tobytes()
            for shift in offsets
        }
        if len(distinct_blocks) < 2:
            return False

        motif_h = motif.shape[0]
        suffix_h = g.shape[0] - suffix_start
        if suffix_h < 2 * motif_h:
            return False
        if np.any(out[:motif_h, 0] != bg):
            return False
        if np.any(out[np.argwhere(g == RED)[:, 0], np.argwhere(g == RED)[:, 1]] != bg):
            return False
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_input()
        if grid is not None and _validate_example(grid):
            return grid
    raise RuntimeError("failed to generate valid puzzle input")
