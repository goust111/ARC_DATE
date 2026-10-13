from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7
MAROON = 9

BLOCK = 3
TILE_COLORS = (BLUE, RED, GREEN, ORANGE)


def _nonempty_rows(grid: np.ndarray) -> np.ndarray:
    return np.any(grid != BLACK, axis=1)


def _find_stripe_row(grid: np.ndarray) -> int:
    h, _w = grid.shape
    stripe_rows = [r for r in range(h) if np.all(grid[r] == MAROON)]
    assert len(stripe_rows) == 1, "expected one maroon full-width stripe"
    stripe_row = int(stripe_rows[0])
    assert stripe_row == h - 1, "stripe must be on the bottom row"
    return stripe_row


def _find_tile_region(grid: np.ndarray, stripe_row: int) -> tuple[int, int, int, int]:
    rows = _nonempty_rows(grid[:stripe_row])
    assert rows.any(), "expected a non-empty tile band above the stripe"

    top = int(np.argmax(rows))
    bottom = top
    while bottom + 1 < stripe_row and rows[bottom + 1]:
        bottom += 1
    assert (bottom - top + 1) % BLOCK == 0, "tile band height must be a multiple of 3"

    band = grid[top : bottom + 1]
    cols = np.any(band != BLACK, axis=0)
    assert cols.any(), "expected active columns inside the tile band"
    left = int(np.argmax(cols))
    right = int(len(cols) - 1 - np.argmax(cols[::-1]))
    assert (right - left + 1) % BLOCK == 0, "tile band width must be a multiple of 3"
    return top, left, bottom, right


def _collapse_block(block: np.ndarray) -> int:
    colors = [int(c) for c in np.unique(block) if int(c) not in (BLACK, GREY)]
    assert len(colors) <= 1, "each tile must contain at most one real color"
    if not colors:
        assert np.all((block == BLACK) | (block == GREY)), "background tile may only contain black/grey"
        return BLACK

    color = int(colors[0])
    assert int(np.count_nonzero(block == color)) >= 8, "colored tile must keep at least 8 main-color pixels"
    assert np.all((block == color) | (block == GREY)), "colored tile noise must be grey only"
    return color


def _collapse_tile_region(grid: np.ndarray, tile_bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = tile_bbox
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = patch.shape
    out = np.full((h // BLOCK, w // BLOCK), BLACK, dtype=int)
    for br in range(h // BLOCK):
        for bc in range(w // BLOCK):
            block = patch[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
            out[br, bc] = _collapse_block(block)
    return out


def _color_frequencies(coarse: np.ndarray) -> dict[int, int]:
    freq: dict[int, int] = {}
    for color in TILE_COLORS:
        count = int(np.count_nonzero(coarse == color))
        if count:
            freq[int(color)] = count
    return freq


def _extract_segments(grid: np.ndarray, start_row: int, stripe_row: int) -> list[dict[str, int]]:
    segments: list[dict[str, int]] = []
    for r in range(start_row, stripe_row):
        cols = np.flatnonzero(grid[r] != BLACK)
        if cols.size == 0:
            continue
        colors = np.unique(grid[r, cols])
        assert colors.size == 1, "each segment row should contain one segment color"
        color = int(colors[0])
        assert color in TILE_COLORS, "segment colors must come from the tile palette"
        c0, c1 = int(cols[0]), int(cols[-1])
        assert int(cols.size) == c1 - c0 + 1, "segment must be contiguous"
        segments.append({"row": int(r), "color": color, "length": int(cols.size)})

    assert 2 <= len(segments) <= 4, "expected 2-4 line segments"
    assert len({seg["color"] for seg in segments}) == len(segments), "segment colors must be distinct"
    assert len({seg["length"] for seg in segments}) == len(segments), "segment lengths must be distinct"
    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """Denoise a 3x3 tile picture, use coarse color frequencies as lookup keys, and restack matching segments.

    Input structure:
    - Black is background and the bottom row is a full-width maroon stripe.
    - The upper part contains a rectangular 3x3-tile matrix; each tile is either
      a noisy monochrome tile in blue/red/green/orange or a black/grey background tile.
    - Below the tile matrix and above the stripe are 2-4 one-pixel-thick horizontal
      segments in distinct colors chosen from the same blue/red/green/orange palette.

    Rule:
    - Collapse each 3x3 tile by ignoring grey noise and keeping its unique non-black color,
      or black if the tile contains only black/grey.
    - Count the occurrences of each non-black color in that coarse grid.
    - Use the coarse-grid frequency of a segment's color as that segment's sorting key.
    - Clear the original segment positions, then place the segments in ascending key order
      on consecutive rows immediately above the maroon stripe, right-aligned to the grid edge.
    - Leave the noisy tile matrix and the maroon stripe unchanged.

    Key insight:
    - The top tile mosaic is a hidden lookup table: after denoising and compression, each
      segment color maps to an external frequency key that determines the segment stack order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    stripe_row = _find_stripe_row(g)
    tile_bbox = _find_tile_region(g, stripe_row)
    coarse = _collapse_tile_region(g, tile_bbox)
    freq = _color_frequencies(coarse)

    tile_bottom = int(tile_bbox[2])
    segments = _extract_segments(g, tile_bottom + 1, stripe_row)
    for seg in segments:
        assert seg["color"] in freq, "every segment color must exist in the coarse grid"

    out = g.copy()
    out[tile_bottom + 1 : stripe_row, :] = BLACK

    ordered = sorted(segments, key=lambda seg: (freq[seg["color"]], seg["length"]))
    for i, seg in enumerate(ordered):
        r = stripe_row - len(ordered) + i
        out[r, g.shape[1] - seg["length"] :] = seg["color"]
    return out


def _sample_dimensions(n_colors: int) -> tuple[int, int]:
    options = [(4, 5), (4, 6), (5, 5), (5, 6)]
    random.shuffle(options)
    for rows, cols in options:
        cells = rows * cols
        min_needed = 20 + (n_colors - 2) * (n_colors - 1) // 2
        if cells >= min_needed:
            return rows, cols
    raise RuntimeError("no feasible coarse dimensions")


def _sample_counts(n_colors: int, total_cells: int) -> list[int]:
    min_extra = sum(range(1, n_colors - 1))
    feasible_m = [m for m in range(9, 12) if 2 * m + 1 + min_extra <= total_cells - 1]
    if not feasible_m:
        raise RuntimeError("no feasible critical-frequency pair")

    m = random.choice(feasible_m)
    critical = [m + 1, m]
    if n_colors == 2:
        return critical

    pool = [v for v in range(1, 8) if v not in critical]
    while True:
        extras = random.sample(pool, n_colors - 2)
        if sum(critical) + sum(extras) <= total_cells - 1:
            return critical + extras


def _sample_coarse_grid(rows: int, cols: int, color_counts: dict[int, int]) -> np.ndarray:
    total = rows * cols
    cells = []
    for color, count in color_counts.items():
        cells.extend([int(color)] * int(count))
    cells.extend([BLACK] * (total - len(cells)))
    assert cells.count(BLACK) >= 1, "need at least one background tile"

    for _ in range(500):
        random.shuffle(cells)
        coarse = np.array(cells, dtype=int).reshape(rows, cols)
        if np.any(np.all(coarse == BLACK, axis=1)):
            continue
        if np.any(np.all(coarse == BLACK, axis=0)):
            continue
        return coarse
    raise RuntimeError("failed to place coarse colors with row/column coverage")


def _make_noisy_block(color: int, grey_count: int) -> np.ndarray:
    block = np.full((BLOCK, BLOCK), color, dtype=int)
    if grey_count:
        for idx in random.sample(range(BLOCK * BLOCK), grey_count):
            block[idx // BLOCK, idx % BLOCK] = GREY
    return block


def _render_upper_grid(coarse: np.ndarray, noise_plan: dict[tuple[int, int], int]) -> np.ndarray:
    rows, cols = coarse.shape
    upper = np.zeros((rows * BLOCK, cols * BLOCK), dtype=int)
    for r in range(rows):
        for c in range(cols):
            color = int(coarse[r, c])
            grey_count = int(noise_plan[(r, c)])
            if color == BLACK:
                block = np.zeros((BLOCK, BLOCK), dtype=int)
                if grey_count:
                    for idx in random.sample(range(BLOCK * BLOCK), grey_count):
                        block[idx // BLOCK, idx % BLOCK] = GREY
            else:
                block = _make_noisy_block(color, grey_count)
            r0 = r * BLOCK
            c0 = c * BLOCK
            upper[r0 : r0 + BLOCK, c0 : c0 + BLOCK] = block
    return upper


def _raw_upper_color_counts(upper: np.ndarray) -> dict[int, int]:
    return {color: int(np.count_nonzero(upper == color)) for color in TILE_COLORS if np.any(upper == color)}


def _frequency_order(colors: list[int], freq: dict[int, int]) -> list[int]:
    return sorted(colors, key=lambda color: freq[color])


def _choose_segment_lengths(width: int, seg_colors: list[int], freq: dict[int, int]) -> dict[int, int]:
    max_len = min(width - 2, 9)
    assert max_len >= len(seg_colors), "grid must be wide enough for unique segment lengths"
    values = sorted(random.sample(range(1, max_len + 1), len(seg_colors)))
    freq_order = _frequency_order(seg_colors, freq)
    while True:
        perm = list(range(len(seg_colors)))
        random.shuffle(perm)
        if perm != list(range(len(seg_colors))):
            break
    return {freq_order[i]: values[perm[i]] for i in range(len(seg_colors))}


def _target_rows(stripe_row: int, n_segments: int) -> list[int]:
    return list(range(stripe_row - n_segments, stripe_row))


def _build_output(
    upper: np.ndarray,
    top_pad: int,
    left_pad: int,
    width: int,
    height: int,
    seg_lengths: dict[int, int],
    seg_colors: list[int],
    freq: dict[int, int],
) -> np.ndarray:
    out = np.zeros((height, width), dtype=int)
    tile_h, tile_w = upper.shape
    out[top_pad : top_pad + tile_h, left_pad : left_pad + tile_w] = upper
    out[-1, :] = MAROON

    ordered = _frequency_order(seg_colors, freq)
    for r, color in zip(_target_rows(height - 1, len(ordered)), ordered):
        length = seg_lengths[color]
        out[r, width - length : width] = color
    return out


def _scatter_input(
    out: np.ndarray,
    tile_bottom: int,
    seg_lengths: dict[int, int],
    seg_colors: list[int],
    freq: dict[int, int],
) -> np.ndarray:
    inp = out.copy()
    stripe_row = inp.shape[0] - 1
    target_rows = set(_target_rows(stripe_row, len(seg_colors)))
    inp[tile_bottom + 1 : stripe_row, :] = BLACK

    candidate_rows = [r for r in range(tile_bottom + 2, stripe_row - len(seg_colors)) if r not in target_rows]
    assert len(candidate_rows) >= len(seg_colors), "need enough scatter rows"

    for _ in range(200):
        inp[tile_bottom + 1 : stripe_row, :] = BLACK
        rows = sorted(random.sample(candidate_rows, len(seg_colors)))
        row_colors = random.sample(seg_colors, len(seg_colors))
        ordered_by_row = [color for _row, color in sorted(zip(rows, row_colors))]
        if ordered_by_row == _frequency_order(seg_colors, freq):
            continue

        ok = True
        for r, color in zip(rows, row_colors):
            length = seg_lengths[color]
            starts = list(range(0, inp.shape[1] - length + 1))
            random.shuffle(starts)
            start = starts[0]
            inp[r, start : start + length] = color
        if ok and not np.array_equal(inp, out):
            return inp
    raise RuntimeError("failed to scatter segments")


def _validate_semantics(inp: np.ndarray) -> bool:
    stripe_row = _find_stripe_row(inp)
    tile_bbox = _find_tile_region(inp, stripe_row)
    coarse = _collapse_tile_region(inp, tile_bbox)
    freq = _color_frequencies(coarse)
    tile_bottom = tile_bbox[2]
    segments = _extract_segments(inp, tile_bottom + 1, stripe_row)

    seg_colors = [seg["color"] for seg in segments]
    seg_lengths = {seg["color"]: seg["length"] for seg in segments}
    if len({freq[color] for color in seg_colors}) != len(seg_colors):
        return False

    freq_order = _frequency_order(seg_colors, freq)
    len_order = [color for color, _length in sorted(seg_lengths.items(), key=lambda item: item[1])]
    if freq_order == len_order:
        return False

    upper = inp[tile_bbox[0] : tile_bbox[2] + 1, tile_bbox[1] : tile_bbox[3] + 1]
    raw_counts = _raw_upper_color_counts(upper)
    raw_order = [color for color in sorted(seg_colors, key=lambda color: raw_counts[color])]
    if raw_order == freq_order:
        return False

    bg_tiles = int(np.count_nonzero(coarse == BLACK))
    if bg_tiles < 1:
        return False

    upper_blocks = upper.reshape(coarse.shape[0], BLOCK, coarse.shape[1], BLOCK).transpose(0, 2, 1, 3)
    colored_with_noise = 0
    bg_with_noise = 0
    for r in range(coarse.shape[0]):
        for c in range(coarse.shape[1]):
            block = upper_blocks[r, c]
            grey_cells = int(np.count_nonzero(block == GREY))
            if coarse[r, c] == BLACK and grey_cells > 0:
                bg_with_noise += 1
            if coarse[r, c] != BLACK and grey_cells > 0:
                colored_with_noise += 1
    if colored_with_noise < 1 or bg_with_noise < 1:
        return False

    out = main(inp)
    if np.array_equal(inp, out):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_segments = random.randint(2, 4)
    n_colors = random.randint(n_segments, len(TILE_COLORS))
    coarse_rows, coarse_cols = _sample_dimensions(n_colors)
    colors = random.sample(TILE_COLORS, n_colors)

    counts = _sample_counts(n_colors, coarse_rows * coarse_cols)
    color_counts = {color: count for color, count in zip(colors, counts)}
    coarse = _sample_coarse_grid(coarse_rows, coarse_cols, color_counts)

    critical_high, critical_low = colors[0], colors[1]
    positions_by_color = {color: list(map(tuple, np.argwhere(coarse == color))) for color in colors}
    bg_positions = list(map(tuple, np.argwhere(coarse == BLACK)))
    if not bg_positions:
        return None

    noise_plan = {(r, c): 0 for r in range(coarse_rows) for c in range(coarse_cols)}
    for pos in positions_by_color[critical_high]:
        noise_plan[pos] = 1
    bg_noise_pos = random.choice(bg_positions)
    noise_plan[bg_noise_pos] = random.choice((1, 2))

    other_colored = [pos for color in colors[2:] for pos in positions_by_color[color]]
    random.shuffle(other_colored)
    for pos in other_colored[: random.randint(0, len(other_colored))]:
        noise_plan[pos] = random.choice((0, 1))
    if not any(noise_plan[pos] > 0 for pos in other_colored) and other_colored:
        noise_plan[random.choice(other_colored)] = 1

    upper = _render_upper_grid(coarse, noise_plan)
    raw_counts = _raw_upper_color_counts(upper)
    if raw_counts[critical_high] >= raw_counts[critical_low]:
        return None

    seg_colors = [critical_high, critical_low]
    remaining = [color for color in colors if color not in seg_colors]
    if n_segments > 2:
        seg_colors.extend(random.sample(remaining, n_segments - 2))
    random.shuffle(seg_colors)

    top_pad = random.randint(0, 1)
    left_pad = random.randint(0, 2)
    right_pad = random.randint(0, 2)
    width = upper.shape[1] + left_pad + right_pad

    seg_lengths = _choose_segment_lengths(width, seg_colors, color_counts)
    freq_order = _frequency_order(seg_colors, color_counts)
    len_order = [color for color, _length in sorted(seg_lengths.items(), key=lambda item: item[1])]
    if freq_order == len_order:
        return None

    sep_gap = random.randint(1, 2)
    scatter_extra = random.randint(1, 2)
    lower_rows = 2 * n_segments + scatter_extra
    height = top_pad + upper.shape[0] + sep_gap + lower_rows + 1
    if height > 30 or width > 30:
        return None

    out = _build_output(
        upper=upper,
        top_pad=top_pad,
        left_pad=left_pad,
        width=width,
        height=height,
        seg_lengths=seg_lengths,
        seg_colors=seg_colors,
        freq=color_counts,
    )
    tile_bottom = top_pad + upper.shape[0] - 1
    inp = _scatter_input(out, tile_bottom, seg_lengths, seg_colors, color_counts)

    if not _validate_semantics(inp):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
